"""Recall benchmark against brute-force ground truth (Phase 4, Day 27-28).

The whole point of an approximate index is that it trades a little recall for a
lot of speed, so the number that actually matters is how much recall we are
keeping. This module measures that by building an exact brute-force index over
the current live vectors and checking how much of its top-k the HNSW search
recovers.

Run periodically, it tracks recall drift as the index grows and as deletes pile
up, and it feeds the vectorforge_recall_at_k gauge so the Grafana dashboards
have a live quality signal.

It can run two ways:
  - In-process, driven by RecallBenchmark, updating a Metrics gauge. The API
    starts one of these on a timer when VECTORFORGE_BENCHMARK_INTERVAL is set.
  - From the command line against a saved index file, for a one-off recall
    check: `python -m vectorforge.benchmark path/to/index.vfidx`.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from numpy.typing import NDArray

from vectorforge.brute_force import BruteForceIndex
from vectorforge.hnsw import HNSWIndex
from vectorforge.metrics import Metrics


def compute_recall_at_k(
    index: HNSWIndex,
    queries: list[NDArray[np.float32]],
    k: int = 10,
    ef: int | None = None,
) -> float:
    """Mean recall@k of *index* over *queries*, against brute-force truth.

    Recall for one query is the fraction of the true top-k that HNSW also
    returns. When the corpus holds fewer than k vectors we divide by however
    many true neighbours actually exist, so a small index does not look like a
    recall failure. Returns 0.0 for an empty index or an empty query set.
    """
    live = list(index.live_items())
    if not live or not queries:
        return 0.0

    brute = BruteForceIndex(dim=index.dim)
    for vector_id, vector in live:
        brute.add(vector_id, vector)

    hits = 0
    total = 0
    for query in queries:
        true_ids = {vid for vid, _ in brute.search(query, k=k)}
        approx_ids = {vid for vid, _ in index.search(query, k=k, ef=ef)}
        hits += len(true_ids & approx_ids)
        total += len(true_ids)

    return hits / total if total else 0.0


def sample_queries(
    index: HNSWIndex, n: int, seed: int = 0
) -> list[NDArray[np.float32]]:
    """Pick *n* live vectors at random to use as query points.

    Sampling from the index itself means the benchmark needs no external query
    set and follows the data distribution as it drifts. If the index holds
    fewer than n vectors we use all of them.
    """
    live = [vec for _, vec in index.live_items()]
    if not live:
        return []
    rng = np.random.default_rng(seed)
    n = min(n, len(live))
    chosen = rng.choice(len(live), size=n, replace=False)
    return [live[i] for i in chosen]


class RecallBenchmark:
    """Runs the recall measurement and (optionally) publishes it to a gauge."""

    def __init__(
        self,
        index: HNSWIndex,
        metrics: Metrics | None = None,
        k: int = 10,
        n_queries: int = 100,
        seed: int = 0,
    ) -> None:
        self.index = index
        self.metrics = metrics
        self.k = k
        self.n_queries = n_queries
        self.seed = seed

    def run_once(self) -> float:
        """Measure recall once and, if a Metrics object was given, record it."""
        queries = sample_queries(self.index, self.n_queries, self.seed)
        recall = compute_recall_at_k(self.index, queries, k=self.k)
        if self.metrics is not None:
            self.metrics.set_recall(recall)
        return recall


def main(argv: list[str] | None = None) -> int:
    """Load a saved index and print its recall@k. Handy as a sidecar/CronJob."""
    parser = argparse.ArgumentParser(description="Measure HNSW recall@k.")
    parser.add_argument("index_path", help="Path to a .vfidx index file.")
    parser.add_argument("-k", type=int, default=10, help="Neighbours per query.")
    parser.add_argument(
        "-n", "--n-queries", type=int, default=100, help="Number of sampled queries."
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = parser.parse_args(argv)

    # Imported here so the module has no hard dependency on persistence for the
    # in-process path.
    from vectorforge.persistence import load

    index = load(args.index_path)
    recall = RecallBenchmark(
        index, k=args.k, n_queries=args.n_queries, seed=args.seed
    ).run_once()
    print(f"recall@{args.k} over {args.n_queries} queries: {recall:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
