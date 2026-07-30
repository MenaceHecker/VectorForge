"""Recall-vs-latency sweep for the README benchmark table (Phase 6, Day 40).

Builds an HNSW index and a brute-force ground truth over the same vectors, then
sweeps the search beam width `ef`. For each `ef` it measures recall@k against the
ground truth and the per-query latency percentiles. Higher `ef` buys recall at
the cost of latency, which is the tradeoff the README chart shows.

    python scripts/run_benchmark.py --n 10000 --dim 128 --out bench.json

The numbers are honest but local: synthetic vectors, single node, one machine.
The resume-grade numbers come from the real embeddings dataset on the sharded
deployment, which is why the README labels these as a local single-node run.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from vectorforge.brute_force import BruteForceIndex
from vectorforge.hnsw import HNSWIndex


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def run(
    n: int, dim: int, q: int, k: int, ef_values: list[int], seed: int
) -> list[dict]:
    rng = np.random.default_rng(seed)
    data = rng.random((n, dim)).astype(np.float32)

    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=200)
    brute = BruteForceIndex(dim=dim)
    for i in range(n):
        hnsw.add(str(i), data[i])
        brute.add(str(i), data[i])

    queries = rng.random((q, dim)).astype(np.float32)
    truth = [{vid for vid, _ in brute.search(query, k=k)} for query in queries]

    rows: list[dict] = []
    for ef in ef_values:
        hits = 0
        latencies: list[float] = []
        for query, true_ids in zip(queries, truth, strict=True):
            start = time.perf_counter()
            got = {vid for vid, _ in hnsw.search(query, k=k, ef=ef)}
            latencies.append((time.perf_counter() - start) * 1000.0)
            hits += len(got & true_ids)
        rows.append(
            {
                "ef": ef,
                "recall": round(hits / (q * k), 4),
                "p50_ms": round(_pct(latencies, 50), 3),
                "p95_ms": round(_pct(latencies, 95), 3),
                "p99_ms": round(_pct(latencies, 99), 3),
            }
        )
        print(
            f"ef={ef:>4}  recall@{k}={rows[-1]['recall']:.3f}  "
            f"p50={rows[-1]['p50_ms']:.2f}ms  p95={rows[-1]['p95_ms']:.2f}ms  "
            f"p99={rows[-1]['p99_ms']:.2f}ms"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recall vs latency sweep.")
    parser.add_argument("--n", type=int, default=10000, help="Number of vectors.")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimensionality.")
    parser.add_argument("--q", type=int, default=200, help="Number of queries.")
    parser.add_argument("-k", type=int, default=10, help="Neighbours per query.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    ef_values = [10, 20, 40, 80, 160, 320]
    print(f"building index: n={args.n}, dim={args.dim}, k={args.k}, queries={args.q}")
    rows = run(args.n, args.dim, args.q, args.k, ef_values, args.seed)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"config": vars(args), "rows": rows}, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
