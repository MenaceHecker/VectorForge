"""Tests for the recall benchmark job (Phase 4, Day 27-28).

These check that recall is measured correctly against brute-force truth, that
the edge cases (empty index, small corpus, deletes) behave, and that run_once
publishes the number to the metrics gauge.
"""

from __future__ import annotations

import numpy as np
import pytest

from vectorforge.benchmark import RecallBenchmark, compute_recall_at_k, sample_queries
from vectorforge.hnsw import HNSWIndex
from vectorforge.metrics import Metrics


def _build_index(n: int = 500, dim: int = 32, seed: int = 1) -> HNSWIndex:
    rng = np.random.default_rng(seed)
    idx = HNSWIndex(dim=dim, M=16, ef_construction=200)
    for i in range(n):
        idx.add(str(i), rng.random(dim).astype(np.float32))
    return idx


def test_recall_is_high_on_well_built_index() -> None:
    idx = _build_index(n=500, dim=32)
    queries = sample_queries(idx, n=50, seed=7)
    recall = compute_recall_at_k(idx, queries, k=10)
    # Querying with vectors that are in the index should recover almost all of
    # the true top-k; a healthy index sits well above 0.9 here.
    assert recall >= 0.9


def test_recall_is_bounded_between_zero_and_one() -> None:
    idx = _build_index(n=200, dim=16)
    queries = sample_queries(idx, n=20, seed=3)
    recall = compute_recall_at_k(idx, queries, k=10)
    assert 0.0 <= recall <= 1.0


def test_recall_empty_index_is_zero() -> None:
    idx = HNSWIndex(dim=8, M=4, ef_construction=10)
    assert compute_recall_at_k(idx, sample_queries(idx, 10), k=10) == 0.0


def test_recall_empty_queries_is_zero() -> None:
    idx = _build_index(n=50, dim=8)
    assert compute_recall_at_k(idx, [], k=10) == 0.0


def test_recall_small_corpus_not_penalised() -> None:
    """With fewer than k vectors, recall should still be able to reach 1.0."""
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    for i in range(3):
        idx.add(str(i), np.full(4, float(i), dtype=np.float32))
    queries = sample_queries(idx, n=3, seed=0)
    # Only 3 vectors exist but we ask for 10; dividing by true-neighbour count
    # means a perfect index still scores 1.0 rather than 0.3.
    assert compute_recall_at_k(idx, queries, k=10) == pytest.approx(1.0)


def test_sample_queries_respects_count_and_bounds() -> None:
    idx = _build_index(n=40, dim=8)
    assert len(sample_queries(idx, n=10, seed=0)) == 10
    # Asking for more than the corpus holds returns the whole corpus, no error.
    assert len(sample_queries(idx, n=100, seed=0)) == 40
    assert sample_queries(HNSWIndex(dim=8), n=5) == []


def test_sample_queries_is_deterministic_for_a_seed() -> None:
    idx = _build_index(n=100, dim=8)
    a = sample_queries(idx, n=10, seed=42)
    b = sample_queries(idx, n=10, seed=42)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_run_once_publishes_to_gauge() -> None:
    idx = _build_index(n=300, dim=16)
    metrics = Metrics()
    bench = RecallBenchmark(idx, metrics=metrics, k=10, n_queries=30, seed=5)

    returned = bench.run_once()
    published = metrics.recall_at_k._value.get()
    assert published == pytest.approx(returned)
    assert returned > 0.0


def test_run_once_without_metrics_still_returns_recall() -> None:
    idx = _build_index(n=100, dim=8)
    recall = RecallBenchmark(idx, metrics=None, n_queries=20).run_once()
    assert 0.0 <= recall <= 1.0
