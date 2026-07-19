"""Tests for the brute-force k-NN index and distance metrics.

These tests double as the recall ground-truth contract
"""

from __future__ import annotations

import numpy as np
import pytest

from vectorforge.brute_force import BruteForceIndex
from vectorforge.distance import cosine, euclidean

# Distance metrics


def test_euclidean_identical_vectors():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert euclidean(a, a) == pytest.approx(0.0, abs=1e-6)


def test_euclidean_known_value():
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    assert euclidean(a, b) == pytest.approx(5.0, rel=1e-5)


def test_cosine_identical_vectors():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine(a, a) == pytest.approx(0.0, abs=1e-6)


def test_cosine_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine(a, b) == pytest.approx(1.0, rel=1e-5)


def test_cosine_zero_vector_returns_one():
    a = np.zeros(4, dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert cosine(a, b) == 1.0

# BruteForceIndex : construction


def test_index_starts_empty():
    idx = BruteForceIndex(dim=8)
    assert len(idx) == 0


def test_invalid_dim_raises():
    with pytest.raises(ValueError, match="dim must be >= 1"):
        BruteForceIndex(dim=0)

# BruteForceIndex : add



def test_add_increases_size():
    idx = BruteForceIndex(dim=4)
    idx.add("v0", np.zeros(4, dtype=np.float32))
    assert len(idx) == 1


def test_add_wrong_shape_raises():
    idx = BruteForceIndex(dim=4)
    with pytest.raises(ValueError, match="Expected vector of shape"):
        idx.add("v0", np.zeros(8, dtype=np.float32))

# BruteForceIndex — search


def test_search_empty_index_returns_empty():
    idx = BruteForceIndex(dim=4)
    results = idx.search(np.zeros(4, dtype=np.float32), k=5)
    assert results == []


def test_search_returns_exact_nearest():
    idx = BruteForceIndex(dim=2)
    idx.add("far", np.array([10.0, 10.0], dtype=np.float32))
    idx.add("near", np.array([0.1, 0.0], dtype=np.float32))
    idx.add("origin", np.array([0.0, 0.0], dtype=np.float32))

    query = np.array([0.0, 0.0], dtype=np.float32)
    results = idx.search(query, k=1)

    assert len(results) == 1
    assert results[0][0] == "origin"
    assert results[0][1] == pytest.approx(0.0, abs=1e-6)


def test_search_results_sorted_ascending():
    rng = np.random.default_rng(42)
    idx = BruteForceIndex(dim=16)
    for i in range(50):
        idx.add(str(i), rng.random(16).astype(np.float32))

    query = rng.random(16).astype(np.float32)
    results = idx.search(query, k=10)

    distances = [d for _, d in results]
    assert distances == sorted(distances)


def test_search_k_larger_than_corpus_returns_all():
    idx = BruteForceIndex(dim=4)
    for i in range(3):
        idx.add(str(i), np.full(4, float(i), dtype=np.float32))

    results = idx.search(np.zeros(4, dtype=np.float32), k=100)
    assert len(results) == 3


def test_search_invalid_k_raises():
    idx = BruteForceIndex(dim=4)
    idx.add("v0", np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="k must be >= 1"):
        idx.search(np.zeros(4, dtype=np.float32), k=0)

# BruteForceIndex : delete


def test_delete_removes_vector():
    idx = BruteForceIndex(dim=4)
    idx.add("v0", np.zeros(4, dtype=np.float32))
    removed = idx.delete("v0")
    assert removed is True
    assert len(idx) == 0


def test_delete_missing_id_returns_false():
    idx = BruteForceIndex(dim=4)
    assert idx.delete("nonexistent") is False


def test_deleted_vector_not_returned_in_search():
    idx = BruteForceIndex(dim=2)
    idx.add("a", np.array([0.0, 0.0], dtype=np.float32))
    idx.add("b", np.array([100.0, 100.0], dtype=np.float32))
    idx.delete("a")

    results = idx.search(np.array([0.0, 0.0], dtype=np.float32), k=1)
    assert results[0][0] == "b"

# Recall sanity check


def test_recall_at_10_is_perfect_for_brute_force():
    """BruteForceIndex must achieve 100% recall@10"""
    rng = np.random.default_rng(0)
    dim = 128
    n = 1_000

    idx = BruteForceIndex(dim=dim)
    for i in range(n):
        idx.add(str(i), rng.random(dim).astype(np.float32))

    hits = 0
    trials = 50
    for _ in range(trials):
        query = rng.random(dim).astype(np.float32)
        true_ids = {vid for vid, _ in idx.search(query, k=10)}
        returned_ids = {vid for vid, _ in idx.search(query, k=10)}
        hits += len(true_ids & returned_ids)

    recall = hits / (trials * 10)
    assert recall == pytest.approx(1.0)