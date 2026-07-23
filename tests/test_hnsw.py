from __future__ import annotations
 
import numpy as np
import pytest
 
from vectorforge.brute_force import BruteForceIndex
from vectorforge.hnsw import HNSWIndex
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
 
def recall_at_k(
    hnsw: HNSWIndex,
    brute: BruteForceIndex,
    queries: list[np.ndarray],
    k: int,
) -> float:
    """Compute mean recall@k across *queries*.
 
    recall@k for a single query = |HNSW top-k ∩ brute-force top-k| / k
    """
    hits = 0
    for query in queries:
        true_ids = {vid for vid, _ in brute.search(query, k=k)}
        approx_ids = {vid for vid, _ in hnsw.search(query, k=k)}
        hits += len(true_ids & approx_ids)
    return hits / (len(queries) * k)
 
 
# ---------------------------------------------------------------------------
# Construction and parameter validation
# ---------------------------------------------------------------------------
 
 
def test_invalid_dim_raises():
    with pytest.raises(ValueError, match="dim must be >= 1"):
        HNSWIndex(dim=0)
 
 
def test_invalid_M_raises():
    with pytest.raises(ValueError, match="M must be >= 2"):
        HNSWIndex(dim=4, M=1)
 
 
def test_ef_construction_less_than_M_raises():
    with pytest.raises(ValueError, match="ef_construction"):
        HNSWIndex(dim=4, M=16, ef_construction=8)
 
 
def test_starts_empty():
    idx = HNSWIndex(dim=8)
    assert len(idx) == 0
 
 
# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------
 
 
def test_add_increases_size():
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    idx.add("v0", np.zeros(4, dtype=np.float32))
    assert len(idx) == 1
 
 
def test_add_wrong_shape_raises():
    idx = HNSWIndex(dim=4)
    with pytest.raises(ValueError, match="Expected vector of shape"):
        idx.add("v0", np.zeros(8, dtype=np.float32))
 
 
def test_add_duplicate_id_raises():
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    v = np.zeros(4, dtype=np.float32)
    idx.add("v0", v)
    with pytest.raises(ValueError, match="already exists"):
        idx.add("v0", v)
 
 
def test_add_many_vectors():
    rng = np.random.default_rng(0)
    idx = HNSWIndex(dim=32, M=8, ef_construction=50)
    for i in range(200):
        idx.add(str(i), rng.random(32).astype(np.float32))
    assert len(idx) == 200
 
 
# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
 
 
def test_search_empty_index_returns_empty():
    idx = HNSWIndex(dim=4)
    result = idx.search(np.zeros(4, dtype=np.float32), k=5)
    assert result == []
 
 
def test_search_returns_k_results():
    rng = np.random.default_rng(1)
    idx = HNSWIndex(dim=16, M=4, ef_construction=20)
    for i in range(50):
        idx.add(str(i), rng.random(16).astype(np.float32))
    results = idx.search(rng.random(16).astype(np.float32), k=5)
    assert len(results) == 5
 
 
def test_search_results_sorted_ascending():
    rng = np.random.default_rng(2)
    idx = HNSWIndex(dim=16, M=8, ef_construction=50)
    for i in range(100):
        idx.add(str(i), rng.random(16).astype(np.float32))
    results = idx.search(rng.random(16).astype(np.float32), k=10)
    distances = [d for _, d in results]
    assert distances == sorted(distances)
 
 
def test_search_k_larger_than_corpus_returns_all():
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    for i in range(5):
        idx.add(str(i), np.full(4, float(i), dtype=np.float32))
    results = idx.search(np.zeros(4, dtype=np.float32), k=100)
    assert len(results) == 5
 
 
def test_search_finds_exact_neighbour_in_small_index():
    """With a tiny, well-separated corpus HNSW must return the exact nearest."""
    idx = HNSWIndex(dim=2, M=4, ef_construction=20)
    idx.add("origin", np.array([0.0, 0.0], dtype=np.float32))
    idx.add("far", np.array([100.0, 100.0], dtype=np.float32))
    idx.add("mid", np.array([5.0, 5.0], dtype=np.float32))
 
    results = idx.search(np.array([0.1, 0.0], dtype=np.float32), k=1)
    assert results[0][0] == "origin"
 
 
def test_search_invalid_k_raises():
    idx = HNSWIndex(dim=4)
    idx.add("v0", np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="k must be >= 1"):
        idx.search(np.zeros(4, dtype=np.float32), k=0)
 
 
def test_search_wrong_query_shape_raises():
    idx = HNSWIndex(dim=4)
    with pytest.raises(ValueError, match="Expected query of shape"):
        idx.search(np.zeros(8, dtype=np.float32), k=1)
 
 
# ---------------------------------------------------------------------------
# Delete (tombstone)
# ---------------------------------------------------------------------------
 
 
def test_delete_reduces_len():
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    idx.add("v0", np.zeros(4, dtype=np.float32))
    assert idx.delete("v0") is True
    assert len(idx) == 0
 
 
def test_delete_missing_returns_false():
    idx = HNSWIndex(dim=4)
    assert idx.delete("nonexistent") is False
 
 
def test_deleted_vector_not_in_results():
    """The tombstoned vector must not appear in search results."""
    rng = np.random.default_rng(7)
    idx = HNSWIndex(dim=32, M=8, ef_construction=50)
    for i in range(100):
        idx.add(str(i), rng.random(32).astype(np.float32))
 
    # Find the true nearest neighbour, tombstone it, confirm it disappears.
    query = rng.random(32).astype(np.float32)
    top_before = idx.search(query, k=1)[0][0]
    idx.delete(top_before)
 
    top_after_ids = {vid for vid, _ in idx.search(query, k=5)}
    assert top_before not in top_after_ids
 
 
# ---------------------------------------------------------------------------
# Recall@10 vs brute-force ground truth  
# ---------------------------------------------------------------------------
 
 
def test_recall_at_10_single_layer_above_threshold():
    """Single-layer HNSW must achieve >= 80% recall@10 on 5K random vectors.
 
    This is the Phase 1 milestone gate.  Phase 2 (multi-layer + tuned M /
    ef_construction) must push recall above 90% on 100K vectors.
    """
    rng = np.random.default_rng(42)
    dim = 128
    n_vectors = 5_000
    n_queries = 100
    k = 10
 
    # Build both indexes on the same corpus
    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=200)
    brute = BruteForceIndex(dim=dim)
 
    for i in range(n_vectors):
        v = rng.random(dim).astype(np.float32)
        hnsw.add(str(i), v)
        brute.add(str(i), v)
 
    queries = [rng.random(dim).astype(np.float32) for _ in range(n_queries)]
    recall = recall_at_k(hnsw, brute, queries, k=k)
 
    print(f"\nRecall@{k} (single-layer, {n_vectors} vectors, dim={dim}): {recall:.3f}")
    assert recall >= 0.80, f"Expected recall >= 0.80, got {recall:.3f}"
 
 
def test_recall_improves_with_higher_ef():
    """Searching with a larger ef must not decrease recall."""
    rng = np.random.default_rng(99)
    dim = 64
    n_vectors = 2_000
    k = 10
 
    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=200)
    brute = BruteForceIndex(dim=dim)
    for i in range(n_vectors):
        v = rng.random(dim).astype(np.float32)
        hnsw.add(str(i), v)
        brute.add(str(i), v)
 
    queries = [rng.random(dim).astype(np.float32) for _ in range(50)]
 
    recall_low_ef = recall_at_k(
        hnsw, brute, queries, k=k
    )
    # Re-run search with explicitly higher ef
    hits_high = 0
    for query in queries:
        true_ids = {vid for vid, _ in brute.search(query, k=k)}
        approx_ids = {vid for vid, _ in hnsw.search(query, k=k, ef=500)}
        hits_high += len(true_ids & approx_ids)
    recall_high_ef = hits_high / (len(queries) * k)
 
    print(f"\nRecall@{k} low-ef:  {recall_low_ef:.3f}")
    print(f"Recall@{k} high-ef: {recall_high_ef:.3f}")
    assert recall_high_ef >= recall_low_ef