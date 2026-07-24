"""Tests for metadata attachment and predicate filtering.

Covers:
1. add() accepts metadata; get_metadata reads it back (as a copy)
2. get_metadata returns None when absent, raises KeyError when id unknown
3. Stored metadata is isolated from later caller-side mutation
4. search(predicate=...) returns only matching vectors, still distance-sorted
5. A predicate matching nothing returns []
6. Filtered search matches brute-force ground truth over the same subset
7. Filtering does not exceed k, and can return fewer than k when scarce
8. Metadata survives a persistence round-trip and filtered results are identical
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vectorforge.brute_force import BruteForceIndex
from vectorforge.hnsw import HNSWIndex
from vectorforge.persistence import load, save

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_path_file(tmp_path: Path) -> Path:
    return tmp_path / "index.vfidx"


def _build_labelled_index(
    n: int = 500,
    dim: int = 32,
    n_groups: int = 5,
    seed: int = 42,
) -> tuple[HNSWIndex, np.random.Generator]:
    """Build an index where each vector carries a ``group`` in 0..n_groups-1."""
    rng = np.random.default_rng(seed)
    idx = HNSWIndex(dim=dim, M=16, ef_construction=100)
    for i in range(n):
        idx.add(
            str(i),
            rng.random(dim).astype(np.float32),
            metadata={"group": i % n_groups, "even": i % 2 == 0},
        )
    return idx, rng


# ---------------------------------------------------------------------------
# Metadata storage / accessors
# ---------------------------------------------------------------------------


def test_get_metadata_roundtrips_value() -> None:
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    idx.add("v0", np.zeros(4, dtype=np.float32), metadata={"color": "red", "size": 3})
    assert idx.get_metadata("v0") == {"color": "red", "size": 3}


def test_get_metadata_none_when_absent() -> None:
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    idx.add("v0", np.zeros(4, dtype=np.float32))
    assert idx.get_metadata("v0") is None


def test_get_metadata_unknown_id_raises() -> None:
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    with pytest.raises(KeyError):
        idx.get_metadata("nope")


def test_metadata_isolated_from_caller_mutation() -> None:
    """A caller mutating the dict it passed must not change the stored copy."""
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    meta = {"color": "red"}
    idx.add("v0", np.zeros(4, dtype=np.float32), metadata=meta)
    meta["color"] = "blue"
    assert idx.get_metadata("v0") == {"color": "red"}


def test_get_metadata_returns_copy() -> None:
    """Mutating the returned dict must not corrupt index state."""
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    idx.add("v0", np.zeros(4, dtype=np.float32), metadata={"color": "red"})
    returned = idx.get_metadata("v0")
    returned["color"] = "green"
    assert idx.get_metadata("v0") == {"color": "red"}


# ---------------------------------------------------------------------------
# Predicate filtering
# ---------------------------------------------------------------------------


def test_search_predicate_returns_only_matches() -> None:
    idx, rng = _build_labelled_index()
    query = rng.random(32).astype(np.float32)
    results = idx.search(query, k=10, predicate=lambda m: m["group"] == 0)
    for vid, _ in results:
        assert idx.get_metadata(vid)["group"] == 0


def test_search_predicate_still_distance_sorted() -> None:
    idx, rng = _build_labelled_index()
    query = rng.random(32).astype(np.float32)
    results = idx.search(query, k=10, predicate=lambda m: m["even"])
    distances = [d for _, d in results]
    assert distances == sorted(distances)


def test_search_predicate_matching_nothing_returns_empty() -> None:
    idx, rng = _build_labelled_index()
    query = rng.random(32).astype(np.float32)
    results = idx.search(query, k=10, predicate=lambda m: m["group"] == 999)
    assert results == []


def test_search_predicate_never_exceeds_k() -> None:
    idx, rng = _build_labelled_index()
    query = rng.random(32).astype(np.float32)
    results = idx.search(query, k=5, predicate=lambda m: m["even"])
    assert len(results) <= 5


def test_predicate_sees_empty_dict_for_missing_metadata() -> None:
    """Nodes added without metadata must not crash a predicate; they see {}."""
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    idx.add("a", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    idx.add("b", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), metadata={"keep": True})

    results = idx.search(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        k=2,
        predicate=lambda m: m.get("keep", False),
    )
    ids = {vid for vid, _ in results}
    assert ids == {"b"}


def test_no_predicate_returns_all_nearest() -> None:
    """Omitting the predicate must behave exactly like an unfiltered search."""
    idx, rng = _build_labelled_index()
    query = rng.random(32).astype(np.float32)
    assert len(idx.search(query, k=10)) == 10


# ---------------------------------------------------------------------------
# Recall of filtered search vs brute-force ground truth over the subset
# ---------------------------------------------------------------------------


def test_filtered_recall_matches_brute_force_subset() -> None:
    """Filtered HNSW search should recover the true nearest neighbours that
    satisfy the predicate, measured against a brute-force index built over
    only the matching vectors."""
    dim, n, k = 48, 800, 10
    rng = np.random.default_rng(7)

    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=200)
    brute = BruteForceIndex(dim=dim)  # ground truth over group==0 only
    for i in range(n):
        v = rng.random(dim).astype(np.float32)
        group = i % 4
        hnsw.add(str(i), v, metadata={"group": group})
        if group == 0:
            brute.add(str(i), v)

    queries = [rng.random(dim).astype(np.float32) for _ in range(30)]
    hits = 0
    for q in queries:
        true_ids = {vid for vid, _ in brute.search(q, k=k)}
        approx_ids = {
            vid for vid, _ in hnsw.search(q, k=k, predicate=lambda m: m["group"] == 0)
        }
        hits += len(true_ids & approx_ids)
    recall = hits / (len(queries) * k)

    print(f"\nFiltered recall@{k} (group==0, 1/4 of {n}): {recall:.3f}")
    assert recall >= 0.90, f"Expected filtered recall >= 0.90, got {recall:.3f}"


def test_filter_interacts_with_tombstone() -> None:
    """A deleted vector must not be returned even if it matches the predicate."""
    idx, rng = _build_labelled_index(n=200)
    query = rng.random(32).astype(np.float32)

    top = idx.search(query, k=1, predicate=lambda m: m["group"] == 0)[0][0]
    idx.delete(top)

    after = {vid for vid, _ in idx.search(query, k=10, predicate=lambda m: m["group"] == 0)}
    assert top not in after


# ---------------------------------------------------------------------------
# Persistence of metadata
# ---------------------------------------------------------------------------


def test_metadata_survives_roundtrip(tmp_path_file: Path) -> None:
    idx, _ = _build_labelled_index(n=200)
    save(idx, tmp_path_file)
    idx2 = load(tmp_path_file)

    for i in range(200):
        assert idx2.get_metadata(str(i)) == idx.get_metadata(str(i))


def test_filtered_search_identical_after_roundtrip(tmp_path_file: Path) -> None:
    idx, rng = _build_labelled_index(n=300)
    query = rng.random(32).astype(np.float32)
    before = idx.search(query, k=10, predicate=lambda m: m["group"] == 2)

    save(idx, tmp_path_file)
    after = load(tmp_path_file).search(query, k=10, predicate=lambda m: m["group"] == 2)

    assert before == after


def test_none_metadata_roundtrips_as_none(tmp_path_file: Path) -> None:
    idx = HNSWIndex(dim=8, M=4, ef_construction=10)
    for i in range(20):
        idx.add(str(i), np.full(8, float(i), dtype=np.float32))
    save(idx, tmp_path_file)
    idx2 = load(tmp_path_file)
    assert all(idx2.get_metadata(str(i)) is None for i in range(20))


def test_unicode_and_nested_metadata_roundtrips(tmp_path_file: Path) -> None:
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    meta = {"café": "☕", "tags": ["a", "b"], "nested": {"n": 1}, "score": 1.5}
    idx.add("v0", np.zeros(4, dtype=np.float32), metadata=meta)
    save(idx, tmp_path_file)
    assert load(tmp_path_file).get_metadata("v0") == meta
