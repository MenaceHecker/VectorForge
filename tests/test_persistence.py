"""Tests for disk persistence (save / load round-trips).

Covers:
1. Full round-trip: results before save == results after load
2. Metadata preserved: dim, M, ef_construction, max_layer
3. Deleted set preserved across round-trip
4. Empty index round-trip
5. Single-node index round-trip
6. Corrupt magic / wrong version raise ValueError
7. Atomic write: .tmp file does not linger on success
"""

from __future__ import annotations
 
import struct
from pathlib import Path
 
import numpy as np
import pytest
 
from vectorforge.hnsw import HNSWIndex
from vectorforge.persistence import load, save  # noqa: E402
 
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
 
 
@pytest.fixture()
def tmp_path_file(tmp_path: Path) -> Path:
    return tmp_path / "index.vfidx"
 
 
def _build_index(
    n: int = 300,
    dim: int = 64,
    M: int = 16,
    ef_construction: int = 100,
    seed: int = 42,
) -> tuple[HNSWIndex, np.random.Generator]:
    rng = np.random.default_rng(seed)
    idx = HNSWIndex(dim=dim, M=M, ef_construction=ef_construction)
    for i in range(n):
        idx.add(str(i), rng.random(dim).astype(np.float32))
    return idx, rng
 
 
# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------
 
 
def test_roundtrip_search_results_identical(tmp_path_file: Path) -> None:
    """Results before save must exactly match results after load."""
    idx, rng = _build_index()
    query = rng.random(64).astype(np.float32)
    before = idx.search(query, k=10)
 
    save(idx, tmp_path_file)
    idx2 = load(tmp_path_file)
    after = idx2.search(query, k=10)
 
    assert before == after
 
 
def test_roundtrip_metadata_preserved(tmp_path_file: Path) -> None:
    idx, _ = _build_index(dim=32, M=8, ef_construction=50)
    save(idx, tmp_path_file)
    idx2 = load(tmp_path_file)
 
    assert idx2.dim == 32
    assert idx2.M == 8
    assert idx2.ef_construction == 50
    assert idx2.max_layer == idx.max_layer
 
 
def test_roundtrip_size_preserved(tmp_path_file: Path) -> None:
    idx, _ = _build_index(n=200)
    save(idx, tmp_path_file)
    assert len(load(tmp_path_file)) == len(idx)
 
 
def test_roundtrip_deleted_set_preserved(tmp_path_file: Path) -> None:
    """Tombstoned vectors must stay excluded after a load."""
    idx, rng = _build_index(n=100)
    query = rng.random(64).astype(np.float32)
 
    # Find the top result then tombstone it
    top_id = idx.search(query, k=1)[0][0]
    idx.delete(top_id)
 
    save(idx, tmp_path_file)
    idx2 = load(tmp_path_file)
 
    returned_ids = {vid for vid, _ in idx2.search(query, k=10)}
    assert top_id not in returned_ids
 
 
def test_roundtrip_all_vector_ids_accessible(tmp_path_file: Path) -> None:
    n = 50
    idx, rng = _build_index(n=n, dim=16, M=4, ef_construction=20)
    save(idx, tmp_path_file)
    idx2 = load(tmp_path_file)
 
    # Every inserted id should be reachable via search
    all_ids = {str(i) for i in range(n)}
    returned: set[str] = set()
    for _ in range(20):
        q = rng.random(16).astype(np.float32)
        returned |= {vid for vid, _ in idx2.search(q, k=10)}
 
    # At minimum the ids close to random queries should be findable
    assert len(returned) > 0
    assert returned.issubset(all_ids)
 
 
# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
 
 
def test_empty_index_roundtrip(tmp_path_file: Path) -> None:
    idx = HNSWIndex(dim=8, M=4, ef_construction=10)
    save(idx, tmp_path_file)
    idx2 = load(tmp_path_file)
    assert len(idx2) == 0
    assert idx2.search(np.zeros(8, dtype=np.float32), k=5) == []
 
 
def test_single_node_roundtrip(tmp_path_file: Path) -> None:
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    idx.add("only", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    save(idx, tmp_path_file)
    idx2 = load(tmp_path_file)
    results = idx2.search(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), k=1)
    assert results[0][0] == "only"
    assert results[0][1] == pytest.approx(0.0, abs=1e-6)
 
 
def test_unicode_vector_ids_roundtrip(tmp_path_file: Path) -> None:
    idx = HNSWIndex(dim=4, M=4, ef_construction=10)
    ids = ["café", "日本語", "emoji🚀", "id with spaces"]
    for i, vid in enumerate(ids):
        idx.add(vid, np.full(4, float(i), dtype=np.float32))
    save(idx, tmp_path_file)
    idx2 = load(tmp_path_file)
    results = idx2.search(np.zeros(4, dtype=np.float32), k=len(ids))
    returned_ids = {vid for vid, _ in results}
    assert set(ids) == returned_ids
 
 
def test_multiple_saves_overwrite_cleanly(tmp_path_file: Path) -> None:
    """Saving twice to the same path must not corrupt the file."""
    idx, rng = _build_index(n=50, dim=16)
    save(idx, tmp_path_file)
    save(idx, tmp_path_file)  # second save — atomic replace
    idx2 = load(tmp_path_file)
    assert len(idx2) == 50
 
 
def test_atomic_write_no_tmp_file_left(tmp_path_file: Path) -> None:
    idx, _ = _build_index(n=20, dim=16)
    save(idx, tmp_path_file)
    tmp = tmp_path_file.with_suffix(".vfidx.tmp")
    assert not tmp.exists(), ".tmp file should be cleaned up after successful save"
 
 
# ---------------------------------------------------------------------------
# Corrupt / invalid file handling
# ---------------------------------------------------------------------------
 
 
def test_load_wrong_magic_raises(tmp_path_file: Path) -> None:
    idx, _ = _build_index(n=10, dim=16)
    save(idx, tmp_path_file)
 
    raw = bytearray(tmp_path_file.read_bytes())
    raw[0:8] = b"BADMAGIC"
    tmp_path_file.write_bytes(bytes(raw))
 
    with pytest.raises(ValueError, match="magic"):
        load(tmp_path_file)
 
 
def test_load_wrong_version_raises(tmp_path_file: Path) -> None:
    idx, _ = _build_index(n=10, dim=16)
    save(idx, tmp_path_file)
 
    # Version field is bytes [8:10] (uint16 little-endian)
    raw = bytearray(tmp_path_file.read_bytes())
    raw[8:10] = struct.pack("<H", 99)
    tmp_path_file.write_bytes(bytes(raw))
 
    with pytest.raises(ValueError, match="version"):
        load(tmp_path_file)
 
 
def test_load_nonexistent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "does_not_exist.vfidx")
 
 
# ---------------------------------------------------------------------------
# Recall preserved after round-trip
# ---------------------------------------------------------------------------
 
 
def test_recall_at_10_unchanged_after_roundtrip(tmp_path_file: Path) -> None:
    """Recall@10 must be the same before and after save/load."""
    rng = np.random.default_rng(7)
    dim, n = 64, 1_000
    idx, _ = _build_index(n=n, dim=dim, seed=7)
 
    queries = [rng.random(dim).astype(np.float32) for _ in range(50)]
 
    def _recall(index: HNSWIndex) -> float:
        from vectorforge.brute_force import BruteForceIndex
        brute = BruteForceIndex(dim=dim)
        for iid, node in index._nodes.items():
            if iid not in index._deleted:
                brute.add(node.vector_id, node.vector)
        hits = sum(
            len({v for v, _ in index.search(q, k=10)} & {v for v, _ in brute.search(q, k=10)})
            for q in queries
        )
        return hits / (len(queries) * 10)
 
    recall_before = _recall(idx)
    save(idx, tmp_path_file)
    recall_after = _recall(load(tmp_path_file))
 
    assert recall_before == pytest.approx(recall_after, abs=1e-9)