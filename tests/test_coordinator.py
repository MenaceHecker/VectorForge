"""Tests for the query coordinator (Phase 5, Day 31-32).

The important guarantees:
  - writes are routed to one shard by id, and that routing is stable,
  - a vector is findable through the coordinator no matter which shard stores
    it (search fans out to all shards), and
  - the merged result matches what a single index over all the data would
    return.

Most tests use LocalShardClient for speed and determinism. One end-to-end test
drives real gRPC servers to prove the wire path, and skips itself if the stubs
have not been generated.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import numpy as np
import pytest

from vectorforge.brute_force import BruteForceIndex
from vectorforge.coordinator import Coordinator, LocalShardClient
from vectorforge.hnsw import HNSWIndex

DIM = 32


def _local_cluster(n_shards: int = 3) -> Coordinator:
    shards = {
        f"shard-{i}": LocalShardClient(HNSWIndex(dim=DIM, M=16, ef_construction=200))
        for i in range(n_shards)
    }
    return Coordinator(shards)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_index_routes_by_id_and_is_stable() -> None:
    coord = _local_cluster()
    rng = _rng(1)
    for i in range(200):
        shard = coord.index(str(i), rng.random(DIM).astype(np.float32))
        # The reported owner must match what the ring says, every time.
        assert shard == coord.owner_of(str(i))


def test_index_spreads_data_across_all_shards() -> None:
    coord = _local_cluster(3)
    rng = _rng(2)
    for i in range(600):
        coord.index(str(i), rng.random(DIM).astype(np.float32))

    sizes = {
        sid: len(client.index_obj)
        for sid, client in coord._shards.items()
    }
    assert sum(sizes.values()) == 600
    # No shard should be empty with 600 keys over 3 shards.
    assert all(size > 0 for size in sizes.values())


def test_empty_cluster_search_returns_empty() -> None:
    coord = Coordinator({})
    assert coord.search(np.zeros(DIM, dtype=np.float32), k=5) == []


def test_index_into_empty_cluster_raises() -> None:
    coord = Coordinator({})
    with pytest.raises(RuntimeError, match="no shards"):
        coord.index("a", np.zeros(DIM, dtype=np.float32))


# ---------------------------------------------------------------------------
# Fan-out search correctness
# ---------------------------------------------------------------------------


def test_vector_is_findable_regardless_of_owning_shard() -> None:
    coord = _local_cluster()
    rng = _rng(3)
    vectors = {str(i): rng.random(DIM).astype(np.float32) for i in range(300)}
    for vid, vec in vectors.items():
        coord.index(vid, vec)

    # Querying with an indexed vector must return it as the nearest, even though
    # the coordinator does not know which shard holds it.
    for vid in ("7", "150", "299"):
        top = coord.search(vectors[vid], k=1)
        assert top[0][0] == vid
        assert top[0][1] == pytest.approx(0.0, abs=1e-5)


def test_merged_results_match_single_index_ground_truth() -> None:
    """Coordinator top-k over sharded data should match a brute-force index
    built over the same vectors."""
    coord = _local_cluster(3)
    brute = BruteForceIndex(dim=DIM)
    rng = _rng(4)
    for i in range(400):
        vec = rng.random(DIM).astype(np.float32)
        coord.index(str(i), vec)
        brute.add(str(i), vec)

    k = 10
    queries = [rng.random(DIM).astype(np.float32) for _ in range(30)]
    hits = 0
    for q in queries:
        truth = {vid for vid, _ in brute.search(q, k=k)}
        got = {vid for vid, _ in coord.search(q, k=k)}
        hits += len(truth & got)
    recall = hits / (len(queries) * k)
    assert recall >= 0.9


def test_results_are_sorted_and_capped_at_k() -> None:
    coord = _local_cluster()
    rng = _rng(5)
    for i in range(300):
        coord.index(str(i), rng.random(DIM).astype(np.float32))

    results = coord.search(rng.random(DIM).astype(np.float32), k=10)
    assert len(results) == 10
    distances = [d for _, d in results]
    assert distances == sorted(distances)


def test_delete_through_coordinator_removes_from_results() -> None:
    coord = _local_cluster()
    rng = _rng(6)
    vectors = {str(i): rng.random(DIM).astype(np.float32) for i in range(200)}
    for vid, vec in vectors.items():
        coord.index(vid, vec)

    assert coord.delete("42") is True
    got = {vid for vid, _ in coord.search(vectors["42"], k=10)}
    assert "42" not in got


def test_metadata_filter_fans_out() -> None:
    coord = _local_cluster()
    rng = _rng(7)
    for i in range(300):
        coord.index(str(i), rng.random(DIM).astype(np.float32), metadata={"group": i % 3})

    results = coord.search(rng.random(DIM).astype(np.float32), k=10, filter={"group": 1})
    assert results
    for vid, _ in results:
        assert int(vid) % 3 == 1


# ---------------------------------------------------------------------------
# End-to-end over real gRPC (skipped until stubs exist)
# ---------------------------------------------------------------------------

pytest.importorskip("vectorforge.vectorforge_pb2", reason="run protoc to generate stubs")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def grpc_cluster() -> Iterator[Coordinator]:
    from vectorforge.coordinator import GrpcShardClient
    from vectorforge.grpc_server import serve

    servers = []
    clients = {}
    for i in range(2):
        port = _free_port()
        servers.append(serve(HNSWIndex(dim=DIM, M=8, ef_construction=50), port=port))
        clients[f"shard-{i}"] = GrpcShardClient(f"localhost:{port}")

    coord = Coordinator(clients)
    try:
        yield coord
    finally:
        coord.close()
        for client in clients.values():
            client.close()
        for server in servers:
            server.stop(grace=None)


def test_grpc_fanout_end_to_end(grpc_cluster: Coordinator) -> None:
    rng = _rng(11)
    vectors = {str(i): rng.random(DIM).astype(np.float32) for i in range(120)}
    for vid, vec in vectors.items():
        grpc_cluster.index(vid, vec)

    # A vector indexed on whichever shard owns it must come back via fan-out.
    top = grpc_cluster.search(vectors["55"], k=1)
    assert top[0][0] == "55"
    assert top[0][1] == pytest.approx(0.0, abs=1e-4)
