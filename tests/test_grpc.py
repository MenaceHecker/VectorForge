"""Integration tests for the gRPC servicer (Phase 3, Day 20–21).

These run a real in-process gRPC server over a loopback socket and drive it
through a generated client stub, so they exercise serialization, the Struct
metadata round-trip, and the status-code mapping end to end.

The generated stubs (``vectorforge_pb2`` / ``vectorforge_pb2_grpc``) are not
checked in; ``pytest.importorskip`` skips this file cleanly until codegen has
run (see the README for the protoc command).
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import grpc
import numpy as np
import pytest

pytest.importorskip("vectorforge.vectorforge_pb2", reason="run protoc to generate stubs")

from vectorforge import vectorforge_pb2 as pb  # noqa: E402
from vectorforge import vectorforge_pb2_grpc as pb_grpc  # noqa: E402
from vectorforge.grpc_server import serve  # noqa: E402
from vectorforge.hnsw import HNSWIndex  # noqa: E402

DIM = 8


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture()
def stub() -> Iterator[pb_grpc.VectorForgeStub]:
    port = _free_port()
    server = serve(HNSWIndex(dim=DIM, M=4, ef_construction=20), port=port)
    channel = grpc.insecure_channel(f"localhost:{port}")
    try:
        yield pb_grpc.VectorForgeStub(channel)
    finally:
        channel.close()
        server.stop(grace=None)


def _vec(seed: int) -> list[float]:
    return np.random.default_rng(seed).random(DIM).tolist()


# ---------------------------------------------------------------------------
# Index / Health
# ---------------------------------------------------------------------------


def test_index_and_health(stub: pb_grpc.VectorForgeStub) -> None:
    assert stub.Health(pb.HealthRequest()).size == 0
    resp = stub.Index(pb.IndexRequest(id="a", vector=_vec(1)))
    assert resp.id == "a" and resp.status == "indexed"

    health = stub.Health(pb.HealthRequest())
    assert health.size == 1
    assert health.dim == DIM
    assert health.status == "ok"


def test_duplicate_id_aborts_already_exists(stub: pb_grpc.VectorForgeStub) -> None:
    stub.Index(pb.IndexRequest(id="a", vector=_vec(1)))
    with pytest.raises(grpc.RpcError) as exc:
        stub.Index(pb.IndexRequest(id="a", vector=_vec(2)))
    assert exc.value.code() == grpc.StatusCode.ALREADY_EXISTS


def test_wrong_dimensionality_aborts_invalid_argument(stub: pb_grpc.VectorForgeStub) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        stub.Index(pb.IndexRequest(id="a", vector=[1.0, 2.0]))
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_returns_sorted_neighbors(stub: pb_grpc.VectorForgeStub) -> None:
    for i in range(20):
        stub.Index(pb.IndexRequest(id=str(i), vector=_vec(i)))
    resp = stub.Search(pb.SearchRequest(vector=_vec(0), k=5))
    assert len(resp.results) == 5
    distances = [r.distance for r in resp.results]
    assert distances == sorted(distances)
    assert resp.results[0].id == "0"


def test_search_metadata_filter(stub: pb_grpc.VectorForgeStub) -> None:
    for i in range(30):
        req = pb.IndexRequest(id=str(i), vector=_vec(i))
        req.metadata.update({"group": i % 3})
        stub.Index(req)

    search = pb.SearchRequest(vector=_vec(0), k=10)
    search.filter.update({"group": 1})
    resp = stub.Search(search)

    ids = [r.id for r in resp.results]
    assert ids and all(int(i) % 3 == 1 for i in ids)


def test_search_honours_optional_ef(stub: pb_grpc.VectorForgeStub) -> None:
    for i in range(50):
        stub.Index(pb.IndexRequest(id=str(i), vector=_vec(i)))
    # ef is a proto3 optional; setting it must be accepted and change nothing
    # about the result contract (still <= k, still sorted).
    resp = stub.Search(pb.SearchRequest(vector=_vec(0), k=10, ef=100))
    assert len(resp.results) == 10


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_reports_flag_and_removes(stub: pb_grpc.VectorForgeStub) -> None:
    for i in range(10):
        stub.Index(pb.IndexRequest(id=str(i), vector=_vec(i)))

    resp = stub.Delete(pb.DeleteRequest(id="0"))
    assert resp.deleted is True

    hits = stub.Search(pb.SearchRequest(vector=_vec(0), k=10))
    assert "0" not in [r.id for r in hits.results]


def test_delete_unknown_id_reports_false(stub: pb_grpc.VectorForgeStub) -> None:
    resp = stub.Delete(pb.DeleteRequest(id="ghost"))
    assert resp.deleted is False
