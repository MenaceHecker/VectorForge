"""Integration tests for the coordinator REST service (Phase 5, Day 33).

Driven through TestClient against a Coordinator backed by in-process shards, so
they check the full HTTP path (routing, fan-out, error codes) without needing a
running gRPC cluster.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorforge.coordinator import Coordinator, LocalShardClient
from vectorforge.coordinator_api import create_coordinator_app
from vectorforge.hnsw import HNSWIndex

DIM = 8


@pytest.fixture()
def client() -> TestClient:
    shards = {
        f"shard-{i}": LocalShardClient(HNSWIndex(dim=DIM, M=4, ef_construction=20))
        for i in range(3)
    }
    return TestClient(create_coordinator_app(Coordinator(shards)))


def _vec(seed: int) -> list[float]:
    return np.random.default_rng(seed).random(DIM).tolist()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_lists_shards(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["shards"] == ["shard-0", "shard-1", "shard-2"]


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_index_reports_owning_shard(client: TestClient) -> None:
    resp = client.post("/index", json={"id": "a", "vector": _vec(1)})
    assert resp.status_code == 201
    assert resp.json()["shard"].startswith("shard-")


def test_index_duplicate_is_409(client: TestClient) -> None:
    client.post("/index", json={"id": "a", "vector": _vec(1)})
    resp = client.post("/index", json={"id": "a", "vector": _vec(2)})
    assert resp.status_code == 409


def test_index_wrong_dimensionality_is_400(client: TestClient) -> None:
    resp = client.post("/index", json={"id": "a", "vector": [1.0, 2.0]})
    assert resp.status_code == 400


def test_index_missing_field_is_422(client: TestClient) -> None:
    resp = client.post("/index", json={"vector": _vec(1)})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Search fan-out
# ---------------------------------------------------------------------------


def test_search_finds_vector_from_any_shard(client: TestClient) -> None:
    vectors = {str(i): _vec(i) for i in range(60)}
    for vid, vec in vectors.items():
        client.post("/index", json={"id": vid, "vector": vec})

    resp = client.post("/search", json={"vector": vectors["37"], "k": 1})
    assert resp.status_code == 200
    top = resp.json()["results"][0]
    assert top["id"] == "37"
    assert top["distance"] == pytest.approx(0.0, abs=1e-5)


def test_search_results_sorted_and_capped(client: TestClient) -> None:
    for i in range(60):
        client.post("/index", json={"id": str(i), "vector": _vec(i)})
    results = client.post("/search", json={"vector": _vec(0), "k": 10}).json()["results"]
    assert len(results) == 10
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


def test_search_metadata_filter(client: TestClient) -> None:
    for i in range(60):
        client.post(
            "/index",
            json={"id": str(i), "vector": _vec(i), "metadata": {"group": i % 3}},
        )
    resp = client.post(
        "/search", json={"vector": _vec(0), "k": 10, "filter": {"group": 2}}
    )
    ids = [r["id"] for r in resp.json()["results"]]
    assert ids and all(int(i) % 3 == 2 for i in ids)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_removes_from_fanout_results(client: TestClient) -> None:
    vectors = {str(i): _vec(i) for i in range(30)}
    for vid, vec in vectors.items():
        client.post("/index", json={"id": vid, "vector": vec})

    assert client.delete("/vectors/5").status_code == 200
    hits = client.post("/search", json={"vector": vectors["5"], "k": 10}).json()["results"]
    assert "5" not in [r["id"] for r in hits]


def test_delete_unknown_id_is_404(client: TestClient) -> None:
    assert client.delete("/vectors/ghost").status_code == 404
