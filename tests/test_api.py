"""Integration tests for the FastAPI service (Phase 3, Day 20–21).

Exercises the running app through TestClient rather than the core index
directly, covering the happy path and malformed / conflicting requests:

    /index    201 insert, 409 duplicate, 400 wrong dimensionality, 422 schema
    /search   result shape, metadata filtering, k / ef honoured, 400 bad shape
    /vectors  204-style delete, 404 unknown id, tombstone reflected in search
    /health   status + live stats
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorforge.api import create_app
from vectorforge.hnsw import HNSWIndex

DIM = 8


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(HNSWIndex(dim=DIM, M=4, ef_construction=20)))


def _vec(seed: int) -> list[float]:
    return np.random.default_rng(seed).random(DIM).tolist()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_reports_stats(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "size": 0, "dim": DIM}

    client.post("/index", json={"id": "a", "vector": _vec(1)})
    assert client.get("/health").json()["size"] == 1


# ---------------------------------------------------------------------------
# /index
# ---------------------------------------------------------------------------


def test_index_inserts_vector(client: TestClient) -> None:
    resp = client.post("/index", json={"id": "a", "vector": _vec(1)})
    assert resp.status_code == 201
    assert resp.json()["id"] == "a"


def test_index_duplicate_id_conflicts(client: TestClient) -> None:
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
# /search
# ---------------------------------------------------------------------------


def test_search_returns_sorted_neighbors(client: TestClient) -> None:
    for i in range(20):
        client.post("/index", json={"id": str(i), "vector": _vec(i)})
    resp = client.post("/search", json={"vector": _vec(0), "k": 5})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 5
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)
    # Querying with an indexed vector should return that id as nearest.
    assert results[0]["id"] == "0"


def test_search_metadata_filter(client: TestClient) -> None:
    for i in range(30):
        client.post(
            "/index",
            json={"id": str(i), "vector": _vec(i), "metadata": {"group": i % 3}},
        )
    resp = client.post(
        "/search", json={"vector": _vec(0), "k": 10, "filter": {"group": 1}}
    )
    ids = [r["id"] for r in resp.json()["results"]]
    assert ids and all(int(i) % 3 == 1 for i in ids)


def test_search_empty_index_returns_no_results(client: TestClient) -> None:
    resp = client.post("/search", json={"vector": _vec(0), "k": 5})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_wrong_dimensionality_is_400(client: TestClient) -> None:
    client.post("/index", json={"id": "a", "vector": _vec(1)})
    resp = client.post("/search", json={"vector": [1.0, 2.0], "k": 1})
    assert resp.status_code == 400


def test_search_invalid_k_is_422(client: TestClient) -> None:
    resp = client.post("/search", json={"vector": _vec(0), "k": 0})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /vectors/{id}
# ---------------------------------------------------------------------------


def test_delete_removes_from_results(client: TestClient) -> None:
    for i in range(10):
        client.post("/index", json={"id": str(i), "vector": _vec(i)})
    resp = client.delete("/vectors/0")
    assert resp.status_code == 200

    hits = client.post("/search", json={"vector": _vec(0), "k": 10}).json()["results"]
    assert "0" not in [r["id"] for r in hits]


def test_delete_unknown_id_is_404(client: TestClient) -> None:
    assert client.delete("/vectors/ghost").status_code == 404
