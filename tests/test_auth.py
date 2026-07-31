"""Tests for API key auth on the write endpoints (Phase 6, Day 41).

Writes (/index, DELETE /vectors/{id}) require the key when one is configured;
search and health stay open. With no key configured, everything is open, which
is what keeps the rest of the suite and local dev working without a key.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorforge.api import create_app
from vectorforge.auth import API_KEY_HEADER
from vectorforge.coordinator import Coordinator, LocalShardClient
from vectorforge.coordinator_api import create_coordinator_app
from vectorforge.hnsw import HNSWIndex

DIM = 8
KEY = "s3cret-key"


def _vec(seed: int) -> list[float]:
    return np.random.default_rng(seed).random(DIM).tolist()


@pytest.fixture()
def secured_client() -> TestClient:
    return TestClient(create_app(HNSWIndex(dim=DIM, M=4, ef_construction=20), api_key=KEY))


@pytest.fixture()
def secured_coordinator_client() -> TestClient:
    shards = {"shard-0": LocalShardClient(HNSWIndex(dim=DIM, M=4, ef_construction=20))}
    return TestClient(create_coordinator_app(Coordinator(shards), api_key=KEY))


# ---------------------------------------------------------------------------
# Single-node API
# ---------------------------------------------------------------------------


def test_index_without_key_is_401(secured_client: TestClient) -> None:
    resp = secured_client.post("/index", json={"id": "a", "vector": _vec(1)})
    assert resp.status_code == 401


def test_index_with_wrong_key_is_401(secured_client: TestClient) -> None:
    resp = secured_client.post(
        "/index", json={"id": "a", "vector": _vec(1)}, headers={API_KEY_HEADER: "nope"}
    )
    assert resp.status_code == 401


def test_index_with_correct_key_succeeds(secured_client: TestClient) -> None:
    resp = secured_client.post(
        "/index", json={"id": "a", "vector": _vec(1)}, headers={API_KEY_HEADER: KEY}
    )
    assert resp.status_code == 201


def test_delete_requires_key(secured_client: TestClient) -> None:
    secured_client.post(
        "/index", json={"id": "a", "vector": _vec(1)}, headers={API_KEY_HEADER: KEY}
    )
    assert secured_client.delete("/vectors/a").status_code == 401
    ok = secured_client.delete("/vectors/a", headers={API_KEY_HEADER: KEY})
    assert ok.status_code == 200


def test_search_and_health_stay_open(secured_client: TestClient) -> None:
    # Read paths must not need the key.
    assert secured_client.get("/health").status_code == 200
    assert secured_client.post("/search", json={"vector": _vec(1), "k": 5}).status_code == 200


def test_no_key_configured_leaves_writes_open() -> None:
    client = TestClient(create_app(HNSWIndex(dim=DIM, M=4, ef_construction=20)))
    assert client.post("/index", json={"id": "a", "vector": _vec(1)}).status_code == 201


# ---------------------------------------------------------------------------
# Coordinator API
# ---------------------------------------------------------------------------


def test_coordinator_write_requires_key(secured_coordinator_client: TestClient) -> None:
    unauth = secured_coordinator_client.post("/index", json={"id": "a", "vector": _vec(1)})
    assert unauth.status_code == 401
    auth = secured_coordinator_client.post(
        "/index", json={"id": "a", "vector": _vec(1)}, headers={API_KEY_HEADER: KEY}
    )
    assert auth.status_code == 201


def test_coordinator_search_stays_open(secured_coordinator_client: TestClient) -> None:
    assert secured_coordinator_client.post(
        "/search", json={"vector": _vec(1), "k": 5}
    ).status_code == 200
