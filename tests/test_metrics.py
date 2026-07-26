"""Tests for the Prometheus instrumentation (Phase 4, Day 24).

These check that the /metrics endpoint exposes the three metrics we care about,
that they move in response to real traffic, and that building more than one app
in the same process does not blow up on duplicate metric registration.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from vectorforge.api import create_app
from vectorforge.hnsw import HNSWIndex

DIM = 8


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(HNSWIndex(dim=DIM, M=4, ef_construction=20)))


def _vec(seed: int) -> list[float]:
    return np.random.default_rng(seed).random(DIM).tolist()


def _sample(body: str, name: str) -> float:
    """Pull a single sample value by name out of the exposition text."""
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name == name:
                return sample.value
    raise AssertionError(f"metric {name!r} not found in /metrics output")


def test_metrics_endpoint_exposes_prometheus_text(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    assert "vectorforge_query_latency_seconds" in body
    assert "vectorforge_index_size_total" in body
    assert "vectorforge_recall_at_k" in body


def test_index_size_gauge_tracks_live_vectors(client: TestClient) -> None:
    for i in range(5):
        client.post("/index", json={"id": str(i), "vector": _vec(i)})
    client.delete("/vectors/0")

    body = client.get("/metrics").text
    assert _sample(body, "vectorforge_index_size_total") == 4.0


def test_query_latency_histogram_counts_searches(client: TestClient) -> None:
    for i in range(10):
        client.post("/index", json={"id": str(i), "vector": _vec(i)})

    before = _sample(client.get("/metrics").text, "vectorforge_query_latency_seconds_count")
    for _ in range(3):
        client.post("/search", json={"vector": _vec(0), "k": 5})
    after = _sample(client.get("/metrics").text, "vectorforge_query_latency_seconds_count")

    assert after == before + 3


def test_recall_gauge_defaults_to_zero_then_reflects_setter(client: TestClient) -> None:
    body = client.get("/metrics").text
    assert _sample(body, "vectorforge_recall_at_k") == 0.0

    # The benchmark job (Day 27-28) will call this; simulate one measurement.
    client.app.state.metrics.set_recall(0.94)
    body = client.get("/metrics").text
    assert _sample(body, "vectorforge_recall_at_k") == pytest.approx(0.94)


def test_two_apps_have_isolated_registries() -> None:
    """Per-app registries mean no duplicate-timeseries error and no leakage."""
    app_a = create_app(HNSWIndex(dim=DIM, M=4, ef_construction=20))
    app_b = create_app(HNSWIndex(dim=DIM, M=4, ef_construction=20))
    client_a, client_b = TestClient(app_a), TestClient(app_b)

    client_a.post("/index", json={"id": "x", "vector": _vec(1)})

    size_a = _sample(client_a.get("/metrics").text, "vectorforge_index_size_total")
    size_b = _sample(client_b.get("/metrics").text, "vectorforge_index_size_total")
    assert size_a == 1.0
    assert size_b == 0.0
