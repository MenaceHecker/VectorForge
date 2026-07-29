"""Locust load test for the VectorForge coordinator (Phase 6, Day 36-37).

Drives the public REST surface (the coordinator, or the single-node API, they
share request shapes) under concurrent load so we can record p50/p95/p99 search
latency at each level. The plan sweeps 10, 50, 100, and 200 users.

Latency comes straight from Locust. Recall does not: it needs brute-force ground
truth, so measure it separately against a snapshot with
`python -m vectorforge.benchmark <index.vfidx>` and pair the two numbers per
load level in the results table.

Run it interactively:

    uvicorn vectorforge.coordinator_api:app &   # or the single-node api:app
    locust -f locust/locustfile.py --host http://localhost:8000

Or headless, one run per load level, writing CSVs the README table reads from:

    for u in 10 50 100 200; do
      locust -f locust/locustfile.py --host http://localhost:8000 \\
        --headless -u "$u" -r "$u" -t 2m --csv "results/load-$u"
    done

Config via environment:
    VECTORFORGE_DIM              vector dimensionality (default 128)
    VECTORFORGE_SEARCH_K         neighbours per query (default 10)
    VECTORFORGE_SEED_PER_USER    vectors each user inserts on start (default 50)
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

DIM = int(os.environ.get("VECTORFORGE_DIM", "128"))
SEARCH_K = int(os.environ.get("VECTORFORGE_SEARCH_K", "10"))
SEED_PER_USER = int(os.environ.get("VECTORFORGE_SEED_PER_USER", "50"))


def random_vector(dim: int = DIM) -> list[float]:
    """A random query/embedding vector. Gaussian components are a fine stand-in
    for real embeddings when the goal is measuring throughput, not recall."""
    return [random.gauss(0.0, 1.0) for _ in range(dim)]


class VectorForgeUser(HttpUser):
    """A client that mostly searches, with a light trickle of writes.

    The 9:1 search-to-index ratio reflects a read-heavy retrieval workload,
    which is what a vector store usually serves.
    """

    wait_time = between(0.0, 0.05)

    def on_start(self) -> None:
        # Give this user's traffic something to find. Ids carry a random prefix
        # so users (and distributed workers) never collide on the same id.
        self._prefix = f"{random.getrandbits(48):012x}"
        for i in range(SEED_PER_USER):
            self.client.post(
                "/index",
                json={"id": f"{self._prefix}-{i}", "vector": random_vector()},
                name="/index (seed)",
            )

    @task(9)
    def search(self) -> None:
        with self.client.post(
            "/search",
            json={"vector": random_vector(), "k": SEARCH_K},
            name="/search",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"search returned {resp.status_code}")

    @task(1)
    def index(self) -> None:
        self._counter = getattr(self, "_counter", SEED_PER_USER) + 1
        self.client.post(
            "/index",
            json={"id": f"{self._prefix}-{self._counter}", "vector": random_vector()},
            name="/index",
        )
