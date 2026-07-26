"""Prometheus metrics for the VectorForge service (Phase 4, Day 24).

We track the three things that actually tell you how the index is behaving in
production:

    vectorforge_query_latency_seconds   how long searches take (histogram)
    vectorforge_index_size_total        how many live vectors it holds (gauge)
    vectorforge_recall_at_k             quality vs brute force (gauge)

The histogram is the interesting one. Its buckets let Prometheus compute p50,
p95, and p99 latency without us storing every sample, and the boundaries are
chosen around the sub-15ms target so the percentiles we care about land in
their own buckets instead of getting lumped into one wide range.

Each Metrics object owns its own CollectorRegistry rather than using the global
default. That keeps the metric names from colliding when more than one app is
built in the same process, which is exactly what the test suite does when it
spins up a fresh app per test.

recall_at_k is populated later by the periodic benchmark job (Day 27-28), so it
sits at 0 until something calls set_recall(). It is defined here now so the
metric exists and dashboards can graph it from day one.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, Histogram

# Latency buckets in seconds, clustered around the sub-15ms goal so p95/p99 are
# readable. Anything slower than 1s falls into the +Inf bucket.
_LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)


class Metrics:
    """A self-contained set of Prometheus collectors for one service instance."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        self.query_latency = Histogram(
            "vectorforge_query_latency_seconds",
            "Search request latency in seconds.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        # Named with a _total suffix to match the project plan and the Grafana
        # dashboards that reference it, even though it is a gauge (it can go
        # down when vectors are deleted), not a monotonic counter.
        self.index_size = Gauge(
            "vectorforge_index_size_total",
            "Number of live (non-tombstoned) vectors in the index.",
            registry=self.registry,
        )
        self.recall_at_k = Gauge(
            "vectorforge_recall_at_k",
            "Most recent measured recall@k against brute-force ground truth.",
            registry=self.registry,
        )

    def set_index_size(self, size: int) -> None:
        """Refresh the index-size gauge, usually right before a scrape."""
        self.index_size.set(size)

    def set_recall(self, recall: float) -> None:
        """Record the latest recall measurement from the benchmark job."""
        self.recall_at_k.set(recall)
