"""Thread-safety tests for HNSWIndex (Phase 6).

The Locust load test surfaced a race: searches and inserts run on different
threads (FastAPI's worker pool), and a search iterating a neighbour set while an
insert mutated it crashed with "set changed size during iteration". These tests
hammer the index from many threads and assert it stays crash-free and
consistent, so that regression cannot come back unnoticed.
"""

from __future__ import annotations

import threading

import numpy as np

from vectorforge.hnsw import HNSWIndex


def _seed(idx: HNSWIndex, n: int, dim: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        idx.add(f"seed-{i}", rng.random(dim).astype(np.float32))


def test_concurrent_search_and_insert_stay_crash_free() -> None:
    dim = 16
    idx = HNSWIndex(dim=dim, M=8, ef_construction=50)
    _seed(idx, 200, dim)

    errors: list[Exception] = []
    stop = threading.Event()

    def searcher() -> None:
        rng = np.random.default_rng()
        try:
            while not stop.is_set():
                idx.search(rng.random(dim).astype(np.float32), k=10)
        except Exception as exc:  # noqa: BLE001 - the whole point is to catch any
            errors.append(exc)

    def writer(base: int) -> None:
        rng = np.random.default_rng(base + 1)
        try:
            for i in range(200):
                idx.add(f"w{base}-{i}", rng.random(dim).astype(np.float32))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    searchers = [threading.Thread(target=searcher) for _ in range(4)]
    writers = [threading.Thread(target=writer, args=(b,)) for b in range(3)]
    for t in searchers + writers:
        t.start()
    for w in writers:
        w.join()
    stop.set()
    for s in searchers:
        s.join()

    assert not errors, f"concurrent access raised: {errors[:3]}"
    # 200 seed + 3 writers x 200 inserts, all unique ids.
    assert len(idx) == 200 + 3 * 200


def test_concurrent_delete_and_search_stay_consistent() -> None:
    dim = 16
    idx = HNSWIndex(dim=dim, M=8, ef_construction=50)
    _seed(idx, 300, dim)

    errors: list[Exception] = []
    stop = threading.Event()

    def searcher() -> None:
        rng = np.random.default_rng()
        try:
            while not stop.is_set():
                idx.search(rng.random(dim).astype(np.float32), k=5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def deleter() -> None:
        try:
            for i in range(150):
                idx.delete(f"seed-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    searchers = [threading.Thread(target=searcher) for _ in range(4)]
    d = threading.Thread(target=deleter)
    for t in [*searchers, d]:
        t.start()
    d.join()
    stop.set()
    for s in searchers:
        s.join()

    assert not errors, f"concurrent delete/search raised: {errors[:3]}"
    assert len(idx) == 300 - 150
