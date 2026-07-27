"""Tests for the consistent hash ring (Phase 5, Day 29-30).

Beyond the basic mapping behaviour, these pin down the two properties that make
consistent hashing worth the trouble:
  - load is spread fairly across shards (thanks to virtual nodes), and
  - adding or removing a shard only moves a small fraction of keys, and only to
    or from the shard that changed (never a reshuffle among the survivors).
"""

from __future__ import annotations

from collections import Counter

import pytest

from vectorforge.hashing import ConsistentHashRing

KEYS = [f"vector-{i}" for i in range(10_000)]


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_empty_ring_returns_none() -> None:
    assert ConsistentHashRing().get_node("anything") is None


def test_single_node_owns_every_key() -> None:
    ring = ConsistentHashRing(["shard-0"])
    assert all(ring.get_node(k) == "shard-0" for k in KEYS[:100])


def test_assignment_is_deterministic() -> None:
    ring_a = ConsistentHashRing(["a", "b", "c"])
    ring_b = ConsistentHashRing(["c", "b", "a"])  # insertion order must not matter
    assert all(ring_a.get_node(k) == ring_b.get_node(k) for k in KEYS[:500])


def test_invalid_virtual_nodes_raises() -> None:
    with pytest.raises(ValueError, match="virtual_nodes must be >= 1"):
        ConsistentHashRing(virtual_nodes=0)


def test_membership_helpers() -> None:
    ring = ConsistentHashRing(["a", "b"])
    assert "a" in ring
    assert "z" not in ring
    assert len(ring) == 2
    assert ring.nodes == {"a", "b"}
    # nodes property returns a copy, mutating it must not affect the ring
    ring.nodes.add("c")
    assert "c" not in ring


def test_add_and_remove_are_idempotent() -> None:
    ring = ConsistentHashRing(["a"])
    ring.add_node("a")           # already there
    assert len(ring) == 1
    ring.remove_node("ghost")    # not there
    assert len(ring) == 1


# ---------------------------------------------------------------------------
# Load distribution
# ---------------------------------------------------------------------------


def test_load_is_roughly_balanced_across_shards() -> None:
    ring = ConsistentHashRing(["s0", "s1", "s2", "s3"], virtual_nodes=200)
    counts = Counter(ring.get_node(k) for k in KEYS)

    expected = len(KEYS) / 4
    # With 200 virtual nodes per shard every shard should land within ~25% of
    # a perfectly even split. This is the whole point of virtual nodes.
    for shard in ("s0", "s1", "s2", "s3"):
        assert 0.75 * expected <= counts[shard] <= 1.25 * expected


def test_more_virtual_nodes_smooth_the_distribution() -> None:
    def imbalance(virtual_nodes: int) -> float:
        ring = ConsistentHashRing(["a", "b", "c"], virtual_nodes=virtual_nodes)
        counts = Counter(ring.get_node(k) for k in KEYS)
        return (max(counts.values()) - min(counts.values())) / len(KEYS)

    # More virtual nodes should not make balance worse; typically much better.
    assert imbalance(200) <= imbalance(1) + 0.02


# ---------------------------------------------------------------------------
# Rebalancing (the Day 34-35 milestone property)
# ---------------------------------------------------------------------------


def test_adding_a_shard_moves_only_a_small_fraction() -> None:
    ring = ConsistentHashRing(["s0", "s1", "s2"], virtual_nodes=200)
    before = {k: ring.get_node(k) for k in KEYS}

    ring.add_node("s3")
    after = {k: ring.get_node(k) for k in KEYS}

    moved = sum(1 for k in KEYS if before[k] != after[k])
    fraction = moved / len(KEYS)
    # Theory says about 1/4 of keys move when going from 3 to 4 shards. Allow
    # slack for hashing variance, but it must be nowhere near a full reshuffle.
    assert 0.15 <= fraction <= 0.35


def test_added_shard_only_takes_keys_never_swaps_survivors() -> None:
    """Keys that move on an add must move *to* the new shard, and keys that stay
    keep their old owner. Existing shards never trade keys among themselves."""
    ring = ConsistentHashRing(["s0", "s1", "s2"], virtual_nodes=200)
    before = {k: ring.get_node(k) for k in KEYS}

    ring.add_node("s3")
    for k in KEYS:
        now = ring.get_node(k)
        if now != before[k]:
            assert now == "s3"


def test_removing_a_shard_only_touches_its_own_keys() -> None:
    ring = ConsistentHashRing(["s0", "s1", "s2", "s3"], virtual_nodes=200)
    before = {k: ring.get_node(k) for k in KEYS}

    ring.remove_node("s3")
    for k in KEYS:
        now = ring.get_node(k)
        if before[k] != "s3":
            # Keys that were not on the removed shard keep their owner.
            assert now == before[k]
        else:
            # Keys from the removed shard land on one of the survivors.
            assert now in {"s0", "s1", "s2"}
