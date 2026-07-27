"""Consistent hash ring for assigning vectors to shards (Phase 5, Day 29-30).

The problem this solves: with S shards, the naive `hash(key) % S` spreads keys
evenly, but the moment S changes (a shard is added or dies) almost every key
lands on a different shard, so nearly the whole dataset has to move. That is a
disaster for a stateful store.

Consistent hashing fixes that. Hash every shard to one or more points on a
fixed ring (here the 64-bit hash space), hash each key to a point too, and walk
clockwise from the key to the first shard point you hit. Now adding or removing
a shard only disturbs the keys in the arc next to it, so on average only about
1/S of the keys move instead of all of them.

Virtual nodes
-------------
If each shard were a single point, three shards would carve the ring into three
arcs of wildly different sizes and load would be lopsided. Instead each shard is
placed at many points (`virtual_nodes` of them). Averaging over many small arcs
smooths the distribution so every shard gets a fair share, and when a shard
leaves, its keys scatter across all the survivors rather than dumping entirely
onto one neighbour.

The hashing primitive is stdlib blake2b; the ring logic (placement, lookup,
rebalancing) is written here from scratch, which is the part worth explaining in
an interview.
"""

from __future__ import annotations

import bisect
import hashlib
from collections.abc import Iterable


class ConsistentHashRing:
    """A hash ring mapping arbitrary string keys to a set of nodes (shards).

    Parameters
    ----------
    nodes:
        Optional iterable of node identifiers to seed the ring with.
    virtual_nodes:
        How many points each node occupies on the ring. Higher means smoother
        load distribution at the cost of more memory and slightly slower
        add/remove. 150 is a common, well-behaved default.
    """

    def __init__(
        self, nodes: Iterable[str] | None = None, virtual_nodes: int = 150
    ) -> None:
        if virtual_nodes < 1:
            raise ValueError(f"virtual_nodes must be >= 1, got {virtual_nodes}")
        self._virtual_nodes = virtual_nodes
        self._ring: dict[int, str] = {}      # ring point -> node
        self._sorted_points: list[int] = []  # ring points, ascending, for bisect
        self._nodes: set[str] = set()
        if nodes:
            for node in nodes:
                self.add_node(node)

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(key: str) -> int:
        """Map a string to a point in the 64-bit ring space.

        blake2b is fast and spreads keys uniformly, which is all the ring needs
        (no cryptographic strength required, just a good distribution).
        """
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big")

    def _point(self, node: str, replica: int) -> int:
        """Ring point for the *replica*-th virtual node of *node*."""
        return self._hash(f"{node}#{replica}")

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def add_node(self, node: str) -> None:
        """Add *node* to the ring. A no-op if it is already present."""
        if node in self._nodes:
            return
        self._nodes.add(node)
        for replica in range(self._virtual_nodes):
            point = self._point(node, replica)
            # 64-bit collisions are astronomically unlikely; if one ever
            # happens the later node simply wins that point, which is harmless.
            self._ring[point] = node
            bisect.insort(self._sorted_points, point)

    def remove_node(self, node: str) -> None:
        """Remove *node* from the ring. A no-op if it is not present."""
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        for replica in range(self._virtual_nodes):
            point = self._point(node, replica)
            # Only drop the point if this node actually owns it (guards against
            # the rare shared-point case noted in add_node).
            if self._ring.get(point) == node:
                del self._ring[point]
                idx = bisect.bisect_left(self._sorted_points, point)
                if idx < len(self._sorted_points) and self._sorted_points[idx] == point:
                    self._sorted_points.pop(idx)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_node(self, key: str) -> str | None:
        """Return the node responsible for *key*, or None if the ring is empty.

        Walk clockwise from the key's point to the first node point at or after
        it, wrapping around the end of the ring back to the start.
        """
        if not self._sorted_points:
            return None
        point = self._hash(key)
        idx = bisect.bisect(self._sorted_points, point)
        if idx == len(self._sorted_points):
            idx = 0  # wrap around
        return self._ring[self._sorted_points[idx]]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> set[str]:
        """The set of physical nodes currently on the ring (a copy)."""
        return set(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: str) -> bool:
        return node in self._nodes

    def __repr__(self) -> str:
        return (
            f"ConsistentHashRing(nodes={len(self._nodes)}, "
            f"virtual_nodes={self._virtual_nodes})"
        )
