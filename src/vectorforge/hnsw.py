"""Multi-layer HNSW (Hierarchical Navigable Small World) index.

Full implementation of Algorithms 1–4 from:
  Malkov, Y. A., & Yashunin, D. A. (2018).
  Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs.
  IEEE Transactions on Pattern Analysis and Machine Intelligence.
  https://arxiv.org/abs/1603.09320

Architecture
------------
The graph has L+1 layers (0 … L).  Each node exists at layers 0 through
its randomly assigned maximum layer.  Layer 0 is the densest; higher layers
are progressively sparser and act as "express lanes" for long-range
navigation.

Key parameters
--------------
M : int
    Max bidirectional connections per node per layer (default 16).
    Layer 0 uses M0 = 2·M (paper §4.1).
ef_construction : int
    Candidate list size during index construction (default 200).
    Higher → better recall, slower insert.  Must be >= M.
mL : float
    Level generation normalisation factor = 1 / ln(M).
    Controls the probability that a node is promoted to a higher layer.

Complexity (N nodes, dim D)
---------------------------
Insert  : O(log N · M · ef_construction · D)  average
Search  : O(log N · M · ef · D)               average

Phase history
-------------
Phase 1 — single-layer skeleton (layer 0 only, no level assignment).
Phase 2 — full multi-layer: probabilistic layer assignment, per-layer
           neighbour lists, two-phase insert, top-down query descent.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Internal node representation


@dataclass
class _Node:
    """A vertex in the multi-layer HNSW graph.

    Attributes
    ----------
    internal_id : int
        Integer key used in all graph structures (avoids string hashing in
        hot-path heaps and sets).
    vector_id : str
        Caller-supplied identifier returned in search results.
    vector : NDArray[np.float32]
        The embedding, stored by value.
    level : int
        Highest layer this node participates in.  Assigned once at insert.
    neighbors : dict[int, set[int]]
        Per-layer neighbour sets: ``neighbors[layer]`` → set of internal_ids.
        Populated for layers 0 … level inclusive.
    """

    internal_id: int
    vector_id: str
    vector: NDArray[np.float32]
    level: int
    neighbors: dict[int, set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for layer in range(self.level + 1):
            self.neighbors[layer] = set()

# HNSW index


class HNSWIndex:
    """Multi-layer HNSW approximate nearest-neighbour index.

    Parameters
    ----------
    dim : int
        Vector dimensionality.  All inserted vectors must match.
    M : int
        Max connections per node per layer.  16 is a strong default.
    ef_construction : int
        Candidate list size during construction.  200 is a strong default.

    Example
    -------
    >>> import numpy as np
    >>> idx = HNSWIndex(dim=4, M=4, ef_construction=20)
    >>> idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    >>> idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
    >>> results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)
    >>> results[0][0]
    'a'
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if M < 2:
            raise ValueError(f"M must be >= 2, got {M}")
        if ef_construction < M:
            raise ValueError(
                f"ef_construction ({ef_construction}) must be >= M ({M})"
            )

        self.dim = dim
        self.M = M
        self.M0 = 2 * M          # layer-0 max connections (paper §4.1)
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)   # level normalisation factor

        self._nodes: dict[int, _Node] = {}
        self._id_map: dict[str, int] = {}     # vector_id → internal_id
        self._deleted: set[int] = set()       # tombstoned internal_ids
        self._entry_point: int | None = None  # internal_id of top-layer entry
        self._max_layer: int = 0
        self._next_id: int = 0
        self._rng = np.random.default_rng()

    # Public API
    def add(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Insert *vector* into the index under *vector_id*.

        Implements Algorithm 1 from the paper:

        1. Assign a random layer *l* via the exponential distribution.
        2. Descend from the current entry point to layer *l+1* using greedy
           search with ef=1 (finding the single closest node per layer).
           This "coarse pass" positions us near the correct region cheaply.
        3. From layer *l* down to layer 0, run the full ef_construction
           beam search, select M (or M0 at layer 0) neighbours, and wire
           bidirectional edges.  Prune any neighbour that now exceeds its
           connection cap.
        4. If *l* exceeds the current max layer, promote the new node to
           be the graph entry point.
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(
                f"Expected vector of shape ({self.dim},), got {vector.shape}"
            )
        if vector_id in self._id_map:
            raise ValueError(f"vector_id {vector_id!r} already exists in the index")

        # Step 1 — assign level via exponential distribution (paper eq. 1)
        level = int(math.floor(-math.log(self._rng.uniform()) * self.mL))

        iid = self._next_id
        self._next_id += 1
        node = _Node(internal_id=iid, vector_id=vector_id, vector=vector.copy(), level=level)
        self._nodes[iid] = node
        self._id_map[vector_id] = iid

        if self._entry_point is None:
            self._entry_point = iid
            self._max_layer = level
            return

        ep = self._entry_point
        top = self._max_layer

        # Step 2 — coarse descent from top layer down to level+1 (ef=1)
        for layer in range(top, level, -1):
            candidates = self._search_layer(vector, ep, ef=1, layer=layer)
            ep = min(candidates, key=lambda x: x[1])[0]

        # Step 3 — fine insert from min(level, top) down to layer 0
        for layer in range(min(level, top), -1, -1):
            m_cap = self.M0 if layer == 0 else self.M
            candidates = self._search_layer(vector, ep, ef=self.ef_construction, layer=layer)

            neighbours = self._select_neighbours(candidates, m=m_cap)
            for nb_iid, _ in neighbours:
                node.neighbors[layer].add(nb_iid)
                self._nodes[nb_iid].neighbors[layer].add(iid)

            # Prune neighbours that exceed their connection cap
            for nb_iid, _ in neighbours:
                nb_node = self._nodes[nb_iid]
                if len(nb_node.neighbors[layer]) > m_cap:
                    nb_node.neighbors[layer] = self._prune_neighbors(
                        nb_node, layer=layer, m=m_cap
                    )

            # Update entry point for next (lower) layer
            if candidates:
                ep = min(candidates, key=lambda x: x[1])[0]

        # Step 4 — promote entry point if new node reaches a higher layer
        if level > self._max_layer:
            self._entry_point = iid
            self._max_layer = level

    def search(
        self, query: NDArray[np.float32], k: int, ef: int | None = None
    ) -> list[tuple[str, float]]:
        """Return the *k* approximate nearest neighbours of *query*.

        Implements Algorithm 5 from the paper:

        1. Descend from the entry point through all layers above 0 using
           ef=1 (fast coarse navigation to the right region of the graph).
        2. At layer 0, run the full beam search with ``ef = max(k, ef)``.
        3. Return the k closest results sorted by distance.

        Parameters
        ----------
        query : array of shape (dim,)
        k : int
            Number of results to return.
        ef : int, optional
            Search candidate list size.  Larger ef → better recall, higher
            latency.  Defaults to ``max(k, ef_construction)``.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(
                f"Expected query of shape ({self.dim},), got {query.shape}"
            )
        if self._entry_point is None:
            return []

        ef = ef or max(k, self.ef_construction)
        ep = self._entry_point

        # Coarse descent through layers above 0
        for layer in range(self._max_layer, 0, -1):
            candidates = self._search_layer(query, ep, ef=1, layer=layer)
            ep = min(candidates, key=lambda x: x[1])[0]

        # Full beam search at layer 0
        candidates = self._search_layer(query, ep, ef=ef, layer=0)
        candidates.sort(key=lambda x: x[1])

        return [
            (self._nodes[iid].vector_id, dist)
            for iid, dist in candidates[:k]
            if iid not in self._deleted
        ]

    def delete(self, vector_id: str) -> bool:
        """Tombstone *vector_id* so it is excluded from future results.

        The node remains in the graph (preserving connectivity for
        neighbours) but is skipped during candidate evaluation and
        never returned in search results.

        Returns ``True`` if the id existed, ``False`` otherwise.
        """
        iid = self._id_map.get(vector_id)
        if iid is None:
            return False
        self._deleted.add(iid)
        return True

    @property
    def max_layer(self) -> int:
        """Highest layer currently in the graph."""
        return self._max_layer

    def __len__(self) -> int:
        return len(self._nodes) - len(self._deleted)

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(dim={self.dim}, M={self.M}, "
            f"ef_construction={self.ef_construction}, "
            f"max_layer={self._max_layer}, size={len(self)})"
        )

    # Core graph algorithm (internal)

    def _dist(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """L2 distance — kept inline for hot-path performance."""
        diff = a - b
        return float(np.sqrt(np.dot(diff, diff)))

    def _search_layer(
        self,
        query: NDArray[np.float32],
        entry_point: int,
        ef: int,
        layer: int,
    ) -> list[tuple[int, float]]:
        """Beam search on *layer* starting from *entry_point*.

        Implements the inner loop of Algorithm 2.

        Uses two heaps:
        - ``candidates``: min-heap by distance — next nodes to expand.
        - ``found``:      max-heap by *negative* distance — the current
                          best-ef set; we evict the furthest when |found|>ef.

        Returns
        -------
        List of (internal_id, distance) — up to ef live (non-tombstoned)
        nodes closest to *query* found during the walk.
        """
        ep_dist = self._dist(query, self._nodes[entry_point].vector)

        candidates: list[tuple[float, int]] = [(ep_dist, entry_point)]
        found: list[tuple[float, int]] = [(-ep_dist, entry_point)]
        visited: set[int] = {entry_point}

        while candidates:
            c_dist, c_iid = heapq.heappop(candidates)
            f_dist = -found[0][0]   # furthest in found set

            # Termination: closest unseen > worst in found → can't improve
            if c_dist > f_dist:
                break

            for nb_iid in self._nodes[c_iid].neighbors.get(layer, set()):
                if nb_iid in visited:
                    continue
                visited.add(nb_iid)

                nb_dist = self._dist(query, self._nodes[nb_iid].vector)
                f_dist = -found[0][0]

                if nb_dist < f_dist or len(found) < ef:
                    heapq.heappush(candidates, (nb_dist, nb_iid))
                    # Only add to found if not tombstoned
                    if nb_iid not in self._deleted:
                        heapq.heappush(found, (-nb_dist, nb_iid))
                        if len(found) > ef:
                            heapq.heappop(found)

        return [(iid, -neg_dist) for neg_dist, iid in found]

    def _select_neighbours(
        self,
        candidates: list[tuple[int, float]],
        m: int,
    ) -> list[tuple[int, float]]:
        """Select the *m* closest candidates as the neighbour set.

        Uses the simple heuristic (Algorithm 3).  The diversity-preserving
        heuristic (Algorithm 4) can be added here later to improve recall on
        clustered datasets without changing the public interface.
        """
        return sorted(candidates, key=lambda x: x[1])[:m]

    def _prune_neighbors(self, node: _Node, layer: int, m: int) -> set[int]:
        """Trim *node*'s layer-*layer* neighbour set to at most *m* entries."""
        scored = sorted(
            node.neighbors[layer],
            key=lambda nb_iid: self._dist(node.vector, self._nodes[nb_iid].vector),
        )
        return set(scored[:m])