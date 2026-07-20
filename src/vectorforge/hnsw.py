"""Single-layer HNSW (Hierarchical Navigable Small World) index.

Implements §4 Algorithm 1 (INSERT) and Algorithm 2 (K-NN-SEARCH) from:
  Malkov, Y. A., & Yashunin, D. A. (2018).
  Efficient and robust approximate nearest neighbor search using
  Hierarchical Navigable Small World graphs.
  IEEE Transactions on Pattern Analysis and Machine Intelligence.
  https://arxiv.org/abs/1603.09320

Phase 1 builds a *single-layer* graph (layer 0 only).
Phase 2 will extend this to the full multi-layer structure by adding
probabilistic layer assignment and per-layer neighbour lists.

Key parameters
--------------
m : int
    Maximum number of bidirectional connections per node (default 16).
    Higher m → better recall, more memory, slower insert.
ef_construction : int
    Size of the dynamic candidate list used during index construction
    (default 200). Higher → better recall at query time, slower insert.
    Must be >= m.

Complexity (single-layer, N nodes, dim D)
-----------------------------------------
Insert  : O(m · ef_construction · D)  average
Search  : O(m · ef · D)               average
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Internal node representation


@dataclass
class _Node:
    """A single vertex in the HNSW graph.

    Attributes
    ----------
    internal_id : int
        Auto-assigned integer used for all internal graph edges.
        Keeps hot-path data structures (heaps, sets) free of strings.
    vector_id : str
        Caller-supplied identifier returned in search results.
    vector : NDArray[np.float32]
        The raw embedding stored by value (copy on insert).
    neighbors : set[int]
        Layer-0 neighbour set (internal_ids).  Capped at M connections.
    """

    internal_id: int
    vector_id: str
    vector: NDArray[np.float32]
    neighbors: set[int] = field(default_factory=set)

# HNSW index


class HNSWIndex:
    """Single-layer HNSW approximate nearest-neighbour index.

    Parameters
    ----------
    dim : int
        Vector dimensionality.  All inserted vectors must match.
    m : int
        Max connections per node.  16 is a good default for most datasets.
    ef_construction : int
        Candidate list size during construction.  200 is a good default.

    Example
    -------
    >>> import numpy as np
    >>> idx = HNSWIndex(dim=4, m=4, ef_construction=20)
    >>> idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    >>> idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
    >>> results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)
    >>> results[0][0]
    'a'
    """

    def __init__(self, dim: int, m: int = 16, ef_construction: int = 200) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if ef_construction < m:
            raise ValueError(f"ef_construction ({ef_construction}) must be >= m ({m})")

        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction

        self._nodes: dict[int, _Node] = {}          # internal_id → _Node
        self._id_map: dict[str, int] = {}           # vector_id   → internal_id
        self._deleted: set[int] = set()             # tombstoned internal_ids
        self._entry_point: int | None = None        # internal_id of entry node
        self._next_id: int = 0

    # Public API

    def add(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Insert *vector* into the index under *vector_id*.

        Follows Algorithm 1 (single-layer variant):
        1. Assign an internal id and create a node.
        2. If the graph is empty, make this node the entry point and return.
        3. Greedily search layer 0 for ef_construction neighbours.
        4. Select the m closest as bidirectional connections.
        5. Prune any neighbour whose connection count exceeds m.
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(
                f"Expected vector of shape ({self.dim},), got {vector.shape}"
            )
        if vector_id in self._id_map:
            raise ValueError(f"vector_id {vector_id!r} already exists in the index")

        # Step 1 : create node
        iid = self._next_id
        self._next_id += 1
        node = _Node(internal_id=iid, vector_id=vector_id, vector=vector.copy())
        self._nodes[iid] = node
        self._id_map[vector_id] = iid

        # Step 2 : first node becomes the entry point
        if self._entry_point is None:
            self._entry_point = iid
            return

        # Step 3 : find ef_construction nearest live neighbours
        candidates = self._search_layer(
            query=vector,
            entry_point=self._entry_point,
            ef=self.ef_construction,
        )

        # Step 4 : select m closest and wire bidirectional edges
        neighbours = self._select_neighbours(candidates, m=self.m)
        for nb_iid, _ in neighbours:
            node.neighbors.add(nb_iid)
            self._nodes[nb_iid].neighbors.add(iid)

        # Step 5 : prune any neighbour that now exceeds m connections
        for nb_iid, _ in neighbours:
            nb_node = self._nodes[nb_iid]
            if len(nb_node.neighbors) > self.m:
                nb_node.neighbors = self._prune_neighbors(nb_node, m=self.m)

    def search(
        self, query: NDArray[np.float32], k: int, ef: int | None = None
    ) -> list[tuple[str, float]]:
        """Return the *k* approximate nearest neighbours of *query*.

        Parameters
        ----------
        query : array of shape (dim,)
        k : int
            Number of results to return.
        ef : int, optional
            Candidate list size.  Defaults to max(k, ef_construction).
            Increasing ef improves recall at the cost of latency.

        Returns
        -------
        list of (vector_id, distance) sorted ascending by distance.
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

        candidates = self._search_layer(
            query=query,
            entry_point=self._entry_point,
            ef=ef,
        )

        # Sort ascending, take top k, map internal ids → caller ids
        candidates.sort(key=lambda x: x[1])
        return [
            (self._nodes[iid].vector_id, dist)
            for iid, dist in candidates[:k]
        ]

    def delete(self, vector_id: str) -> bool:
        """Tombstone *vector_id* so it is excluded from future search results.

        Phase 2 will add full graph-repair on delete.  For now, tombstoning
        is sufficient: the node remains in the graph structure but is skipped
        during candidate evaluation and never returned in results.

        Returns ``True`` if the id existed, ``False`` otherwise.
        """
        iid = self._id_map.get(vector_id)
        if iid is None:
            return False
        self._deleted.add(iid)
        return True

    def __len__(self) -> int:
        return len(self._nodes) - len(self._deleted)

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(dim={self.dim}, m={self.m}, "
            f"ef_construction={self.ef_construction}, size={len(self)})"
        )

    # Core graph algorithm (internal)

    def _dist(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """L2 distance between two vectors (hot path, kept inline)."""
        diff = a - b
        return float(np.sqrt(np.dot(diff, diff)))

    def _search_layer(
        self,
        query: NDArray[np.float32],
        entry_point: int,
        ef: int,
    ) -> list[tuple[int, float]]:
        """Greedy beam search on the layer-0 graph.

        Implements the inner loop of Algorithm 2 from the paper.

        Uses two heaps:
        - ``candidates`` : min-heap ordered by distance (to expand next)
        - ``found``      : max-heap ordered by *negative* distance (to evict
                           the furthest element when |found| > ef)

        Returns
        -------
        List of (internal_id, distance) for the ef closest live nodes found.
        """
        ep_dist = self._dist(query, self._nodes[entry_point].vector)

        # min-heap of (dist, internal_id) — nodes to explore
        candidates: list[tuple[float, int]] = [(ep_dist, entry_point)]
        # max-heap stored as (-dist, internal_id)
        found: list[tuple[float, int]] = [(-ep_dist, entry_point)]

        visited: set[int] = {entry_point}

        while candidates:
            c_dist, c_iid = heapq.heappop(candidates)

            # Furthest element currently in found set
            f_dist = -found[0][0]

            # Paper termination: if the closest unseen candidate is further
            # than the worst in our result set, we cannot improve => stop.
            if c_dist > f_dist:
                break

            # Expand neighbours of the current candidate
            for nb_iid in self._nodes[c_iid].neighbors:
                if nb_iid in visited:
                    continue
                visited.add(nb_iid)

                # Skip tombstoned nodes
                if nb_iid in self._deleted:
                    continue

                nb_dist = self._dist(query, self._nodes[nb_iid].vector)
                f_dist = -found[0][0]

                if nb_dist < f_dist or len(found) < ef:
                    heapq.heappush(candidates, (nb_dist, nb_iid))
                    heapq.heappush(found, (-nb_dist, nb_iid))
                    if len(found) > ef:
                        heapq.heappop(found)  # evict furthest

        return [(iid, -neg_dist) for neg_dist, iid in found]

    def _select_neighbours(
        self,
        candidates: list[tuple[int, float]],
        m: int,
    ) -> list[tuple[int, float]]:
        """Return the *m* closest candidates as the neighbour set.

        Simple heuristic (Algorithm 3, "simple" variant from the paper).
        Phase 2 will optionally add the diversity-preserving heuristic
        (Algorithm 4) which improves recall on clustered data.
        """
        return sorted(candidates, key=lambda x: x[1])[:m]

    def _prune_neighbors(self, node: _Node, m: int) -> set[int]:
        """Trim *node*'s neighbour set to at most *m* closest connections."""
        scored = sorted(
            node.neighbors,
            key=lambda nb_iid: self._dist(node.vector, self._nodes[nb_iid].vector),
        )
        return set(scored[:m])