"""Brute-force k-nearest-neighbour index.

This is the ground-truth baseline for the entire project. Every recall@k
measurement made against HNSW (and later the distributed coordinator) is
computed by comparing results against what this index returns.

Design notes:
- Stores vectors as a (N, D) float32 NumPy array; vectorised distance
  computation over the full corpus in one NumPy call makes this fast enough
  to serve as a benchmark reference on datasets up to ~500K vectors.
- IDs are caller-supplied strings so the API matches what HNSW will expose.
- No external dependencies beyond NumPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BruteForceIndex:
    """Exact k-NN index using exhaustive linear scan.

    Parameters
    ----------
    dim:
        Dimensionality of the vectors this index will store.  All vectors
        passed to :meth:`add` must have exactly this many elements.

    Example
    -------
    >>> idx = BruteForceIndex(dim=4)
    >>> idx.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    >>> idx.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
    >>> results = idx.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)
    >>> results[0][0]
    'a'
    """

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self._ids: list[str] = []
        self._vectors: list[NDArray[np.float32]] = []

    # Mutation

    def add(self, vector_id: str, vector: NDArray[np.float32]) -> None:
        """Insert *vector* into the index under *vector_id*.

        Raises
        ------
        ValueError
            If the vector's shape does not match self.dim.
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(
                f"Expected vector of shape ({self.dim},), got {vector.shape}"
            )
        self._ids.append(vector_id)
        self._vectors.append(vector)

    def delete(self, vector_id: str) -> bool:
        """Remove the vector with *vector_id* from the index.

        Returns True if the id was found and removed, False otherwise.
        """
        try:
            idx = self._ids.index(vector_id)
        except ValueError:
            return False
        self._ids.pop(idx)
        self._vectors.pop(idx)
        return True

    # Query

    def search(
        self, query: NDArray[np.float32], k: int
    ) -> list[tuple[str, float]]:
        """Return the *k* nearest neighbours of *query* as ``(id, distance)`` pairs.

        Results are sorted ascending by distance (closest first).

        Raises
        ------
        ValueError
            If *k* < 1 or the query shape does not match self.dim.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(
                f"Expected query of shape ({self.dim},), got {query.shape}"
            )

        n = len(self._ids)
        if n == 0:
            return []

        k = min(k, n)

        # Vectorised L2: compute all distances in one NumPy broadcast.
        matrix = np.stack(self._vectors)          # (N, D)
        diff = matrix - query                     # (N, D)
        distances: NDArray[np.float32] = np.sqrt((diff * diff).sum(axis=1))

        # argpartition gives top-k indices in O(N) average time.
        top_k_idx = np.argpartition(distances, k - 1)[:k]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]

        return [(self._ids[i], float(distances[i])) for i in top_k_idx]

    # Introspection

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return f"BruteForceIndex(dim={self.dim}, size={len(self)})"