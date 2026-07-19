"""Distance metrics for vector similarity search.

All functions operate on 1-D NumPy float32 arrays. 
"""

import numpy as np
from numpy.typing import NDArray


def euclidean(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Return the Euclidean (L2) distance between vectors *a* and *b*.

    This is the primary metric used by HNSW throughout the project.
    Lower distance == more similar.
    """
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Return the cosine distance (1 - cosine_similarity) between *a* and *b*.

    Range is [0, 2]. Lower == more similar. Returns 1.0 for zero-magnitude
    vectors rather than raising, so callers never need to guard against NaN.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))