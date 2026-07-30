"""Build a realistic benchmark dataset from a text corpus (Phase 6, Day 38).

Synthetic random vectors are fine for smoke tests, but they do not look like
real embeddings: real ones are clustered, correlated, and live on a lower
-dimensional manifold, which is exactly what stresses an ANN index. So for the
headline benchmark numbers we embed an actual text corpus with
sentence-transformers and index those vectors.

sentence-transformers drags in torch, which is heavy and not something the core
package or CI should depend on, so it is an optional extra:

    pip install -e ".[embeddings]"

To keep the pipeline testable without that dependency, the embedding step is
injected as a callable. The default one loads a sentence-transformers model; a
test can pass a stand-in that returns random vectors and exercise everything
else (corpus loading, index building, persistence) for free.

Command line:

    python -m vectorforge.dataset --corpus corpus.txt --out data/wiki.vfidx
    python -m vectorforge.dataset --out data/sample.vfidx      # bundled sample

The output is a .vfidx index (loadable by the benchmark) plus a sibling .npz of
raw vectors, ids, and texts for building brute-force ground truth.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vectorforge.hnsw import HNSWIndex
from vectorforge.persistence import save

Embedder = Callable[[list[str]], NDArray[np.float32]]

# A tiny, topically varied corpus so the tool runs with no download or corpus
# file. Real runs should pass --corpus with something like Wikipedia abstracts.
SAMPLE_CORPUS: list[str] = [
    "The mitochondria is the powerhouse of the cell.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The French Revolution began in 1789 and reshaped Europe.",
    "Jazz emerged in New Orleans in the early twentieth century.",
    "A black hole is a region of spacetime where gravity is extreme.",
    "Consistent hashing spreads keys evenly across a changing set of nodes.",
    "Espresso is brewed by forcing hot water through finely ground coffee.",
    "The Great Barrier Reef is the world's largest coral reef system.",
    "Neural networks learn representations by adjusting connection weights.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "The printing press accelerated the spread of ideas in the Renaissance.",
    "Sharks have existed for hundreds of millions of years.",
]


def load_corpus(path: str | Path) -> list[str]:
    """Read a corpus file, one document per line, skipping blank lines."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def default_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Return an embedder backed by a sentence-transformers model.

    The model is loaded once, lazily, so importing this module never pulls in
    torch. Raises a clear error if the optional dependency is missing.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
        raise ModuleNotFoundError(
            "sentence-transformers is not installed. Install the embeddings extra "
            "with: pip install -e '.[embeddings]'"
        ) from exc

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> NDArray[np.float32]:
        return np.asarray(model.encode(texts), dtype=np.float32)

    return embed


def build_index(
    ids: list[str],
    texts: list[str],
    vectors: NDArray[np.float32],
    M: int = 16,
    ef_construction: int = 200,
) -> HNSWIndex:
    """Build an HNSW index from embeddings, keeping each source text as metadata."""
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D (n, dim), got shape {vectors.shape}")
    if not (len(ids) == len(texts) == len(vectors)):
        raise ValueError("ids, texts, and vectors must be the same length")

    index = HNSWIndex(dim=vectors.shape[1], M=M, ef_construction=ef_construction)
    for vector_id, text, vector in zip(ids, texts, vectors, strict=True):
        index.add(vector_id, vector, metadata={"text": text})
    return index


def generate(
    out_path: str | Path,
    corpus_path: str | Path | None = None,
    embed_fn: Embedder | None = None,
    M: int = 16,
    ef_construction: int = 200,
) -> HNSWIndex:
    """Embed a corpus, build an index, and save it (plus a raw-vector .npz).

    *embed_fn* defaults to a sentence-transformers model; pass your own to avoid
    that dependency (tests do this).
    """
    texts = load_corpus(corpus_path) if corpus_path is not None else list(SAMPLE_CORPUS)
    if not texts:
        raise ValueError("corpus is empty")

    embed = embed_fn or default_embedder()
    vectors = np.asarray(embed(texts), dtype=np.float32)
    ids = [f"doc-{i}" for i in range(len(texts))]

    index = build_index(ids, texts, vectors, M=M, ef_construction=ef_construction)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(index, out_path)
    # Raw vectors alongside the index, for brute-force ground truth in benchmarks.
    np.savez(
        out_path.with_suffix(".npz"),
        ids=np.array(ids),
        texts=np.array(texts, dtype=object),
        vectors=vectors,
    )
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a benchmark dataset from text.")
    parser.add_argument("--out", required=True, help="Output .vfidx path.")
    parser.add_argument(
        "--corpus", default=None, help="Corpus file, one document per line (optional)."
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model.")
    parser.add_argument("-M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    args = parser.parse_args(argv)

    index = generate(
        args.out,
        corpus_path=args.corpus,
        embed_fn=default_embedder(args.model),
        M=args.M,
        ef_construction=args.ef_construction,
    )
    print(f"wrote {len(index)} vectors (dim {index.dim}) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
