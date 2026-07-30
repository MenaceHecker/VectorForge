"""Tests for the embeddings dataset builder (Phase 6, Day 38).

The actual embedding step needs sentence-transformers (and torch), which is an
optional extra we do not install in CI. So these inject a fake embedder that
returns random vectors, which exercises the whole pipeline (corpus loading,
index building, metadata, persistence, the .npz sidecar) without the heavy
dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vectorforge.dataset import (
    SAMPLE_CORPUS,
    build_index,
    generate,
    load_corpus,
)
from vectorforge.persistence import load


def _fake_embedder(dim: int = 24, seed: int = 0):
    """A stand-in embedder: deterministic random vectors, one per text."""
    rng = np.random.default_rng(seed)

    def embed(texts: list[str]) -> np.ndarray:
        return rng.random((len(texts), dim)).astype(np.float32)

    return embed


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def test_load_corpus_reads_lines_and_skips_blanks(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("first line\n\n  second line  \n\n", encoding="utf-8")
    assert load_corpus(corpus) == ["first line", "second line"]


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def test_build_index_attaches_text_metadata() -> None:
    texts = ["alpha", "beta", "gamma"]
    ids = [f"doc-{i}" for i in range(3)]
    vectors = _fake_embedder(dim=8)(texts)

    index = build_index(ids, texts, vectors)
    assert len(index) == 3
    assert index.dim == 8
    assert index.get_metadata("doc-1") == {"text": "beta"}


def test_build_index_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_index(["a"], ["x", "y"], np.zeros((1, 4), dtype=np.float32))


def test_build_index_requires_2d_vectors() -> None:
    with pytest.raises(ValueError, match="2D"):
        build_index(["a"], ["x"], np.zeros(4, dtype=np.float32))


# ---------------------------------------------------------------------------
# End-to-end generate()
# ---------------------------------------------------------------------------


def test_generate_writes_loadable_index_and_npz(tmp_path: Path) -> None:
    out = tmp_path / "data" / "sample.vfidx"
    index = generate(out, embed_fn=_fake_embedder(dim=16))

    # Index file loads back and keeps size, dim, and metadata.
    assert out.exists()
    reloaded = load(out)
    assert len(reloaded) == len(SAMPLE_CORPUS)
    assert reloaded.dim == 16
    assert reloaded.get_metadata("doc-0") == {"text": SAMPLE_CORPUS[0]}

    # Sidecar .npz has aligned ids, texts, and vectors for ground truth.
    npz = np.load(out.with_suffix(".npz"), allow_pickle=True)
    assert list(npz["ids"]) == [f"doc-{i}" for i in range(len(SAMPLE_CORPUS))]
    assert npz["vectors"].shape == (len(SAMPLE_CORPUS), 16)
    assert list(npz["texts"]) == SAMPLE_CORPUS
    assert index.dim == 16


def test_generate_from_corpus_file(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("one\ntwo\nthree\n", encoding="utf-8")
    index = generate(
        tmp_path / "out.vfidx", corpus_path=corpus, embed_fn=_fake_embedder(dim=12)
    )
    assert len(index) == 3


def test_generate_empty_corpus_raises(tmp_path: Path) -> None:
    corpus = tmp_path / "empty.txt"
    corpus.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        generate(tmp_path / "out.vfidx", corpus_path=corpus, embed_fn=_fake_embedder())
