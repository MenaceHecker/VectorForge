"""FastAPI service wrapping the HNSW core index.

Exposes the Phase 3 REST surface (plan Day 15–16):

    POST   /index          insert a vector (+ optional metadata)
    POST   /search         k-NN search, optional metadata filter
    DELETE /vectors/{id}   tombstone a vector
    GET    /health         liveness + basic index stats

Design
------
The core :class:`~vectorforge.hnsw.HNSWIndex` is stateful and not thread-safe
for concurrent writes, so it lives behind a single service object rather than
being touched by handlers directly.  Handlers only translate HTTP ↔ core
calls and map core ``ValueError``/``KeyError`` to the right status codes:

    400  malformed vector (wrong dimensionality)
    404  delete of an unknown id
    409  insert of a duplicate id

Metadata filtering over HTTP cannot carry a Python callable, so ``/search``
accepts a JSON ``filter`` object and compiles it into an equality predicate
(every key must match).  This keeps the wire format declarative and forward
-compatible with a richer predicate grammar later.

Use :func:`create_app` to inject an index (tests do this); importing ``app``
builds one from ``VECTORFORGE_DIM`` / ``VECTORFORGE_M`` /
``VECTORFORGE_EF_CONSTRUCTION`` environment variables for ``uvicorn``.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from vectorforge.hnsw import HNSWIndex
from vectorforge.metrics import Metrics

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    id: str = Field(..., min_length=1, description="Caller-supplied vector id.")
    vector: list[float] = Field(..., description="Embedding; length must equal index dim.")
    metadata: dict | None = Field(
        default=None, description="Optional JSON payload filterable at search time."
    )


class SearchRequest(BaseModel):
    vector: list[float] = Field(..., description="Query embedding; length must equal dim.")
    k: int = Field(default=10, ge=1, description="Number of neighbours to return.")
    ef: int | None = Field(
        default=None, ge=1, description="Search beam width; larger = better recall, slower."
    )
    filter: dict | None = Field(
        default=None,
        description="Equality filter; a result is kept only if every key matches its metadata.",
    )


class Neighbor(BaseModel):
    id: str
    distance: float


class SearchResponse(BaseModel):
    results: list[Neighbor]


class HealthResponse(BaseModel):
    status: str
    size: int
    dim: int


# ---------------------------------------------------------------------------
# Filter compilation
# ---------------------------------------------------------------------------


def _compile_filter(spec: dict | None) -> Callable[[dict], bool] | None:
    """Turn a JSON equality ``spec`` into a metadata predicate.

    ``{"group": 0, "lang": "en"}`` → keep vectors whose metadata has
    ``group == 0`` **and** ``lang == "en"``.  ``None``/empty means no filter.
    """
    if not spec:
        return None

    def predicate(meta: dict) -> bool:
        return all(meta.get(key) == value for key, value in spec.items())

    return predicate


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(index: HNSWIndex) -> FastAPI:
    """Build a FastAPI app serving *index*."""
    app = FastAPI(
        title="VectorForge",
        description="Distributed vector search engine — HNSW core, REST surface.",
        version="0.1.0",
    )
    app.state.index = index
    metrics = Metrics()
    app.state.metrics = metrics

    def _as_vector(values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    @app.post("/index", status_code=201)
    def index_vector(req: IndexRequest) -> dict[str, str]:
        try:
            index.add(req.id, _as_vector(req.vector), metadata=req.metadata)
        except ValueError as exc:
            # Duplicate id vs. shape mismatch carry different semantics.
            status = 409 if "already exists" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return {"id": req.id, "status": "indexed"}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        try:
            # Time only the core search so the latency metric reflects the
            # index, not request parsing or JSON serialization.
            with metrics.query_latency.time():
                hits = index.search(
                    _as_vector(req.vector),
                    k=req.k,
                    ef=req.ef,
                    predicate=_compile_filter(req.filter),
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SearchResponse(results=[Neighbor(id=vid, distance=dist) for vid, dist in hits])

    @app.delete("/vectors/{vector_id}")
    def delete_vector(vector_id: str) -> dict[str, str]:
        if not index.delete(vector_id):
            raise HTTPException(status_code=404, detail=f"Unknown vector id {vector_id!r}")
        return {"id": vector_id, "status": "deleted"}

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", size=len(index), dim=index.dim)

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        # Refresh the size gauge at scrape time so it is always current without
        # having to touch it on every insert and delete.
        metrics.set_index_size(len(index))
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def _index_from_env() -> HNSWIndex:
    return HNSWIndex(
        dim=int(os.environ.get("VECTORFORGE_DIM", "128")),
        M=int(os.environ.get("VECTORFORGE_M", "16")),
        ef_construction=int(os.environ.get("VECTORFORGE_EF_CONSTRUCTION", "200")),
    )


# Module-level ASGI app for ``uvicorn vectorforge.api:app``.
app = create_app(_index_from_env())
