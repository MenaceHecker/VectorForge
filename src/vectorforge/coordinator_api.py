"""FastAPI front door for a sharded VectorForge cluster (Phase 5, Day 33).

This is the public REST surface that sits in front of the shards. It speaks the
same request and response shapes as the single-node API, so a client cannot tell
whether it is talking to one index or a fanned-out cluster. Under the hood every
call goes through a Coordinator: writes route to one shard by id, searches fan
out to all shards and merge.

Build it with create_coordinator_app(coordinator) in tests (inject a Coordinator
over LocalShardClients). Importing `app` wires up GrpcShardClients from the
VECTORFORGE_SHARDS environment variable, for example:

    VECTORFORGE_SHARDS="vf-shard-0:50051,vf-shard-1:50051,vf-shard-2:50051"
"""

from __future__ import annotations

import os

import grpc
import numpy as np
from fastapi import Depends, FastAPI, HTTPException

from vectorforge.api import IndexRequest, Neighbor, SearchRequest, SearchResponse
from vectorforge.auth import make_api_key_dependency
from vectorforge.coordinator import Coordinator, GrpcShardClient

# Map the shard-side error codes onto HTTP, whether they arrive as a local
# ValueError (in-process shards) or a gRPC error (remote shards).
_GRPC_TO_HTTP = {
    grpc.StatusCode.ALREADY_EXISTS: 409,
    grpc.StatusCode.INVALID_ARGUMENT: 400,
}


def _http_error_from_shard(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        status = 409 if "already exists" in str(exc) else 400
        return HTTPException(status_code=status, detail=str(exc))
    if isinstance(exc, grpc.RpcError):
        status = _GRPC_TO_HTTP.get(exc.code(), 502)
        return HTTPException(status_code=status, detail=exc.details() or str(exc))
    raise exc


def create_coordinator_app(coordinator: Coordinator, api_key: str | None = None) -> FastAPI:
    """Build a FastAPI app that serves requests through *coordinator*.

    *api_key* guards the write endpoints, defaulting to ``VECTORFORGE_API_KEY``
    from the environment; unset on both means the write endpoints are open.
    """
    app = FastAPI(
        title="VectorForge Coordinator",
        description="Public REST surface fanning out to VectorForge shards.",
        version="0.1.0",
    )
    app.state.coordinator = coordinator

    effective_key = api_key if api_key is not None else os.environ.get("VECTORFORGE_API_KEY")
    require_write = Depends(make_api_key_dependency(effective_key))

    def _as_vector(values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    @app.post("/index", status_code=201, dependencies=[require_write])
    def index_vector(req: IndexRequest) -> dict[str, str]:
        try:
            shard = coordinator.index(req.id, _as_vector(req.vector), metadata=req.metadata)
        except (ValueError, grpc.RpcError) as exc:
            raise _http_error_from_shard(exc) from exc
        return {"id": req.id, "status": "indexed", "shard": shard}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        try:
            hits = coordinator.search(
                _as_vector(req.vector), k=req.k, ef=req.ef, filter=req.filter
            )
        except (ValueError, grpc.RpcError) as exc:
            raise _http_error_from_shard(exc) from exc
        return SearchResponse(results=[Neighbor(id=vid, distance=dist) for vid, dist in hits])

    @app.delete("/vectors/{vector_id}", dependencies=[require_write])
    def delete_vector(vector_id: str) -> dict[str, str]:
        if not coordinator.delete(vector_id):
            raise HTTPException(status_code=404, detail=f"Unknown vector id {vector_id!r}")
        return {"id": vector_id, "status": "deleted"}

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "shards": sorted(coordinator.shard_ids)}

    return app


def _coordinator_from_env() -> Coordinator:
    targets = [t.strip() for t in os.environ.get("VECTORFORGE_SHARDS", "").split(",") if t.strip()]
    shards = {target: GrpcShardClient(target) for target in targets}
    return Coordinator(shards)


# Module-level ASGI app for `uvicorn vectorforge.coordinator_api:app`.
app = create_coordinator_app(_coordinator_from_env())
