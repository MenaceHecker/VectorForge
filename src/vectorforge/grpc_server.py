"""gRPC servicer wrapping the HNSW core index.

The low-latency internal counterpart to the REST layer in
:mod:`vectorforge.api`, implementing the service defined in
``protos/vectorforge.proto``.  Each shard runs one of these; the Phase 5
coordinator is just another gRPC client fanning out to them.

Prerequisites
-------------
Generate the protobuf stubs before importing this module::

    python -m grpc_tools.protoc -I protos \\
      --python_out=src/vectorforge --grpc_python_out=src/vectorforge \\
      protos/vectorforge.proto

and add ``grpcio`` + ``grpcio-tools`` to the project dependencies.

Error mapping (mirrors the REST status codes)
---------------------------------------------
    ALREADY_EXISTS    duplicate id on Index
    INVALID_ARGUMENT  wrong vector dimensionality
    (Delete reports absence via the ``deleted`` flag rather than aborting,
     since a missing id is a normal, expected result for internal callers.)
"""

from __future__ import annotations

import logging
import os
from concurrent import futures

import grpc
import numpy as np
from google.protobuf.json_format import MessageToDict

from vectorforge.filtering import compile_equality_filter
from vectorforge.hnsw import HNSWIndex

logger = logging.getLogger(__name__)

# Generated stubs live next to this module once protoc has run.  Fall back to a
# top-level import so either --python_out layout (package dir vs. src root)
# works without editing this file.
try:  # pragma: no cover - import shim, exercised only after codegen
    from vectorforge import vectorforge_pb2 as pb
    from vectorforge import vectorforge_pb2_grpc as pb_grpc
except ImportError:  # pragma: no cover
    import vectorforge_pb2 as pb  # type: ignore[no-redef]
    import vectorforge_pb2_grpc as pb_grpc  # type: ignore[no-redef]


def _struct_to_dict(struct) -> dict:
    """Convert a ``google.protobuf.Struct`` field to a plain Python dict."""
    return MessageToDict(struct)


class VectorForgeServicer(pb_grpc.VectorForgeServicer):
    """Adapts the stateful :class:`HNSWIndex` to the generated service ABC."""

    def __init__(self, index: HNSWIndex) -> None:
        self._index = index

    def Index(self, request, context):  # noqa: N802 - gRPC method name
        metadata = _struct_to_dict(request.metadata) if request.HasField("metadata") else None
        try:
            self._index.add(
                request.id,
                np.asarray(request.vector, dtype=np.float32),
                metadata=metadata,
            )
        except ValueError as exc:
            code = (
                grpc.StatusCode.ALREADY_EXISTS
                if "already exists" in str(exc)
                else grpc.StatusCode.INVALID_ARGUMENT
            )
            context.abort(code, str(exc))
        return pb.IndexResponse(id=request.id, status="indexed")

    def Search(self, request, context):  # noqa: N802 - gRPC method name
        predicate = compile_equality_filter(
            _struct_to_dict(request.filter) if request.HasField("filter") else None
        )
        try:
            hits = self._index.search(
                np.asarray(request.vector, dtype=np.float32),
                k=request.k,
                ef=request.ef if request.HasField("ef") else None,
                predicate=predicate,
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return pb.SearchResponse(
            results=[pb.Neighbor(id=vid, distance=dist) for vid, dist in hits]
        )

    def Delete(self, request, context):  # noqa: N802 - gRPC method name
        deleted = self._index.delete(request.id)
        return pb.DeleteResponse(id=request.id, deleted=deleted)

    def Health(self, request, context):  # noqa: N802 - gRPC method name
        return pb.HealthResponse(status="ok", size=len(self._index), dim=self._index.dim)


def serve(index: HNSWIndex, port: int = 50051, max_workers: int = 8) -> grpc.Server:
    """Start (but do not block on) a gRPC server bound to *port*.

    Returns the running :class:`grpc.Server`; the caller owns its lifecycle
    (``server.wait_for_termination()`` to block, ``server.stop(grace)`` to
    shut down).
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb_grpc.add_VectorForgeServicer_to_server(VectorForgeServicer(index), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server


def main() -> None:
    """Run a shard as a standalone gRPC server until killed.

    Reads the index geometry and port from the environment so one image can be
    deployed many times as separate shards. This is the container entrypoint
    for a shard pod: `python -m vectorforge.grpc_server`.
    """
    logging.basicConfig(level=logging.INFO)
    index = HNSWIndex(
        dim=int(os.environ.get("VECTORFORGE_DIM", "128")),
        M=int(os.environ.get("VECTORFORGE_M", "16")),
        ef_construction=int(os.environ.get("VECTORFORGE_EF_CONSTRUCTION", "200")),
    )
    port = int(os.environ.get("VECTORFORGE_GRPC_PORT", "50051"))
    server = serve(index, port=port)
    logger.info("shard gRPC server listening on :%d", port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
