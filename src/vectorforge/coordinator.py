"""Query coordinator over a set of shards (Phase 5, Day 31-32).

The coordinator is the front door to a sharded VectorForge cluster. It uses the
consistent hash ring to decide which shard owns a given vector, and it merges
results back together for the client.

The key asymmetry, and the thing worth explaining in an interview:

  - Writes and deletes are keyed by vector id. The ring hashes the id to exactly
    one shard, so an insert or delete goes to a single shard.

  - Search has no id to hash. A query vector's nearest neighbours could sit on
    any shard, so search fans out to *every* shard in parallel, then merges the
    per-shard top-k lists and keeps the global top-k by distance. This is
    correct because the global nearest neighbours are always a subset of the
    union of the per-shard nearest neighbours.

The coordinator talks to shards through the small ShardClient protocol below, so
the fan-out and merge logic can be tested in-process with LocalShardClient and
run in production over gRPC with GrpcShardClient, with no change to Coordinator
itself.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from vectorforge.filtering import compile_equality_filter
from vectorforge.hashing import ConsistentHashRing
from vectorforge.hnsw import HNSWIndex


class ShardClient(Protocol):
    """What the coordinator needs from a single shard.

    Both the in-process and gRPC implementations satisfy this, and filters
    travel as plain dicts so the same call works locally or over the wire.
    """

    def index(
        self, vector_id: str, vector: NDArray[np.float32], metadata: dict | None = None
    ) -> None: ...

    def search(
        self,
        vector: NDArray[np.float32],
        k: int,
        ef: int | None = None,
        filter: dict | None = None,
    ) -> list[tuple[str, float]]: ...

    def delete(self, vector_id: str) -> bool: ...


class LocalShardClient:
    """A ShardClient backed by an in-process HNSWIndex.

    Useful for running several shards inside one process (tests, a single-node
    dev cluster) without any networking.
    """

    def __init__(self, index: HNSWIndex) -> None:
        self.index_obj = index

    def index(
        self, vector_id: str, vector: NDArray[np.float32], metadata: dict | None = None
    ) -> None:
        self.index_obj.add(vector_id, vector, metadata=metadata)

    def search(
        self,
        vector: NDArray[np.float32],
        k: int,
        ef: int | None = None,
        filter: dict | None = None,
    ) -> list[tuple[str, float]]:
        return self.index_obj.search(
            vector, k=k, ef=ef, predicate=compile_equality_filter(filter)
        )

    def delete(self, vector_id: str) -> bool:
        return self.index_obj.delete(vector_id)


class GrpcShardClient:
    """A ShardClient that talks to a remote shard over gRPC.

    The generated stubs are imported lazily so this module (and LocalShardClient
    with it) stays importable before protoc has been run.
    """

    def __init__(self, target: str) -> None:
        import grpc

        # Local import: these only exist after codegen (see README).
        try:
            from vectorforge import vectorforge_pb2 as pb
            from vectorforge import vectorforge_pb2_grpc as pb_grpc
        except ImportError:  # pragma: no cover
            import vectorforge_pb2 as pb  # type: ignore[no-redef]
            import vectorforge_pb2_grpc as pb_grpc  # type: ignore[no-redef]

        self._pb = pb
        self._channel = grpc.insecure_channel(target)
        self._stub = pb_grpc.VectorForgeStub(self._channel)

    def index(
        self, vector_id: str, vector: NDArray[np.float32], metadata: dict | None = None
    ) -> None:
        req = self._pb.IndexRequest(id=vector_id, vector=list(map(float, vector)))
        if metadata:
            req.metadata.update(metadata)
        self._stub.Index(req)

    def search(
        self,
        vector: NDArray[np.float32],
        k: int,
        ef: int | None = None,
        filter: dict | None = None,
    ) -> list[tuple[str, float]]:
        req = self._pb.SearchRequest(vector=list(map(float, vector)), k=k)
        if ef is not None:
            req.ef = ef
        if filter:
            req.filter.update(filter)
        resp = self._stub.Search(req)
        return [(n.id, n.distance) for n in resp.results]

    def delete(self, vector_id: str) -> bool:
        return self._stub.Delete(self._pb.DeleteRequest(id=vector_id)).deleted

    def close(self) -> None:
        self._channel.close()


class Coordinator:
    """Routes writes by id and fans search out across all shards.

    Parameters
    ----------
    shards:
        Mapping of shard id to its ShardClient.
    virtual_nodes:
        Passed through to the hash ring; controls load smoothing.
    max_workers:
        Thread-pool size for the search fan-out. Shard calls are I/O bound
        (a network round trip, or a search that releases the GIL), so threads
        give real parallelism here. Defaults to one worker per shard.
    """

    def __init__(
        self,
        shards: dict[str, ShardClient],
        virtual_nodes: int = 150,
        max_workers: int | None = None,
    ) -> None:
        self._shards = dict(shards)
        self._ring = ConsistentHashRing(self._shards.keys(), virtual_nodes=virtual_nodes)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or max(1, len(self._shards)),
            thread_name_prefix="coordinator-fanout",
        )

    # ------------------------------------------------------------------
    # Writes: routed to the single owning shard
    # ------------------------------------------------------------------

    def index(
        self, vector_id: str, vector: NDArray[np.float32], metadata: dict | None = None
    ) -> str:
        """Insert a vector on the shard that owns its id. Returns that shard id."""
        node = self._ring.get_node(vector_id)
        if node is None:
            raise RuntimeError("cannot index into a cluster with no shards")
        self._shards[node].index(vector_id, vector, metadata=metadata)
        return node

    def delete(self, vector_id: str) -> bool:
        """Delete a vector from its owning shard. False if there are no shards."""
        node = self._ring.get_node(vector_id)
        if node is None:
            return False
        return self._shards[node].delete(vector_id)

    # ------------------------------------------------------------------
    # Search: fanned out to every shard, then merged
    # ------------------------------------------------------------------

    def search(
        self,
        vector: NDArray[np.float32],
        k: int,
        ef: int | None = None,
        filter: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Query all shards in parallel and return the merged global top-k."""
        if not self._shards:
            return []

        futures = [
            self._pool.submit(client.search, vector, k, ef, filter)
            for client in self._shards.values()
        ]

        merged: list[tuple[str, float]] = []
        for future in futures:
            merged.extend(future.result())

        # Each shard already returned its own nearest k; the global nearest k
        # are the closest of that union by distance.
        merged.sort(key=lambda pair: pair[1])
        return merged[:k]

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    def owner_of(self, vector_id: str) -> str | None:
        """Which shard owns *vector_id* (without touching the shard)."""
        return self._ring.get_node(vector_id)

    @property
    def shard_ids(self) -> set[str]:
        return set(self._shards)

    def close(self) -> None:
        """Shut down the fan-out thread pool."""
        self._pool.shutdown(wait=True)

    def __enter__(self) -> Coordinator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
