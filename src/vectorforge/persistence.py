"""Disk persistence for HNSWIndex.

Saves and loads the full graph state in a custom binary format.  No pickle —
using pickle would hide the serialisation cost and be unsafe for untrusted
files.  This implementation uses :mod:`struct` for fixed-width header fields
and ``ndarray.tobytes()`` / ``numpy.frombuffer`` for vector data.

Binary format (little-endian throughout)
-----------------------------------------

HEADER  (40 bytes, fixed)
  [0:8]   magic      8s      b"VFIDX\\x00\\x01\\x00"
  [8:10]  version    H       1
  [10:14] dim        I
  [14:18] M          I
  [18:22] ef_constr  I
  [22:26] max_layer  I
  [26:34] num_nodes  Q       total nodes including tombstoned
  [34:42] num_del    Q       number of tombstoned internal_ids

ENTRY POINT  (8 bytes)
  entry_point        q       signed int64; -1 means empty index

NODE RECORDS  (num_nodes entries, variable length)
  internal_id        Q
  level              I
  vid_len            H
  vid                <vid_len bytes, UTF-8>
  vector             <dim × 4 bytes, float32 little-endian>
  meta_len           I       (v2+ only) byte length of the JSON blob; 0 = None
  meta               <meta_len bytes, UTF-8 JSON>   (v2+ only, omitted if 0)
  for layer in 0 … level:
    num_neighbors    I
    neighbors        <num_neighbors × 8 bytes, uint64>

DELETED SET  (at end of file)
  num_deleted        Q       (same value as header num_del, for verification)
  deleted_ids        <num_deleted × 8 bytes, uint64>

Versioning
  v1 — no per-node metadata (the meta_len/meta fields are absent).
  v2 — adds the length-prefixed JSON metadata blob shown above.  v1 files
       still load correctly (their nodes come back with metadata=None); new
       saves always write v2.

Why this design?
  - Fixed-length header allows O(1) metadata reads without parsing the body.
  - Length-prefixed strings (vid_len) support arbitrary UTF-8 ids.
  - Vectors stored as raw float32 bytes — zero copy on load via frombuffer.
  - Neighbour lists stored per layer so multi-layer structure is preserved.
  - Metadata stored as UTF-8 JSON (not pickle) — safe to load from untrusted
    files and human-inspectable, at the cost of a small serialisation step.
  - Deleted set appended last so saves stay append-friendly in future.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from vectorforge.hnsw import HNSWIndex, _Node

# Format constants

MAGIC = b"VFIDX\x00\x01\x00"   # 8 bytes
VERSION = 2                    # current write version
SUPPORTED_VERSIONS = frozenset({1, 2})

# Header layout (little-endian)
_HEADER_FMT = "<8sHIIIIQQ"       # magic, version, dim, M, ef, max_layer, num_nodes, num_del
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)   # == 42 bytes
_EP_FMT = "<q"                   # entry point (signed, -1 = None)
_EP_SIZE = struct.calcsize(_EP_FMT)


# Public API


def save(index: HNSWIndex, path: str | Path) -> None:
    """Serialise *index* to *path* in the VectorForge binary format.

    The file is written atomically: data goes to ``<path>.tmp`` first,
    then renamed over the target so a crash mid-write never leaves a
    corrupt index on disk.

    Parameters
    ----------
    index : HNSWIndex
        The index to persist.  May be queried concurrently during save
        (this function holds no locks — callers that need consistency
        must coordinate externally).
    path : str or Path
        Destination file path.  Parent directories must already exist.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")

    num_nodes = len(index._nodes)
    num_deleted = len(index._deleted)
    ep = index._entry_point if index._entry_point is not None else -1

    with tmp.open("wb") as fh:
        # --- Header ---
        fh.write(
            struct.pack(
                _HEADER_FMT,
                MAGIC,
                VERSION,
                index.dim,
                index.M,
                index.ef_construction,
                index._max_layer,
                num_nodes,
                num_deleted,
            )
        )
        # Entry point
        fh.write(struct.pack(_EP_FMT, ep))

        # Node records
        for node in index._nodes.values():
            vid_bytes = node.vector_id.encode("utf-8")
            fh.write(struct.pack("<QIH", node.internal_id, node.level, len(vid_bytes)))
            fh.write(vid_bytes)
            fh.write(node.vector.astype("<f4").tobytes())

            # Metadata (v2): length-prefixed compact JSON; 0-length means None.
            if node.metadata:
                meta_bytes = json.dumps(node.metadata, separators=(",", ":")).encode("utf-8")
            else:
                meta_bytes = b""
            fh.write(struct.pack("<I", len(meta_bytes)))
            fh.write(meta_bytes)

            for layer in range(node.level + 1):
                nb_ids = list(node.neighbors.get(layer, set()))
                fh.write(struct.pack("<I", len(nb_ids)))
                if nb_ids:
                    fh.write(np.array(nb_ids, dtype="<u8").tobytes())

        # Deleted set
        fh.write(struct.pack("<Q", num_deleted))
        if num_deleted:
            fh.write(np.array(list(index._deleted), dtype="<u8").tobytes())

    tmp.replace(path)


def load(path: str | Path) -> HNSWIndex:
    """Deserialise an HNSWIndex from *path*.

    Returns
    -------
    HNSWIndex
        A fully reconstructed index ready to serve queries.

    Raises
    ------
    ValueError
        If the file magic bytes or version do not match expectations.
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)

    with path.open("rb") as fh:
        # Header
        raw = fh.read(_HEADER_SIZE)
        (magic, version, dim, M, ef_construction, max_layer, num_nodes, num_deleted) = (
            struct.unpack(_HEADER_FMT, raw)
        )
        _validate_header(magic, version)

        # Entry point
        (ep_raw,) = struct.unpack(_EP_FMT, fh.read(_EP_SIZE))
        entry_point: int | None = None if ep_raw == -1 else ep_raw

        # Reconstruct the index shell (skip __init__ side effects on mutable
        # state by calling __init__ then overwriting the fields we control).
        index = HNSWIndex(dim=dim, M=M, ef_construction=ef_construction)
        index._max_layer = max_layer
        index._entry_point = entry_point

        # Node records
        for _ in range(num_nodes):
            internal_id, level, vid_len = struct.unpack("<QIH", fh.read(14))
            vector_id = fh.read(vid_len).decode("utf-8")
            vector = np.frombuffer(fh.read(dim * 4), dtype="<f4").copy()

            # Metadata (v2+): absent in v1 files, which load as metadata=None.
            metadata: dict | None = None
            if version >= 2:
                (meta_len,) = struct.unpack("<I", fh.read(4))
                if meta_len:
                    metadata = json.loads(fh.read(meta_len).decode("utf-8"))

            node = _Node(
                internal_id=internal_id,
                vector_id=vector_id,
                vector=vector,
                level=level,
                metadata=metadata,
            )

            for layer in range(level + 1):
                (num_nb,) = struct.unpack("<I", fh.read(4))
                if num_nb:
                    nb_bytes = fh.read(num_nb * 8)
                    nb_ids = np.frombuffer(nb_bytes, dtype="<u8").tolist()
                    node.neighbors[layer] = set(int(x) for x in nb_ids)

            index._nodes[internal_id] = node
            index._id_map[vector_id] = internal_id
            index._next_id = max(index._next_id, internal_id + 1)

        # --- Deleted set ---
        (num_del_check,) = struct.unpack("<Q", fh.read(8))
        if num_del_check != num_deleted:
            raise ValueError(
                f"Deleted-set count mismatch: header={num_deleted}, "
                f"footer={num_del_check}"
            )
        if num_deleted:
            del_bytes = fh.read(num_deleted * 8)
            index._deleted = set(int(x) for x in np.frombuffer(del_bytes, dtype="<u8"))

    return index

# Internal helpers


def _validate_header(magic: bytes, version: int) -> None:
    if magic != MAGIC:
        raise ValueError(
            f"Unrecognised file magic {magic!r}; expected {MAGIC!r}. "
            "Is this a VectorForge index file?"
        )
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported index version {version}; this build supports versions "
            f"{sorted(SUPPORTED_VERSIONS)}."
        )