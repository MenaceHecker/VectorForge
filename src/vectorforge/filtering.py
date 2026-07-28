"""Shared metadata-filter compilation.

The wire formats (REST JSON, gRPC Struct) all carry a metadata filter as a plain
object like ``{"lang": "en", "group": 1}``. The core index, though, wants a
``dict -> bool`` predicate. This is the one place that translation lives, so the
REST layer, the gRPC servicer, and the coordinator's local shard client all
agree on exactly what a filter means: keep a vector only if every key in the
filter matches its metadata.
"""

from __future__ import annotations

from collections.abc import Callable


def compile_equality_filter(spec: dict | None) -> Callable[[dict], bool] | None:
    """Compile an equality ``spec`` into a metadata predicate (None if empty)."""
    if not spec:
        return None

    def predicate(meta: dict) -> bool:
        return all(meta.get(key) == value for key, value in spec.items())

    return predicate
