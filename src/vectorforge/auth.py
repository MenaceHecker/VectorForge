"""API key authentication for the write endpoints (Phase 6, Day 41).

Reads and deletes change state, so they sit behind an API key; search and health
stay open. Auth is opt-in: if no key is configured the dependency is a no-op, so
local development and the test suite work without ceremony. Set one key and the
write endpoints start requiring it.

The key travels in an `X-API-Key` header and is compared in constant time so a
timing side channel cannot leak it character by character.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"


def make_api_key_dependency(expected_key: str | None) -> Callable:
    """Build a FastAPI dependency that enforces *expected_key* on a request.

    When *expected_key* is ``None`` (or empty), the dependency allows everything,
    which is how auth stays off until a key is configured. Otherwise a request
    must present a matching key in the ``X-API-Key`` header or it is rejected
    with 401.
    """
    header_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

    def verify(provided: str | None = Security(header_scheme)) -> None:
        if not expected_key:
            return  # auth disabled
        if provided is None or not hmac.compare_digest(provided, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    return verify
