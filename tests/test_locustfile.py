"""Sanity tests for the Locust load test (Phase 6, Day 36-37).

A locustfile is not really unit-testable (it drives a live server), but we can
still guard against it drifting: confirm it imports, exposes a user class with
the expected tasks, and that its vector helper produces correctly shaped input.

Importantly, this runs in a *subprocess*. Importing `locust` calls
gevent.monkey.patch_all(), which globally patches the stdlib threading
primitives for the whole process. Doing that inside the pytest process would
break every threaded test that runs afterwards (TestClient, the gRPC servers,
the index's own reader-writer lock), so we isolate the import here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LOCUSTFILE = REPO / "locust" / "locustfile.py"

# Validation script run in a fresh interpreter. Prints one token so the parent
# can tell "locust missing" (skip) apart from "locustfile broken" (fail).
_CHECK = f"""
import importlib.util, sys
try:
    import locust
    from locust import HttpUser
except ModuleNotFoundError:
    print("NO_LOCUST"); sys.exit(0)

spec = importlib.util.spec_from_file_location("vf_locustfile", r"{LOCUSTFILE}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

assert issubclass(mod.VectorForgeUser, HttpUser), "user is not an HttpUser"
vec = mod.random_vector(64)
assert len(vec) == 64 and all(isinstance(x, float) for x in vec), "bad vector"
names = {{t.__name__ for t in mod.VectorForgeUser.tasks}}
assert {{"search", "index"}} <= names, f"missing tasks: {{names}}"
print("OK")
"""


def test_locustfile_is_well_formed() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO / "src")},
    )
    if "NO_LOCUST" in result.stdout:
        pytest.skip("locust not installed")
    assert "OK" in result.stdout, f"locustfile check failed:\n{result.stdout}\n{result.stderr}"
