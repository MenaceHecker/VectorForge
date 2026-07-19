"""Smoke test: confirms the package is installed and importable.

This is intentionally trivial — its only job on Day 1 is to give the CI
pipeline something real to run before any actual index logic exists.
"""

import vectorforge


def test_package_imports():
    assert vectorforge.__version__ == "0.1.0"
