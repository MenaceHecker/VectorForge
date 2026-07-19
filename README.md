# VectorForge

A distributed vector search engine, built from scratch — HNSW indexing,
a FastAPI/gRPC query layer, consistent-hash sharding, and a recall/latency
benchmark suite.

**Status:** Phase 1 — Core Index (in progress)

## Stack

Python 3.11 · NumPy/Numba · FastAPI · gRPC · Kubernetes · Prometheus · Terraform

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Roadmap

- [x] Phase 1 — Core index & brute-force baseline (in progress)
- [ ] Phase 2 — Multi-layer HNSW & persistence
- [ ] Phase 3 — FastAPI + gRPC API layer
- [ ] Phase 4 — Kubernetes & observability
- [ ] Phase 5 — Distributed sharding
- [ ] Phase 6 — Benchmarks, polish, launch

> Architecture diagram and benchmark results will be added here in Phase 6.
