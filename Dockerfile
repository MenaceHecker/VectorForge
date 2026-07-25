# VectorForge — multi-stage image for the FastAPI query service.
#
# Stage 1 (builder) compiles the package and its dependencies into an isolated
# virtualenv.  Stage 2 (runtime) copies only that venv onto a slim base, so the
# final image carries no build toolchain, caches, or source tree.  This keeps
# the image well under 500 MB despite NumPy + Numba (llvmlite) being heavy.
#
#   docker build -t vectorforge .
#   docker run -p 8000:8000 -e VECTORFORGE_DIM=128 vectorforge

# ---------------------------------------------------------------------------
# Stage 1 — builder: install into a self-contained venv
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Build the wheel/deps in an isolated venv we can copy wholesale to runtime.
ENV VENV=/opt/venv
RUN python -m venv "$VENV"
ENV PATH="$VENV/bin:$PATH"

WORKDIR /build

# pyproject reads README.md (project.readme), so it must be present to build.
COPY pyproject.toml README.md ./
COPY src ./src

# --no-cache-dir keeps no pip cache in the layer; the venv is all we carry on.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim base + the prebuilt venv only
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Fail fast, no .pyc clutter, unbuffered logs for container stdout.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Default index geometry; override per-deployment (k8s env / docker -e).
ENV VECTORFORGE_DIM=128 \
    VECTORFORGE_M=16 \
    VECTORFORGE_EF_CONSTRUCTION=200

COPY --from=builder /opt/venv /opt/venv

# Run as an unprivileged user — never serve as root.
RUN useradd --create-home --uid 1000 vectorforge
USER vectorforge

EXPOSE 8000

# Liveness probe using the stdlib (slim has no curl/wget).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"]

CMD ["uvicorn", "vectorforge.api:app", "--host", "0.0.0.0", "--port", "8000"]
