ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MPLBACKEND=Agg
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash build-essential ca-certificates git libgomp1 pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt docker/constraints.txt /tmp/dependencies/
RUN python -m pip install --no-cache-dir -r /tmp/dependencies/requirements.txt \
    -c /tmp/dependencies/constraints.txt \
    && python -m pip freeze > /opt/pipeline-packages.txt
COPY scripts/install_leanctx.py /app/scripts/install_leanctx.py
# Runs inside the target Linux architecture; never copies the host's binary.
RUN python scripts/install_leanctx.py \
    && .tools/leanctx/3.10.1/lean-ctx --version
COPY wp1 /app/wp1
COPY scripts /app/scripts
COPY docker /app/docker
COPY data/smoke_sympy_17139.json /app/data/smoke_sympy_17139.json
RUN mkdir -p /app/data/repos /app/results \
    && python -m compileall -q wp1 scripts \
    && graphify --help > /dev/null
ENTRYPOINT ["python3", "/app/docker/run.py"]

FROM base AS test
RUN python -m pip install --no-cache-dir pytest
COPY tests /app/tests
COPY pytest.ini /app/pytest.ini
RUN python -m pytest -q

FROM base AS runtime
