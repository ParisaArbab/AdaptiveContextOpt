# Docker validation — 2026-09-10

Completed locally:

- 36 regression tests passed, including the real installed LeanCTX binary,
  platform asset selection, GPU-allocation rejection, model reuse, preserved
  pipeline arguments, saved startup errors, and rejection of incomplete results.
- CPU and NVIDIA Compose configurations validate with Docker Compose 2.31.0.
- Custom model/checkouts bind paths, including spaces, validate.
- Official Python and Ollama image manifests include Linux AMD64 and ARM64.
- Python compilation, shell syntax, and `git diff --check` pass.

Actual image build was attempted after starting Docker Desktop. It was blocked
before dependency installation by DNS in this host's Docker VM. The configured
`docker.iranserver.com` mirror failed to resolve; a second attempt using
`registry-1.docker.io` directly also failed with `no such host`. No global
Docker configuration was changed to bypass this. This is not a passing image
build or a completed A100 test.

Independent Linux AMD64/ARM64 dependency-resolution checks were also attempted
with `uv pip compile`. Both were blocked by PyPI connection timeouts after
three retries. They do not establish successful dependency resolution; the
server's test build remains required.

The image contains a `test` build stage. On the target server run:

```bash
docker build --target test -t adaptive-context-pipeline:test .
PIPELINE_DEVICE=nvidia bash scripts/run_docker.sh
```

The second command checks GPU availability before the smoke benchmark and
records Ollama's actual VRAM allocation. Inspect the resulting
`results/docker_<timestamp>/environment.json` and the Full/RAW outcome files
to verify that server's execution. The host needs a working NVIDIA driver
and Container Toolkit; these cannot be validated on this Mac.

Local logs/config snapshots are in `results/docker_validation_20260910/`.
