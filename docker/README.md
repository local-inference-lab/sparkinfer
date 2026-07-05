# Docker: build b12x from source into a vLLM serving image

The published `voipmonitor/vllm` images ship a pinned b12x. This Dockerfile
overlays the current source checkout on top of one of those images, so kernel
changes can be tested end-to-end (vLLM serve, CUDA graphs, TP) without
rebuilding vLLM itself.

## How b12x "builds"

There is no ahead-of-time kernel compilation. b12x is a pure-Python package of
CuTe DSL kernels (`nvidia-cutlass-dsl`); they are JIT-compiled for the target
GPU at engine startup via `CUTE_DSL_ARCH=sm_120a`. First engine start after a
kernel change spends a few extra minutes in JIT. Installing from source is a
plain pip install:

```bash
pip install --no-deps --force-reinstall .
```

(`--no-deps` because the vLLM base image already pins torch / cutlass-dsl /
tvm-ffi — only the b12x package itself should be replaced.)

## Build

From the repo root:

```bash
docker build -f docker/Dockerfile -t b12x-dev .
# or against a different base:
docker build -f docker/Dockerfile --build-arg BASE=voipmonitor/vllm:<tag> -t b12x-dev .
```

## Run the test suite

```bash
docker run --rm --gpus '"device=0"' --ipc=host b12x-dev \
  bash -c "/opt/venv/bin/pip install -q pytest && /opt/venv/bin/python -m pytest /src/b12x/tests -q"
```

## Serve a model

The base image's serve scripts and vLLM integration are unchanged — use them
as documented for the base image, e.g.:

```bash
docker run --rm --gpus all --ipc=host --network host \
  -v /path/to/model:/models/<name>:ro \
  --entrypoint /serve/<serve-script>.sh b12x-dev --port 8001
```
