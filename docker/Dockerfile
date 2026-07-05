# Build a vLLM serving image with b12x installed from this source checkout.
#
# The published voipmonitor/vllm images ship a pinned b12x; this overlays the
# repo's working tree on top so kernel changes can be tested end-to-end
# without rebuilding vLLM. b12x kernels are pure-Python CuTe DSL — there is
# no ahead-of-time compile step; they JIT for the target GPU at engine start
# (CUTE_DSL_ARCH=sm_120a).
#
# Build (from the repo root):
#   docker build -f docker/Dockerfile -t b12x-dev .
#
# Pick a different vLLM base:
#   docker build -f docker/Dockerfile --build-arg BASE=voipmonitor/vllm:<tag> -t b12x-dev .
ARG BASE=voipmonitor/vllm:eldritch-enlightenment-v45c1582-b12xf3686b5-pc1441b5-cu132-20260704
FROM ${BASE}

COPY . /src/b12x
# --no-deps: torch / cutlass-dsl / tvm-ffi are already pinned in the base
# image; only the b12x package itself is replaced.
RUN /opt/venv/bin/pip install --no-deps --force-reinstall /src/b12x \
    && /opt/venv/bin/python -c "import b12x; print('b12x', b12x.__version__ if hasattr(b12x, '__version__') else 'installed')"

ENV CUTE_DSL_ARCH=sm_120a
