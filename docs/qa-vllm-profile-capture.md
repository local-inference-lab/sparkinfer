# QA vLLM Profile Capture

Capture a short rank-0 Torch trace from a busy vLLM server.

## vLLM Args

Add these args to the `vllm serve ...` command:

```bash
--profiler-config.profiler=torch \
--profiler-config.torch_profiler_dir=/tmp/vllm-profile/qa-run \
--profiler-config.torch_profiler_with_stack=true \
--profiler-config.torch_profiler_record_shapes=false \
--profiler-config.torch_profiler_with_memory=false \
--profiler-config.torch_profiler_with_flops=false \
--profiler-config.torch_profiler_use_gzip=true \
--profiler-config.torch_profiler_dump_cuda_time_total=false \
--profiler-config.ignore_frontend=true \
--profiler-config.delay_iterations=0 \
--profiler-config.max_iterations=4 \
--profiler-config.warmup_iterations=0 \
--profiler-config.active_iterations=5 \
--profiler-config.wait_iterations=0
```

Use a unique `torch_profiler_dir` per run.

## Setup

```bash
export BASE=http://127.0.0.1:8000
export MODEL=<served-model-name>
export PROFILE_DIR=/tmp/vllm-profile/qa-run
```

## Prefill Capture

Start a long prefill request from an external client: very long input, tiny output, for example `max_tokens=1`.
Trigger profiling immediately after submitting the request.

```bash
curl -fsS -X POST "$BASE/start_profile" -H 'Content-Type: application/json' -d '{}'
```

## Decode Capture

Start a long decode request from an external client: short input, large output, for example thousands of generated tokens.
Trigger profiling after tokens begin streaming.

```bash
curl -fsS -X POST "$BASE/start_profile" -H 'Content-Type: application/json' -d '{}'
```

## Result

The profiler auto-stops after 4 iterations. The trace appears under:

```bash
find "$PROFILE_DIR" -type f \( -name '*.pt.trace.json.gz' -o -name '*.pt.trace.json' \) -print
```

If no trace appears, make the prefill prompt or decode output longer and repeat.
