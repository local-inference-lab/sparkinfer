---
name: vllm-take-capture
description: Capture a short live vLLM profile from an already-running workload. Use when the user says "take a capture", "capture vLLM", "profile vLLM", or asks to grab a GPU+CPU vLLM trace from the local OpenAI server at 127.0.0.1:8000, automatically bounded to 4 scheduler/engine steps, then copy rank 0 to lukea@orion.local:~/Desktop without probing or health-checking the instance.
---

# vLLM Take Capture

## Default Workflow

Run the helper immediately:

```bash
python skills/vllm-take-capture/scripts/take_vllm_capture.py
```

Do not query `/health`, `/v1/models`, metrics, or other discovery endpoints first. Assume the workload is already running and the server is `http://127.0.0.1:8000`.

The helper:

- Posts directly to `/start_profile` with CPU+CUDA and 4-step hints.
- Does not send inference traffic.
- Relies on the running vLLM profiler config to auto-stop after 4 iterations.
- Finds the newest rank-0 trace emitted after the start request.
- Runs `scp -r <rank0 trace> lukea@orion.local:~/Desktop`.

## If Defaults Fail

Only add options when the direct helper reports a concrete file-location problem:

```bash
python skills/vllm-take-capture/scripts/take_vllm_capture.py --profile-dir /absolute/profile/dir
```

Use `--wait-timeout` if the four-step capture needs more time to flush. Do not manually call `/stop_profile` unless the user explicitly asks for that override; the intended path is the profiler's configured `max_iterations=4` auto-stop.

## Notes

vLLM's public OpenAI server profiling route starts/stops the profiler over HTTP; CPU+CUDA activity selection and step bounds are normally part of the server's profiler config. The helper still sends those values in the start payload because some local builds accept them, but do not spend time verifying the server's route shape before taking the capture.
