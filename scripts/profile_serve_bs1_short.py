#!/usr/bin/env python3
"""Profile a short-context bs=1 serve request with CPU+GPU timelines.

Outputs one Chrome/Perfetto trace per TP rank plus text summaries and a
request-level metrics JSON file.

Example:
  /home/luke/projects/sglang/.venv/bin/python scripts/profile_serve_bs1_short.py \
      --model /data/models/Qwen3.5-397B-A17B-NVFP4-BF16shared \
      --gpu-ids 0,1,2,3 \
      --out-dir /tmp/b12x-serve-profile
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serve.tp.launch import launch_tp

torch.set_grad_enabled(False)


def _profile_worker(
    tp_group,
    model_path: str,
    out_dir: str,
    context_length: int,
    max_new_tokens: int,
    mode: str,
    with_stack: bool,
    stack_ops: tuple[str, ...],
    compile_layers: bool,
) -> None:
    rank = tp_group.rank if tp_group else 0
    device = f"cuda:{tp_group.device.index}" if tp_group else "cuda"

    from serve.engine.sampling import SamplingParams
    from serve.engine.serving import ServingEngine
    from tests.serve.test_qwen35_perf_integration import _build_prompt_ids

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    engine = ServingEngine(
        model_path,
        device=device,
        tp_group=tp_group,
        warmup_prefill_lengths=[128],
        graph_batch_sizes=[1, 2, 4, 8],
        compile_layers=compile_layers,
    )

    prompt_ids = _build_prompt_ids(
        engine.tokenizer,
        target_tokens=context_length,
        code="0100",
    )

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    profile_kwargs = dict(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    )

    if rank != 0:
        with profile(**profile_kwargs) as prof:
            engine.run_follower()
        prof.export_chrome_trace(str(out_path / f"trace_rank{rank}.json"))
        (out_path / f"summary_rank{rank}.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80) + "\n"
        )
        if with_stack and stack_ops:
            _write_stack_report(prof, out_path / f"stacks_rank{rank}.txt", stack_ops)
        return

    # Warm up the exact path before profiling so the trace is not dominated by
    # startup and first-use compilation noise.
    engine.generate_batch([prompt_ids], SamplingParams.greedy(max_new_tokens=2))
    torch.cuda.synchronize()

    if mode == "full_request":
        with profile(**profile_kwargs) as prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = engine.generate_batch(
                [prompt_ids],
                SamplingParams.greedy(max_new_tokens=max_new_tokens),
            )
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t0

        result = results[0]
        decoded = engine.tokenizer.decode(result.generated_ids, skip_special_tokens=True)
        metrics = {
            "mode": mode,
            "model_path": model_path,
            "context_length": context_length,
            "max_new_tokens": max_new_tokens,
            "elapsed_s": elapsed_s,
            "aggregate_tok_per_s": (
                len(result.generated_ids) / elapsed_s if elapsed_s > 0 else 0.0
            ),
            "generated_tokens": len(result.generated_ids),
            "ttft_ms": result.time_to_first_token_ms,
            "total_time_ms": result.total_time_ms,
            "finish_reason": result.finish_reason,
            "decoded_text": decoded,
            "trace_files": [f"trace_rank{i}.json" for i in range(tp_group.world_size)],
        }
    elif mode == "decode_step_replay":
        params = SamplingParams.greedy(max_new_tokens=2)
        req = engine.submit(prompt_ids, params)
        while not req.is_finished and not req.output_ids:
            engine._step()
        assert req.output_ids, "prefill did not emit the first token"
        assert not req.is_finished, "request finished before decode-step profiling"

        with profile(**profile_kwargs) as prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine._step()
            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t0

        result = engine._to_result(req)
        decoded = engine.tokenizer.decode(result.generated_ids, skip_special_tokens=True)
        metrics = {
            "mode": mode,
            "model_path": model_path,
            "context_length": context_length,
            "max_new_tokens": max_new_tokens,
            "elapsed_s": elapsed_s,
            "generated_tokens_after_profiled_step": len(result.generated_ids),
            "ttft_ms": result.time_to_first_token_ms,
            "total_time_ms": result.total_time_ms,
            "finish_reason": result.finish_reason,
            "decoded_text": decoded,
            "trace_files": [f"trace_rank{i}.json" for i in range(tp_group.world_size)],
        }
    else:
        raise ValueError(f"unsupported mode: {mode}")

    prof.export_chrome_trace(str(out_path / "trace_rank0.json"))
    (out_path / "summary_rank0.txt").write_text(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=120) + "\n"
    )
    if with_stack and stack_ops:
        _write_stack_report(prof, out_path / "stacks_rank0.txt", stack_ops)
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    engine.shutdown()


def _write_stack_report(prof, path: Path, stack_ops: tuple[str, ...]) -> None:
    lines: list[str] = []
    for op_name in stack_ops:
        lines.append(f"== {op_name} ==")
        rows = prof.key_averages(group_by_stack_n=8)
        matched = [
            evt for evt in rows
            if evt.key == op_name and getattr(evt, "stack", None)
        ]
        matched.sort(key=lambda evt: evt.cpu_time_total, reverse=True)
        if not matched:
            lines.append("no stack samples recorded")
            lines.append("")
            continue
        for idx, evt in enumerate(matched[:10], start=1):
            lines.append(
                f"{idx}. cpu_total_us={evt.cpu_time_total:.3f} self_cpu_us={evt.self_cpu_time_total:.3f} "
                f"calls={evt.count}"
            )
            for frame in evt.stack:
                lines.append(f"   {frame}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/data/models/Qwen3.5-397B-A17B-NVFP4-BF16shared",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out-dir", default="/tmp/b12x-serve-profile")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--compile-layers", action="store_true")
    parser.add_argument(
        "--stack-ops",
        default="aten::to,aten::_to_copy",
        help="Comma-separated op names to dump stack-grouped summaries for when --with-stack is set.",
    )
    parser.add_argument(
        "--mode",
        choices=("full_request", "decode_step_replay"),
        default="decode_step_replay",
    )
    args = parser.parse_args()

    os.environ.setdefault("B12X_ENABLE_PROFILE_RANGES", "1")

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    launch_tp(
        _profile_worker,
        world_size=len(gpu_ids),
        args=(
            args.model,
            args.out_dir,
            args.context_length,
            args.max_new_tokens,
            args.mode,
            args.with_stack,
            tuple(x.strip() for x in args.stack_ops.split(",") if x.strip()),
            args.compile_layers,
        ),
        gpu_ids=gpu_ids,
    )

    print(f"Profile written to {args.out_dir}", flush=True)
    print(f"Open trace_rank0.json in https://ui.perfetto.dev", flush=True)


if __name__ == "__main__":
    main()
