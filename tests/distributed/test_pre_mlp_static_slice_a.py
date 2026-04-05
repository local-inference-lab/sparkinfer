from __future__ import annotations

import os

import torch
import torch.distributed as dist

from b12x.distributed.pcie_oneshot import PCIeOneshotAllReduce
from b12x.moe.fused.pre_mlp_static import (
    UnifiedPreMLPIPC,
    UnifiedPreMLPStaticLaunchConfig,
    slice_a_allreduce_residual_gemma_rmsnorm,
)
from b12x.moe.fused.reference import compare_to_reference


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _rank0_print(msg: str) -> None:
    if _rank() == 0:
        print(msg, flush=True)


def main() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", _local_rank()))
    torch.cuda.set_device(_local_rank())
    device = torch.device("cuda", _local_rank())
    torch.set_grad_enabled(False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(f"Requires SM120, got sm_{major}{minor}")

    num_tokens = 4
    hidden_size = 256
    byte_count = num_tokens * hidden_size * torch.empty((), dtype=torch.bfloat16).element_size()
    runtime = PCIeOneshotAllReduce.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=byte_count,
    )
    try:
        hidden_states = torch.randn(
            num_tokens,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        residual = torch.randn_like(hidden_states)
        norm_weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)

        ipc = UnifiedPreMLPIPC.from_oneshot_runtime(runtime, inp=hidden_states)
        launch_config = UnifiedPreMLPStaticLaunchConfig()
        slice_a = slice_a_allreduce_residual_gemma_rmsnorm(
            hidden_states,
            residual,
            norm_weight,
            eps=1e-6,
            ipc=ipc,
            launch_config=launch_config,
        )
        ref_out, ref_residual = runtime.allreduce_gemma_rmsnorm(
            hidden_states,
            residual,
            norm_weight,
            1e-6,
            peer_input_ptrs=ipc.peer_input_ptrs,
        )
        torch.cuda.synchronize()

        out_metrics = compare_to_reference(slice_a.normalized, ref_out)
        residual_metrics = compare_to_reference(slice_a.residual_out, ref_residual)
        if out_metrics.max_abs > 0.0 or out_metrics.cos <= 0.999999:
            raise AssertionError(
                f"normalized mismatch: max_abs={out_metrics.max_abs:.3e} cos={out_metrics.cos:.6f}"
            )
        if residual_metrics.max_abs > 0.0 or residual_metrics.cos <= 0.999999:
            raise AssertionError(
                f"residual mismatch: max_abs={residual_metrics.max_abs:.3e} cos={residual_metrics.cos:.6f}"
            )
        _rank0_print(
            "slice_a validate: "
            f"normalized(max_abs={out_metrics.max_abs:.3e}, cos={out_metrics.cos:.6f}) "
            f"residual(max_abs={residual_metrics.max_abs:.3e}, cos={residual_metrics.cos:.6f})"
        )
    finally:
        runtime.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
