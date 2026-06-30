#!/usr/bin/env python3
"""Benchmark the fused b12x mHC residual post_pre (Gram) kernel vs vLLM."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pathlib
import statistics
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.common import (
    bench_cuda_graph,
    bench_gpu_ms,
    capture_cuda_graph,
    make_l2_flush_fn,
    require_sm120,
)
from b12x.integration import (
    B12XMHCScratchCaps,
    b12x_mhc_post_pre,
    plan_mhc_scratch,
)


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    y_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    y = y.to(residual.dtype if y_dtype is None else y_dtype)
    return y, post, comb


def _mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(x.dtype)


def _post_pre_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the fused post_pre. RMSNorm variance is taken in fp32 from
    the collapsed activation (matching the Gram kernel and vLLM, which both
    compute the norm in fp32 rather than from the bf16-rounded activation)."""
    residual_out = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_raw_fp32, post, comb = _mhc_pre_reference(
        residual_out,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        y_dtype=torch.float32,
    )
    if norm_weight is not None:
        rms = torch.rsqrt(y_raw_fp32.square().mean(dim=-1, keepdim=True) + norm_eps)
        y = (
            y_raw_fp32.to(torch.bfloat16).float() * rms * norm_weight.float()
        ).to(torch.bfloat16)
    else:
        y = y_raw_fp32.to(torch.bfloat16)
    return residual_out, y, post, comb


def _make_inputs(
    *,
    tokens: int,
    hidden_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    residual = (
        torch.randn((tokens, 4, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 3
    ).to(torch.bfloat16)
    x = (
        torch.randn((tokens, hidden_size), generator=gen, dtype=torch.float32).to(device)
        / 4
    ).to(torch.bfloat16)
    fn = torch.randn((24, 4 * hidden_size), generator=gen, dtype=torch.float32).to(device) / 64
    scale = torch.randn((3,), generator=gen, dtype=torch.float32).to(device) / 3
    bias = torch.randn((24,), generator=gen, dtype=torch.float32).to(device) / 5
    return residual.contiguous(), x.contiguous(), fn.contiguous(), scale.contiguous(), bias.contiguous()


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = actual.float() - expected.float()
    return float(diff.abs().max().item()), float(torch.sqrt(torch.mean(diff * diff)).item())


def _bench_graph(fn, *, warmup: int, iters: int, l2_flush) -> tuple[float, float]:
    graph = capture_cuda_graph(fn, warmup=warmup)
    stats = bench_cuda_graph(graph, replays=iters, l2_flush=l2_flush)
    samples = stats["replay_us"]
    return statistics.median(samples), min(samples)


def _bench_eager(fn, *, warmup: int, iters: int, l2_flush) -> tuple[float, float]:
    samples = []
    for _ in range(warmup):
        if l2_flush is not None:
            l2_flush()
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        samples.append(bench_gpu_ms(fn, warmup=0, iters=1, l2_flush=l2_flush) * 1000.0)
    return statistics.median(samples), min(samples)


def _register_vllm_mhc_tilelang(vllm_path: pathlib.Path) -> None:
    vllm_root = str(vllm_path.expanduser().resolve())
    if vllm_root not in sys.path:
        sys.path.insert(1, vllm_root)

    try:
        import tilelang  # noqa: F401
    except ModuleNotFoundError:
        vllm_site = next(
            (
                path
                for path in (pathlib.Path(vllm_root) / ".venv" / "lib").glob(
                    "python*/site-packages"
                )
                if (path / "tilelang").exists()
            ),
            None,
        )
        if vllm_site is None:
            raise
        sys.path.append(str(vllm_site))
        import tilelang  # noqa: F401

    def _package(name: str) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(_package(parent_name), child_name, mod)
        return mod

    for package in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.kernels",
        "vllm.model_executor.kernels.mhc",
        "vllm.platforms",
        "vllm.utils",
    ):
        _package(package)

    class _CurrentPlatform:
        def is_cuda_alike(self) -> bool:
            return True

        def is_cuda(self) -> bool:
            return True

        def is_arch_support_pdl(self) -> bool:
            return True

    sys.modules["vllm.platforms"].current_platform = _CurrentPlatform()

    import_utils = types.ModuleType("vllm.utils.import_utils")
    import_utils.has_tilelang = lambda: True
    sys.modules["vllm.utils.import_utils"] = import_utils
    setattr(sys.modules["vllm.utils"], "import_utils", import_utils)

    math_utils = types.ModuleType("vllm.utils.math_utils")
    math_utils.cdiv = lambda a, b: int(math.ceil(a / b))
    sys.modules["vllm.utils.math_utils"] = math_utils
    setattr(sys.modules["vllm.utils"], "math_utils", math_utils)

    deep_gemm = types.ModuleType("vllm.utils.deep_gemm")
    deep_gemm.is_deep_gemm_supported = lambda: False

    def _no_deep_gemm(*_args, **_kwargs):
        raise RuntimeError("DeepGEMM is disabled in the direct vLLM MHC loader")

    deep_gemm.tf32_hc_prenorm_gemm = _no_deep_gemm
    sys.modules["vllm.utils.deep_gemm"] = deep_gemm
    setattr(sys.modules["vllm.utils"], "deep_gemm", deep_gemm)

    torch_utils = types.ModuleType("vllm.utils.torch_utils")

    def direct_register_custom_op(
        *,
        op_name: str,
        op_func,
        mutates_args,
        fake_impl=None,
    ) -> None:
        registered = torch.library.custom_op(
            f"vllm::{op_name}",
            mutates_args=tuple(mutates_args),
        )(op_func)
        if fake_impl is not None:
            registered.register_fake(fake_impl)

    torch_utils.direct_register_custom_op = direct_register_custom_op
    sys.modules["vllm.utils.torch_utils"] = torch_utils
    setattr(sys.modules["vllm.utils"], "torch_utils", torch_utils)

    def _load_module(module_name: str, path: pathlib.Path) -> None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    mhc_dir = pathlib.Path(vllm_root) / "vllm" / "model_executor" / "kernels" / "mhc"
    _load_module(
        "vllm.model_executor.kernels.mhc.tilelang_kernels",
        mhc_dir / "tilelang_kernels.py",
    )
    _load_module(
        "vllm.model_executor.kernels.mhc.tilelang",
        mhc_dir / "tilelang.py",
    )

    if not hasattr(torch.ops.vllm, "mhc_fused_post_pre_tilelang"):
        raise RuntimeError("vLLM mhc_fused_post_pre_tilelang custom op was not registered")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--split-k", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--block-h", type=int, default=512)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--hc-eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--l2-flush", action="store_true")
    parser.add_argument("--l2-flush-bytes", type=int, default=0)
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    parser.add_argument(
        "--expected-m",
        type=int,
        default=None,
        help=(
            "Expected/capture M used for mHC dispatch policy. Live --tokens may "
            "be smaller; scratch is sized for max(tokens, expected_m)."
        ),
    )
    parser.add_argument(
        "--prefill-bf16-mma",
        action="store_true",
        help="Enable native BF16 tensor-core prefill projection when TF32 is disabled.",
    )
    parser.add_argument(
        "--prefill-tf32-mma",
        action="store_true",
        help="Enable native TF32 tensor-core prefill projection path.",
    )
    parser.add_argument("--no-prefill-tf32-mma", action="store_true")
    parser.add_argument(
        "--prefill-block-m",
        action="store_true",
        help="Enable the block-M scalar prefill projection path explicitly.",
    )
    parser.add_argument("--no-prefill-block-m", action="store_true")
    parser.add_argument("--prefill-block-m-size", type=int, default=2)
    parser.add_argument("--prefill-tile-n", type=int, default=24)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=91_500)
    parser.add_argument(
        "--compare-vllm",
        action="store_true",
        help="Also benchmark vLLM's DeepSeek TileLang fused post_pre op.",
    )
    parser.add_argument(
        "--vllm-path",
        type=pathlib.Path,
        default=pathlib.Path("~/projects/vllm-other"),
        help="Path to the vLLM checkout used for --compare-vllm.",
    )
    args = parser.parse_args()
    if args.no_prefill_tf32_mma:
        os.environ["B12X_MHC_PREFILL_TF32_MMA"] = "0"
    elif args.prefill_tf32_mma:
        os.environ["B12X_MHC_PREFILL_TF32_MMA"] = "1"
    if args.prefill_bf16_mma:
        os.environ["B12X_MHC_PREFILL_BF16_MMA"] = "1"
    if args.no_prefill_block_m:
        os.environ["B12X_MHC_PREFILL_BLOCK_M"] = "0"
    elif args.prefill_block_m:
        os.environ["B12X_MHC_PREFILL_BLOCK_M"] = "1"
    if args.prefill_block_m or not args.no_prefill_block_m:
        os.environ["B12X_MHC_PREFILL_BLOCK_M_SIZE"] = str(args.prefill_block_m_size)
        os.environ["B12X_MHC_PREFILL_TILE_N"] = str(args.prefill_tile_n)
    prefill_tf32_enabled = (
        os.environ.get(
            "B12X_MHC_PREFILL_TF32_MMA",
            os.environ.get("B12X_MHC_PREFILL_BF16_MMA", "1"),
        )
        != "0"
    )
    prefill_bf16_enabled = os.environ.get("B12X_MHC_PREFILL_BF16_MMA", "1") != "0"
    prefill_block_m_enabled = os.environ.get("B12X_MHC_PREFILL_BLOCK_M", "1") != "0"
    prefill_gram_threads = int(
        os.environ.get(
            "B12X_MHC_PREFILL_GRAM_THREADS",
            os.environ.get("B12X_MHC_PREFILL_THREADS", "1024"),
        )
    )
    prefill_finalize_threads = int(
        os.environ.get("B12X_MHC_PREFILL_FINALIZE_THREADS", "256")
    )
    prefill_tf32_tma_m_warps = int(
        os.environ.get(
            "B12X_MHC_PREFILL_TF32_TMA_M_WARPS",
            os.environ.get(
                "B12X_MHC_PREFILL_TF32_TMA_WARPS",
                os.environ.get("B12X_MHC_PREFILL_TMA_WARPS", "1"),
            ),
        )
    )
    prefill_tf32_tma_n_warps = int(
        os.environ.get("B12X_MHC_PREFILL_TF32_TMA_N_WARPS", "1")
    )
    prefill_tf32_tma_m = int(
        os.environ.get(
            "B12X_MHC_PREFILL_TF32_TMA_TILE_M",
            os.environ.get("B12X_MHC_PREFILL_TMA_TILE_M", "16"),
        )
    )
    prefill_tf32_tma_n = int(
        os.environ.get("B12X_MHC_PREFILL_TF32_TMA_TILE_N", "8")
    )
    prefill_tf32_tma_k = int(
        os.environ.get(
            "B12X_MHC_PREFILL_TF32_TMA_TILE_K",
            os.environ.get("B12X_MHC_PREFILL_TMA_TILE_K", "256"),
        )
    )
    prefill_tf32_tma_stages = int(
        os.environ.get(
            "B12X_MHC_PREFILL_TF32_TMA_STAGES",
            os.environ.get("B12X_MHC_PREFILL_TMA_STAGES", "1"),
        )
    )
    prefill_tf32_tma_chunk_min_tokens = int(
        os.environ.get("B12X_MHC_PREFILL_TF32_TMA_CHUNK_MIN_TOKENS", "4096")
    )
    prefill_tf32_tma_chunk_geometry = (
        args.tokens >= prefill_tf32_tma_chunk_min_tokens
    )
    if prefill_tf32_tma_chunk_geometry:
        prefill_tf32_tma_m_warps = int(
            os.environ.get("B12X_MHC_PREFILL_TF32_TMA_CHUNK_M_WARPS", "2")
        )
        prefill_tf32_tma_m = int(
            os.environ.get("B12X_MHC_PREFILL_TF32_TMA_CHUNK_TILE_M", "32")
        )

    device = require_sm120()
    if args.compare_vllm:
        _register_vllm_mhc_tilelang(args.vllm_path)

    residual, x, fn, scale, bias = _make_inputs(
        tokens=args.tokens,
        hidden_size=args.hidden_size,
        seed=args.seed,
        device=device,
    )
    fn_bf16 = fn.to(torch.bfloat16).contiguous() if args.prefill_bf16_mma else None
    scratch_tokens = max(args.tokens, args.expected_m or args.tokens)
    fused_plan = plan_mhc_scratch(
        B12XMHCScratchCaps(
            device=device,
            max_tokens=scratch_tokens,
            hidden_size=args.hidden_size,
            split_k=args.split_k,
        )
    )
    fused_scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in fused_plan.shapes_and_dtypes()
    )
    fused_y = torch.empty((args.tokens, args.hidden_size), dtype=torch.bfloat16, device=device)
    fused_post = torch.empty((args.tokens, 4), dtype=torch.float32, device=device)
    fused_comb = torch.empty((args.tokens, 4, 4), dtype=torch.float32, device=device)
    fused_out = torch.empty((args.tokens, 4, args.hidden_size), dtype=torch.bfloat16, device=device)
    fused_binding = fused_plan.bind(
        scratch=fused_scratch,
        tokens=args.tokens,
        expected_m=args.expected_m,
        y=fused_y,
        post=fused_post,
        comb=fused_comb,
        out=fused_out,
    )
    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=args.rms_eps,
        hc_eps=args.hc_eps,
        sinkhorn_iters=args.sinkhorn_iters,
    )
    prev_post = prev_post.contiguous()
    prev_comb = prev_comb.contiguous()
    norm_weight = None
    if args.fuse_rmsnorm:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.seed + 17)
        norm_weight = (
            torch.randn((args.hidden_size,), generator=gen, dtype=torch.float32)
            .to(device)
            .to(torch.bfloat16)
            .contiguous()
        )

    def run_fused() -> None:
        b12x_mhc_post_pre(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=args.norm_eps,
            fn_bf16=fn_bf16,
            binding=fused_binding,
            split_k=args.split_k,
            block_k=args.block_k,
            block_h=args.block_h,
        )

    vllm_out = vllm_post = vllm_comb = vllm_y = None

    def run_vllm_fused() -> None:
        nonlocal vllm_out, vllm_post, vllm_comb, vllm_y
        vllm_out, vllm_post, vllm_comb, vllm_y = torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            args.rms_eps,
            args.hc_eps,
            args.hc_eps,
            2.0,
            args.sinkhorn_iters,
            1,
            1,
            norm_weight,
            args.norm_eps if norm_weight is not None else 0.0,
        )

    run_fused()
    if args.compare_vllm:
        run_vllm_fused()
    torch.cuda.synchronize()

    if not args.skip_check:
        out_ref, y_ref, post_ref, comb_ref = _post_pre_reference(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            rms_eps=args.rms_eps,
            hc_eps=args.hc_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=args.norm_eps,
        )
        fused_y_max, fused_y_rmse = _error_stats(fused_y, y_ref)
        fused_post_max, _ = _error_stats(fused_post, post_ref)
        fused_comb_max, _ = _error_stats(fused_comb, comb_ref)
        fused_out_max, fused_out_rmse = _error_stats(fused_out, out_ref)
        if args.prefill_bf16_mma:
            assert fn_bf16 is not None
            _, y_ref_bf16, post_ref_bf16, comb_ref_bf16 = _post_pre_reference(
                x,
                residual,
                prev_post,
                prev_comb,
                fn_bf16.float(),
                scale,
                bias,
                rms_eps=args.rms_eps,
                hc_eps=args.hc_eps,
                sinkhorn_iters=args.sinkhorn_iters,
                norm_weight=norm_weight,
                norm_eps=args.norm_eps,
            )
            bf16ref_y_max, bf16ref_y_rmse = _error_stats(fused_y, y_ref_bf16)
            bf16ref_post_max, _ = _error_stats(fused_post, post_ref_bf16)
            bf16ref_comb_max, _ = _error_stats(fused_comb, comb_ref_bf16)
        else:
            bf16ref_y_max = bf16ref_y_rmse = float("nan")
            bf16ref_post_max = bf16ref_comb_max = float("nan")
        if args.compare_vllm:
            assert vllm_out is not None and vllm_post is not None
            assert vllm_comb is not None and vllm_y is not None
            vllm_y_max, vllm_y_rmse = _error_stats(vllm_y, y_ref)
            vllm_post_max, _ = _error_stats(vllm_post.squeeze(-1), post_ref)
            vllm_comb_max, _ = _error_stats(vllm_comb, comb_ref)
            vllm_out_max, vllm_out_rmse = _error_stats(vllm_out, out_ref)
        else:
            vllm_y_max = vllm_y_rmse = vllm_post_max = vllm_comb_max = float("nan")
            vllm_out_max = vllm_out_rmse = float("nan")
    else:
        fused_y_max = fused_y_rmse = fused_post_max = fused_comb_max = float("nan")
        fused_out_max = fused_out_rmse = float("nan")
        bf16ref_y_max = bf16ref_y_rmse = float("nan")
        bf16ref_post_max = bf16ref_comb_max = float("nan")
        vllm_y_max = vllm_y_rmse = vllm_post_max = vllm_comb_max = float("nan")
        vllm_out_max = vllm_out_rmse = float("nan")

    l2_flush = make_l2_flush_fn(args.l2_flush, args.l2_flush_bytes)
    bench = _bench_eager if args.eager else _bench_graph
    fused_median, fused_min = bench(
        run_fused, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush
    )
    if args.compare_vllm:
        vllm_median, vllm_min = bench(
            run_vllm_fused, warmup=args.warmup, iters=args.iters, l2_flush=l2_flush
        )
    else:
        vllm_median = vllm_min = float("nan")
    mode = "eager" if args.eager else "graph"

    line = (
        "residual_mhc "
        f"mode={mode} tokens={args.tokens} expected_m={args.expected_m} "
        f"hidden={args.hidden_size} "
        f"split_k={args.split_k} block_k={args.block_k} block_h={args.block_h} "
        f"prefill_tf32_mma={prefill_tf32_enabled} "
        f"prefill_tf32_tma=m{prefill_tf32_tma_m}n{prefill_tf32_tma_n}"
        f"k{prefill_tf32_tma_k}s{prefill_tf32_tma_stages}"
        f"wm{prefill_tf32_tma_m_warps}wn{prefill_tf32_tma_n_warps} "
        f"prefill_tf32_chunk_geometry={prefill_tf32_tma_chunk_geometry} "
        f"prefill_gram_threads={prefill_gram_threads} "
        f"prefill_finalize_threads={prefill_finalize_threads} "
        f"prefill_bf16_mma={prefill_bf16_enabled} "
        f"prefill_block_m={prefill_block_m_enabled} "
        f"prefill_block_m_size={args.prefill_block_m_size} "
        f"prefill_tile_n={args.prefill_tile_n} "
        f"post_pre_us={fused_median:.2f}/{fused_min:.2f} "
        f"fused_y_max={fused_y_max:.3g} fused_y_rmse={fused_y_rmse:.3g} "
        f"fused_post_max={fused_post_max:.3g} fused_comb_max={fused_comb_max:.3g} "
        f"fused_out_max={fused_out_max:.3g} fused_out_rmse={fused_out_rmse:.3g}"
    )
    if args.prefill_bf16_mma:
        line += (
            f" bf16ref_y_max={bf16ref_y_max:.3g} "
            f"bf16ref_y_rmse={bf16ref_y_rmse:.3g} "
            f"bf16ref_post_max={bf16ref_post_max:.3g} "
            f"bf16ref_comb_max={bf16ref_comb_max:.3g}"
        )
    if args.compare_vllm:
        line += (
            f" vllm_post_pre_us={vllm_median:.2f}/{vllm_min:.2f} "
            f"b12x_vs_vllm={vllm_median / fused_median:.3f}x "
            f"vllm_y_max={vllm_y_max:.3g} vllm_y_rmse={vllm_y_rmse:.3g} "
            f"vllm_post_max={vllm_post_max:.3g} vllm_comb_max={vllm_comb_max:.3g} "
            f"vllm_out_max={vllm_out_max:.3g} vllm_out_rmse={vllm_out_rmse:.3g}"
        )
    print(line)


if __name__ == "__main__":
    main()
