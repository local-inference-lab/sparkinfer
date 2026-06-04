import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from benchmarks.benchmark_moe import (
    ModelSpec, make_shape_only_expert_weights, get_scale_contract_params,
)
from b12x.integration.tp_moe import (
    allocate_tp_moe_workspace_pool, b12x_moe_fp4, clear_tp_moe_caches,
)

def run(I_tp, M=8, K=6144, E=16, top_k=4):
    spec = ModelSpec(hidden_size=K, intermediate_size=I_tp, num_experts=E,
                     top_k=top_k, tp_size=1, tp_rank=0)
    w = make_shape_only_expert_weights(spec, layer_idx=0, activation="silu")
    sp = get_scale_contract_params(w, "shared")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    tw, ti = torch.topk(logits, top_k, dim=-1)
    tw = torch.softmax(tw, dim=-1).to(torch.float32)
    ti = ti.to(torch.int32)
    ws = allocate_tp_moe_workspace_pool()
    clear_tp_moe_caches()
    out = b12x_moe_fp4(x, sp.a1_gscale, w.w13_weight, w.w13_blockscale_swizzled,
                       sp.g1_alphas, sp.a2_gscale, w.w2_weight,
                       w.w2_blockscale_swizzled, sp.g2_alphas, tw, ti,
                       workspace=ws, input_scales_static=True)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(out).all().item())
    nan = int(torch.isnan(out.float()).sum().item())
    print(f"I_tp={I_tp:4d} M={M} routed_rows={M*top_k:4d} static-path: "
          f"finite={finite} nan_count={nan} sample={out.flatten()[:4].float().tolist()}")

for I_tp in (256, 384, 512):
    run(I_tp)
