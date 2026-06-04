import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from benchmarks.benchmark_moe import ModelSpec, make_shape_only_expert_weights, get_scale_contract_params
from b12x.integration import tp_moe

def run(I_tp, fill, M=8, K=6144, E=16, top_k=4):
    spec = ModelSpec(hidden_size=K, intermediate_size=I_tp, num_experts=E, top_k=top_k, tp_size=1, tp_rank=0)
    w = make_shape_only_expert_weights(spec, layer_idx=0, activation="silu")
    sp = get_scale_contract_params(w, "shared")
    dev = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    logits = torch.randn(M, E, device=dev); tw, ti = torch.topk(logits, top_k, -1)
    tw = torch.softmax(tw, -1).float(); ti = ti.int()
    plan = tp_moe.plan_tp_moe_scratch(tp_moe.TPMoEScratchCaps(
        max_tokens=max(M,1), weight_E=E, k=K, n=I_tp, num_topk=top_k, device=dev,
        dtype=torch.bfloat16, core_token_counts=(max(M,1),), route_num_experts=0,
        quant_mode="nvfp4", activation="silu", apply_router_weight_on_input=False,
        swiglu_limit=None, source_format="modelopt_nvfp4", w13_layout="w13",
        w4a16_weight_layout=None, w4a16_scale_format=None, frozen=True))
    nb = plan.scratch_specs()[0].shape[0]
    scratch = torch.empty(nb, dtype=torch.uint8, device=dev)
    scratch.fill_(0xFF) if fill == "ff" else scratch.zero_()
    out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
    b = plan.bind(scratch=scratch, a=x, a1_gscale=sp.a1_gscale, w1_fp4=w.w13_weight,
        w1_blockscale=w.w13_blockscale_swizzled, w1_alphas=sp.g1_alphas, a2_gscale=sp.a2_gscale,
        w2_fp4=w.w2_weight, w2_blockscale=w.w2_blockscale_swizzled, w2_alphas=sp.g2_alphas,
        topk_weights=tw, topk_ids=ti, apply_router_weight_on_input=False, output=out,
        input_scales_are_reciprocal=True, input_scales_static=True, activation="silu",
        quant_mode="nvfp4", unit_scale_contract=False, source_format="modelopt_nvfp4",
        w13_layout="w13", prepared_w4a16=None, swiglu_limit=None)
    tp_moe.b12x_moe_fp4(binding=b); torch.cuda.synchronize()
    nan = int(torch.isnan(out.float()).sum())
    print(f"n={I_tp} ({I_tp//16}x16, %64={I_tp%64}) fill={fill}: nan={nan} finite={bool(torch.isfinite(out).all())}")

for I_tp in (320, 352, 368, 384):   # 320=5x64 aligned; 352/368 = 16- but not 64-aligned
    for fill in ("zero", "ff"):
        run(I_tp, fill)
