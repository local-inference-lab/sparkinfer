import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from benchmarks.benchmark_moe import ModelSpec, make_shape_only_expert_weights, get_scale_contract_params
from b12x.integration import tp_moe

def build(n, M=8, K=6144, E=16, top_k=4):
    dev = torch.device("cuda", torch.cuda.current_device())
    spec = ModelSpec(hidden_size=K, intermediate_size=n, num_experts=E, top_k=top_k, tp_size=1, tp_rank=0)
    w = make_shape_only_expert_weights(spec, layer_idx=0, activation="silu")
    g = torch.Generator(device=dev).manual_seed(123)
    w.w13_weight.random_(0, 256, generator=g)   # randomize fp4 weights -> discriminating
    w.w2_weight.random_(0, 256, generator=g)
    sp = get_scale_contract_params(w, "shared")
    torch.manual_seed(7)
    x = (torch.randn(M, K, device=dev) * 0.3).to(torch.bfloat16)
    logits = torch.randn(M, E, device=dev); tw, ti = torch.topk(logits, top_k, -1)
    tw = torch.softmax(tw, -1).float(); ti = ti.int()
    return dict(n=n, M=M, K=K, E=E, top_k=top_k, dev=dev, w=w, sp=sp, x=x, tw=tw, ti=ti)

def run(b, cutover):
    dev = b["dev"]
    tp_moe.clear_tp_moe_caches()
    tp_moe._STATIC_COMPACT_CUTOVER_PAIRS_CACHE["nvfp4"] = cutover
    plan = tp_moe.plan_tp_moe_scratch(tp_moe.TPMoEScratchCaps(
        max_tokens=max(b["M"],1), weight_E=b["E"], k=b["K"], n=b["n"], num_topk=b["top_k"],
        device=dev, dtype=torch.bfloat16, core_token_counts=(max(b["M"],1),), route_num_experts=0,
        quant_mode="nvfp4", activation="silu", apply_router_weight_on_input=False, swiglu_limit=None,
        source_format="modelopt_nvfp4", w13_layout="w13", w4a16_weight_layout=None,
        w4a16_scale_format=None, frozen=True))
    nb = plan.scratch_specs()[0].shape[0]
    scratch = torch.empty(nb, dtype=torch.uint8, device=dev).zero_()
    out = torch.empty(b["M"], b["K"], dtype=torch.bfloat16, device=dev)
    w, sp = b["w"], b["sp"]
    bind = plan.bind(scratch=scratch, a=b["x"], a1_gscale=sp.a1_gscale, w1_fp4=w.w13_weight,
        w1_blockscale=w.w13_blockscale_swizzled, w1_alphas=sp.g1_alphas, a2_gscale=sp.a2_gscale,
        w2_fp4=w.w2_weight, w2_blockscale=w.w2_blockscale_swizzled, w2_alphas=sp.g2_alphas,
        topk_weights=b["tw"], topk_ids=b["ti"], apply_router_weight_on_input=False, output=out,
        input_scales_are_reciprocal=True, input_scales_static=True, activation="silu",
        quant_mode="nvfp4", unit_scale_contract=False, source_format="modelopt_nvfp4",
        w13_layout="w13", prepared_w4a16=None, swiglu_limit=None)
    tp_moe.b12x_moe_fp4(binding=bind); torch.cuda.synchronize()
    return out.float().clone()

for n in (256, 320, 384, 352):  # 256,384=128-aligned; 320=64-but-not-128; 352=32
    b = build(n)
    s = run(b, 100000)   # compact-static (fixed)
    d = run(b, 0)        # dynamic (oracle)
    diff = (s - d).abs().max().item(); scale = d.abs().max().item()
    print(f"n={n}: static_finite={bool(torch.isfinite(s).all())} dyn_finite={bool(torch.isfinite(d).all())} "
          f"max|static-dyn|={diff:.4g} (dyn max={scale:.4g}, rel={diff/(scale+1e-9):.3g})")
