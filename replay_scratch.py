import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
from b12x.integration import tp_moe

def replay(path, scratch_fill):
    d = torch.load(path, map_location="cpu")
    dev = torch.device("cuda", torch.cuda.current_device())
    def g(k):
        v = d[k]; return v.to(dev) if torch.is_tensor(v) else v
    M, K, n, E, topk = d["tokens"], d["k"], d["n"], d["weight_E"], d["topk"]
    plan = tp_moe.plan_tp_moe_scratch(tp_moe.TPMoEScratchCaps(
        max_tokens=max(M,1), weight_E=E, k=K, n=n, num_topk=topk, device=dev,
        dtype=torch.bfloat16, core_token_counts=(max(M,1),), route_num_experts=0,
        quant_mode=d["quant_mode"], activation=d["activation"],
        apply_router_weight_on_input=d["apply_router_weight_on_input"],
        swiglu_limit=d["swiglu_limit"], source_format=d["source_format"],
        w13_layout=d["w13_layout"], w4a16_weight_layout=None,
        w4a16_scale_format=None, frozen=True))
    nbytes = plan.scratch_specs()[0].shape[0]
    scratch = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    if scratch_fill == "zero":
        scratch.zero_()
    elif scratch_fill == "ff":
        scratch.fill_(0xFF)   # bf16/fp32 view -> NaN/Inf bytes
    elif scratch_fill == "7f":
        scratch.fill_(0x7F)
    output = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
    binding = plan.bind(
        scratch=scratch, a=g("a"), a1_gscale=g("a1_gscale"), w1_fp4=g("w1"),
        w1_blockscale=g("w1_blockscale"), w1_alphas=g("w1_alphas"),
        a2_gscale=g("a2_gscale"), w2_fp4=g("w2"), w2_blockscale=g("w2_blockscale"),
        w2_alphas=g("w2_alphas"), topk_weights=g("topk_weights"), topk_ids=g("topk_ids"),
        apply_router_weight_on_input=d["apply_router_weight_on_input"], output=output,
        input_scales_are_reciprocal=True, input_scales_static=d["input_scales_static"],
        activation=d["activation"], quant_mode=d["quant_mode"],
        unit_scale_contract=d["unit_scale_contract"], source_format=d["source_format"],
        w13_layout=d["w13_layout"], prepared_w4a16=None, swiglu_limit=d["swiglu_limit"])
    tp_moe.b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    nan = int(torch.isnan(output.float()).sum())
    print(f"rank{d['rank']} scratch={scratch_fill:5s}: out_nan={nan} finite={bool(torch.isfinite(output).all())}")

for fill in ("empty", "zero", "7f", "ff"):
    replay("/tmp/b12x_moe_nan_dump.rank5.pt", fill)
