#!/usr/bin/env python
"""Verify GLM-5.1 routed-NVFP4 MoE virtual-TP sharding + equal-split options.

No GPU / no weights: pure config + arithmetic. Cross-checks the real
vllm `_make_virtual_axis` against a local reimplementation, then enumerates
the achievable *equal* per-shard splits and their memory cost.
"""

import math

MODEL = "/models/GLM-5.1-NVFP4-MTP-NVFP4"
TP = 6

# --- real config -----------------------------------------------------------
try:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    tcfg = getattr(cfg, "text_config", cfg)
    HIDDEN = int(tcfg.hidden_size)
    MOE_INTER = int(tcfg.moe_intermediate_size)
    N_ROUTED = int(getattr(tcfg, "n_routed_experts", 0))
    N_LAYERS = int(tcfg.num_hidden_layers)
    FIRST_DENSE = int(getattr(tcfg, "first_k_dense_replace", 0))
    print(f"[config] {MODEL}")
    print(
        f"  hidden={HIDDEN} moe_intermediate={MOE_INTER} "
        f"n_routed={N_ROUTED} layers={N_LAYERS} first_dense={FIRST_DENSE}"
    )
except Exception as e:  # pragma: no cover - fall back to known GLM-5.1 numbers
    print(f"[config] AutoConfig failed ({e}); using known GLM-5.1 values")
    HIDDEN, MOE_INTER, N_ROUTED, N_LAYERS, FIRST_DENSE = 6144, 2048, 256, 78, 3

MOE_LAYERS = N_LAYERS - FIRST_DENSE


# --- replicate _make_virtual_axis -----------------------------------------
def local_axis(orig, tp, align):
    local = math.ceil(orig / tp)
    local = math.ceil(local / align) * align
    return {"local": local, "padded": local * tp, "waste_global": local * tp - orig}


# --- cross-check against the real vllm implementation ----------------------
real = None
try:
    from vllm.config.virtual_tp import _make_virtual_axis

    real = _make_virtual_axis(MOE_INTER, TP, 32)
    print(f"[vllm _make_virtual_axis align=32] {real}")
except Exception as e:
    print(f"[vllm] could not import _make_virtual_axis ({e}); using local only")

mine = local_axis(MOE_INTER, TP, 32)
print(f"[local align=32] {mine}")
if real is not None:
    ok = real["local_size"] == mine["local"] and real["padded_size"] == mine["padded"]
    print(f"[cross-check] local matches vllm: {ok}")


# --- equal-split options ----------------------------------------------------
# compact-static gated NVFP4 needs per-shard %128; dynamic needs %16 (SF block).
print("\n[equal-split options] global moe_intermediate =", MOE_INTER, "TP =", TP)
print(f"  exact even share = {MOE_INTER}/{TP} = {MOE_INTER/TP:.2f} (not integer)")
print(f"  {'align':>6} {'local':>6} {'padded':>7} {'waste/GPU':>10} {'kernel':>16}")
for A in (16, 32, 64, 128, 256):
    ax = local_axis(MOE_INTER, TP, A)
    kern = "compact-static" if A % 128 == 0 else "dynamic (n%16)"
    waste_per_gpu = ax["local"] - MOE_INTER // TP  # extra elems each rank stores
    print(
        f"  {A:>6} {ax['local']:>6} {ax['padded']:>7} "
        f"{ax['waste_global']:>10} {kern:>16}"
    )


# --- natural-128-boundary uneven slice (the alternative) -------------------
tiles = MOE_INTER // 128
print(f"\n[natural-128 uneven slice] {MOE_INTER} = {tiles} tiles of 128")
base, rem = divmod(tiles, TP)
shards = [(base + 1) * 128 if r < rem else base * 128 for r in range(TP)]
print(f"  per-rank shards = {shards}  sum = {sum(shards)}  waste = {sum(shards)-MOE_INTER}")
print("  -> every shard %128==0 (compact-static OK), zero padding, but UNEQUAL")


# --- memory cost of L=384 vs L=352 (routed FFN, NVFP4) ---------------------
def routed_ffn_bytes_per_gpu(L):
    # per expert, per shard: w13 [2L,H] fp4 + w2 [H,L] fp4 (2 fp4/byte)
    w13 = (2 * L) * HIDDEN // 2
    w2 = HIDDEN * L // 2
    # NVFP4 block scales: 1 fp8 byte per 16 elems
    sf = ((2 * L) * HIDDEN + HIDDEN * L) // 16
    per_expert = w13 + w2 + sf
    return per_expert * N_ROUTED * MOE_LAYERS


print("\n[routed FFN weight memory per GPU]")
for L in (352, 384):
    gb = routed_ffn_bytes_per_gpu(L) / 1e9
    print(f"  L={L}: {gb:6.2f} GB")
delta = (routed_ffn_bytes_per_gpu(384) - routed_ffn_bytes_per_gpu(352)) / 1e9
print(f"  delta (384 - 352) = {delta:.2f} GB/GPU saved by a 352 equal split")
