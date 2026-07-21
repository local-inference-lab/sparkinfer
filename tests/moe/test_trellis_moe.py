from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("cutlass")

from sparkinfer.moe import trellis_moe
from sparkinfer.moe._shared.kernels.w4a16.host import plan_w4a16_buffers


_MCG = np.uint64(0xCBAC1FED)
_MASK = np.uint32(0x8FFF8FFF)
_ORC = np.uint32(0x3B603B60)


def _caps(**overrides) -> trellis_moe.Caps:
    values = {
        "max_tokens": 32,
        "num_topk": 8,
        "num_experts": 192,
        "hidden_size": 6144,
        "intermediate_size": 512,
        "route_num_experts": 256,
        "block_size_m": 8,
        "input_dtype": torch.bfloat16,
        "device": "cuda:0",
    }
    values.update(overrides)
    return trellis_moe.Caps(**values)


def test_caps_preserve_exact_block_m_and_input_dtypes() -> None:
    bf16 = _caps(block_size_m=8, input_dtype=torch.bfloat16)
    fp16 = _caps(block_size_m=64, input_dtype=torch.float16)

    assert bf16.block_size_m == 8
    assert bf16.route_num_experts == 256
    assert bf16.input_dtype == torch.bfloat16
    assert fp16.block_size_m == 64
    assert fp16.input_dtype == torch.float16


@pytest.mark.parametrize("block_size_m", [0, 4, 12, 128])
def test_caps_reject_unsupported_block_m(block_size_m: int) -> None:
    with pytest.raises(ValueError, match="block_size_m"):
        _caps(block_size_m=block_size_m)


def test_low_level_buffer_plan_honors_explicit_block_m() -> None:
    prepared = SimpleNamespace(
        num_experts=256,
        hidden_size=6144,
        intermediate_size=512,
        is_gated=True,
    )
    automatic = plan_w4a16_buffers(
        prepared,
        m=1024,
        topk=8,
        route_num_experts=256,
        sms=120,
    )
    exact = plan_w4a16_buffers(
        prepared,
        m=1024,
        topk=8,
        route_num_experts=256,
        sms=120,
        full_rotation=True,
        block_size_m=8,
    )

    assert automatic.block_size_m != 8
    assert exact.block_size_m == 8
    assert exact.rotation_a_elements == 1024 * 8 * 6144
    with pytest.raises(ValueError, match="block_size_m"):
        plan_w4a16_buffers(
            prepared,
            m=1,
            topk=8,
            route_num_experts=256,
            sms=120,
            block_size_m=4,
        )


def _cpu_weight_tensors() -> tuple[torch.Tensor, ...]:
    w13 = torch.zeros((2, 1, 8, 8, 48), dtype=torch.int16)
    w2 = torch.zeros((1, 8, 8, 48), dtype=torch.int16)
    edge = torch.ones((1, 128), dtype=torch.float16)
    intermediate = torch.ones((1, 384), dtype=torch.float16)
    return w13, w2, edge, edge.clone(), intermediate, edge.clone()


def test_prepare_weights_rejects_non_mcg_before_cuda_work() -> None:
    w13, w2, gate_suh, up_suh, intermediate, down_svh = _cpu_weight_tensors()
    with pytest.raises(NotImplementedError, match="only the MCG"):
        trellis_moe.prepare_weights(
            w13,
            w2,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate,
            down_svh=down_svh,
            codebook="mul1",
            tile_config=(64, 128, 64, 128),
        )
    with pytest.raises(ValueError, match="unexpected MCG marker"):
        trellis_moe.prepare_weights(
            w13,
            w2,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate,
            down_svh=down_svh,
            mcg=0x83DCD12D,
            tile_config=(64, 128, 64, 128),
        )


def test_prepare_weights_validates_all_rotation_shapes() -> None:
    w13, w2, gate_suh, up_suh, intermediate, down_svh = _cpu_weight_tensors()
    with pytest.raises(ValueError, match="intermediate_rotations must have shape"):
        trellis_moe.prepare_weights(
            w13,
            w2,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate[:, :-1].contiguous(),
            down_svh=down_svh,
            tile_config=(64, 128, 64, 128),
        )


def _decode_3inst_fp16(window: np.ndarray) -> np.ndarray:
    value = window.astype(np.uint64)
    value = ((value * _MCG) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    value = np.uint32((value & _MASK) ^ _ORC)
    low = (value & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (
        ((value >> np.uint32(16)) & np.uint32(0xFFFF))
        .astype(np.uint16)
        .view(np.float16)
    )
    return (low.astype(np.float16) + high.astype(np.float16)).astype(np.float16)


def _decode_lane(tile_words: np.ndarray, lane: int, bits: int) -> np.ndarray:
    width = 8 * bits
    values = []
    for weight in range(8):
        end_bit = (lane * 8 + weight + 257) * bits
        start_bit = end_bit - 16
        first_word = start_bit // 32
        last_word = (end_bit - 1) // 32
        shift = (last_word + 1) * 32 - end_bit
        first = tile_words[..., first_word % width].astype(np.uint64)
        last = tile_words[..., last_word % width].astype(np.uint64)
        merged = (first << np.uint64(32)) | last
        window = ((merged >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.uint32)
        values.append(_decode_3inst_fp16(window))
    return np.stack(values, axis=-1).astype(np.float16)


def _reconstruct_native(trellis: torch.Tensor) -> torch.Tensor:
    native = trellis.detach().cpu().numpy()
    bits = int(native.shape[-1]) // 16
    k_tiles, n_tiles, _ = native.shape
    packed = native.view(np.uint16).reshape(k_tiles, n_tiles, 8 * bits, 2)
    words = packed[..., 0].astype(np.uint32) | (
        packed[..., 1].astype(np.uint32) << np.uint32(16)
    )
    output = np.zeros((k_tiles * 16, n_tiles * 16), dtype=np.float16)
    for k_tile in range(k_tiles):
        for n_tile in range(n_tiles):
            lanes = np.stack(
                [_decode_lane(words[k_tile, n_tile], lane, bits) for lane in range(32)]
            )
            block = np.zeros((16, 16), dtype=np.float16)
            for lane in range(32):
                row0 = (lane % 4) * 2
                rows = (row0, row0 + 1, row0 + 8, row0 + 9)
                col0 = lane // 8
                col1 = col0 + 4
                parity = (lane >> 2) & 1
                for weight in range(8):
                    block[
                        rows[weight % 4], 2 * (col0 if weight < 4 else col1) + parity
                    ] = lanes[lane, weight]
            output[
                k_tile * 16 : (k_tile + 1) * 16,
                n_tile * 16 : (n_tile + 1) * 16,
            ] = block
    return torch.from_numpy(output)


def _hadamard_128(device: torch.device) -> torch.Tensor:
    indices = torch.arange(128, dtype=torch.int64)
    rows = []
    for row in range(128):
        parity = torch.tensor(
            [(row & int(col)).bit_count() & 1 for col in indices],
            dtype=torch.bool,
        )
        rows.append(torch.where(parity, -1.0, 1.0))
    return (torch.stack(rows) / (128.0**0.5)).to(device)


def _had128(
    value: torch.Tensor,
    hadamard: torch.Tensor,
    *,
    suh: torch.Tensor | None = None,
    svh: torch.Tensor | None = None,
    store_fp16: bool,
) -> torch.Tensor:
    work = value.float()
    if suh is not None:
        work = (work * suh.float()).to(torch.float16).float()
    rows, width = work.shape
    work = (work.view(rows, width // 128, 128) @ hadamard).view(rows, width)
    if svh is not None:
        work = work * svh.float()
    return work.to(torch.float16) if store_fp16 else work


def _reference_full_rotation(
    x: torch.Tensor,
    local_ids: torch.Tensor,
    router_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
) -> torch.Tensor:
    device = x.device
    experts = int(w2.shape[0])
    intermediate = int(w2.shape[1]) * 16
    hidden = int(w2.shape[2]) * 16
    hadamard = _hadamard_128(device)
    gate_weights = torch.stack(
        [_reconstruct_native(w13[0, expert]) for expert in range(experts)]
    ).to(device)
    up_weights = torch.stack(
        [_reconstruct_native(w13[1, expert]) for expert in range(experts)]
    ).to(device)
    down_weights = torch.stack(
        [_reconstruct_native(w2[expert]) for expert in range(experts)]
    ).to(device)
    gate_svh = intermediate_rotations[:, :intermediate]
    up_svh = intermediate_rotations[:, intermediate : 2 * intermediate]
    down_suh = intermediate_rotations[:, 2 * intermediate :]
    output = torch.zeros((int(x.shape[0]), hidden), dtype=torch.float32, device=device)
    for token in range(int(x.shape[0])):
        for slot in range(int(local_ids.shape[1])):
            expert = int(local_ids[token, slot])
            source = x[token : token + 1].to(torch.float16)
            gate_a = _had128(
                source,
                hadamard,
                suh=gate_suh[expert],
                store_fp16=True,
            )
            up_a = _had128(
                source,
                hadamard,
                suh=up_suh[expert],
                store_fp16=True,
            )
            gate = (gate_a.float() @ gate_weights[expert].float()).to(torch.float16)
            up = (up_a.float() @ up_weights[expert].float()).to(torch.float16)
            gate = _had128(
                gate,
                hadamard,
                svh=gate_svh[expert],
                store_fp16=True,
            )
            up = _had128(
                up,
                hadamard,
                svh=up_svh[expert],
                store_fp16=True,
            )
            activated = (gate.float() * torch.sigmoid(gate.float()) * up.float()).to(
                torch.float16
            )
            down_a = _had128(
                activated,
                hadamard,
                suh=down_suh[expert],
                store_fp16=True,
            )
            down = (down_a.float() @ down_weights[expert].float()).to(torch.float16)
            route = _had128(
                down,
                hadamard,
                svh=down_svh[expert],
                store_fp16=False,
            )
            output[token] += router_weights[token, slot] * route[0]
    return output


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 12 and minor in (0, 1)


@pytest.mark.skipif(not _sm12x_available(), reason="requires an SM120/SM121 GPU")
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
def test_planned_full_rotation_matches_reference_and_captures(
    input_dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260721)
    device = torch.device("cuda", torch.cuda.current_device())
    experts, hidden, intermediate = 2, 128, 128
    bits = 3
    tile_config = (64, 128, 64, 128)
    w13 = torch.randint(
        -32768,
        32767,
        (2, experts, hidden // 16, intermediate // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (experts, intermediate // 16, hidden // 16, 16 * bits),
        dtype=torch.int16,
        device=device,
    )

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    gate_suh = scales((experts, hidden)).contiguous()
    up_suh = scales((experts, hidden)).contiguous()
    intermediate_rotations = scales((experts, 3 * intermediate)).contiguous()
    down_svh = scales((experts, hidden)).contiguous()
    weights = trellis_moe.prepare_weights(
        w13,
        w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        codebook="mcg",
        mcg=0xCBAC1FED,
        tile_config=tile_config,
    )
    assert weights.w13.data_ptr() == w13.data_ptr()
    assert weights.w2.data_ptr() == w2.data_ptr()

    plan = trellis_moe.plan(
        trellis_moe.Caps(
            max_tokens=2,
            num_topk=2,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            route_num_experts=4,
            block_size_m=8,
            trellis_bits=bits,
            tile_config=tile_config,
            input_dtype=input_dtype,
            device=device,
        )
    )
    assert plan.fused_launch.moe_block_size == 8
    assert plan.fused_launch.full_rotation
    assert {launch.route_ids_dtype for launch in plan.identity_sums} == {
        torch.int32,
        torch.int64,
    }
    assert all(launch.full_rotation for launch in plan.mapped_sums)

    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    x = (torch.randn((2, hidden), device=device) * 1.0e-3).to(input_dtype)
    local_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    global_ids = torch.tensor([[1, 3], [3, 1]], dtype=torch.int64, device=device)
    router_weights = torch.tensor(
        [[0.65, 0.35], [0.2, 0.8]], dtype=torch.float32, device=device
    )
    route_map = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=device)
    output_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32, device=device)
    external_output = torch.empty((2, hidden), dtype=torch.float32, device=device)

    mapped = trellis_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        weights=weights,
        topk_weights=router_weights,
        topk_ids=global_ids,
        route_expert_map=route_map,
        output_expert_map=output_map,
        output=external_output,
    )
    mapped_output = trellis_moe.run(binding=mapped)
    torch.cuda.synchronize(device)
    assert mapped_output.data_ptr() == external_output.data_ptr()
    mapped_eager = mapped_output.clone()

    identity = trellis_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        weights=weights,
        topk_weights=router_weights,
        topk_ids=local_ids,
    )
    identity_output = identity.run()
    torch.cuda.synchronize(device)
    assert identity_output.dtype == torch.float32
    assert torch.allclose(identity_output, mapped_eager, rtol=2.0e-3, atol=2.0e-3)

    reference = _reference_full_rotation(
        x,
        local_ids,
        router_weights,
        w13,
        w2,
        gate_suh,
        up_suh,
        intermediate_rotations,
        down_svh,
    )
    relative_error = (mapped_eager - reference).norm() / reference.norm().clamp_min(
        1.0e-9
    )
    cosine = torch.nn.functional.cosine_similarity(
        mapped_eager.flatten(), reference.flatten(), dim=0
    )
    assert float(relative_error) <= 2.0e-2
    assert float(cosine) >= 0.999

    mapped = trellis_moe.bind(
        plan,
        scratch=scratch,
        a=x,
        weights=weights,
        topk_weights=router_weights,
        topk_ids=global_ids,
        route_expert_map=route_map,
        output_expert_map=output_map,
        output=external_output,
    )
    mapped.run()
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    mapped.run()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == allocated_before

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = mapped.run()
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_output.data_ptr() == external_output.data_ptr()
    assert torch.allclose(captured_output, mapped_eager, rtol=2.0e-3, atol=2.0e-3)
