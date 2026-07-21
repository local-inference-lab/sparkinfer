from __future__ import annotations

from collections.abc import Mapping

import pytest
import sparkinfer
import torch

from sparkinfer import gemm
from tests._reference.helpers import require_sparkinfer

# The two qualified geometries, represented as zero-copy views of one
# larger rowwise-MXFP8 allocation.
BATCH, PACK_ROWS, K_N_MAJOR, K_K_MAJOR = 16, 448, 192, 512
N_N_MAJOR, N_K_MAJOR_OUT = 512, 256
WARM_M_VALUES = (1, 2, 4, 8, 16, 25, 32)

BASE_SPEC = dict(
    a_dtype="bfloat16",
    b_dtype="float8_e4m3fn",
    sf_dtype="float8_e8m0fnu",
    c_dtype="bfloat16",
    sf_vec_size=32,
)


def _spec(b_major: str) -> dict[str, object]:
    return {**BASE_SPEC, "b_major": b_major, "sf_axis": b_major}


def _make_pack(seed: int = 7, batch: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = (
        torch.randn(
            batch * PACK_ROWS,
            K_K_MAJOR,
            device="cuda",
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    scales = torch.randint(
        118,
        132,
        (batch * PACK_ROWS, K_K_MAJOR // 32),
        device="cuda",
        generator=generator,
        dtype=torch.uint8,
    )
    return values, scales


def _rhs_views(
    values: torch.Tensor, scales: torch.Tensor, batch: int = BATCH
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    values_3d = values.view(batch, PACK_ROWS, K_K_MAJOR)
    scales_3d = scales.view(batch, PACK_ROWS, K_K_MAJOR // 32)
    return {
        # B-major N: logical/physical [B,K,N], scales grouped along N.
        "n": (
            values_3d[:, :K_N_MAJOR, :],
            scales_3d[:, :K_N_MAJOR, :],
        ),
        # B-major K: logical B is the transpose of physical [B,N,K].
        "k": (
            values_3d[:, K_N_MAJOR : K_N_MAJOR + N_K_MAJOR_OUT, :],
            scales_3d[:, K_N_MAJOR : K_N_MAJOR + N_K_MAJOR_OUT, :],
        ),
    }


def _dequant_physical(
    rhs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    values, scales = rhs
    scale_values = scales.view(torch.float8_e8m0fnu).to(torch.bfloat16)
    return values.to(torch.bfloat16) * scale_values.repeat_interleave(32, dim=-1)


def _logical_b(rhs: tuple[torch.Tensor, torch.Tensor], b_major: str) -> torch.Tensor:
    physical = _dequant_physical(rhs)
    return physical if b_major == "n" else physical.transpose(1, 2)


def _graph_output(
    *, b_major: str, m: int, n: int, batch: int = BATCH
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if b_major == "k":
        backing = torch.full(
            (m, batch, n + 8),
            torch.nan,
            device="cuda",
            dtype=torch.bfloat16,
        )
        return backing[..., :n].transpose(0, 1), backing[..., n:]
    return (
        torch.empty(batch, m, n, device="cuda", dtype=torch.bfloat16),
        None,
    )


@pytest.fixture(scope="module")
def packed_rhs() -> Mapping[str, tuple[torch.Tensor, torch.Tensor]]:
    require_sparkinfer()
    values, scales = _make_pack()
    rhs_by_major = _rhs_views(values, scales)
    for major, rhs in rhs_by_major.items():
        assert gemm.prewarm_bmm(rhs, WARM_M_VALUES, **_spec(major)) == len(
            WARM_M_VALUES
        )
    return rhs_by_major


def test_registry_import_order_preserves_flat_bmm_function() -> None:
    sparkinfer.list_ops()
    flat_bmm = gemm.bmm
    assert callable(flat_bmm)
    assert not hasattr(flat_bmm, "mm")

    sparkinfer.list_ops()
    assert gemm.bmm is flat_bmm


def test_public_contract_is_generic() -> None:
    meta = sparkinfer.find_op("gemm.bmm")
    assert meta.qualname == "gemm.bmm"
    assert set(meta.entry_points) == {
        "bmm",
        "prewarm_bmm",
        "can_implement_bmm",
        "is_bmm_supported",
    }
    assert callable(gemm.mm)
    assert gemm.mm is gemm.blockscaled.mm
    assert callable(gemm.bmm)
    assert not hasattr(gemm.bmm, "mm")
    assert not hasattr(gemm, "Weight")
    assert not hasattr(gemm, "qbmm_absorb")


def test_rhs_views_are_zero_copy() -> None:
    require_sparkinfer()
    values, scales = _make_pack(seed=11)
    rhs_by_major = _rhs_views(values, scales)
    n_values, n_scales = rhs_by_major["n"]
    k_values, k_scales = rhs_by_major["k"]

    assert n_values.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
    assert k_values.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
    assert n_scales.untyped_storage().data_ptr() == scales.untyped_storage().data_ptr()
    assert k_scales.untyped_storage().data_ptr() == scales.untyped_storage().data_ptr()
    assert n_values.shape == (BATCH, K_N_MAJOR, N_N_MAJOR)
    assert k_values.shape == (BATCH, N_K_MAJOR_OUT, K_K_MAJOR)
    assert n_values.stride(0) == PACK_ROWS * K_K_MAJOR
    assert k_values.storage_offset() == K_N_MAJOR * K_K_MAJOR
    assert k_scales.storage_offset() == K_N_MAJOR * (K_K_MAJOR // 32)


@pytest.mark.parametrize("b_major", ["n", "k"])
def test_dequant_is_bitwise_equal_to_bf16_reference(packed_rhs, b_major: str) -> None:
    """One-hot rows turn the BMM into a direct dequantization readout."""
    rhs = packed_rhs[b_major]
    physical = _dequant_physical(rhs)
    k = K_N_MAJOR if b_major == "n" else K_K_MAJOR
    n = N_N_MAJOR if b_major == "n" else N_K_MAJOR_OUT

    for k0 in range(0, k, 32):
        lhs = torch.zeros(BATCH, 32, k, device="cuda", dtype=torch.bfloat16)
        rows = torch.arange(32, device="cuda")
        lhs[:, rows, k0 + rows] = 1
        out = torch.empty(BATCH, 32, n, device="cuda", dtype=torch.bfloat16)
        returned = gemm.bmm(lhs, rhs, out, **_spec(b_major))
        expected = (
            physical[:, k0 : k0 + 32, :]
            if b_major == "n"
            else physical[:, :, k0 : k0 + 32].transpose(1, 2)
        )
        assert returned is out
        assert torch.equal(out, expected), f"dequant mismatch at K={k0}"


@pytest.mark.parametrize("b_major", ["n", "k"])
def test_e8m0_boundary_bytes_match_bf16_reference(packed_rhs, b_major: str) -> None:
    base_values, base_scales = packed_rhs[b_major]
    values = torch.ones_like(base_values)
    scales = torch.full_like(base_scales, 127)
    k = K_N_MAJOR if b_major == "n" else K_K_MAJOR
    n = N_N_MAJOR if b_major == "n" else N_K_MAJOR_OUT
    lhs = torch.zeros(BATCH, 1, k, device="cuda", dtype=torch.bfloat16)
    lhs[:, :, 0] = 1

    for scale_byte in (0, 1, 254, 255):
        scales.fill_(127)
        scales[:, :, 0] = scale_byte
        rhs = (values, scales)
        out = torch.empty(BATCH, 1, n, device="cuda", dtype=torch.bfloat16)
        gemm.bmm(lhs, rhs, out, **_spec(b_major))

        physical = _dequant_physical(rhs)
        expected = (
            physical[:, :1, :] if b_major == "n" else physical[:, :, :1].transpose(1, 2)
        )
        expected_nan = torch.isnan(expected)
        assert expected_nan.any().item() == (scale_byte == 255)
        assert torch.equal(torch.isnan(out), expected_nan)
        out_bits = out[~expected_nan].contiguous().view(torch.uint16)
        expected_bits = expected[~expected_nan].contiguous().view(torch.uint16)
        assert torch.equal(out_bits, expected_bits)


@pytest.mark.parametrize("b_major", ["n", "k"])
@pytest.mark.parametrize("m", WARM_M_VALUES)
def test_qualified_envelope_matches_bmm_error(packed_rhs, b_major: str, m: int) -> None:
    rhs = packed_rhs[b_major]
    logical_b = _logical_b(rhs, b_major)
    k = int(logical_b.shape[1])
    n = int(logical_b.shape[2])
    lhs = torch.randn(BATCH, m, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(BATCH, m, n, device="cuda", dtype=torch.bfloat16)

    gemm.bmm(lhs, rhs, out, **_spec(b_major))
    reference64 = torch.bmm(lhs.double(), logical_b.double())
    candidate_error = (out.double() - reference64).abs().max().item()
    cublas_error = (torch.bmm(lhs, logical_b).double() - reference64).abs().max().item()

    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out) > 0
    assert candidate_error <= cublas_error * 1.05 + 1e-12


@pytest.mark.parametrize("b_major", ["n", "k"])
def test_repeated_eager_launches_are_deterministic(packed_rhs, b_major: str) -> None:
    rhs = packed_rhs[b_major]
    logical_b = _logical_b(rhs, b_major)
    lhs = torch.randn(BATCH, 8, logical_b.shape[1], device="cuda", dtype=torch.bfloat16)
    first = torch.empty(
        BATCH, 8, logical_b.shape[2], device="cuda", dtype=torch.bfloat16
    )
    second = torch.empty_like(first)
    gemm.bmm(lhs, rhs, first, **_spec(b_major))
    gemm.bmm(lhs, rhs, second, **_spec(b_major))
    assert torch.equal(first, second)


@pytest.mark.parametrize("b_major", ["n", "k"])
@pytest.mark.parametrize("m", [4, 25])
def test_cuda_graph_replays_fresh_input_into_stable_output(
    packed_rhs, b_major: str, m: int
) -> None:
    rhs = packed_rhs[b_major]
    logical_b = _logical_b(rhs, b_major)
    lhs = torch.zeros(BATCH, m, logical_b.shape[1], device="cuda", dtype=torch.bfloat16)
    out, padding = _graph_output(
        b_major=b_major,
        m=m,
        n=int(logical_b.shape[2]),
    )
    out_ptr = out.data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = gemm.bmm(lhs, rhs, out, **_spec(b_major))
    assert returned is out

    fresh = torch.randn_like(lhs)
    lhs.copy_(fresh)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    eager = torch.empty_like(out)
    gemm.bmm(fresh, rhs, eager, **_spec(b_major))
    assert out.data_ptr() == out_ptr
    assert torch.equal(out, eager)
    if padding is not None:
        assert torch.isnan(padding).all()


@pytest.mark.parametrize("b_major", ["n", "k"])
def test_capture_compile_miss_raises(packed_rhs, b_major: str) -> None:
    rhs = packed_rhs[b_major]
    m = 13  # deliberately absent from WARM_M_VALUES
    k = K_N_MAJOR if b_major == "n" else K_K_MAJOR
    n = N_N_MAJOR if b_major == "n" else N_K_MAJOR_OUT
    lhs = torch.zeros(BATCH, m, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(BATCH, m, n, device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match=r"compile miss during CUDA-graph capture"),
        torch.cuda.graph(graph),
    ):
        gemm.bmm(lhs, rhs, out, **_spec(b_major))


def test_dtype_and_axis_dispatch_rejects_unsupported_specializations(
    packed_rhs,
) -> None:
    rhs = packed_rhs["n"]
    lhs = torch.zeros(BATCH, 1, K_N_MAJOR, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(BATCH, 1, N_N_MAJOR, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="supports only"):
        gemm.bmm(lhs, rhs, out, **{**_spec("n"), "b_dtype": "bfloat16"})
    with pytest.raises(NotImplementedError, match="sf_axis to match b_major"):
        gemm.bmm(lhs, rhs, out, **{**_spec("n"), "sf_axis": "k"})


def test_tensor_dtype_validation_matches_declared_specialization(packed_rhs) -> None:
    values, scales = packed_rhs["n"]
    lhs = torch.zeros(BATCH, 1, K_N_MAJOR, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(BATCH, 1, N_N_MAJOR, device="cuda", dtype=torch.bfloat16)
    wrong_values = values.to(torch.bfloat16)

    with pytest.raises(ValueError, match="rhs values must be float8_e4m3fn"):
        gemm.bmm(lhs, (wrong_values, scales), out, **_spec("n"))


def test_output_with_internal_overlap_is_rejected(packed_rhs) -> None:
    rhs = packed_rhs["n"]
    lhs = torch.zeros(BATCH, 2, K_N_MAJOR, device="cuda", dtype=torch.bfloat16)
    storage = torch.empty(1024, device="cuda", dtype=torch.bfloat16)
    out = torch.as_strided(storage, (BATCH, 2, N_N_MAJOR), (2, 2, 1))

    with pytest.raises(ValueError, match="out must not have internal storage overlap"):
        gemm.bmm(lhs, rhs, out, **_spec("n"))


def test_can_implement_reports_only_qualified_specialization() -> None:
    device = require_sparkinfer()
    kwargs = dict(
        batch=BATCH,
        max_m=32,
        n=N_N_MAJOR,
        k=K_N_MAJOR,
        device=device,
        **_spec("n"),
    )
    assert gemm.can_implement_bmm(**kwargs)
    assert gemm.can_implement_bmm(**{**kwargs, "batch": 8})
    assert not gemm.can_implement_bmm(**{**kwargs, "max_m": 33})
    assert not gemm.can_implement_bmm(**{**kwargs, "batch": BATCH - 1})
    assert not gemm.can_implement_bmm(**{**kwargs, "batch": 7})
    assert not gemm.can_implement_bmm(**{**kwargs, "sf_axis": "k"})
    assert not gemm.can_implement_bmm(**{**kwargs, "b_dtype": "bfloat16"})


@pytest.mark.parametrize("b_major", ["n", "k"])
def test_tp8_geometry_matches_reference_and_cuda_graph(b_major: str) -> None:
    """The GLM TP8 shard uses eight MLA heads per rank."""
    require_sparkinfer()
    batch = 8
    m = 4
    values, scales = _make_pack(seed=23, batch=batch)
    rhs = _rhs_views(values, scales, batch=batch)[b_major]
    logical_b = _logical_b(rhs, b_major)
    k = int(logical_b.shape[1])
    n = int(logical_b.shape[2])
    assert gemm.can_implement_bmm(
        batch=batch,
        max_m=32,
        n=n,
        k=k,
        device=values.device,
        **_spec(b_major),
    )
    assert gemm.prewarm_bmm(rhs, (m,), **_spec(b_major)) == 1

    lhs = torch.zeros(batch, m, k, device="cuda", dtype=torch.bfloat16)
    out, padding = _graph_output(
        b_major=b_major,
        m=m,
        n=n,
        batch=batch,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = gemm.bmm(lhs, rhs, out, **_spec(b_major))
    assert returned is out

    fresh = torch.randn_like(lhs)
    lhs.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.bmm(fresh, logical_b)
    reference64 = torch.bmm(fresh.double(), logical_b.double())
    candidate_error = (out.double() - reference64).abs().max().item()
    cublas_error = (expected.double() - reference64).abs().max().item()
    assert torch.isfinite(out).all()
    assert candidate_error <= cublas_error * 1.05 + 1e-12
    if padding is not None:
        assert torch.isnan(padding).all()


@pytest.mark.parametrize("b_major", ["n", "k"])
def test_non_current_stream_prewarm_and_mm(packed_rhs, b_major: str) -> None:
    rhs = packed_rhs[b_major]
    logical_b = _logical_b(rhs, b_major)
    launch_stream = torch.cuda.Stream(device=rhs[0].device)
    current_stream = torch.cuda.current_stream(rhs[0].device)
    assert launch_stream.cuda_stream != current_stream.cuda_stream
    assert (
        gemm.prewarm_bmm(
            rhs,
            (3,),
            stream=launch_stream,
            synchronize=False,
            **_spec(b_major),
        )
        == 1
    )

    lhs = torch.randn(BATCH, 3, logical_b.shape[1], device="cuda", dtype=torch.bfloat16)
    out = torch.empty(BATCH, 3, logical_b.shape[2], device="cuda", dtype=torch.bfloat16)
    launch_stream.wait_stream(current_stream)
    returned = gemm.bmm(lhs, rhs, out, stream=launch_stream, **_spec(b_major))
    current_stream.wait_stream(launch_stream)

    expected = torch.empty_like(out)
    gemm.bmm(lhs, rhs, expected, **_spec(b_major))
    assert returned is out
    assert torch.equal(out, expected)


@pytest.mark.parametrize("b_major", ["n", "k"])
def test_torch_compile_preserves_out_mutation_for_fresh_buffers(
    packed_rhs, b_major: str
) -> None:
    rhs = packed_rhs[b_major]
    logical_b = _logical_b(rhs, b_major)
    k = int(logical_b.shape[1])
    n = int(logical_b.shape[2])

    def run(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return gemm.bmm(a, rhs, c, **_spec(b_major))

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
    cases = [
        (
            torch.randn(BATCH, 2, k, device="cuda", dtype=torch.bfloat16),
            torch.empty(BATCH, 2, n, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(2)
    ]
    assert cases[0][0].data_ptr() != cases[1][0].data_ptr()
    assert cases[0][1].data_ptr() != cases[1][1].data_ptr()

    for lhs, out in cases:
        expected = torch.empty_like(out)
        gemm.bmm(lhs, rhs, expected, **_spec(b_major))
        returned = compiled(lhs, out)
        assert returned is out
        assert torch.equal(out, expected)
