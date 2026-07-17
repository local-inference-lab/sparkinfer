from __future__ import annotations

import pytest

from b12x.moe.fused.w4a16.kernel import (
    w4a16_hybrid_mapped_grid188_mapping_proof,
    w4a16_hybrid_mapped_grid188_task_map,
)


def test_grid188_task_map_is_an_exact_partition() -> None:
    proof = w4a16_hybrid_mapped_grid188_mapping_proof()

    assert proof["grid_x"] == 188
    assert sorted(proof["fc1_tasks"]) == list(range(128))
    assert sorted(proof["fc2_tasks"]) == list(range(768))
    assert proof["fc1_per_cta_counts"] == (1,) * 128 + (0,) * 60
    assert proof["fc2_per_cta_counts"] == (5,) * 16 + (4,) * 172
    assert proof["fc1_idle_ctas"] == tuple(range(128, 188))


@pytest.mark.parametrize("task_count", [-1, 0, 127, 129, 767, 769])
def test_grid188_task_map_rejects_other_geometries(task_count: int) -> None:
    with pytest.raises(ValueError, match="exactly 128 or 768"):
        w4a16_hybrid_mapped_grid188_task_map(task_count)
