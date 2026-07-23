"""Public preparation and execution surfaces for CuTeDSL W4A16 kernels."""

from .kernel import run_trellis256_dense
from .prepare import (
    PreparedTrellis256DenseWeight,
    prepare_trellis256_dense_weight,
)

__all__ = [
    "PreparedTrellis256DenseWeight",
    "prepare_trellis256_dense_weight",
    "run_trellis256_dense",
]
