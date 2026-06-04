"""Generated MoE decode MAX_ACTIVE_CLUSTERS tuning data."""

from .registry import register_max_active_clusters_policy

# micro:   routed_rows < 64 (direct-micro selection cutover)
# dynamic: otherwise

register_max_active_clusters_policy(
    regime="decode",
    backend="micro",
    ladder=(
        (2, 84),
        (4, 127),
        (8, 107),
        (10, 84),
        (16, 63),
        (20, 84),
    ),
)

register_max_active_clusters_policy(
    regime="decode",
    backend="dynamic",
    ladder=(
        (640, 188),
        (1024, 147),
    ),
)
