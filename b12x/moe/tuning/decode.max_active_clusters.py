"""Generated MoE decode MAX_ACTIVE_CLUSTERS tuning data."""

from .registry import register_max_active_clusters_policy

# micro: routed_rows <= 20
# static: 20 < routed_rows <= 640
# dynamic: routed_rows > 640

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
    backend="static",
    ladder=(
        (24, 148),
        (32, 169),
        (40, 132),
        (48, 149),
        (64, 134),
        (80, 175),
        (96, 171),
        (120, 125),
        (128, 130),
        (160, 171),
        (192, 166),
        (256, 141),
        (320, 158),
        (512, 175),
        (640, 188),
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
