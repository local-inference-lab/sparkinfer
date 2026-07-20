"""Environment-variable conventions for sparkinfer.

All sparkinfer tuning knobs use the ``SPARKINFER_`` prefix.  Deployments
tuned against upstream sparkinfer may still carry ``SPARKINFER_*`` variables; on first use
we copy those values onto their new names (never overwriting an explicitly set
new-style variable) and emit a single DeprecationWarning naming the mapping.
"""

from __future__ import annotations

import os
import warnings

PREFIX = "SPARKINFER_"
_LEGACY_PREFIX = "SPARKINFER_"

# Legacy suffixes whose new-style suffix is not a verbatim carry-over.
_LEGACY_COMPILE_PREFIX = "CUTE_COMPILE_"  # SPARKINFER_CUTE_COMPILE_X -> ..._COMPILE_X
_LEGACY_SPECIAL = {
    "VLLM_ENGINE_STARTED": "ENGINE_STARTED",
}

_synced = False


def _new_name(legacy_suffix: str) -> str:
    if legacy_suffix in _LEGACY_SPECIAL:
        return PREFIX + _LEGACY_SPECIAL[legacy_suffix]
    if legacy_suffix.startswith(_LEGACY_COMPILE_PREFIX):
        return PREFIX + "COMPILE_" + legacy_suffix[len(_LEGACY_COMPILE_PREFIX) :]
    return PREFIX + legacy_suffix


def sync_legacy_env() -> None:
    """Copy ``SPARKINFER_*`` values onto ``SPARKINFER_*`` names (once).

    Explicitly-set new-style variables always win.  Runs at first import of
    the sparkinfer compiler so every ported ``os.environ`` read observes the
    mapped values.
    """
    global _synced
    if _synced:
        return
    _synced = True
    mapped: list[str] = []
    for key in sorted(os.environ):
        if not key.startswith(_LEGACY_PREFIX):
            continue
        new = _new_name(key[len(_LEGACY_PREFIX) :])
        if new not in os.environ:
            os.environ[new] = os.environ[key]
            mapped.append(f"{key} -> {new}")
    if mapped:
        warnings.warn(
            "sparkinfer picked up legacy sparkinfer environment "
            "variables and mapped them to their new names: "
            + "; ".join(mapped)
            + ". Rename these to their SPARKINFER_* forms.",
            DeprecationWarning,
            stacklevel=2,
        )


def env_raw(suffix: str) -> str | None:
    """Read an sparkinfer knob by suffix, e.g. ``env_raw("MLA_FORCE_SPLIT")``."""
    sync_legacy_env()
    return os.environ.get(PREFIX + suffix)


def env_flag(suffix: str, *, default: bool = False) -> bool:
    raw = env_raw(suffix)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


__all__ = ["PREFIX", "sync_legacy_env", "env_raw", "env_flag"]
