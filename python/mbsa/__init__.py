# SPDX-License-Identifier: Apache-2.0
"""Blackwell block-sparse prefill with FA4 Fixed and Fireworks Dynamic."""

from importlib import import_module


__all__ = ["prepare_cutedsl_pipeline"]

_BACKENDS = {
    "prepare_cutedsl_pipeline": ".api",
}


def __getattr__(name: str):
    """Load accelerator dependencies only when their backend is requested."""
    if name not in _BACKENDS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_BACKENDS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
