"""Maxwell Block Sparse Attention sparse attention: Python reference, Triton baseline, CuTe DSL."""

from importlib import import_module


__all__ = ["reference_pipeline", "triton_pipeline", "prepare_cutedsl_pipeline"]

_BACKENDS = {
    "reference_pipeline": ".reference",
    "triton_pipeline": ".triton_backend",
    "prepare_cutedsl_pipeline": ".cutedsl_backend",
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
