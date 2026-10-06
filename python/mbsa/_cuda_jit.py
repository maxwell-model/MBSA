# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
# Modified for MBSA: shared by the CUDA C++ extensions; see root NOTICE.

"""Loading and building MBSA's CUDA C++ extensions.

A packaging build (vLLM's) compiles the extensions ahead of time and installs
each as ``<package>.<name>`` next to the Python module that loads it;
:func:`load_extension` imports that prebuilt module and only falls back to
compiling with ``torch.utils.cpp_extension.load``. The remaining helpers supply
the include paths and nvcc flags the JIT builds share.
"""

from __future__ import annotations

import glob
import importlib.util
import os

import torch

#: Environment variable naming a directory that holds ``cute/`` and ``cutlass/``.
CUTLASS_INCLUDE_ENV = "MBSA_CUTLASS_INCLUDE"
#: Environment variable choosing how extensions load: ``auto`` (prebuilt,
#: else JIT), ``prebuilt`` (never JIT) or ``jit`` (ignore prebuilt modules).
EXTENSIONS_ENV = "MBSA_EXTENSIONS"
#: How each loaded extension was obtained, ``"prebuilt"`` or ``"jit"``, by name.
LOADED = {}


def load_extension(package, name, jit_build):
    """The CUDA extension ``name``: the prebuilt ``<package>.<name>``, else JIT.

    Args:
      package: Package the prebuilt module is installed in (the loader's
        ``__package__``, so a vendored copy finds its own).
      name: Module name; a prebuilt module is compiled with
        ``TORCH_EXTENSION_NAME`` set to it.
      jit_build: Zero-argument callable that compiles and returns the module.

    Raises:
      RuntimeError: if ``MBSA_EXTENSIONS=prebuilt`` and no prebuilt module is
        installed.
      ValueError: if ``MBSA_EXTENSIONS`` has another value than the three
        above.
    """
    mode = os.environ.get(EXTENSIONS_ENV, "auto")
    if mode not in ("auto", "prebuilt", "jit"):
        raise ValueError(f"{EXTENSIONS_ENV} must be auto, prebuilt or jit, got {mode!r}")
    if mode != "jit":
        qualified = f"{package}.{name}"
        try:
            module = importlib.import_module(qualified)
        except ModuleNotFoundError as exc:
            if exc.name != qualified:
                raise
            if mode == "prebuilt":
                raise RuntimeError(
                    f"{EXTENSIONS_ENV}=prebuilt, but {qualified} is not installed; "
                    "build it ahead of time or unset the variable to JIT-compile it."
                ) from exc
        else:
            LOADED[name] = "prebuilt"
            return module
    module = jit_build()
    LOADED[name] = "jit"
    return module


def cuda_home_include():
    for var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(var)
        if root and os.path.isdir(os.path.join(root, "include")):
            return os.path.join(root, "include")
    return "/usr/local/cuda/include" if os.path.isdir("/usr/local/cuda/include") else None


def nvidia_wheel_include_dirs():
    """A shim include dir for CUDA headers that live in pip wheels, not CUDA_HOME.

    Containers built on the pip CUDA stack (``nvidia-cu13``, ``nvidia-cusparse``,
    ...) keep some headers under ``site-packages/nvidia/**/include`` while
    ``/usr/local/cuda/include`` carries only the toolkit's own.  ``ATen/cuda/
    CUDAContext.h`` pulls in ``cusparse.h``, so without help the extension fails
    to compile with a bare "No such file or directory".

    Adding the wheel dirs wholesale does not work: they also ship
    ``crt/host_runtime.h`` and friends, and ``torch.utils.cpp_extension`` places
    ``extra_include_paths`` *ahead* of ``-isystem $CUDA_HOME/include``.  A wheel
    whose patch level differs from the installed ``nvcc`` then shadows the
    matching runtime headers and cudafe's generated stubs stop compiling
    (``macro "__cudaLaunch" passed 2 arguments``).

    So symlink only the headers CUDA_HOME is actually missing into one shim
    directory and expose that.
    """
    try:
        import nvidia  # noqa: PLC0415 - optional, only present on pip CUDA stacks
    except ImportError:
        return []

    cuda_include = cuda_home_include()
    if cuda_include is None:
        return []

    wheel_dirs = []
    for root in getattr(nvidia, "__path__", []):
        # Both the flat (nvidia/cu13/include) and per-component
        # (nvidia/cusparse/include) wheel layouts are in the wild.
        for pattern in ("*/include", "*/*/include"):
            wheel_dirs.extend(sorted(glob.glob(os.path.join(root, pattern))))
    wheel_dirs = [d for d in dict.fromkeys(wheel_dirs) if os.path.isdir(d)]
    if not wheel_dirs:
        return []

    shim = os.path.join(
        os.path.expanduser("~/.cache/minfer"), "cuda_wheel_include_shim"
    )
    os.makedirs(shim, exist_ok=True)
    for wheel_dir in wheel_dirs:
        for header in sorted(glob.glob(os.path.join(wheel_dir, "*.h"))):
            name = os.path.basename(header)
            if os.path.exists(os.path.join(cuda_include, name)):
                continue  # CUDA_HOME already owns it — never shadow.
            link = os.path.join(shim, name)
            if os.path.lexists(link):
                continue
            try:
                os.symlink(header, link)
            except OSError:
                pass  # concurrent build won the race, or a read-only cache
    return [shim]


def cuda_arch_flag():
    """``-arch`` for the running device, falling back to the SM100 baseline.

    For plain CUDA C++ (no tcgen05) the family target is enough — no ``a``
    suffix needed — and matching the device avoids a PTX JIT at every process
    start on SM103 parts.
    """
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - no device visible at import time
        return "-arch=sm_100"
    if major < 10:
        return "-arch=sm_100"
    return f"-arch=sm_{major}{minor}"


def template_stub_flags():
    """Keep template ``__global__`` stubs externally linked.

    CUDA 12.8 flipped ``-static-global-template-stub`` to ``true`` by default,
    which breaks the generated launch stubs for template kernels declared in an
    anonymous namespace (``__cudaLaunch was not declared in this scope``).  The
    repo's csrc JIT already pins this flag; the CSR builder needs it for the
    same reason.  The option does not exist before 12.8, so it is gated.
    """
    raw = torch.version.cuda or ""
    try:
        major, minor = (int(p) for p in raw.split(".")[:2])
    except ValueError:
        return []
    return ["-static-global-template-stub=false"] if (major, minor) >= (12, 8) else []


def _has_cutlass(path):
    return bool(path) and all(
        os.path.isdir(os.path.join(path, name)) for name in ("cute", "cutlass")
    )


def _package_dir(name):
    """The directory of an importable package, without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return next(iter(spec.submodule_search_locations), None)


def cutlass_include_candidates():
    """``(source, path)`` pairs searched for CUTLASS/CuTe headers, in order."""
    candidates = []
    if os.environ.get("CUTLASS_HOME"):
        candidates.append(
            ("CUTLASS_HOME", os.path.join(os.environ["CUTLASS_HOME"], "include"))
        )
    for package, subdir in (
        ("vllm", ("third_party", "deep_gemm", "include")),
        ("deep_gemm", ("include",)),
        ("cutlass_library", ("source", "include")),
    ):
        root = _package_dir(package)
        if root is not None:
            candidates.append((f"the {package} package", os.path.join(root, *subdir)))
    return candidates


def cutlass_include_dir():
    """A directory holding the CUTLASS ``cute/`` and ``cutlass/`` headers.

    ``$MBSA_CUTLASS_INCLUDE`` wins and must be valid when set; otherwise the
    first of :func:`cutlass_include_candidates` that holds both trees is used.

    Raises:
        RuntimeError: if no candidate holds the headers.
    """
    explicit = os.environ.get(CUTLASS_INCLUDE_ENV)
    if explicit:
        if not _has_cutlass(explicit):
            raise RuntimeError(
                f"{CUTLASS_INCLUDE_ENV}={explicit} does not contain cute/ and cutlass/"
            )
        return explicit
    searched = cutlass_include_candidates()
    for _, path in searched:
        if _has_cutlass(path):
            return path
    tried = "".join(f"\n  {source}: {path}" for source, path in searched)
    raise RuntimeError(
        "MBSA's CUDA extensions need the CUTLASS/CuTe headers; set "
        f"{CUTLASS_INCLUDE_ENV} to a directory containing cute/ and cutlass/."
        + (f" Searched:{tried}" if tried else "")
    )

