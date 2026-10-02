"""Import-path bootstrap for every script in this project.

Each bench / test / tool script starts with the same preamble:

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N]))
    import _paths  # noqa: F401

(`N` = how deep the script sits below the project root.) Importing this module
registers:

    <root>/kernels    the four kernels this project benchmarks
    <root>/bench      so `from common.bench_utils import ...` works

Everything else the benches need (`deep_gemm`, `tilelang`, `cutlass`, FA4,
`fmha_sm100`) comes from the isolated venvs in the delivery image; see
docs/token_sparse/ENVIRONMENT.md and docs/block_sparse/ENVIRONMENT.md.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
KERNELS = ROOT / "kernels"
BENCH = ROOT / "bench"
VENDORED_MSA_PYTHON = ROOT / "third_party" / "msa" / "python"


def _prepend(p: pathlib.Path) -> None:
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


for _p in (BENCH, KERNELS, VENDORED_MSA_PYTHON):
    _prepend(_p)
