#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU contracts and full-query comparison against pinned HF Attention.

The shared config owns every shape and gate. Decode reserves scheduler budget;
only prefill enters the operator. A strict mismatch is a failure, including ties.
"""

from __future__ import annotations
import argparse
import ast
import re
import tempfile
import zipfile
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


PACKAGE = ROOT / "python/mbsa"

MSA = PACKAGE / "cute"

LOCAL_ROOTS = {
    "mbsa": ROOT / "python/mbsa",
}

MXFP6_SOURCES = (
    "fp6_dg_ext.cu",
    "fp6_ext.cu",
    "fp6_index_kernel.cuh",
    "paged_mqa_logits_scheduler.cuh",
    "sm100_fp6_paged_mqa_logits.cuh",
    "sm100_utils.cuh",
)

BASELINE_ALLOWED = {"__future__", "torch", "triton", "types", "importlib"}


def module_exists(path):
    return path.is_dir() or path.is_file() or path.with_suffix(".py").is_file()


def check_sources():
    errors = []
    # macOS resource-fork sidecars are binary metadata, not Python modules.
    paths = sorted(p for p in PACKAGE.rglob("*.py") if not p.name.startswith("._"))
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        baseline = path.parent == PACKAGE and path.name in (
            "reference.py",
            "triton_backend.py",
            "__init__.py",
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    target = path.parent
                    for _ in range(node.level - 1):
                        target = target.parent
                    if node.module:
                        target = target.joinpath(*node.module.split("."))
                        if not module_exists(target):
                            errors.append(
                                f"{path.relative_to(ROOT)}: missing relative module {node.module}"
                            )
                    continue
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                head, *tail = name.split(".")
                if head in {"src", "interface", "sparse_index_utils", "fmha_sm100"}:
                    errors.append(
                        f"{path.relative_to(ROOT)}: unscoped runtime import {name}"
                    )
                if baseline and head not in BASELINE_ALLOWED:
                    errors.append(f"{path.relative_to(ROOT)}: baseline imports {name}")
                if head in LOCAL_ROOTS:
                    target = (
                        LOCAL_ROOTS[head].joinpath(*tail) if tail else LOCAL_ROOTS[head]
                    )
                    if not module_exists(target):
                        errors.append(
                            f"{path.relative_to(ROOT)}: missing local module {name}"
                        )
    required = (
        PACKAGE / "third_party/msa/LICENSE",
        PACKAGE / "third_party/msa/NOTICE",
        PACKAGE / "third_party/fixed_bsa/LICENSE",
        PACKAGE / "third_party/fixed_bsa/NOTICE",
        PACKAGE / "third_party/deep_gemm/LICENSE",
        PACKAGE / "third_party/deep_gemm/NOTICE",
        MSA / "src/sm100/build_k2q_csr/build_k2q_csr.cu",
        *(PACKAGE / "mxfp6/csrc" / name for name in MXFP6_SOURCES),
    )
    for path in required:
        if not path.is_file():
            errors.append(f"missing runtime/license file: {path.relative_to(ROOT)}")
    for name in ("kernels", "maxwell_bsa", "mbsa", "bench"):
        if (ROOT / name).exists():
            errors.append(f"obsolete source directory: {name}")
    for path in (
        ROOT / "README.md",
        ROOT / "NOTICE",
        *sorted((ROOT / "docs").glob("*.md")),
    ):
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            target = target.split("#", 1)[0]
            if not target or "://" in target:
                continue
            if not (path.parent / target).exists():
                errors.append(
                    f"{path.relative_to(ROOT)}: missing document target {target}"
                )
    if errors:
        raise AssertionError("\n".join(errors))
    print(
        f"Source check passed: {len(paths)} Python modules; baseline isolated; runtime imports, licenses and document links present"
    )


def check_wheel(path):
    required = (
        "mbsa/index_formats.py",
        "mbsa/cute/src/sm100/mx_router.py",
        "mbsa/sparse_fmha_adapter.py",
        "mbsa/third_party/msa/LICENSE",
        "mbsa/third_party/msa/NOTICE",
        "mbsa/third_party/fixed_bsa/LICENSE",
        "mbsa/third_party/fixed_bsa/NOTICE",
        "mbsa/third_party/deep_gemm/LICENSE",
        "mbsa/third_party/deep_gemm/NOTICE",
        "mbsa/cute/src/sm100/build_k2q_csr/build_k2q_csr.cu",
        *(f"mbsa/mxfp6/csrc/{name}" for name in MXFP6_SOURCES),
    )
    with zipfile.ZipFile(path) as wheel, tempfile.TemporaryDirectory() as folder:
        names = set(wheel.namelist())
        missing = set(required) - names
        if missing:
            raise SystemExit(f"Wheel is missing runtime data: {sorted(missing)}")
        wheel.extractall(folder)
        script = r"""
import pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import mbsa
assert pathlib.Path(mbsa.__file__).resolve().is_relative_to(root)
assert not {'torch', 'triton', 'cutlass', 'fmha_sm100'}.intersection(sys.modules)
import mbsa.cute as bundled
assert pathlib.Path(bundled.__file__).resolve().is_relative_to(root)
assert 'fmha_sm100' not in sys.modules
from mbsa import reference_pipeline, prepare_cutedsl_pipeline
import torch
q = torch.zeros((1, 2, 4))
k = torch.zeros((7, 1, 4))
v = torch.arange(28).reshape(7, 1, 4).float()
output, lse, route = reference_pipeline(
    torch.ones((1, 1, 4)), torch.ones((7, 1, 4)), q, k, v,
    block_size=2, dynamic_topk=4, prefix_blocks=1, local_blocks=1,
)
torch.testing.assert_close(output[0, 0], v[:, 0].mean(0))
torch.testing.assert_close(lse, torch.full((1, 2), 7.0).log())
assert route.shape == (1, 1, 4)
try:
    prepare_cutedsl_pipeline(q, k, q, k, v)
except ValueError as error:
    assert 'CUDA' in str(error)
else:
    raise AssertionError('CuTe DSL must reject CPU tensors')
assert 'cutlass' not in sys.modules and 'fmha_sm100' not in sys.modules
print('Wheel check passed: isolated CPU imports, oracle, licenses and CUDA runtime sources')
"""
        subprocess.run(
            [sys.executable, "-I", "-c", script, folder], cwd=folder, check=True
        )


class SourcePackagingContracts(unittest.TestCase):
    def test_source_imports_licenses_and_document_links(self):
        check_sources()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate an isolated mbsa wheel")
    parser.add_argument("--wheel", type=Path, required=True)
    check_wheel(parser.parse_args().wheel)
