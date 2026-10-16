#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Runtime imports and distributable package contracts."""

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

FIREWORKS_REQUIRED = (
    "mbsa/cute/kvouter/__init__.py",
    "mbsa/cute/kvouter/prepared.py",
    "mbsa/third_party/fireworks/LICENSE",
    "mbsa/third_party/fireworks/NOTICE",
    "mbsa/third_party/fireworks/msa/LICENSE",
    "mbsa/third_party/fireworks/flash_attention/LICENSE",
    "mbsa/third_party/fireworks/flash_attention/AUTHORS",
)


FA4_REQUIRED = (
    "mbsa/cute/fixed_attention.py",
    "mbsa/cute/fixed_sync.py",
    "mbsa/third_party/fa4/LICENSE",
    "mbsa/third_party/fa4/AUTHORS",
)

DEEP_GEMM_REQUIRED = (
    "mbsa/third_party/deep_gemm/LICENSE",
    "mbsa/third_party/deep_gemm/NOTICE",
)

VLLM_REQUIRED = (
    "mbsa/third_party/vllm/NOTICE",
)


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
        ROOT / "LICENSE",
        ROOT / "NOTICE",
        PACKAGE / "third_party/msa/LICENSE",
        PACKAGE / "third_party/msa/NOTICE",
        *(ROOT / "python" / name for name in DEEP_GEMM_REQUIRED),
        *(ROOT / "python" / name for name in VLLM_REQUIRED),
        *(PACKAGE / "mxfp6/csrc" / name for name in MXFP6_SOURCES),
        *(ROOT / "python" / name for name in FIREWORKS_REQUIRED + FA4_REQUIRED),
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
        "mbsa/third_party/msa/LICENSE",
        "mbsa/third_party/msa/NOTICE",
        *DEEP_GEMM_REQUIRED,
        *VLLM_REQUIRED,
        *(f"mbsa/mxfp6/csrc/{name}" for name in MXFP6_SOURCES),
        *FIREWORKS_REQUIRED,
        *FA4_REQUIRED,
    )
    with zipfile.ZipFile(path) as wheel, tempfile.TemporaryDirectory() as folder:
        names = set(wheel.namelist())
        retired = [name for name in names if name.startswith("mbsa/cute/src/sm100/fixed/") or name == "mbsa/cute/src/sm100/fwd/config.py"]
        retired += [name for name in names if name in ("mbsa/reference.py", "mbsa/triton_backend.py", "mbsa/sparse.py", "mbsa/cute/interface.py")]
        if retired:
            raise AssertionError(f"Wheel contains retired private runtime: {retired}")
        missing = set(required) - names
        if missing:
            raise SystemExit(f"Wheel is missing runtime data: {sorted(missing)}")
        assert "mbsa/cute/fixed_recorded.py" not in names, "Retired recorded runtime must not ship"
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1
        dist_info = metadata_paths[0].rsplit("/", 1)[0]
        for filename in ("LICENSE", "NOTICE"):
            if not {f"{dist_info}/{filename}", f"{dist_info}/licenses/{filename}"} & names:
                raise AssertionError(f"Wheel is missing project license material: {filename}")
        metadata = wheel.read(metadata_paths[0]).decode()
        assert "Provides-Extra: fireworks" in metadata
        requirements = [line.split(":", 1)[1].strip() for line in metadata.splitlines()
                        if line.startswith("Requires-Dist:")]
        assert not any(re.match(r"flash[-_]attn[-_]4(?:\[|\s|[<>=!~;]|$)", requirement)
                       for requirement in requirements), "FA4 b20 must remain a separate --no-deps install"
        fireworks = {requirement.split(";", 1)[0].strip() for requirement in requirements
                     if re.search(r"extra\s*==\s*['\"]fireworks['\"]", requirement)}
        assert {"nvidia-cutlass-dsl[cu13]==4.6.0", "quack-kernels==0.6.3"} <= fireworks
        ffi = next(requirement for requirement in fireworks if requirement.startswith("apache-tvm-ffi"))
        assert set(ffi.removeprefix("apache-tvm-ffi").split(",")) == {">=0.1.12", "<0.2"}
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
import torch
baseline_modules = set(sys.modules)
heavy_modules = {'cutlass', 'quack', 'flash_attn', 'cuda'}
from mbsa import prepare_cutedsl_pipeline
from mbsa.cute.kvouter import PreparedKVOuter
assert not heavy_modules.intersection(set(sys.modules) - baseline_modules)
try:
    PreparedKVOuter(
        torch.empty(1,32,128,dtype=torch.bfloat16), torch.empty(1,4,128,128,dtype=torch.bfloat16), torch.empty(1,4,128,128,dtype=torch.bfloat16),
        torch.full((1,4,1),-1,dtype=torch.int32), torch.zeros(1,1,dtype=torch.int32),
        torch.tensor([0,1],dtype=torch.int64), torch.tensor([128],dtype=torch.int32),
    )
except ValueError as error:
    assert 'CUDA' in str(error)
else:
    raise AssertionError('PreparedKVOuter must reject CPU tensors before importing CUDA dependencies')
assert not heavy_modules.intersection(set(sys.modules) - baseline_modules)
q = torch.zeros((1, 32, 128))
k = v = torch.zeros((7, 4, 128))
try:
    prepare_cutedsl_pipeline(q, k, q, k, v)
except ValueError as error:
    assert 'CUDA' in str(error)
else:
    raise AssertionError('CuTe DSL must reject CPU tensors')
assert not {'cutlass', 'fmha_sm100'}.intersection(set(sys.modules) - baseline_modules)
print('Wheel check passed: isolated CPU imports, licenses and CUDA runtime sources')
"""
        subprocess.run(
            [sys.executable, "-I", "-c", script, folder], cwd=folder, check=True
        )


class SourcePackagingContracts(unittest.TestCase):
    def test_source_imports_licenses_and_document_links(self):
        check_sources()

    def test_fireworks_import_and_cpu_rejection_stay_lazy(self):
        script = r"""
import sys
sys.path.insert(0, sys.argv[1])
import torch
baseline_modules = set(sys.modules)
heavy_modules = {'cutlass', 'quack', 'flash_attn', 'cuda'}
from mbsa.cute.kvouter import PreparedKVOuter
assert not heavy_modules.intersection(set(sys.modules) - baseline_modules)
try:
    PreparedKVOuter(
        torch.empty(1,32,128,dtype=torch.bfloat16), torch.empty(1,4,128,128,dtype=torch.bfloat16), torch.empty(1,4,128,128,dtype=torch.bfloat16),
        torch.full((1,4,1),-1,dtype=torch.int32), torch.zeros(1,1,dtype=torch.int32),
        torch.tensor([0,1],dtype=torch.int64), torch.tensor([128],dtype=torch.int32),
    )
except ValueError as error:
    assert 'CUDA' in str(error)
else:
    raise AssertionError('PreparedKVOuter must reject CPU tensors')
assert not heavy_modules.intersection(set(sys.modules) - baseline_modules)
"""
        subprocess.run([sys.executable, "-I", "-c", script, str(ROOT / "python")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate an isolated mbsa wheel")
    parser.add_argument("--wheel", type=Path, required=True)
    check_wheel(parser.parse_args().wheel)
