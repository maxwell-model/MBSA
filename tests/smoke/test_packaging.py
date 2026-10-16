#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source manifests, runtime imports and distributable package contracts."""

from __future__ import annotations
import argparse
import ast
import hashlib
import json
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
    "mbsa/third_party/fireworks/PROVENANCE.md",
    "mbsa/third_party/fireworks/UPSTREAM.json",
    "mbsa/third_party/fireworks/msa/LICENSE",
    "mbsa/third_party/fireworks/flash_attention/LICENSE",
    "mbsa/third_party/fireworks/flash_attention/AUTHORS",
)


FA4_REQUIRED = (
    "mbsa/cute/fixed_attention.py",
    "mbsa/cute/fixed_sync.py",
    "mbsa/third_party/fa4/LICENSE",
    "mbsa/third_party/fa4/AUTHORS",
    "mbsa/third_party/fa4/PROVENANCE.md",
    "mbsa/third_party/fa4/UPSTREAM.json",
)

DEEP_GEMM_REQUIRED = (
    "mbsa/third_party/deep_gemm/LICENSE",
    "mbsa/third_party/deep_gemm/NOTICE",
    "mbsa/third_party/deep_gemm/PROVENANCE.md",
    "mbsa/third_party/deep_gemm/UPSTREAM.json",
)

VLLM_REQUIRED = (
    "mbsa/third_party/vllm/LICENSE",
    "mbsa/third_party/vllm/NOTICE",
    "mbsa/third_party/vllm/PROVENANCE.md",
    "mbsa/third_party/vllm/UPSTREAM.json",
)


def check_msa_manifest(read_bytes):
    manifest = json.loads(read_bytes("mbsa/third_party/msa/UPSTREAM.json"))
    assert manifest["repository"] == "https://github.com/MiniMax-AI/MSA"
    assert manifest["commit"] == "80434d7f67877c6570ca19cac444b84bc9855dac"
    for record in manifest["files"]:
        target = record["path"].removeprefix("python/")
        assert re.fullmatch(r"[0-9a-f]{40}", record["upstream_git_blob"]), target
        assert hashlib.sha256(read_bytes(target)).hexdigest() == record["local_sha256"], target


def check_deep_gemm_manifest(read_bytes):
    manifest = json.loads(read_bytes("mbsa/third_party/deep_gemm/UPSTREAM.json"))
    assert manifest["official_repository"] == "https://github.com/deepseek-ai/DeepGEMM"
    assert manifest["license"] == "MIT"
    for record in manifest["files"]:
        target = record["path"].removeprefix("python/")
        assert hashlib.sha256(read_bytes(target)).hexdigest() == record["local_sha256"], target
        if "upstream_git_blob" in record:
            assert re.fullmatch(r"[0-9a-f]{40}", record["upstream_git_blob"]), target
        for upstream in record.get("upstream_files", ()):
            assert re.fullmatch(r"[0-9a-f]{40}", upstream["git_blob"]), upstream["path"]
    for target in DEEP_GEMM_REQUIRED:
        assert read_bytes(target), target
    assert b"MIT License" in read_bytes("mbsa/third_party/deep_gemm/LICENSE")


def check_vllm_manifest(read_bytes):
    manifest = json.loads(read_bytes("mbsa/third_party/vllm/UPSTREAM.json"))
    assert manifest["base_project"] == "https://github.com/vllm-project/vllm"
    assert manifest["source_commit"] == "950d80710610ebffc41128dad89845346262af48"
    assert manifest["license"] == "Apache-2.0"
    assert len(manifest["files"]) == 5
    for record in manifest["files"]:
        target = record["path"].removeprefix("python/")
        source = read_bytes(target)
        assert hashlib.sha256(source).hexdigest() == record["local_sha256"], target
        assert b"SPDX-License-Identifier: Apache-2.0" in source, target
        assert b"Copyright (c) 2026 Maxwell Team" in source, target
        for upstream in record["upstream_files"]:
            assert re.fullmatch(r"[0-9a-f]{40}", upstream["git_blob"]), upstream["path"]
            assert re.fullmatch(r"[0-9a-f]{64}", upstream["sha256"]), upstream["path"]
    for target in VLLM_REQUIRED:
        assert read_bytes(target), target
    assert b"Apache License" in read_bytes("mbsa/third_party/vllm/LICENSE")


def check_fa4_manifest(read_bytes):
    """Require public synchronization source, exact local hash and its license."""
    manifest = json.loads(read_bytes("mbsa/third_party/fa4/UPSTREAM.json"))
    assert manifest["package"] == "flash-attn-4[cu13]"
    assert manifest["version"] == "4.0.0b20"
    assert manifest["extracted_method"] == "FlashAttentionForwardSm100.kernel"
    assert manifest["extracted_method_lines"] == [799, 1320]
    for field in ("wheel_sha256", "source_file_sha256", "extracted_method_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", manifest[field]), field
    assert manifest["license"] == "BSD-3-Clause AND Apache-2.0"
    target = manifest["local_file"].removeprefix("python/")
    source = read_bytes(target)
    assert hashlib.sha256(source).hexdigest() == manifest["local_sha256"], target
    assert b"SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0" in source
    for path in FA4_REQUIRED:
        assert read_bytes(path), path
    assert b"BSD 3-Clause License" in read_bytes("mbsa/third_party/fa4/LICENSE")
    assert b"respective contributors" in read_bytes("mbsa/third_party/fa4/LICENSE")


def check_fireworks_manifest(read_bytes):
    """Audit shipped source against its public provenance and local hashes."""
    manifest = json.loads(read_bytes("mbsa/third_party/fireworks/UPSTREAM.json"))
    assert manifest["repository"] == "https://github.com/MiniMax-AI/MSA"
    assert manifest["branch"] == "fireworks-msa"
    assert set(manifest["licenses"]) == {"Apache-2.0", "BSD-3-Clause"}
    for record in manifest["files"]:
        target = record["path"].removeprefix("python/")
        data = read_bytes(target)
        assert hashlib.sha256(data).hexdigest() == record["local_sha256"], target
        assert len(record["upstream_git_blob"]) == 40, target
        assert len(record["upstream_sha256"]) == 64, target
        assert record["upstream_path"].startswith("python/fmha_sm100/kvouter/"), target
        assert b"Copyright (c) 2026 Fireworks AI" in data, target
        assert b"SPDX-License-Identifier:" in data, target
    for path, expected in manifest["mbsa_adapter_sha256"].items():
        assert hashlib.sha256(read_bytes(path.removeprefix("python/"))).hexdigest() == expected, path
    for target in FIREWORKS_REQUIRED:
        assert read_bytes(target), target
    assert b"Apache License" in read_bytes("mbsa/third_party/fireworks/LICENSE")
    assert b"BSD 3-Clause License" in read_bytes("mbsa/third_party/fireworks/flash_attention/LICENSE")
    assert b"MIT License" in read_bytes("mbsa/third_party/fireworks/msa/LICENSE")


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
    check_msa_manifest(lambda name: (ROOT / "python" / name).read_bytes())
    check_deep_gemm_manifest(lambda name: (ROOT / "python" / name).read_bytes())
    check_vllm_manifest(lambda name: (ROOT / "python" / name).read_bytes())
    check_fireworks_manifest(lambda name: (ROOT / "python" / name).read_bytes())
    check_fa4_manifest(lambda name: (ROOT / "python" / name).read_bytes())
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
        check_msa_manifest(wheel.read)
        missing = set(required) - names
        if missing:
            raise SystemExit(f"Wheel is missing runtime data: {sorted(missing)}")
        check_deep_gemm_manifest(wheel.read)
        check_vllm_manifest(wheel.read)
        check_fireworks_manifest(wheel.read)
        check_fa4_manifest(wheel.read)
        assert "mbsa/cute/fixed_recorded.py" not in names, "Retired recorded runtime must not ship"
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1
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
