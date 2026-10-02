#!/usr/bin/env python3
"""Reject imports outside the baseline's explicit dependency allowlist."""

from __future__ import annotations

import ast
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE_PATHS = (
    ROOT / "compass_bsa",
    ROOT / "bench" / "bench_opt4_gpu_v1.py",
    ROOT / "bench" / "bench_triton_baseline.py",
    ROOT / "tests" / "test_independent_v1.py",
    ROOT / "tools" / "check_independence.py",
)
ALLOWED_IMPORT_ROOTS = (
    "__future__",
    "argparse",
    "ast",
    "bench_opt4_gpu_v1",
    "compass_bsa",
    "dataclasses",
    "json",
    "math",
    "pathlib",
    "statistics",
    "sys",
    "torch",
    "triton",
    "unittest",
)


def imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.append(node.module)
    return modules


def main() -> int:
    failures: list[str] = []
    paths: list[pathlib.Path] = []
    for scope in BASELINE_PATHS:
        paths.extend(scope.rglob("*.py") if scope.is_dir() else (scope,))
    for path in sorted(paths):
        for module in imported_modules(path):
            root = module.split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                failures.append(f"{path.relative_to(ROOT)} imports non-allowlisted {module}")
    if failures:
        print("Opt4 independence check failed:")
        print("\n".join(f"  - {failure}" for failure in failures))
        return 1
    print("Opt4 independence check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
