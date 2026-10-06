"""GPU qualification entry point; architecture overrides exist only in tests.

On B300, --dispatch-sm100 selects the SM100 policy. CuTe compiles for the
physical GPU by default; the report distinguishes policy coverage from a
B200 hardware qualification. Triton runs without the dispatch override.
"""

from __future__ import annotations

import argparse
import hashlib
from contextlib import nullcontext
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("regression", "benchmark"), default="regression"
    )
    parser.add_argument("--dispatch-sm100", action="store_true")
    parser.add_argument(
        "--result", type=Path, help="Save regression and 96-option CSR evidence"
    )
    args, remaining = parser.parse_known_args()
    physical = torch.cuda.get_device_capability()
    if args.dispatch_sm100 and physical not in ((10, 0), (10, 3)):
        parser.error("SM100 qualification requires Blackwell hardware")
    print(
        json.dumps(
            dict(
                physical_arch=physical,
                dispatch_arch=(10, 0) if args.dispatch_sm100 else physical,
                cute_target=os.environ.get("CUTE_DSL_ARCH", "auto"),
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
            )
        ),
        flush=True,
    )
    context = (
        patch("torch.cuda.get_device_capability", return_value=(10, 0))
        if args.dispatch_sm100
        else nullcontext()
    )
    with context:
        if args.mode == "benchmark":
            # Worker mode avoids propagating the synthetic policy into unrelated
            # subprocesses or introducing an override in the package itself.
            sys.argv = ["bench_sparse_attention_ops.py", *remaining, "--worker"]
            runpy.run_path(str(ROOT / "benchmarks/bench_sparse_attention_ops.py"), run_name="__main__")
            output = Path(remaining[remaining.index("--output") + 1])
            records = [
                json.loads(line)
                for line in (output / "results.jsonl").read_text().splitlines()
            ]
            status = Counter(record["status"] for record in records)
            print(json.dumps(dict(records=len(records), status=status)), flush=True)
            raise SystemExit(
                any(status[k] for k in ("failed", "error", "oom", "unsupported_graph"))
            )
        from tests.regression import test_precision, test_fixed_window, test_sparse_api
        from tests.integration.test_adapter import PagedPrefillIntegration

        suite = unittest.TestSuite()
        for cls in (
            test_precision.BackendAlignment,
            test_fixed_window.FixedWindowRegression,
            PagedPrefillIntegration,
            test_sparse_api.SparseApiTest,
        ):
            for test in unittest.defaultTestLoader.loadTestsFromTestCase(cls):
                if args.dispatch_sm100 and test._testMethodName.startswith(
                    "test_triton"
                ):
                    continue
                suite.addTest(test)
        grid_records = []

        class QualificationResult(unittest.TextTestResult):
            def addSubTest(self, test, subtest, err):
                if test._testMethodName == "test_supported_parameter_grid":
                    item = {
                        k: str(v) if isinstance(v, torch.dtype) else v
                        for k, v in subtest.params.items()
                    }
                    item["status"] = "passed" if err is None else "failed"
                    if err:
                        item["exception"] = str(err[1])
                    grid_records.append(item)
                    print(json.dumps(item), flush=True)
                super().addSubTest(test, subtest, err)

        result = unittest.TextTestRunner(
            verbosity=2, resultclass=QualificationResult
        ).run(suite)
        if args.result:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(
                json.dumps(
                    dict(
                        physical_arch=physical,
                        dispatch_arch=(10, 0) if args.dispatch_sm100 else physical,
                        model=dict(q_heads=32, kv_heads=4, gqa=8, head_dim=128),
                        planned=96,
                        checked=len(grid_records),
                        records=grid_records,
                        regression=dict(
                            tests_run=result.testsRun,
                            successful=result.wasSuccessful(),
                            failures=[
                                dict(test=str(t), traceback=detail)
                                for t, detail in result.failures
                            ],
                            errors=[
                                dict(test=str(t), traceback=detail)
                                for t, detail in result.errors
                            ],
                            skipped=[
                                dict(test=str(t), reason=reason)
                                for t, reason in result.skipped
                            ],
                        ),
                        source_sha256={
                            str(p.relative_to(ROOT)): hashlib.sha256(
                                p.read_bytes()
                            ).hexdigest()
                            for p in sorted((ROOT / "python/mbsa").rglob("*.py"))
                            if not p.name.startswith("._")
                        },
                    ),
                    indent=2,
                )
                + "\n"
            )
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
