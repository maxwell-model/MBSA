#!/usr/bin/env python3
"""CPU/GPU contracts and full-query comparison against pinned HF Attention.

The shared config owns every shape and gate. Decode reserves scheduler budget;
only prefill enters the operator. A strict mismatch is a failure, including ties.
"""

from __future__ import annotations
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
import torch
from mbsa import prepare_cutedsl_pipeline

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


from tests.common import (
    CONFIG_PATH,
    configuration,
    select_cases,
    hf_class,
    make_inputs,
    format_index_inputs,
    oracle_checks,
    route_metrics,
    compare,
    append,
)


def matrix(config, args):
    if not torch.cuda.is_available():
        raise RuntimeError("full-query matrix requires CUDA")
    cases = select_cases(config, args.case_id)
    args.output.mkdir(parents=True, exist_ok=True)
    hf = hf_class(args.hf_source)
    source_sha = hashlib.sha256(args.hf_source.read_bytes()).hexdigest()
    # Verify the same upstream snapshot used for the previous full matrix.
    pinned = config["hf"]["source_sha256"]
    if source_sha != pinned:
        raise ValueError("HF source does not match the pinned snapshot")
    (args.output / "selected-cases.json").write_text(json.dumps(cases, indent=2))
    path = args.output / "precision.jsonl"
    if path.exists():
        raise FileExistsError(path)
    counts = {"passed": 0, "failed": 0}
    for case in cases:
        for mapping in config["options"]["page_mapping"]:
            original_data = make_inputs(case, mapping)
            data, oracle_data = format_index_inputs(original_data, args.index_format)
            probes = oracle_checks(case, oracle_data, hf)
            original_probes = (
                oracle_checks(case, original_data, hf)
                if args.index_format != "bf16"
                else probes
            )
            quantization = [
                route_metrics(p["route"], o["route"])
                for p, o in zip(probes, original_probes)
            ]
            for dtype, overlap in itertools.product(
                config["options"]["partial_dtype"], config["options"]["overlap"]
            ):
                p = prepare_cutedsl_pipeline(
                    data["iq"],
                    data["ik"],
                    data["q"],
                    data["pool"][..., 0, :].permute(0, 2, 1, 3),
                    data["pool"][..., 1, :].permute(0, 2, 1, 3),
                    index_q_scales=data.get("index_q_scales"),
                    cu_seqlens_q=data["cu_q"],
                    cu_seqlens_k=data["cu_k"],
                    index_page_indices=data["pages"],
                    page_table=data["table"],
                    partial_dtype=torch.bfloat16
                    if dtype == "bf16"
                    else torch.float8_e4m3fn,
                    overlap_fixed_score={"auto": None, "off": False, "on": True}[
                        overlap
                    ],
                )
                p.run()
                p.plan.capture_mega_graph()
                for execution in config["options"]["execution"]:
                    if execution == "eager":
                        result = p.run()
                    else:
                        p.plan.run_mega()
                        a = p.plan.attention
                        result = a.out, a.out_lse, a.tile_plan.dynamic_route
                    precision = compare(result, probes, config["tolerances"])
                    record = {
                        "config_id": case["config_id"],
                        "index_format": args.index_format,
                        "quantization_vs_bf16": quantization,
                        "partial_dtype": dtype,
                        "page_mapping": mapping,
                        "overlap": overlap,
                        "execution": execution,
                        "status": precision["status"],
                        "precision": precision,
                    }
                    append(path, record)
                    counts[record["status"]] += 1
                del p, result
            del data, probes, original_data, oracle_data, original_probes, quantization
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps({"case": case["config_id"], "status": counts}), flush=True)
    (args.output / "complete.json").write_text(json.dumps(counts, indent=2))
    if counts["failed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--index-format", choices=("bf16", "fp8", "mxfp8_mxfp6"), default="bf16"
    )
    parser.add_argument("--hf-source", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output", type=Path, default=Path("/tmp/mbsa-precision"))
    args = parser.parse_args()
    matrix(configuration(args.config), args)


if __name__ == "__main__":
    main()
