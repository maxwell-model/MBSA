#!/usr/bin/env python3
"""Render a self-contained Chinese HTML report from actual BS32 JSONL records."""

import argparse
import base64
import csv
import gzip
import hashlib
import html
import json
import itertools
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def read_jsonl(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def canonical(records, *, overlap=None):
    return [
        r
        for r in records
        if r.get("execution") == "eager"
        and r.get("l2") == "warm"
        and r.get("page_mapping") == "identity"
        and (overlap is None or r.get("overlap") == overlap)
    ]


def route_summary(records):
    result = defaultdict(lambda: Counter())
    for r in records:
        if "precision" not in r:
            continue
        precision = r["precision"]
        row = result[r["partial_dtype"]]
        row["configs"] += 1
        row["checked_queries"] += precision["checked_queries"]
        for key, value in precision["route"].items():
            if key.startswith("max_"):
                row[key] = max(row[key], value)
            else:
                row[key] += value
        row["output_max_abs"] = max(row["output_max_abs"], precision["output_max_abs"])
        row["output_max_tolerance_ratio"] = max(
            row["output_max_tolerance_ratio"], precision["output_max_tolerance_ratio"]
        )
    return {k: dict(v) for k, v in result.items()}


def slim(record, cases, kind):
    case = cases[record["config_id"]]
    q = [r["q"] for r in case["requests"]]
    kv = [r["kv"] for r in case["requests"]]
    precision = record.get("precision", {})
    route = precision.get("route", {})
    timing = record.get("timing", {})
    return {
        "kind": kind,
        "id": record["config_id"],
        "family": case["family"],
        "dtype": record["partial_dtype"],
        "map": record["page_mapping"],
        "overlap": record.get("overlap", "—"),
        "execution": record["execution"],
        "l2": record["l2"],
        "status": record["status"],
        "backend": record.get("backend", "mbsa_cutedsl"),
        "D": case.get("reserved_decode_tokens", 0),
        "scheduler_T": case.get("scheduler_total_q", sum(q)),
        "P": sum(r["phase"] == "prefill" for r in case["requests"]),
        "T": sum(q),
        "qmin": min(q),
        "qmax": max(q),
        "kvmin": min(kv),
        "kvmax": max(kv),
        "gpu": timing.get("gpu_median_ms"),
        "wall": timing.get("wall_median_ms"),
        "gpu_p95_ms": timing.get("gpu_p95_ms"),
        "prepare_first_run_ms": record.get("prepare_first_run_ms"),
        "capture_ms": record.get("capture_ms"),
        "gpu_uuid": record.get("gpu_uuid"),
        "peak_allocated_bytes": record.get("memory", {}).get("allocated_peak_bytes"),
        "error": precision.get("output_max_abs"),
        "output_rms": precision.get("output_rms"),
        "ratio": precision.get("output_max_tolerance_ratio"),
        "lse_max_abs": precision.get("lse_max_abs"),
        "lse_max_tolerance_ratio": precision.get("lse_max_tolerance_ratio"),
        "mismatch": route.get("mismatched_query_heads"),
        "heads": route.get("query_head_count"),
        "missing": route.get("missing_blocks"),
        "extra": route.get("extra_blocks"),
        "checked": precision.get("checked_queries"),
        "exception": record.get("exception", ""),
    }


def validate_records(records, cases, *, mbsa):
    dimensions = ("partial_dtype", "page_mapping", "execution", "l2")
    values = (
        ("bf16", "fp8"),
        ("identity", "permuted"),
        ("eager", "prepared_graph"),
        ("warm", "cold"),
    )
    if mbsa:
        dimensions += ("overlap",)
        values += (("auto", "off", "on"),)
    expected = {
        (cid, *option) for cid in cases for option in itertools.product(*values)
    }
    observed = [(r["config_id"], *(r[k] for k in dimensions)) for r in records]
    counts = Counter(observed)
    violations = []
    for r in records:
        case = cases[r["config_id"]]
        if r["status"] in ("passed", "failed") and (
            "precision" not in r or "timing" not in r
        ):
            violations.append([r["config_id"], "missing_precision_or_timing"])
        if (
            r.get("precision", {}).get("checked_queries", case["prefill_total_q"])
            != case["prefill_total_q"]
        ):
            violations.append([r["config_id"], "checked_queries"])
        for key in ("gpu_samples_ms", "wall_samples_ms"):
            if "timing" in r and len(r["timing"][key]) != 19:
                violations.append([r["config_id"], key])
    return {
        "missing_combinations": len(expected - counts.keys()),
        "unexpected_combinations": len(counts.keys() - expected),
        "duplicate_combinations": sum(v - 1 for v in counts.values()),
        "record_contract_violations": violations,
    }


def precision_gates(records):
    result = {}
    for dtype in ("bf16", "fp8"):
        subset = [r for r in records if r["partial_dtype"] == dtype]
        checked = [r for r in subset if "precision" in r]
        output = [
            r for r in checked if r["precision"]["output_max_tolerance_ratio"] <= 1
        ]
        lse = [
            r
            for r in checked
            if r["precision"].get("lse_max_tolerance_ratio", float("inf")) <= 1
        ]
        route = [
            r for r in checked if r["precision"]["route"]["mismatched_query_heads"] == 0
        ]
        result[dtype] = {
            "records": len(subset),
            "checked_records": len(checked),
            "output_passed_records": len(output),
            "lse_passed_records": len(lse),
            "route_passed_records": len(route),
            "strict_passed_records": sum(r["status"] == "passed" for r in subset),
            "failed_records_without_route_difference": sum(
                r["status"] == "failed"
                and r["precision"]["route"]["mismatched_query_heads"] == 0
                for r in checked
            ),
            "output_max_tolerance_ratio": max(
                (r["precision"]["output_max_tolerance_ratio"] for r in checked),
                default=None,
            ),
            "lse_max_tolerance_ratio": max(
                (
                    r["precision"]["lse_max_tolerance_ratio"]
                    for r in checked
                    if "lse_max_tolerance_ratio" in r["precision"]
                ),
                default=None,
            ),
            "output_failed_config_ids": sorted(
                {
                    r["config_id"]
                    for r in checked
                    if not r["precision"]["output_max_tolerance_ratio"] <= 1
                }
            ),
            "lse_failed_config_ids": sorted(
                {
                    r["config_id"]
                    for r in checked
                    if "lse_max_tolerance_ratio" in r["precision"]
                    and not r["precision"]["lse_max_tolerance_ratio"] <= 1
                }
            ),
            "route_failed_config_ids": sorted(
                {
                    r["config_id"]
                    for r in checked
                    if r["precision"]["route"]["mismatched_query_heads"] != 0
                }
            ),
        }
    return result


def latency_figure(mbsa, native, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    styles = [
        (mbsa, "bf16", "mbsa_cutedsl", "MBSA BF16", "#ee4d2d", "-"),
        (mbsa, "fp8", "mbsa_cutedsl", "MBSA FP8 partial", "#b8351d", "--"),
        (native, "bf16", "native_unified", "Native prefill BF16", "#6b7280", "-"),
        (native, "fp8", "native_unified", "Native prefill FP8", "#9ca3af", "--"),
        (
            native,
            "bf16",
            "legacy_sparse_gqa_fallback",
            "Native prefill fallback",
            "#6b7280",
            "-",
        ),
    ]
    for row, q in enumerate((128, 256)):
        for col, execution in enumerate(("eager", "prepared_graph")):
            ax = axes[row, col]
            for records, dtype, backend, label, color, style in styles:
                points = []
                for r in records:
                    if (
                        r.get("partial_dtype") != dtype
                        or r.get("execution") != execution
                        or r.get("l2") != "warm"
                        or r.get("page_mapping") != "identity"
                        or "timing" not in r
                    ):
                        continue
                    if records is mbsa and r.get("overlap") != "auto":
                        continue
                    if r.get("backend", "mbsa_cutedsl") != backend or not r[
                        "config_id"
                    ].startswith(f"uniform-q{q}-kv"):
                        continue
                    kv = int(r["config_id"].split("-kv")[1]) / 1024
                    points.append((kv, r["timing"]["gpu_median_ms"], r["status"]))
                points.sort()
                if points:
                    ax.plot(
                        [p[0] for p in points],
                        [p[1] for p in points],
                        style,
                        color=color,
                        label=label,
                        lw=1.8,
                    )
                    bad = [p for p in points if p[2] != "passed"]
                    if bad:
                        ax.scatter(
                            [p[0] for p in bad],
                            [p[1] for p in bad],
                            marker="x",
                            s=20,
                            color=color,
                        )
            ax.set_title(
                f"BS32, Q/request={q} · {execution}", loc="left", fontweight="bold"
            )
            ax.set_xlabel("Logical KV length / request (K tokens)")
            ax.set_ylabel("GPU median (ms, log scale)")
            ax.set_yscale("log")
            ax.set_xticks((8, 64, 128, 192, 256))
            ax.grid(alpha=0.18, which="both")
            ax.legend(loc="best", fontsize=8, frameon=False)
    fig.savefig(destination / "bs32-latency.svg")
    fig.savefig(destination / "bs32-latency.png", dpi=160)
    plt.close(fig)
    encoded = base64.b64encode((destination / "bs32-latency.svg").read_bytes()).decode()
    return f'<img style="width:100%;height:auto" alt="BS32 latency curves" src="data:image/svg+xml;base64,{encoded}">'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        c["config_id"]: c
        for c in json.loads((args.results / "selected-cases.json").read_text())
    }
    mbsa = read_jsonl(args.results / "results.jsonl")
    native_core = read_jsonl(args.native / "native-bf16.jsonl") + read_jsonl(
        args.native / "native-fp8.jsonl"
    )
    native = native_core
    environment = json.loads((args.results / "environment.json").read_text())
    status_mbsa, status_native = (
        Counter(r["status"] for r in mbsa),
        Counter(r["status"] for r in native),
    )
    summary = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "selected_configs": len(cases),
        "unique_q_kv_geometries": len(
            {tuple((r["q"], r["kv"]) for r in c["requests"]) for c in cases.values()}
        ),
        "planned_mbsa_records": len(cases) * 48,
        "actual_mbsa_records": len(mbsa),
        "planned_native_records": len(cases) * 16,
        "actual_native_records": len(native),
        "mbsa_status": dict(status_mbsa),
        "native_status": dict(status_native),
        "mbsa_routes": route_summary(canonical(mbsa, overlap="auto")),
        "native_routes": route_summary(canonical(native_core)),
        "mbsa_routes_all_page_maps": route_summary(
            [
                r
                for r in mbsa
                if r["execution"] == "eager"
                and r["l2"] == "warm"
                and r["overlap"] == "auto"
            ]
        ),
        "native_routes_all_page_maps": route_summary(
            [r for r in native if r["execution"] == "eager" and r["l2"] == "warm"]
        ),
        "environment": environment,
        "measurement_scope": "prefill only; decode reserves scheduler tokens",
        "configuration_coverage": {
            "selected": len(cases),
            "catalog": 11328,
            "fraction": len(cases) / 11328,
            "uniform_selected": 224,
            "mixed_budget_selected": 192,
        },
        "mbsa_validation": validate_records(mbsa, cases, mbsa=True),
        "native_validation": validate_records(native, cases, mbsa=False),
        "mbsa_precision_gates": precision_gates(mbsa),
        "native_precision_gates": precision_gates(native),
    }
    unit_path = args.output.parent / "unit-tests.log"
    unit_text = unit_path.read_text() if unit_path.exists() else ""
    unit_match = re.search(r"Ran (\d+) tests in ([\d.]+)s", unit_text)
    unit_skips = re.search(r"skipped=(\d+)", unit_text)
    summary["gpu_unit_tests"] = {
        "tests": int(unit_match.group(1)) if unit_match else None,
        "elapsed_seconds": float(unit_match.group(2)) if unit_match else None,
        "status": "passed"
        if unit_match and re.search(r"^OK(?:\s|$)", unit_text, re.M)
        else "not_verified",
        "skipped": int(unit_skips.group(1)) if unit_skips else 0,
        "log": "unit-tests.log" if unit_path.exists() else None,
    }
    audit_path = args.output.parent / "runtime-source-audit.json"
    if audit_path.exists():
        summary["runtime_source_audit"] = json.loads(audit_path.read_text())
    context_faults = [
        r for r in native if "illegal memory access" in r.get("exception", "")
    ]
    summary["native_context_faults"] = {
        "records": len(context_faults),
        "config_ids": sorted({r["config_id"] for r in context_faults}),
        "handling": "error records retained; new process continues unrecorded combinations",
    }
    replay = read_jsonl(args.output.parent / "native-fresh-process-replay.jsonl")
    summary["native_fresh_process_replay"] = {
        "records": len(replay),
        "status": dict(Counter(r["status"] for r in replay)),
        "config_ids": sorted({r["config_id"] for r in replay}),
        "per_dtype_status": {
            dtype: dict(
                Counter(r["status"] for r in replay if r["partial_dtype"] == dtype)
            )
            for dtype in ("bf16", "fp8")
        },
        "scope": "supplemental fresh-process replay, normal asynchronous execution; excluded from matrix counts and speed ratios",
    }
    native_map = defaultdict(list)
    for r in native:
        if "timing" not in r:
            continue
        native_map[
            (
                r["config_id"],
                r["partial_dtype"],
                r["page_mapping"],
                r["execution"],
                r["l2"],
            )
        ].append(r)
    comparisons = defaultdict(list)
    valid_comparisons = defaultdict(list)
    for r in mbsa:
        if r.get("overlap") != "auto" or "timing" not in r:
            continue
        peers = native_map.get(
            (
                r["config_id"],
                r["partial_dtype"],
                r["page_mapping"],
                r["execution"],
                r["l2"],
            )
        )
        for peer in peers or []:
            if (
                r.get("gpu_uuid")
                and peer.get("gpu_uuid")
                and r["gpu_uuid"] != peer["gpu_uuid"]
            ):
                raise ValueError("refuse a cross-GPU timing comparison")
            ratio = peer["timing"]["gpu_median_ms"] / r["timing"]["gpu_median_ms"]
            key = (
                r["partial_dtype"],
                r["execution"],
                r["l2"],
                peer.get("backend", "unknown"),
            )
            comparisons[key].append(ratio)
            if r["status"] == peer["status"] == "passed":
                valid_comparisons[key].append(ratio)
    summary["same_gpu_timing_pairs"] = sum(len(v) for v in comparisons.values())
    summary["strict_gate_definition"] = {
        "mbsa": "output tolerance AND LSE tolerance AND exact remote block set",
        "native": "output tolerance AND exact remote block set; LSE not exposed",
    }
    summary["speed_ratios"] = [
        {
            "dtype": k[0],
            "execution": k[1],
            "l2": k[2],
            "native_backend": k[3],
            "pairs": len(v),
            "native_to_mbsa_median": statistics.median(v),
            "native_to_mbsa_geomean": math.exp(statistics.mean(math.log(x) for x in v)),
            "both_precision_passed_pairs": len(valid_comparisons[k]),
            "both_passed_ratio_median": statistics.median(valid_comparisons[k])
            if valid_comparisons[k]
            else None,
        }
        for k, v in sorted(comparisons.items())
    ]
    historical_path = args.output.parent / "full-real-route-counts.json"
    historical_rows = []
    if historical_path.exists():
        historical = json.loads(historical_path.read_text())
        summary["historical_real_layer_routes"] = historical["totals"]
        for dtype, value in historical["totals"].items():
            examples = [r for r in historical["records"] if r["dtype"] == dtype]
            fraction = value["mismatched_query_heads"] / value["query_head_count"]
            overlap = value["intersection_blocks"] / value["expected_valid_blocks"]
            historical_rows.append(
                f"<tr><td>{dtype}</td><td>{value['query_head_count']:,}</td>"
                f"<td>{value['mismatched_query_heads']:,}（{fraction:.4%}）</td>"
                f"<td>{value['missing_blocks']:,} / {value['extra_blocks']:,}</td>"
                f"<td>{value['missing_blocks'] / value['query_head_count']:.4f} / 16</td>"
                f"<td>{overlap:.4%}</td><td>{max(r['native_vs_hf_derived_oracle']['max_abs_error'] for r in examples):.6g}</td>"
                f"<td>{max(r['mbsa_vs_hf_derived_oracle']['max_abs_error'] for r in examples):.6g}</td></tr>"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = read_jsonl(args.output.parent / "route-diagnostics.jsonl")
    examples = [x for r in diagnostics for x in r["examples"]]
    summary["route_diagnostics"] = {
        "config_map_pairs": len(diagnostics),
        "scored_mismatched_query_heads": len(examples),
        "HF_fp32_exact_score_sum_ties": sum(
            x["HF_fp32_exact_score_sum_tie"] for x in examples
        ),
        "near_tie_abs_score_loss_le_1e_5": sum(
            x["near_tie_abs_score_loss_le_1e_5"] for x in examples
        ),
        "max_abs_selected_score_loss_fp32": max(
            (abs(x["selected_score_loss_fp32"]) for x in examples), default=0
        ),
        "max_abs_selected_score_loss_fp64": max(
            (abs(x["selected_score_loss_fp64"]) for x in examples), default=0
        ),
        "forced_HF_attention_passed": sum(
            x["attention_vs_HF_with_MBSA_route"]["passed"] for x in examples
        ),
    }
    (args.output.parent / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    rows = [slim(r, cases, "MBSA") for r in mbsa] + [
        slim(r, cases, "原生 vLLM") for r in native
    ]
    with (args.output.parent / "results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    artifacts = []
    for path in [
        args.results / "results.jsonl",
        args.native / "native-bf16.jsonl",
        args.native / "native-fp8.jsonl",
    ]:
        if path.exists():
            target = args.output.parent / (path.name + ".gz")
            with target.open("wb") as f:
                f.write(gzip.compress(path.read_bytes(), mtime=0))
            artifacts.append(
                {
                    "name": target.name,
                    "bytes": target.stat().st_size,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
    for name in (
        "hf-source.json",
        "full-real-route-counts.json",
        "unit-tests.log",
        "route-diagnostics.jsonl",
        "gpu-monitor.jsonl",
        "gpu6-monitor.jsonl",
        "selected-cases.json",
        "environment.json",
        "runtime-source-audit.json",
        "native-fresh-process-replay.jsonl",
        "post-test-cleanup.json",
    ):
        path = args.output.parent / name
        if path.exists():
            artifacts.append(
                {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    for path in sorted((args.output.parent / "evidence").rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "name": str(path.relative_to(args.output.parent)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (args.output.parent / "artifact-hashes.json").write_text(
        json.dumps(artifacts, indent=2)
    )
    route_rows = []
    for backend, values in (
        ("MBSA", summary["mbsa_routes"]),
        ("原生 vLLM", summary["native_routes"]),
    ):
        for dtype, value in values.items():
            count = value["query_head_count"]
            mismatch = value["mismatched_query_heads"]
            valid = value["expected_valid_blocks"]
            overlap = value["intersection_blocks"] / valid if valid else 1
            route_rows.append(
                f"<tr><td>{backend}</td><td>{dtype}</td><td>{value['configs']}</td>"
                f"<td>{count:,}</td><td>{mismatch:,} / {mismatch / max(count, 1):.3e}</td>"
                f"<td>{value['missing_blocks']:,} / {value['extra_blocks']:,}</td>"
                f"<td>{overlap:.6%}</td><td>{value['output_max_abs']:.6g}</td></tr>"
            )
    perf_rows = []
    for value in summary["speed_ratios"]:
        passed = value["both_passed_ratio_median"]
        perf_rows.append(
            f"<tr><td>{value['dtype']}</td><td>{value['execution']}</td><td>{value['l2']}</td>"
            f"<td>{value['native_backend']}</td><td>{value['pairs']}</td>"
            f"<td>{value['native_to_mbsa_median']:.3f}×</td>"
            f"<td>{value['both_precision_passed_pairs']}</td><td>{passed:.3f}×</td></tr>"
            if passed is not None
            else f"<tr><td>{value['dtype']}</td><td>{value['execution']}</td><td>{value['l2']}</td>"
            f"<td>{value['native_backend']}</td><td>{value['pairs']}</td><td>{value['native_to_mbsa_median']:.3f}×</td>"
            f"<td>0</td><td>—</td></tr>"
        )
    mbsa_complete = not any(summary["mbsa_validation"].values())
    native_complete = not any(summary["native_validation"].values())
    figure = latency_figure(mbsa, native, args.output.parent)
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace(
        "</", r"<\/"
    )
    shape_data = json.dumps(
        list(cases.values()), ensure_ascii=False, separators=(",", ":")
    ).replace("</", r"<\/")
    doc = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Maxwell Block Sparse Attention · BS32 精度与性能测试</title>
<style>
:root{--orange:#ee4d2d;--ink:#20242a;--muted:#646b75;--line:#e5e8ec;--soft:#fff3ef}*{box-sizing:border-box}body{margin:0;background:#f6f7f9;color:var(--ink);font:15px/1.65 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:36px 28px}h1{font-size:32px;line-height:1.3;margin:12px 0}h2{font-size:22px;margin:30px 0 15px}h3{font-size:17px}p{max-width:1120px}.eyebrow{color:var(--orange);font-weight:700;letter-spacing:.05em}.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:24px 0}.card,section{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}.card{min-width:0;overflow-wrap:anywhere}section{margin:18px 0;min-width:0}.card strong{font-size:28px;display:block}.note{background:var(--soft);border-left:4px solid var(--orange);padding:14px 18px;margin:16px 0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{background:#f3f4f6;font-weight:650;position:sticky;top:0}tr:hover{background:#fff8f5}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}input,select,button{font:inherit;padding:7px 10px;border:1px solid #d4d9df;border-radius:6px;background:white}button{cursor:pointer}.passed{color:#26754a}.failed,.error,.oom{color:#c43c24}.unsupported_graph{color:#805a1d}code,pre{font:12px/1.6 ui-monospace,SFMono-Regular,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#cf4527}details{margin:12px 0}#detail{white-space:pre-wrap;max-height:500px;overflow:auto}footer{margin:26px 0;color:var(--muted)}@media(max-width:800px){main{padding:20px 12px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:25px}}
</style><main>
<div class="eyebrow">MAXWELL BLOCK SPARSE ATTENTION / B300 / BS32</div>
<h1>精度、路由与性能测试结果</h1><p class="muted">生成于 __DATE__（上海时间）。调度负载：32 请求、每步总 Q 预算 8192、Block 128、Prefix-8 + SWA-8、Remote Top-16；主输入和输出 BF16。</p>
<div class="cards"><div class="card">选定配置<strong>__CASES__</strong>__GEOMETRIES__ 个不同 Q/KV geometry</div><div class="card">MBSA 算子记录<strong>__ACTUAL__ / __PLANNED__</strong>__STATUS__</div><div class="card">原生对比记录<strong>__NATIVE_ACTUAL__ / __NATIVE_PLANNED__</strong>__NATIVE_STATUS__</div><div class="card">精度范围<strong>全部 query</strong>MBSA 全配置；原生成功执行项校验完整输出与 route，执行异常单列</div></div>
<section><h2>测试结论</h2><div class="note">__CONCLUSION__</div><p>输出、LSE 和路由门限分别统计如下；一条记录只有三个门限均通过才是 MBSA strict passed。原生 Attention 不提供可比 LSE，原生 strict gate 使用输出与路由。</p><div class="scroll"><table><thead><tr><th>实现</th><th>Partial</th><th>已校验 / 全部记录</th><th>输出通过</th><th>LSE 通过</th><th>路由通过</th><th>严格通过</th></tr></thead><tbody>__GATES__</tbody></table></div><p>__GATE_FINDING__</p><p>__PERF_FINDING__</p></section><section><h2>如何阅读结果</h2><p>mixed batch 中的 D 个 decode 只预留调度预算，不送入被测算子。输入只打包 P 个 prefill：满预算时其总 Q=8192−D，未满预算时更少。表中 D/P 是调度负载，Prefill Q 是实际执行、计时和精度统计的 token 数。纯 decoding 未纳入测试。</p><p>本次按已确认的「KV 主档位 + Q 边界 + 所有 mixed 族」选择有限矩阵，未逐项执行全部 11,328 个清单配置，也未穷举整数 KV 域。每个选定配置展开 48 个算子组合：BF16/FP8 partial × eager/prepared graph × identity/permuted pages × warm/cold L2 × overlap auto/off/on。</p>
<p>矩阵包含 KV 8K、16K、…、256K 全部 32 主档位；等长 Q=16/17/127/128/129/255/256；单 prefill 的 16 个 Q 边界在 KV 8K/64K/128K/256K；P=1–32 的均分、倾斜和随机 varlen（满预算/未满预算）族，种子 17/42/2026、三种请求顺序均有覆盖。</p>
<div class="note">__COMPLETENESS__。__UNIT_TESTS__；性能是合成输入的 Attention 算子结果，不是全模型吞吐。实际 vLLM engine 的 prefix-cache、PIECEWISE/FULL 组合不在此次算子矩阵的完成计数中。</div>
<p>正确性基准直接执行原始 HF 文件中提取的五个未修改方法（<a href="hf-source.json">源文件与 AST 哈希</a>），版本 <a href="https://git.garena.com/shopee/llm/instruction-tuning/compass-training/compassv4_hf_assets/-/blob/d97a6c52967a5ed332e2196a5b985f0b4a475549/modeling_compass_moe.py"><code>d97a6c52967a5ed332e2196a5b985f0b4a475549</code></a>。HF attention 输出与保留 Python 输出逐元素严格一致；LSE 由保留版本的可选返回校验。模型投影、RoPE、MoE FFN、cache 调度未纳入这次 Attention 级比较。</p>
<p>输出门限：<code>|ΔO| ≤ 0.03 + 0.03|O_HF|</code>；LSE：<code>|ΔLSE| ≤ 0.01 + 0.002|LSE_ref|</code>；Remote 路由比较有效 block ID 集合，忽略排列，但要求集合完全一致。原生路径的计时即使完成，HF 精度失败仍保留失败状态。</p></section>
<h2>路由差异的数量级</h2><section><p>下表使用 identity pages、eager/warm、MBSA overlap=auto 的每个配置一次结果。BF16 和 FP8 分开统计，避免把四种计时重复计入路由分母。分母是 query × 4 个 index head；一个 head 的 16 个 remote block 中任意 block 集合变化，即记作一个差异 query/head。多选/缺失按集合计算，Top-K 顺序变化不计为差异。</p>
<div class="scroll"><table><thead><tr><th>实现</th><th>Partial</th><th>配置数</th><th>Query/head 总数</th><th>差异数量 / 比例（科学记数）</th><th>缺失 / 多选 block</th><th>有效 remote block 重合率</th><th>输出最大绝对误差</th></tr></thead><tbody>__ROUTES__</tbody></table></div>
<p>__ROUTE_ALL_MAPS__</p><p>permuted 使用全局 page ID 逆序映射，并按对应 paged 输入重算 HF 基准。Native score-gap 诊断对每个请求最多八个发生差异的 query 重算 FP32 block-max dot scores；路由数量统计仍覆盖全部 query。详细 <code>score_gap</code> 数据在原始 JSONL。零差异表示本次有限输入上的观测值；不推断所有生产输入都无差异。</p></section>
<section><h3>MBSA 严格路由失败的分数与算术诊断</h3><p>以下诊断从失败配置重放，按原始 HF 的 16-query/8192-key GEMM 几何重算发生替换的 block 分数，并以 FP64 dot 检查数值间隔。强制 HF Attention 使用 MBSA 的 route 后再比较输出，用于分离路由选择与 Attention 算术。严格 failed 状态仍保留；近似等分没有自动计为通过。</p><p>__DIAGNOSTIC_FINDING__</p><details><summary>诊断统计 JSON</summary><pre>__DIAGNOSTICS__</pre></details><p><a href="route-diagnostics.jsonl">逐 query/head 分数、block ID 与强制路由输出误差</a></p></section>
<h2>此前真实层完整 chunk 的路由差异</h2><section><p>这是独立的历史数据集：2026-10-03 两个受影响的完整 chunk（Q3836/KV7804、Q3701/KV11893），各四层，BF16/FP8 各 30,148 个 query/layer 条目。捕获的 MBSA Remote 集合已逐 query 证明与 pinned HF-derived Python 一致；本次从完整原始 tensor 捕获重新统计 native 的集合差异，没有只看最大误差窗口。</p>
<div class="scroll"><table><thead><tr><th>Partial</th><th>Query/head</th><th>原生路由差异</th><th>缺失 / 多选 block</th><th>平均缺失 / 每 head</th><th>Block 重合率</th><th>原生输出 MaxAbs</th><th>MBSA 输出 MaxAbs</th></tr></thead><tbody>__HISTORICAL__</tbody></table></div>
<div class="note">真实层的原生路由差异约为 10⁵ 个 query/head、10⁶ 个 block 替换，属于 10⁰ 量级的比例差异。MBSA 在这些完整 chunk 的 HF-derived gate 均通过；原生输出在强制使用其自身 route 的 Python Attention 下通过，支持将大误差定位到路由。新的随机合成矩阵即使通过，也不能抹去这组真实输入上的差异。</div><p><a href="full-real-route-counts.json">完整数量、分布与原始捕获 SHA-256</a></p></section>
<section><h3>原生 CUDA context 异常与独立重放</h3><p>连续 shape 序列中的异常保留为 error。外层 runner 在 CUDA context 损坏后启动新进程，继续未记录的选项。异常配置及数量由下方 JSON 给出。若提供独立重放结果，其 scope 和状态单独保留；重放数据单列，不覆盖原始异常，也不计入主矩阵完成数或时间比。</p><pre>__NATIVE_FAULTS__</pre><p><a href="native-fresh-process-replay.jsonl">独立重放的完整记录</a> · <a href="evidence/README.md">执行快照、异常及恢复日志</a></p></section><h2>性能比较</h2><section><p>每个组合预热 5 次，19 次测量，eager/graph 交替顺序。CUDA events 与同步墙钟同时记录；cold L2 在计时前写入 256 MiB flush buffer 并同步。编译、计划准备和 capture 排除于热运行时间之外；MBSA 另存每个配置的准备/首轮与 capture 时间，原生的冷编译过程保留在日志中。这里的 graph 是完整预备算子计划。MBSA 另检查修改 live Q 后的 graph/eager 输出逐元素一致；两种 page map 各自准备计划。</p>
__FIGURE__<p class="muted">Warm L2、identity pages、auto overlap。× 标记严格精度失败点；其时间仍展示为诊断数据。上排每请求 Q=128，总 prefill Q=4096；下排每请求 Q=256，总 prefill Q=8192。可导出 <a href="bs32-latency.svg">SVG</a> / <a href="bs32-latency.png">PNG</a>。</p>
<p>__ROUTER_FINDING__图中 × 表示严格精度失败，不能据此声称已通过精度资格化。</p><p>比值 = 原生 GPU median / MBSA GPU median；大于 1 表示 MBSA 用时更短。先逐配置配对，再取比值中位数。MBSA 使用 auto overlap；按相同请求 partial 档位、page map、execution 和 L2 配对；fallback 的实际 BF16 精度已单独标明。原生 unified 的 Score plan 需要 <code>max_qo_len≥129</code>；该 plan 包含 index-head 打包，不等于逻辑 query 长度。按实际 API 返回值记录是否走 legacy sparse-GQA fallback；fallback 采用 BF16 Attention，FP8 partial 开关不适用于此路径，表中 FP8 表示请求的对比档位。</p>
<div class="scroll"><table><thead><tr><th>Partial</th><th>执行</th><th>L2</th><th>原生路径</th><th>全部配对</th><th>全部时间比</th><th>双方精度通过配对</th><th>双方通过的时间比</th></tr></thead><tbody>__PERF__</tbody></table></div>
<div class="note">HF 路由不一致的原生记录只能作为时间参考，不能据此声称等语义加速。未提供 packed batch 接口的 Triton 精度基线由 GPU 单元测试验证，不把逐请求串行时间当作 packed BS32 的同等实现。</div></section>
<h2>完整记录（可筛选，点击行查看 Q/KV 明细）</h2><section>
<div class="controls"><input id="search" placeholder="配置 ID / 错误关键词"><select id="kind"><option value="">全部实现</option><option>MBSA</option><option>原生 vLLM</option></select><select id="dtype"><option value="">全部 Partial</option><option>bf16</option><option>fp8</option></select><select id="status"><option value="">全部状态</option><option>passed</option><option>failed</option><option>error</option><option>oom</option><option>unsupported_graph</option></select><select id="execution"><option value="">全部执行</option><option>eager</option><option>prepared_graph</option></select><select id="l2"><option value="">全部 L2</option><option>warm</option><option>cold</option></select><select id="overlap"><option value="">全部 Overlap</option><option>auto</option><option>off</option><option>on</option></select><button id="download">下载筛选 JSON</button></div>
<div class="controls"><button id="previous">上一页</button><span id="count"></span><button id="next">下一页</button></div>
<div class="scroll"><table><thead><tr><th>实现 / 配置</th><th>D/P</th><th>Prefill Q</th><th>KV min–max</th><th>Partial / Map / Overlap</th><th>执行 / L2</th><th>状态</th><th>GPU ms</th><th>Wall ms</th><th>输出 MaxAbs</th><th>路由差异/总数</th></tr></thead><tbody id="records"></tbody></table></div><details open><summary>选中记录的请求长度、误差与异常</summary><pre id="detail">点击一行查看。</pre></details></section>
<h2>环境、覆盖率和原始证据</h2><section><p>测试在物理 GPU 1 和 GPU 6 的空闲 B300 上分片进行，容器 <code>mbsa-bs32-test-min</code> / <code>mbsa-bs32-gpu6-min</code>，镜像 <code>maxwell-refactor-min:overlay</code>。每个配置的 MBSA/native 配对固定在同一张 GPU 上，未比较不同卡的耗时。占卡的 <code>maxwell-ruler-min</code> 和 <code>maxwell-base-min</code> 已按用户授权停止；GPU 7 上的 <code>vllm-qwen35</code> 未停止。GPU 进程记录按 30 秒采样保留。</p>
<p>当前测试矩阵是 __CASES__ 个配置 / __GEOMETRIES__ 个不同 prefill Q/KV geometry；清单中的 8192 等长配置选取 224 个，mixed 预算构造样例选取 192 个。配置覆盖率为 416/11328≈3.67%；32 个主 KV 档位、列出的边界、P=1–32 和算子 48 个选项均展开。中间 Q、每个整数 KV、多次 seed 的完整交叉积以及 full-model/engine 选项仍不在本次完成范围内。</p>
<p>本报告针对 B300/SM103。本次执行源码、共享配置和两个测试入口的哈希由<a href="runtime-source-audit.json">源码审计</a>记录；SM103 的修复后结果与此前报告分开保存。保留的 H100 路径及历史结果属于独立测试范围。</p><p>__SCOPE_DIFFERENCES__</p><details><summary>环境与汇总 JSON</summary><pre>__SUMMARY__</pre></details><p><a href="results.csv">完整 CSV</a> · <a href="summary.json">汇总 JSON</a> · <a href="artifact-hashes.json">压缩原始数据 SHA-256</a></p>__ARTIFACTS__</section>
<footer>结果由实际 JSONL 生成；失败、OOM 和 unsupported 保留。独立计时环境与精度范围以附带日志为准。</footer></main>
<script id="data" type="application/json">__DATA__</script><script id="shapes" type="application/json">__SHAPES__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), cases=new Map(JSON.parse(document.getElementById('shapes').textContent).map(c=>[c.config_id,c]));let page=0,filtered=[];const size=60;const fields=['kind','dtype','status','execution','l2','overlap'];const esc=s=>String(s??'—').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');const num=(n,d=4)=>n==null?'—':Number(n).toFixed(d);
function render(){const query=document.getElementById('search').value.toLowerCase();filtered=data.filter(r=>fields.every(k=>!document.getElementById(k).value||r[k]===document.getElementById(k).value)&&(!query||(r.id+' '+r.family+' '+r.exception).toLowerCase().includes(query)));const max=Math.max(0,Math.ceil(filtered.length/size)-1);page=Math.min(page,max);document.getElementById('count').textContent=`${filtered.length.toLocaleString()} 条 · 第 ${page+1}/${max+1} 页`;document.getElementById('records').innerHTML=filtered.slice(page*size,(page+1)*size).map((r,i)=>`<tr data-index="${page*size+i}"><td>${esc(r.kind)}<br>${esc(r.id)}</td><td>${r.D}/${r.P}</td><td>${r.T}</td><td>${r.kvmin}–${r.kvmax}</td><td>${esc(r.dtype)} / ${esc(r.map)} / ${esc(r.overlap)}</td><td>${esc(r.execution)} / ${esc(r.l2)}</td><td class="${esc(r.status)}">${esc(r.status)}</td><td>${num(r.gpu)}</td><td>${num(r.wall)}</td><td>${num(r.error,6)}</td><td>${r.mismatch??'—'}/${r.heads??'—'}</td></tr>`).join('');for(const row of document.querySelectorAll('tr[data-index]'))row.onclick=()=>{const r=filtered[Number(row.dataset.index)];document.getElementById('detail').textContent=JSON.stringify({record:r,prefill_requests:cases.get(r.id).requests,scheduler_requests:cases.get(r.id).scheduler_requests,scheduler_phases:cases.get(r.id).scheduler_phases},null,2)};}
for(const k of fields)document.getElementById(k).onchange=()=>{page=0;render()};document.getElementById('search').oninput=()=>{page=0;render()};document.getElementById('previous').onclick=()=>{page=Math.max(0,page-1);render()};document.getElementById('next').onclick=()=>{page++;render()};document.getElementById('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(filtered,null,2)],{type:'application/json'}));a.download='mbsa-filtered-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();
</script></html>"""
    gate_rows = []
    for backend, key in (
        ("MBSA", "mbsa_precision_gates"),
        ("原生 vLLM", "native_precision_gates"),
    ):
        for dtype, gate in summary[key].items():
            lse_text = (
                str(gate["lse_passed_records"]) if backend == "MBSA" else "不提供"
            )
            gate_rows.append(
                f"<tr><td>{backend}</td><td>{dtype}</td><td>{gate['checked_records']} / {gate['records']}</td><td>{gate['output_passed_records']}</td><td>{lse_text}</td><td>{gate['route_passed_records']}</td><td>{gate['strict_passed_records']}</td></tr>"
            )
    conclusion = f"MBSA 严格 HF gate：{status_mbsa['passed']:,} 条通过、{status_mbsa['failed']:,} 条失败。原生对比：{status_native['passed']:,} 条通过、{status_native['failed']:,} 条失败、{status_native['error'] + status_native['oom']:,} 条执行异常。"
    if status_mbsa["failed"]:
        conclusion += (
            " 当前 MBSA 尚未通过全部严格 HF 校验；路由与数值失败项保留在明细中。"
        )
    gates = summary["mbsa_precision_gates"]
    diagnostic = summary["route_diagnostics"]
    perf_reference = next(
        (
            r
            for r in summary["speed_ratios"]
            if (r["dtype"], r["execution"], r["l2"], r["native_backend"])
            == ("bf16", "prepared_graph", "warm", "native_unified")
            and r["both_passed_ratio_median"] is not None
        ),
        None,
    )
    substitutions = {
        "__SCOPE_DIFFERENCES__": (
            "本地 SM90/H100 与独立 Sparse API 有后续更新，差异单独列在源码审计中；这些路径未被本次 B300 算子调用，后续改动保留在工作区，本次 commit 使用已验证快照："
            + "、".join(
                html.escape(p)
                for p in list(summary.get("runtime_source_audit", {}).get(
                    "out_of_scope_local_sm90_changes", {}
                )) + list(summary.get("runtime_source_audit", {}).get(
                    "out_of_scope_local_additions", {}
                ))
            )
            if summary.get("runtime_source_audit", {}).get("out_of_scope_local_sm90_changes")
            else "本地源码与执行快照的对应关系见源码审计。"
        ),
        "__ROUTE_ALL_MAPS__": "合并 identity 与 permuted 两种输入，各配置/page-map 仍只计一次："
        + "；".join(
            f"MBSA {dtype.upper()} 有 {r['mismatched_query_heads']:,} / "
            f"{r['query_head_count']:,} 个差异 query/head"
            f"（{r['mismatched_query_heads'] / max(r['query_head_count'], 1):.3e}）"
            for dtype, r in summary["mbsa_routes_all_page_maps"].items()
        )
        + "。原生执行异常项不计入已校验分母；分项统计见汇总 JSON。",
        "__GATE_FINDING__": (
            f"BF16/FP8 分别有 {gates['bf16']['failed_records_without_route_difference']} / "
            f"{gates['fp8']['failed_records_without_route_difference']} 条记录在路由一致时仍因数值门限失败。"
            f"LSE 失败涉及 {len(gates['bf16']['lse_failed_config_ids'])} 个配置，"
            + (
                f"失败配置：{', '.join(gates['bf16']['lse_failed_config_ids'])}。"
                if gates["bf16"]["lse_failed_config_ids"]
                else "全部已执行 BF16 配置的 LSE 门限通过。"
            )
        ),
        "__PERF_FINDING__": (
            "BF16、prepared graph、warm L2、原生 unified、"
            f"双方严格精度通过的 {perf_reference['both_precision_passed_pairs']} 个同卡配对中，"
            f"原生/MBSA 时间比中位数为 {perf_reference['both_passed_ratio_median']:.3f}；"
            f"即 MBSA 耗时约为原生的 {1 / perf_reference['both_passed_ratio_median']:.3f} 倍。"
            "具体 KV 与 batch 档位的性能差异见下方曲线。"
            if perf_reference
            else "当前样本不足以给出双方精度通过的同卡性能结论。"
        ),
        "__DIAGNOSTIC_FINDING__": (
            f"重放 {diagnostic['config_map_pairs']} 个配置/page-map 对，诊断 "
            f"{diagnostic['scored_mismatched_query_heads']} 个差异 query/head；"
            f"其中 {diagnostic['HF_fp32_exact_score_sum_ties']} 个 HF FP32 分数和严格相等，"
            f"全部样本的分数损失绝对值不超过 {diagnostic['max_abs_selected_score_loss_fp32']:.3e}。"
            f"固定 MBSA route 后，{diagnostic['forced_HF_attention_passed']} 个样本的 Attention 输出通过。"
            "这支持将这些样本的输出差异定位到近似等分处的选择；诊断只覆盖所列差异样本，不代表其它输出或 LSE 失败已解决。"
        ),
        "__UNIT_TESTS__": f'空 GPU 上另有 {summary["gpu_unit_tests"]["tests"]} 项单元测试通过、skip={summary["gpu_unit_tests"]["skipped"]}（{summary["gpu_unit_tests"]["elapsed_seconds"]} 秒，<a href="unit-tests.log">日志</a>）'
        if summary["gpu_unit_tests"]["status"] == "passed"
        else "单元测试日志尚未附带或未通过",
        "__NATIVE_FAULTS__": html.escape(
            json.dumps(
                {
                    "context_faults": summary["native_context_faults"],
                    "fresh_process_replay": summary["native_fresh_process_replay"],
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
        "__ROUTER_FINDING__": html.escape(
            summary.get("runtime_source_audit", {}).get(
                "implementation_note", "Router 发射与精修行为以本次源码审计为准。"
            )
        ),
        "__CONCLUSION__": conclusion,
        "__GATES__": "".join(gate_rows),
        "__DATE__": datetime.fromisoformat(summary["generated_at"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "__CASES__": str(len(cases)),
        "__GEOMETRIES__": str(summary["unique_q_kv_geometries"]),
        "__ACTUAL__": str(len(mbsa)),
        "__PLANNED__": str(len(cases) * 48),
        "__STATUS__": f"通过 {status_mbsa['passed']:,} · 失败 {status_mbsa['failed']:,}",
        "__NATIVE_ACTUAL__": str(len(native)),
        "__NATIVE_PLANNED__": str(summary["planned_native_records"]),
        "__NATIVE_STATUS__": f"通过 {status_native['passed']:,} · 失败 {status_native['failed']:,} · 异常 {status_native['error'] + status_native['oom']:,}",
        "__COMPLETENESS__": "选定算子矩阵已完成，原生对比已完成"
        if mbsa_complete and native_complete
        else "测试尚未全部完成；实际记录数与计划数的差额不可计为通过",
        "__ROUTES__": "".join(route_rows),
        "__HISTORICAL__": "".join(historical_rows),
        "__FIGURE__": figure,
        "__DIAGNOSTICS__": html.escape(
            json.dumps(summary["route_diagnostics"], ensure_ascii=False, indent=2)
        ),
        "__PERF__": "".join(perf_rows),
        "__SUMMARY__": html.escape(json.dumps(summary, ensure_ascii=False, indent=2)),
        "__ARTIFACTS__": "".join(
            f'<p><a href="{a["name"]}">{a["name"]}</a>（{a["bytes"]:,} bytes）</p>'
            for a in artifacts
        ),
        "__DATA__": payload,
        "__SHAPES__": shape_data,
    }
    if not diagnostics:
        doc = re.sub(
            r"<section><h3>MBSA 严格路由失败的分数与算术诊断</h3>.*?</section>",
            "",
            doc,
            flags=re.S,
        )
    if not historical_path.exists():
        doc = re.sub(
            r"<h2>此前真实层完整 chunk 的路由差异</h2><section>.*?</section>",
            "",
            doc,
            flags=re.S,
        )
    if not replay:
        doc = doc.replace(
            '<a href="native-fresh-process-replay.jsonl">独立重放的完整记录</a> · ', ""
        )
    for key, value in substitutions.items():
        doc = doc.replace(key, value)
    args.output.write_text(doc)
    print(
        json.dumps(
            {
                "html": str(args.output),
                "mbsa": dict(status_mbsa),
                "native": dict(status_native),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
