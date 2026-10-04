#!/usr/bin/env python3
"""Prefill performance matrix: same shared shapes, full precision gates and samples.

MBSA and native vLLM are separate sequential runs on the same empty GPU.
CUDA context faults remain errors; the parent restarts only unfinished work.
"""

import argparse
from collections import Counter
import gc
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def monitor_command(*command):
    return subprocess.check_output(command, text=True).strip()


def monitor_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-gpu", dest="gpu", type=int, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    container_id = monitor_command(
        "docker", "inspect", "--format", "{{.Id}}", args.container
    )
    uuid = monitor_command(
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"
    )
    while not args.stop_file.exists():
        record = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "gpu": args.gpu,
            "uuid": uuid,
        }
        try:
            record["gpu_fields"] = (
                "memory.used,memory.free,utilization.gpu,clocks.sm,temperature.gpu,power.draw"
            )
            record["gpu_values"] = monitor_command(
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=" + record["gpu_fields"],
                "--format=csv,noheader,nounits",
            )
            processes = []
            for row in monitor_command(
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ).splitlines():
                fields = [x.strip() for x in row.split(",")]
                if fields[0] != uuid:
                    continue
                try:
                    cgroup = Path(f"/proc/{fields[1]}/cgroup").read_text()
                    owned = container_id in cgroup
                except FileNotFoundError:
                    owned = None  # A process may exit between the two reads.
                processes.append(
                    {
                        "pid": fields[1],
                        "name": fields[2],
                        "memory_mib": fields[3],
                        "test_container": owned,
                    }
                )
            record["processes"] = processes
            record["foreign_process_count"] = sum(
                p["test_container"] is False for p in processes
            )
        except Exception as exc:
            record["error"] = str(exc)
        with args.output.open("a") as f:
            f.write(json.dumps(record) + "\n")
        time.sleep(30)


# Host monitoring uses only the standard library, before Torch imports.
if "--monitor-gpu" in sys.argv:
    monitor_cli()
    raise SystemExit(0)


import torch
from mbsa import prepare_cutedsl_pipeline
from test_precision import (
    CONFIG_PATH,
    configuration,
    select_cases,
    make_inputs,
    hf_class,
    oracle_checks,
    compare,
    route_metrics,
    append,
)


def measure(functions, flush, args):
    for _ in range(args.warmups):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()
    result = {}
    for l2 in args.config_data["options"]["l2"]:
        values = {
            name: {"gpu_samples_ms": [], "wall_samples_ms": []} for name in functions
        }
        for sample in range(args.samples):
            items = list(functions.items())
            if sample % 2:
                items.reverse()
            for name, fn in items:
                torch.cuda.synchronize()
                if l2 == "cold":
                    flush.add_(1)
                    torch.cuda.synchronize()
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                started = time.perf_counter()
                begin.record()
                fn()
                end.record()
                end.synchronize()
                values[name]["wall_samples_ms"].append(
                    (time.perf_counter() - started) * 1000
                )
                values[name]["gpu_samples_ms"].append(begin.elapsed_time(end))
        for name, samples in values.items():
            samples["gpu_median_ms"] = statistics.median(samples["gpu_samples_ms"])
            samples["wall_median_ms"] = statistics.median(samples["wall_samples_ms"])
            samples["gpu_p95_ms"] = sorted(samples["gpu_samples_ms"])[
                int((args.samples - 1) * 0.95)
            ]
            result[name, l2] = samples
    return result


def native_modules():
    for stem, prefix in (
        ("maxwell_moe", "MaxwellMoe"),
        ("maxwell_v1", "MaxwellV1"),
        ("compass", "Compass"),
    ):
        name = f"vllm.model_executor.layers.{stem}_bsa"
        if importlib.util.find_spec(name) is not None:
            attention = importlib.import_module(name)
            indexer = importlib.import_module(name + "_indexer")
            return (
                attention,
                getattr(attention, prefix + "BsaSparseGqa"),
                getattr(indexer, prefix + "BsaIndexer"),
            )
    raise RuntimeError("native sparse Attention not found")


def native_compare(out, route, probes, tolerances):
    rows = torch.cat([p["rows"] for p in probes])
    expected = torch.cat([p["out"] for p in probes])
    diff = (out[rows].float() - expected.float()).abs()
    normalized = diff / (
        tolerances["output_atol"] + tolerances["output_rtol"] * expected.float().abs()
    )
    routes = route_metrics(route[:, rows], torch.cat([p["route"] for p in probes], 1))
    return {
        "checked_queries": rows.numel(),
        "output_max_abs": float(diff.max()),
        "output_rms": float(diff.square().mean().sqrt()),
        "output_max_tolerance_ratio": float(normalized.max()),
        "output_close_fraction": float((normalized <= 1).float().mean()),
        "output_passed": bool((normalized <= 1).all()),
        "route": routes,
        "status": "passed"
        if bool((normalized <= 1).all()) and routes["mismatched_query_heads"] == 0
        else "failed",
    }


def remote_from_complete(route, lengths):
    positions = torch.cat(
        [torch.arange(kn - qn, kn, device="cuda") for qn, kn in lengths]
    )
    local_start = (positions // 128 - 7).clamp(min=8)
    valid = (route >= 8) & (route < local_start[None, :, None])
    sorted_remote = torch.where(valid, route, 2**30).sort(-1).values[..., :16]
    return torch.where(sorted_remote == 2**30, -1, sorted_remote)


def score_gap(case, data, probes, route):
    """FP32 max-dot score gap for mismatched sampled routes only."""
    losses, missing_scores, extra_scores = [], [], []
    for p in probes:
        i = p["request"]
        kn = case["requests"][i]["kv"]
        ids = data["table"][i, : data["counts"][i]].long()
        ik = data["ik"].index_select(0, ids).reshape(-1, 128)[:kn]
        # Routing counts are over all Q; score-gap diagnostics use at most
        # eight mismatched queries per request to bound the FP32 temporary.
        actual_all = route[:, p["rows"]]
        mismatch_queries = (
            (actual_all.sort(-1).values != p["route"].sort(-1).values)
            .any(-1)
            .any(0)
            .nonzero()
            .flatten()
        )
        if not mismatch_queries.numel():
            continue
        chosen = mismatch_queries[
            torch.linspace(
                0,
                mismatch_queries.numel() - 1,
                min(8, mismatch_queries.numel()),
                device="cuda",
            ).long()
        ]
        iq = data["iq"][p["rows"][chosen]]
        # Only store block scores, bounded to a handful of probe queries.
        scores = torch.einsum("qgd,kd->gqk", iq.float(), ik.float()) * 128**-0.5
        scores = torch.nn.functional.pad(scores, (0, (-kn) % 128), value=-torch.inf)
        blocks = scores.reshape(4, iq.shape[0], -1, 128).amax(-1)
        actual = actual_all[:, chosen].cpu()
        expected = p["route"][:, chosen].cpu()
        block_cpu = blocks.cpu()
        for g, q in (
            (actual.sort(-1).values != expected.sort(-1).values)
            .any(-1)
            .nonzero()
            .tolist()
        ):
            a = set(actual[g, q].tolist()) - {-1}
            e = set(expected[g, q].tolist()) - {-1}
            missed, added = e - a, a - e
            ms = [
                float(block_cpu[g, q, b]) for b in missed if 0 <= b < blocks.shape[-1]
            ]
            es = [float(block_cpu[g, q, b]) for b in added if 0 <= b < blocks.shape[-1]]
            missing_scores.extend(ms)
            extra_scores.extend(es)
            losses.append(sum(ms) - sum(es))
    return {
        "mismatched_query_heads_scored": len(losses),
        "sum_selected_score_loss": sum(losses),
        "max_selected_score_loss": max(losses, default=0),
        "mean_selected_score_loss": sum(losses) / max(len(losses), 1),
        "missing_block_score_mean": sum(missing_scores) / max(len(missing_scores), 1),
        "extra_block_score_mean": sum(extra_scores) / max(len(extra_scores), 1),
    }


def mbsa_matrix(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cases = chosen_cases(args)
    options = list(
        itertools.product(
            *(
                args.config_data["options"][key]
                for key in ("partial_dtype", "page_mapping", "overlap")
            )
        )
    )
    (args.output / "selected-cases.json").write_text(json.dumps(cases, indent=2))
    if args.selection_only:
        print(json.dumps({"cases": len(cases), "option_results": len(cases) * 48}))
        return
    hf = hf_class(args.hf_source)
    if args.save_oracles:
        args.save_oracles.mkdir(parents=True, exist_ok=True)
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 3)):
        raise RuntimeError("requires SM90 or SM103")
    environment = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "gpu_capability": torch.cuda.get_device_capability(),
        "initial_memory_free_total_bytes": torch.cuda.mem_get_info(),
        "hf_source_sha256": hashlib.sha256(args.hf_source.read_bytes()).hexdigest(),
        "hf_revision": "d97a6c52967a5ed332e2196a5b985f0b4a475549",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "python/mbsa").rglob("*.py"))
        },
        "samples": args.samples,
        "warmups": args.warmups,
        "cold_l2_flush_mib": args.config_data["timing"]["cold_l2_flush_mib"],
        "oracle_scope": "all queries of every selected shape",
        "measurement_scope": "prefill_only_v2",
        "selected_cases": len(cases),
        "planned_operator_results": len(cases) * 48,
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    path = args.output / "results.jsonl"
    finished = set()
    if args.resume and path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            finished.add(
                (
                    r["config_id"],
                    r["partial_dtype"],
                    r["page_mapping"],
                    r["overlap"],
                    r["execution"],
                    r["l2"],
                )
            )
    elif path.exists():
        raise FileExistsError("use --resume or a new output directory")
    flush = torch.empty(
        args.config_data["timing"]["cold_l2_flush_mib"] * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    for number, case in enumerate(cases, 1):
        if all(
            (case["config_id"], dtype, mapping, overlap, execution, l2) in finished
            for dtype, mapping, overlap in options
            for execution, l2 in itertools.product(
                args.config_data["options"]["execution"],
                args.config_data["options"]["l2"],
            )
        ):
            print(
                json.dumps(
                    {
                        "case": number,
                        "of": len(cases),
                        "id": case["config_id"],
                        "resume_skipped": True,
                    }
                ),
                flush=True,
            )
            continue
        case_started = time.perf_counter()
        for mapping in args.config_data["options"]["page_mapping"]:
            data = probes = None
            allocation_error = None
            try:
                data = make_inputs(case, mapping)
                probes = load_oracle(case, mapping, data, hf, args)
            except Exception as exc:
                allocation_error = {
                    "status": "oom"
                    if isinstance(exc, torch.OutOfMemoryError)
                    else "error",
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                }
            for dtype, page_mapping, overlap in options:
                if page_mapping != mapping:
                    continue
                base = {
                    "config_id": case["config_id"],
                    "family": case["family"],
                    "partial_dtype": dtype,
                    "page_mapping": mapping,
                    "overlap": overlap,
                    "q_lens": [r["q"] for r in case["requests"]],
                    "kv_lens": [r["kv"] for r in case["requests"]],
                    "seed": case.get("seed", 17),
                }
                keys = [
                    (case["config_id"], dtype, mapping, overlap, ex, l2)
                    for ex in args.config_data["options"]["execution"]
                    for l2 in args.config_data["options"]["l2"]
                ]
                if all(key in finished for key in keys):
                    continue
                pipeline = None
                try:
                    if allocation_error is not None:
                        raise RuntimeError(allocation_error["exception"])
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    pipeline = prepare_cutedsl_pipeline(
                        data["iq"],
                        data["ik"],
                        data["q"],
                        data["pool"][..., 0, :].permute(0, 2, 1, 3),
                        data["pool"][..., 1, :].permute(0, 2, 1, 3),
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
                    first = pipeline.run()
                    torch.cuda.synchronize()
                    prepare_ms = (time.perf_counter() - started) * 1000
                    precision = compare(first, probes, args.config_data["tolerances"])
                    started = time.perf_counter()
                    pipeline.plan.capture_mega_graph()
                    pipeline.plan.run_mega()
                    torch.cuda.synchronize()
                    capture_ms = (time.perf_counter() - started) * 1000
                    graph_precision = compare(
                        (
                            pipeline.plan.attention.out,
                            pipeline.plan.attention.out_lse,
                            pipeline.plan.attention.tile_plan.dynamic_route,
                        ),
                        probes,
                        args.config_data["tolerances"],
                    )
                    # Explicitly compare a replay after changing live query values.
                    original_q = data["q"].clone()
                    data["q"].add_(0.125)
                    pipeline.plan.run_mega()
                    graph_out = pipeline.plan.attention.out.clone()
                    eager_out = pipeline.run()[0]
                    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)
                    data["q"].copy_(original_q)
                    del original_q, graph_out, eager_out
                    times = measure(
                        {
                            "eager": pipeline.run,
                            "prepared_graph": pipeline.plan.run_mega,
                        },
                        flush,
                        args,
                    )
                    memory = {
                        "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                        "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
                        "free_total_bytes": torch.cuda.mem_get_info(),
                    }
                    for (execution, l2), timing in times.items():
                        check = precision if execution == "eager" else graph_precision
                        save_record(
                            path,
                            {
                                **base,
                                "execution": execution,
                                "l2": l2,
                                "status": check["status"],
                                "precision": check,
                                "timing": timing,
                                "memory": memory,
                                "prepare_first_run_ms": prepare_ms,
                                "capture_ms": capture_ms,
                                "live_query_graph_replay": "passed",
                            },
                        )
                except Exception as exc:
                    failure = allocation_error or {
                        "status": "oom"
                        if isinstance(exc, torch.OutOfMemoryError)
                        else "error",
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    for execution, l2 in itertools.product(
                        args.config_data["options"]["execution"],
                        args.config_data["options"]["l2"],
                    ):
                        save_record(
                            path, {**base, "execution": execution, "l2": l2, **failure}
                        )
                finally:
                    del pipeline
                    gc.collect()
                    torch.cuda.empty_cache()
            del data, probes
            gc.collect()
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "case": number,
                    "of": len(cases),
                    "id": case["config_id"],
                    "elapsed_s": time.perf_counter() - case_started,
                }
            ),
            flush=True,
        )
    (args.output / "complete.json").write_text(
        json.dumps({"cases": len(cases), "status": "complete"})
    )


def native_matrix(args):
    os.environ["MAXWELL_BSA_PREFILL_FP8_PARTIALS"] = (
        "0" if args.partial_dtype == "bf16" else "1"
    )
    os.environ["MBSA_VLLM_PREFILL"] = "0"
    args.output.mkdir(parents=True, exist_ok=True)
    cases = chosen_cases(args)
    hf = hf_class(args.hf_source)
    native, attention_cls, indexer_cls = native_modules()
    (args.output / f"environment-{args.partial_dtype}.json").write_text(
        json.dumps(
            {
                "module": native.__name__,
                "path": native.__file__,
                "source_sha256": hashlib.sha256(
                    Path(native.__file__).read_bytes()
                ).hexdigest(),
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "hf_sha256": hashlib.sha256(args.hf_source.read_bytes()).hexdigest(),
                "dtype": args.partial_dtype,
                "cases": len(cases),
            },
            indent=2,
        )
    )
    native.configure_unified_prefill_capacity(
        max_num_seqs=32, max_model_len=262144, block_size=128
    )
    native.prepare_unified_prefill_storage(
        torch.device("cuda", 0), max_num_batched_tokens=8192
    )
    flush = torch.empty(
        args.config_data["timing"]["cold_l2_flush_mib"] * 1024 * 1024,
        device="cuda",
        dtype=torch.uint8,
    )
    path = args.output / f"native-{args.partial_dtype}.jsonl"
    finished = set()
    if args.resume and path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            finished.add((r["config_id"], r["page_mapping"], r["execution"], r["l2"]))
    elif path.exists():
        raise FileExistsError("use --resume or a new output directory")
    for number, case in enumerate(cases, 1):
        started = time.perf_counter()
        for mapping in args.config_data["options"]["page_mapping"]:
            if all(
                (case["config_id"], mapping, execution, l2) in finished
                for execution, l2 in itertools.product(
                    args.config_data["options"]["execution"],
                    args.config_data["options"]["l2"],
                )
            ):
                continue
            base = {
                "config_id": case["config_id"],
                "family": case["family"],
                "page_mapping": mapping,
                "partial_dtype": args.partial_dtype,
            }
            data = probes = indexer = attention = plan = graph = common = functions = (
                out
            ) = route = None
            try:
                data = make_inputs(case, mapping)
                # Native varlen plans bucket logical KV extent in 128-page
                # groups. Match that metadata contract, retaining valid pages.
                table_width = ((data["table"].shape[1] + 127) // 128) * 128
                if data["table"].shape[1] != table_width:
                    data["table"] = torch.nn.functional.pad(
                        data["table"], (0, table_width - data["table"].shape[1])
                    )
                probes = load_oracle(case, mapping, data, hf, args)
                lengths = [(r["q"], r["kv"]) for r in case["requests"]]
                indexer = indexer_cls(
                    num_index_heads=4,
                    num_kv_heads=4,
                    head_dim=128,
                    block_size=128,
                    shared_route=False,
                    score_reduce="max",
                    init_blocks=8,
                    local_blocks=8,
                )
                indexer._k_cache = data["ik"]
                attention = attention_cls(
                    num_qo_heads=32,
                    num_kv_heads=4,
                    head_dim=128,
                    block_size=128,
                    topk=32,
                    shared_route=False,
                )
                plan = indexer.build_plan(
                    qo_lens=torch.tensor([qn for qn, _ in lengths], dtype=torch.int32),
                    kv_lens=torch.tensor([kn for _, kn in lengths], dtype=torch.int32),
                    qo_offset=torch.tensor(
                        [kn - qn for qn, kn in lengths], dtype=torch.int32
                    ),
                )
                out = torch.empty_like(data["q"])
                common = dict(
                    index_q=data["iq"],
                    index_k_cache=indexer.decode_k_cache(),
                    index_plan=plan,
                    index_page_indices=data["pages"],
                    page_table=data["table"],
                    seq_lens=torch.tensor(
                        [kn for _, kn in lengths], dtype=torch.int32, device="cuda"
                    ),
                    request_lens=lengths,
                    cu_seqlens_q=data["cu_q"],
                    cu_seqlens_k=data["cu_k"],
                    prefix_lens=torch.tensor(
                        [kn - qn for qn, kn in lengths],
                        dtype=torch.int32,
                        device="cuda",
                    ),
                    out=out,
                )

                def unified():
                    return attention.prefill_unified(
                        data["q"],
                        data["pool"][..., 0, :],
                        data["pool"][..., 1, :],
                        **common,
                    )

                first = unified()
                if first is None:
                    base["backend"] = "legacy_sparse_gqa_fallback"
                    base["unified_status"] = "unsupported_index_plan_no_tileq256_owner"
                    visible = torch.cat(
                        [
                            torch.arange(kn - qn, kn, device="cuda", dtype=torch.int32)
                            // 128
                            + 1
                            for qn, kn in lengths
                        ]
                    )

                    def legacy():
                        score = indexer.max_scores(
                            data["iq"], plan=plan, kv_indices=data["pages"]
                        )
                        selection = indexer.select(
                            score,
                            num_valid_pages=max(data["counts"]),
                            topk=32,
                            per_query_valid=visible,
                        )
                        attention.prefill(
                            data["q"],
                            data["pool"][..., 0, :],
                            data["pool"][..., 1, :],
                            selection,
                            cu_seqlens_q=data["cu_q"],
                            cu_seqlens_k=data["cu_k"],
                            seq_lens=common["seq_lens"],
                            page_table=data["table"],
                            max_seqlen_q=max(qn for qn, kn in lengths),
                            max_seqlen_k=max(kn for qn, kn in lengths),
                            total_rows=sum(data["counts"]),
                            total_k=sum(kn for qn, kn in lengths),
                            softmax_scale=128**-0.5,
                            out=out,
                        )
                        return selection

                    selection = legacy()
                    route = remote_from_complete(selection, lengths)
                    fn = legacy
                else:
                    base["backend"] = "native_unified"
                    base["unified_status"] = "supported"
                    route = native.unified_prefill_dynamic_route(
                        native._UNIFIED_PREFILL_STORAGE[0],
                        sum(qn for qn, kn in lengths),
                    ).clone()
                    fn = unified
                precision = native_compare(
                    out, route, probes, args.config_data["tolerances"]
                )
                gaps = score_gap(case, data, probes, route)
                functions = {"eager": fn}
                graph_error = None
                try:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            fn()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured_return = fn()
                    graph.replay()
                    torch.cuda.synchronize()
                    if base["backend"] == "native_unified":
                        graph_route = native.unified_prefill_dynamic_route(
                            native._UNIFIED_PREFILL_STORAGE[0],
                            sum(qn for qn, kn in lengths),
                        )
                    else:
                        graph_route = remote_from_complete(captured_return, lengths)
                    graph_precision = native_compare(
                        out, graph_route, probes, args.config_data["tolerances"]
                    )
                    functions["prepared_graph"] = graph.replay
                except Exception as exc:
                    graph_error = str(exc)
                timings = measure(functions, flush, args)
                for (execution, l2), timing in timings.items():
                    check = precision if execution == "eager" else graph_precision
                    save_record(
                        path,
                        {
                            **base,
                            "execution": execution,
                            "l2": l2,
                            "status": check["status"],
                            "precision": check,
                            "score_gap": gaps,
                            "timing": timing,
                        },
                    )
                if graph_error:
                    for l2 in args.config_data["options"]["l2"]:
                        save_record(
                            path,
                            {
                                **base,
                                "execution": "prepared_graph",
                                "l2": l2,
                                "status": "unsupported_graph",
                                "exception": graph_error,
                            },
                        )
            except Exception as exc:
                for execution, l2 in itertools.product(
                    args.config_data["options"]["execution"],
                    args.config_data["options"]["l2"],
                ):
                    save_record(
                        path,
                        {
                            **base,
                            "execution": execution,
                            "l2": l2,
                            "status": "oom"
                            if isinstance(exc, torch.OutOfMemoryError)
                            else "error",
                            "exception": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
            finally:
                native._UNIFIED_PREFILL_STATE.clear()
                data = probes = indexer = attention = plan = graph = common = (
                    functions
                ) = out = route = None
                gc.collect()
                torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "case": number,
                    "of": len(cases),
                    "id": case["config_id"],
                    "elapsed_s": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    (args.output / f"complete-{args.partial_dtype}.json").write_text(
        json.dumps({"cases": len(cases), "status": "complete"})
    )


_WRITTEN = {}


def save_record(path, record):
    def key(r):
        return tuple(
            r.get(k)
            for k in (
                "config_id",
                "partial_dtype",
                "page_mapping",
                "overlap",
                "execution",
                "l2",
            )
        )

    if path not in _WRITTEN:
        _WRITTEN[path] = (
            {key(json.loads(line)) for line in path.read_text().splitlines()}
            if path.exists()
            else set()
        )
    identity = key(record)
    if identity not in _WRITTEN[path]:
        append(path, record)
        _WRITTEN[path].add(identity)


def chosen_cases(args):
    cases = select_cases(args.config_data, args.case_id)
    cases = cases[args.shard_index :: args.shards]
    return cases[: args.limit] if args.limit else cases


def load_oracle(case, mapping, data, hf, args):
    # Cached results are valid only for identical generated inputs and the
    # pinned upstream source. Legacy bare lists are recomputed once.
    identity = {
        "schema": 1,
        "requests": case["requests"],
        "seed": case.get("seed", 17),
        "page_mapping": mapping,
        "hf_source_sha256": args.config_data["hf"]["source_sha256"],
        "torch_version": str(torch.__version__),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "generator": "make_inputs:BF16-Q32-IQ4-pool4-K1-D128-v1",
    }
    name = f"{case['config_id']}-{mapping}.pt"
    path = args.oracle_cache / name if args.oracle_cache else None
    if path and path.exists():
        cached = torch.load(path, map_location="cuda", weights_only=True)
        if isinstance(cached, dict) and cached.get("identity") == identity:
            return cached["probes"]
        del cached
    probes = oracle_checks(case, data, hf)
    if args.save_oracles:
        args.save_oracles.mkdir(parents=True, exist_ok=True)
        cpu_probes = [
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in probe.items()}
            for probe in probes
        ]
        torch.save(
            {"identity": identity, "probes": cpu_probes}, args.save_oracles / name
        )
    return probes


def completed_keys(path, native):
    if not path.exists():
        return set()
    keys = ("config_id", "partial_dtype", "page_mapping", "execution", "l2")
    if not native:
        keys += ("overlap",)
    return {
        tuple(r[k] for k in keys)
        for r in (json.loads(l) for l in path.read_text().splitlines())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--backend", choices=("mbsa", "native"), default="mbsa")
    parser.add_argument("--hf-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partial-dtype",
        choices=("bf16", "fp8"),
        default="bf16",
        help="native requested partial dtype",
    )
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--oracle-cache", type=Path)
    parser.add_argument("--save-oracles", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.config_data = configuration(args.config)
    args.samples = args.config_data["timing"]["samples"]
    args.warmups = args.config_data["timing"]["warmups"]
    if not 0 <= args.shard_index < args.shards:
        parser.error("shard-index must lie in [0,shards)")
    if (
        hashlib.sha256(args.hf_source.read_bytes()).hexdigest()
        != args.config_data["hf"]["source_sha256"]
    ):
        parser.error("HF source differs from pinned config")
    if args.selection_only:
        print(
            json.dumps(
                {
                    "cases": len(chosen_cases(args)),
                    "results": len(chosen_cases(args))
                    * (8 if args.backend == "native" else 48),
                }
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        (native_matrix if args.backend == "native" else mbsa_matrix)(args)
        return
    native = args.backend == "native"
    contract = {
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "hf_source_sha256": args.config_data["hf"]["source_sha256"],
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "precision_test_sha256": hashlib.sha256(
            (ROOT / "tests/test_precision.py").read_bytes()
        ).hexdigest(),
        "runtime_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "python/mbsa").rglob("*.py"))
            if not p.name.startswith("._")
        },
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch_version": str(torch.__version__),
        "case_ids": [c["config_id"] for c in chosen_cases(args)],
    }
    contract_path = args.output / "run-contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "resume source/configuration/GPU differs; use a fresh output directory"
            )
    else:
        contract_path.write_text(json.dumps(contract, indent=2))
    path = args.output / (
        f"native-{args.partial_dtype}.jsonl" if native else "results.jsonl"
    )
    dimensions = [
        args.config_data["options"][k] for k in ("page_mapping", "execution", "l2")
    ]
    if not native:
        dimensions += [args.config_data["options"]["overlap"]]
    dtypes = (
        [args.partial_dtype] if native else args.config_data["options"]["partial_dtype"]
    )
    expected = {
        (c["config_id"], d, *options)
        for c in chosen_cases(args)
        for d in dtypes
        for options in itertools.product(*dimensions)
    }
    if path.exists() and not args.resume:
        raise FileExistsError("use --resume or a fresh output directory")
    history = []
    while expected - completed_keys(path, native):
        before = completed_keys(path, native)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker",
            "--resume",
        ]
        log = (
            args.output
            / f"worker-{args.backend}-{args.partial_dtype}-{len(history) + 1:03d}.log"
        )
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        after = completed_keys(path, native)
        history.append(
            {
                "exit_code": result.returncode,
                "before": len(expected & before),
                "after": len(expected & after),
                "planned": len(expected),
                "log": log.name,
            }
        )
        (args.output / f"recovery-{args.backend}-{args.partial_dtype}.json").write_text(
            json.dumps(history, indent=2)
        )
        print(json.dumps(history[-1]), flush=True)
        if not expected & (after - before):
            raise RuntimeError(f"worker made no progress; inspect {log}")
    status = Counter(
        r["status"] for r in (json.loads(l) for l in path.read_text().splitlines())
    )
    print(
        json.dumps({"records": len(completed_keys(path, native)), "status": status}),
        flush=True,
    )
    if any(status[s] for s in ("failed", "error", "oom", "unsupported_graph")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
