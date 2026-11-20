#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the MBSA prefill pipeline and FA4 dense GQA.
Sparse totals sum chunk medians.
Preparation, compilation, projections and cache writes are excluded.
FP8 KV: default uses native FP8 QK/PV with timed Q conversion; optimized
decodes FP8 K/V to BF16 each run. Dense baselines use the corresponding
represented operands decoded to BF16, not native FP8 baseline kernels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('default', 'optimized'), default='default',
                        help='explicit MBSA backend; default selection is unchanged')
    parser.add_argument('--lean192', choices=('auto', 'on', 'off'), default='auto',
                        help='optimized BF16 Index policy; actual per-chunk selection is recorded')
    parser.add_argument("--lengths", default="32768,65536,131072,262144,524288,1048576")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--dense-samples", type=int, default=3)
    parser.add_argument(
        "--dense-backend", choices=("flash-attn", "sdpa"), default="flash-attn"
    )
    parser.add_argument("--l2", choices=("cold", "warm"), default="cold")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--kv-dtype", choices=("bf16", "fp8"), default="bf16",
        help="KV precision; default fp8 casts Q and uses FP8 attention, optimized decodes K/V and uses BF16 attention",
    )
    parser.add_argument(
        "--index-dtype", choices=("bf16", "fp16", "fp8"), default="fp8",
        help="Index Q/K precision; fp8 uses E4M3FN with FP32 score accumulation",
    )
    parser.add_argument(
        "--extent", type=int,
        help="input generation length (default: longest measured length)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="plot an existing result JSON without running benchmarks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="result JSON, or figure path when using --input",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="figure path (default: output JSON path with .png suffix)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="save JSON without plotting")
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="measure dense GQA first; use --resume later to add sparse timings",
    )
    args = parser.parse_args()
    if args.lean192 != 'auto' and args.backend != 'optimized':
        parser.error('--lean192 on/off requires --backend optimized')
    if args.lean192 == 'on' and (args.index_dtype != 'bf16' or args.kv_dtype != 'bf16'):
        parser.error('--lean192 on requires BF16 Index and BF16 attention')
    if args.input is not None:
        if args.resume or args.dense_only or args.plot is not None or args.no_plot:
            parser.error(
                "--input cannot be combined with --resume, --dense-only, --plot or --no-plot"
            )
        if args.input.resolve() == args.output.resolve():
            parser.error("input JSON and output figure must have different paths")
        return args
    if args.no_plot and args.plot is not None:
        parser.error("--no-plot cannot be combined with --plot")
    if args.plot is None and not args.dense_only and not args.no_plot:
        args.plot = args.output.with_suffix(".png")
    if args.plot is not None:
        if args.dense_only:
            parser.error("--plot requires complete sparse and dense measurements")
        if args.plot.resolve() == args.output.resolve():
            parser.error("output JSON and figure must have different paths")
    try:
        args.lengths = sorted(set(int(n) for n in args.lengths.split(",")))
    except ValueError:
        parser.error("lengths must be comma-separated integers")
    if not 0 < args.chunk <= 8192 or args.chunk % 128:
        parser.error("chunk must be a multiple of 128 between 128 and 8192")
    if not args.lengths or any(
        n < args.chunk or n > 1048576 or n % args.chunk for n in args.lengths
    ):
        parser.error("lengths must be multiples of chunk, between chunk and 1M")
    if args.warmups < 0 or min(args.samples, args.dense_samples) < 1:
        parser.error("warmups must be nonnegative and sample counts positive")
    if args.extent is None:
        args.extent = max(args.lengths)
    if not max(args.lengths) <= args.extent <= 1048576:
        parser.error("extent must be between the longest measured length and 1M")
    if args.output.exists() and not args.resume:
        parser.error("output already exists; use --resume or a new output path")
    return args


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def measure(torch, fn, flush, warmups, samples):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        if flush is not None:
            flush.add_(1)
            # Drain only the cache-flush work before starting an eager sample;
            # otherwise its queued latency can hide CPU kernel-enqueue gaps.
            torch.cuda.current_stream().synchronize()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(values), "samples_ms": values}




def run(args):
    import torch
    from mbsa import prepare_cutedsl_pipeline
    from tests.reference import reference_pipeline
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError(f"SM100-family GPU required; got {capability}")
    flash = None
    fa4_sources = {}
    if args.dense_backend == "flash-attn":
        import flash_attn.cute as fa4
        from flash_attn.cute import flash_attn_func

        flash = flash_attn_func
        fa4_root = Path(fa4.__file__).parent
        fa4_sources = {
            str(p.relative_to(fa4_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(fa4_root.rglob("*.py"))
        }
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "python/mbsa").rglob("*"))
        if p.is_file()
        and p.suffix in (".py", ".cu", ".cuh")
        and not p.name.startswith("._")
    }
    contract = {
        "gpu": torch.cuda.get_device_name(),
        "backend": args.backend,
        "lean192_requested": args.lean192,
        "lean192_resolved_location": "chunks[].lean192_enabled",
        "measurement_scope": "sum_of_independently_measured_chunk_medians",
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "capability": list(capability),
        "torch": str(torch.__version__),
        "chunk": args.chunk,
        "lengths": args.lengths,
        "seed": args.seed,
        "warmups": args.warmups,
        "samples": args.samples,
        "dense_samples": args.dense_samples,
        "dense_backend": args.dense_backend,
        "dense_implementation": "FA4" if flash is not None else "PyTorch Flash SDPA",
        "fa4_source_sha256": fa4_sources,
        "l2": args.l2,
        "dtype": "bf16",
        "kv_dtype": args.kv_dtype,
        "output_dtype": "bf16",
        "attention_compute_dtype": 'bf16' if args.backend == 'optimized' else args.kv_dtype,
        "kv_quantization": "BF16 cast to E4M3FN; no scale; excluded from timing" if args.kv_dtype == "fp8" else "none",
        "query_quantization": "BF16 to E4M3FN inside each timed MBSA run" if args.kv_dtype == "fp8" and args.backend == 'default' else "none",
        "baseline_dtype": "bf16",
        "baseline_representation": ("represented Q/K/V decoded to BF16 before timing; BF16 attention baseline, not native FP8" if args.kv_dtype == "fp8" else "original BF16 Q/K/V"),
        "index_dtype": args.index_dtype,
        "score_dtype": "fp32",
        "index_quantization": "none" if args.index_dtype == "bf16" else f"BF16 cast to {args.index_dtype}; no scale; excluded from timing",
        "extent": args.extent,
        "partial_dtype": "bf16",
        "heads": {"q": 32, "kv": 4, "dim": 128},
        "source_sha256": sources,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result = {
        "schema": 1,
        "contract": contract,
        "status": "running",
        "chunks": [],
        "dense": [],
    }
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text())
        previous = result["contract"]
        # Dense GQA does not depend on MBSA kernels. A dense-only result may
        # be completed after MBSA changes, while all dense settings stay fixed.
        if not result["chunks"]:
            previous = {**previous, "source_sha256": sources}
        if previous != contract:
            raise ValueError("resume GPU, source, or benchmark settings differ")
        result["contract"] = contract
    result["status"] = "running"
    print(json.dumps({"backend": args.backend, "lean192_requested": args.lean192,
                      "measurement_scope": contract["measurement_scope"],
                      "l2": args.l2}), flush=True)
    save(args.output, result)
    longest = max(args.lengths)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    def rand(*shape):
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, generator=generator
        )

    q, iq = rand(args.extent, 32, 128), rand(args.extent, 4, 128)
    k, v, ik = rand(args.extent, 4, 128), rand(args.extent, 4, 128), rand(args.extent, 1, 128)
    if args.index_dtype != "bf16":
        index_dtype = torch.float16 if args.index_dtype == "fp16" else torch.float8_e4m3fn
        iq, ik = iq.to(index_dtype), ik.to(index_dtype)
    # Baselines receive the exact values represented by native FP8 attention,
    # decoded once; conversion is not part of baseline timing. Preserve BF16 Q
    # for MBSA so its per-run conversion remains part of the measured pipeline.
    if args.kv_dtype == "fp8":
        k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
        baseline_q = (q.to(torch.float8_e4m3fn).to(torch.bfloat16)
                      if args.backend == 'default' else q)
        baseline_k, baseline_v = k.to(torch.bfloat16), v.to(torch.bfloat16)
    else:
        baseline_q, baseline_k, baseline_v = q, k, v
    flush = (
        torch.zeros(256 << 20, device="cuda", dtype=torch.uint8)
        if args.l2 == "cold"
        else None
    )

    def dense(n):
        if flash is not None:
            out = flash(baseline_q[None, :n], baseline_k[None, :n], baseline_v[None, :n], causal=True)
            return out[0] if isinstance(out, tuple) else out
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                baseline_q[:n].transpose(0, 1)[None],
                baseline_k[:n].transpose(0, 1)[None],
                baseline_v[:n].transpose(0, 1)[None],
                is_causal=True,
                enable_gqa=True,
            ).transpose(1, 2)

    with torch.inference_mode():
        n = min(args.chunk, 128)
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.nn.functional.scaled_dot_product_attention(
                baseline_q[:n].transpose(0, 1)[None],
                baseline_k[:n].transpose(0, 1)[None],
                baseline_v[:n].transpose(0, 1)[None],
                is_causal=True,
                enable_gqa=True,
            ).transpose(1, 2)
        torch.testing.assert_close(
            dense(n).float(), expected.float(), atol=0.03, rtol=0.03
        )
        result["dense_check"] = "passed"
        stop = len(result["chunks"]) if args.dense_only else longest // args.chunk
        for chunk in range(len(result["chunks"]), stop):
            begin, end = chunk * args.chunk, (chunk + 1) * args.chunk
            pipeline = prepare_cutedsl_pipeline(
                iq[begin:end],
                ik[:end],
                q[begin:end],
                k[:end],
                v[:end],
                partial_dtype=torch.bfloat16,
                backend=args.backend,
                lean192=args.lean192,
            )
            output, lse, remote = pipeline.run()
            if pipeline.backend != args.backend:
                raise AssertionError('resolved backend differs from the requested benchmark backend')
            if output.dtype != torch.bfloat16:
                raise AssertionError("MBSA prefill benchmark requires BF16 output")
            entry = {"chunk": chunk, "kv": end, "lean192_requested": pipeline.lean192,
                     "lean192_enabled": pipeline.lean192_enabled}
            print(json.dumps({k: entry[k] for k in
                ("chunk", "kv", "lean192_requested", "lean192_enabled")}), flush=True)
            if chunk == 0 or end in args.lengths:
                probe = slice(args.chunk - 16, args.chunk)
                eo, el, er = reference_pipeline(
                    iq[end - 16 : end].float(),
                    ik[:end].float(),
                    baseline_q[end - 16 : end],
                    baseline_k[:end],
                    baseline_v[:end],
                    query_offset=end - 16,
                )
                torch.testing.assert_close(output[probe], eo, atol=0.03, rtol=0.03)
                torch.testing.assert_close(lse[probe], el, atol=0.01, rtol=0.002)
                torch.testing.assert_close(
                    remote[:, probe].sort(-1).values, er.sort(-1).values, atol=0, rtol=0
                )
                entry["precision_probe"] = "passed"
                entry["precision_probe_details"] = {
                    "query_rows": 16,
                    "index_oracle": "FP32 decoded Index Q/K",
                    "o_max_abs": (output[probe].float() - eo.float()).abs().max().item(),
                    "lse_max_abs": (lse[probe] - el).abs().max().item(),
                    "route_set": "exact",
                }
                del eo, el, er
            entry["mbsa"] = measure(
                torch, pipeline.run, flush, args.warmups, args.samples
            )
            result["chunks"].append(entry)
            save(args.output, result)
            print(
                json.dumps(
                    {
                        "chunk": chunk,
                        "kv": end,
                        "mbsa_ms": entry["mbsa"]["median_ms"],
                    }
                ),
                flush=True,
            )
            del pipeline, output, lse, remote
            gc.collect()
            torch.cuda.empty_cache()
        completed = {item["length"] for item in result["dense"]}
        for n in args.lengths:
            if n in completed:
                continue
            timing = measure(
                torch, lambda: dense(n), flush, args.warmups, args.dense_samples
            )
            result["dense"].append({"length": n, **timing})
            save(args.output, result)
            print(
                json.dumps({"dense_length": n, "median_ms": timing["median_ms"]}),
                flush=True,
            )
    if args.dense_only:
        result["status"] = "dense_complete"
        save(args.output, result)
        return
    dense_ms = {item["length"]: item["median_ms"] for item in result["dense"]}
    result["summary"] = []
    for n in args.lengths:
        chunks = result["chunks"][: n // args.chunk]
        mbsa = sum(item["mbsa"]["median_ms"] for item in chunks)
        result["summary"].append(
            {
                "length": n,
                "mbsa_ms": mbsa,
                "dense_gqa_ms": dense_ms[n],
                "speedup": dense_ms[n] / mbsa,
            }
        )
    result["status"] = "complete"
    save(args.output, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


def plot_results(input_path: Path, output_path: Path):
    """Plot a complete result JSON on CPU; import matplotlib only when needed."""
    result = json.loads(input_path.read_text())
    if result["status"] != "complete":
        raise ValueError("benchmark results are incomplete")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    rows = sorted(result["summary"], key=lambda row: row["length"])
    lengths = [row["length"] for row in rows]
    dense_backend = result["contract"].get(
        "dense_implementation", result["contract"]["dense_backend"]
    )
    index_dtype = result["contract"].get("index_dtype", "bf16")
    kv_dtype = result["contract"].get("kv_dtype", "bf16")
    baseline_suffix = " (represented BF16)" if kv_dtype == "fp8" else ""
    fig, ax = plt.subplots(figsize=(10, 6))
    curves = [
        ("dense_gqa_ms", "Dense GQA" + baseline_suffix, "#929292", "--"),
        ("mbsa_ms", "MBSA", "#f04b2d", "-"),
    ]
    for key, label, color, style in curves:
        ax.plot(
            lengths,
            [row[key] for row in rows],
            marker="o",
            color=color,
            linestyle=style,
            label=label,
        )
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: f"{x / 1024:.0f}K" if x < 1048576 else f"{x / 1048576:g}M"
        )
    )
    ax.set(
        xlabel="Sequence length (tokens)",
        ylabel="Execution time (ms)",
        title=f"Maxwell Block Sparse Attention — {result['contract']['gpu']}",
    )
    ax.grid(alpha=0.2)
    if max(lengths) >= 262144:
        ax.set_xticks([0, 262144, 524288, 786432, 1048576])
    ax.legend(loc="upper left")
    peak = max(row[key] for row in rows for key, *_ in curves)
    for row in rows:
        if row["length"] in (262144, 1048576):
            ax.annotate(
                f"{row['speedup']:.2f}× vs. GQA",
                (row["length"], row["mbsa_ms"]),
                xytext=(-15, -65 if row["mbsa_ms"] > 0.85 * peak else 18),
                textcoords="offset points",
                ha="right",
                color="#d93c24",
            )
    if len(rows) > 1:
        inset = ax.inset_axes([0.14, 0.48, 0.48, 0.30])
        for key, label, color, style in curves[1:]:
            inset.plot(
                lengths,
                [row[key] for row in rows],
                marker=".",
                color=color,
                label=label,
            )
        inset.set_title("MBSA prefill", fontsize=10)
        inset.xaxis.set_major_formatter(ax.xaxis.get_major_formatter())
        inset.tick_params(labelsize=8)
        inset.grid(alpha=0.2)
        if max(lengths) >= 262144:
            inset.set_xticks([0, 262144, 524288, 786432, 1048576])
    fig.text(
        0.08,
        0.035,
        f"Q input/output BF16 · KV {kv_dtype} · Index {index_dtype} · Q/KV heads 32/4 · D=128 · chunk={result['contract']['chunk']} · dense={dense_backend}\nSparse totals sum chunk medians. Preparation, compilation and index quantization excluded.",
        fontsize=8,
    )
    if kv_dtype == "fp8":
        fig.text(0.08, 0.012,
                 "Baselines: exact represented Q/K/V decoded to BF16 before timing. MBSA: native FP8 QK/PV; Q conversion timed.",
                 fontsize=7)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = arguments()
    if args.input is None:
        run(args)
    input_path = args.input if args.input is not None else args.output
    figure_path = args.output if args.input is not None else args.plot
    if figure_path is not None:
        try:
            plot_results(input_path, figure_path)
        except ValueError as error:
            raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
