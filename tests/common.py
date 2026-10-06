#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU/GPU contracts and full-query comparison against pinned HF Attention.

The shared config owns every shape and gate. Decode reserves scheduler budget;
only prefill enters the operator. A strict mismatch is a failure, including ties.
"""

from __future__ import annotations
import ast
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import torch
import torch.nn.functional as F

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))
CONFIG_PATH = ROOT / "tests/config.json"


def configuration(path=CONFIG_PATH):
    config = json.loads(path.read_text())
    for case in config["cases"]:
        if "uniform_prefill" in case:
            uniform = case["uniform_prefill"]
            case["requests"] = [
                {"phase": "prefill", "q": uniform["q"], "kv": uniform["kv"]}
                for _ in range(uniform["count"])
            ]
        requests = case["requests"]
        if not requests or any(
            r["phase"] != "prefill" or not 0 < r["q"] <= r["kv"] for r in requests
        ):
            raise ValueError(f"invalid prefill requests: {case['config_id']}")
        if len(case["scheduler_phases"]) != config["scheduler"]["requests"]:
            raise ValueError("scheduler request count differs from config")
        if case["scheduler_phases"].count("decode") != case[
            "reserved_decode_tokens"
        ] or case["scheduler_phases"].count("prefill") != len(requests):
            raise ValueError("mixed phase counts differ from reserved budget")
        if sum(r["q"] for r in requests) != case["prefill_total_q"]:
            raise ValueError("prefill Q sum differs from metadata")
        if (
            not case["prefill_total_q"] + case["reserved_decode_tokens"]
            == case["scheduler_total_q"]
            <= config["scheduler"]["total_query_budget"]
        ):
            raise ValueError("mixed scheduler budget exceeded")
    return config


def select_cases(config=None, ids=None):
    cases = (config or configuration())["cases"]
    if ids is not None:
        missing = set(ids) - {c["config_id"] for c in cases}
        if missing:
            raise ValueError(f"unknown configuration IDs: {sorted(missing)}")
        cases = [c for c in cases if c["config_id"] in ids]
    return cases


def hf_class(source):
    tree = ast.parse(source.read_text())
    names = {
        "_query_positions",
        "_allowed_keys",
        "_select_blocks",
        "_build_block_aligned_route",
        "_sparse_attention",
    }
    candidates = [
        c
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        and names.issubset(
            {n.name for n in c.body if isinstance(n, ast.FunctionDef)}
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one HF attention class with the required methods; found {len(candidates)}"
        )
    original = candidates[0]
    methods = [
        n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(methods) == len(names)
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="PinnedHFAttention",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    cls = namespace["PinnedHFAttention"]
    instance = cls()
    instance.block_size = 128
    instance.num_remote_blocks = 16
    instance.num_prefix_blocks = instance.num_local_blocks = 8
    instance.prefix_window_size = instance.local_window_size = 1024
    instance.query_chunk_size = 16
    instance.softmax_scale = 128**-0.5
    instance.indexer = SimpleNamespace(index_softmax_scale=128**-0.5)
    return instance


def make_inputs(case, mapping):
    requests = case["requests"]
    q_lens, k_lens = ([r[name] for r in requests] for name in ("q", "kv"))
    counts = [(n + 127) // 128 for n in k_lens]
    generator = torch.Generator(device="cuda").manual_seed(case.get("seed", 17))
    rand = lambda shape: torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    q = rand((sum(q_lens), 32, 128))
    iq = rand((sum(q_lens), 4, 128))
    pool = rand((sum(counts), 128, 4, 2, 128))
    ik = rand((sum(counts), 1, 128, 128))
    pages = torch.arange(sum(counts), device="cuda", dtype=torch.int32)
    if mapping == "permuted":
        pages = pages.flip(0)
    table = torch.zeros((len(requests), max(counts)), device="cuda", dtype=torch.int32)
    page_start = 0
    for row, count in enumerate(counts):
        table[row, :count].copy_(pages[page_start : page_start + count])
        page_start += count

    def prefix(values):
        return torch.tensor(
            [0] + list(itertools.accumulate(values)), device="cuda", dtype=torch.int32
        )

    return dict(
        q=q,
        iq=iq,
        pool=pool,
        ik=ik,
        pages=pages,
        table=table,
        cu_q=prefix(q_lens),
        cu_k=prefix(k_lens),
        counts=counts,
    )


def format_index_inputs(data, index_format):
    """Return runtime operands and dequantized HF views of the same index values.

    Dequantization belongs exclusively to the test oracle, outside kernel
    timing. Keep the original input dictionary for separate quantization-loss
    diagnostics; kernel precision is measured against the quantized inputs.
    """
    if index_format == "bf16":
        return data, data
    if index_format == "fp8":
        runtime = {
            **data,
            "iq": data["iq"].to(torch.float8_e4m3fn),
            "ik": data["ik"].to(torch.float8_e4m3fn),
        }
        return runtime, {
            **data,
            "iq": runtime["iq"].bfloat16(),
            "ik": runtime["ik"].bfloat16(),
        }
    if index_format == "mxfp8_mxfp6":
        from mbsa.index_formats import (
            quantize_q_mxfp8,
            quantize_index_k_mxfp6,
            dequantize_index_q_mxfp8,
            dequantize_index_k_mxfp6,
        )

        codes, scales = quantize_q_mxfp8(data["iq"])
        iq = codes[:, :4].view(torch.float8_e4m3fn)
        ik = quantize_index_k_mxfp6(data["ik"])
        runtime = {**data, "iq": iq, "ik": ik, "index_q_scales": scales}
        oracle = {
            **data,
            "iq": dequantize_index_q_mxfp8(iq, scales),
            "ik": dequantize_index_k_mxfp6(ik),
        }
        return runtime, oracle
    raise ValueError(f"unsupported index format: {index_format}")


def oracle_checks(case, data, hf):
    probes = []
    start = 0
    for i, request in enumerate(case["requests"]):
        qn, kn = request["q"], request["kv"]
        ids = data["table"][i, : data["counts"][i]].long()
        positions = list(range(qn))
        rows = torch.tensor([start + p for p in positions], device="cuda")
        q, iq = data["q"][rows], data["iq"][rows]
        selected_pool = data["pool"].index_select(0, ids)
        k = selected_pool[..., 0, :].reshape(-1, 4, 128)[:kn]
        v = selected_pool[..., 1, :].reshape(-1, 4, 128)[:kn]
        ik = data["ik"].index_select(0, ids).reshape(-1, 1, 128)[:kn]
        absolute = torch.tensor([kn - qn + p for p in positions], device="cuda")
        qp = hf._query_positions(
            batch_size=1,
            query_len=len(positions),
            key_len=kn,
            device=q.device,
            cache_position=absolute,
        )
        allowed = hf._allowed_keys(query_positions=qp, key_len=kn, attention_mask=None)
        remote = hf._select_blocks(
            iq.transpose(0, 1)[None], ik[:, 0][None], qp, allowed
        )
        complete = hf._build_block_aligned_route(remote, qp)
        expected = hf._sparse_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            complete,
            allowed,
        )[0].transpose(0, 1)
        # LSE is an optional addition to the retained source, not an HF output.
        from mbsa.reference import MaxwellBlockSparseAttentionReference

        ref = MaxwellBlockSparseAttentionReference(index_head_dim=128, head_dim=128)
        retained_out, lse = ref._sparse_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            complete,
            allowed,
            return_lse=True,
        )
        torch.testing.assert_close(
            retained_out[0].transpose(0, 1), expected, atol=0, rtol=0
        )
        probes.append(
            {
                "request": i,
                "rows": rows,
                "positions": positions,
                "out": expected.contiguous(),
                "lse": lse[0].transpose(0, 1).contiguous(),
                "route": remote[0].permute(1, 0, 2).contiguous(),
            }
        )
        start += qn
        del selected_pool, k, v, ik
    torch.cuda.synchronize()
    return probes


def route_metrics(actual, expected):
    # Compare sets of valid block IDs; -1 padding is excluded from overlap.
    match = (actual.sort(-1).values == expected.sort(-1).values).all(-1)
    intersection = (
        ((actual[..., :, None] == expected[..., None, :]) & (actual[..., :, None] >= 0))
        .any(-1)
        .sum(-1)
    )
    valid_expected = (expected >= 0).sum(-1)
    valid_actual = (actual >= 0).sum(-1)
    missing = valid_expected - intersection
    extra = valid_actual - intersection
    return {
        "query_head_count": match.numel(),
        "mismatched_query_heads": int((~match).sum()),
        "missing_blocks": int(missing.sum()),
        "extra_blocks": int(extra.sum()),
        "expected_valid_blocks": int(valid_expected.sum()),
        "intersection_blocks": int(intersection.sum()),
        "max_missing_blocks_per_query_head": int(missing.max()),
        "max_extra_blocks_per_query_head": int(extra.max()),
    }


def compare(actual, probes, tolerances=None):
    tolerances = tolerances or configuration()["tolerances"]
    out, lse, route = actual
    expected_out = torch.cat([p["out"] for p in probes])
    rows = torch.cat([p["rows"] for p in probes])
    actual_out = out[rows]
    diff = (actual_out.float() - expected_out.float()).abs()
    normalized = diff / (
        tolerances["output_atol"]
        + tolerances["output_rtol"] * expected_out.float().abs()
    )
    expected_lse = torch.cat([p["lse"] for p in probes])
    ld = (lse[rows].float() - expected_lse.float()).abs()
    ln = ld / (
        tolerances["lse_atol"] + tolerances["lse_rtol"] * expected_lse.float().abs()
    )
    expected_route = torch.cat([p["route"] for p in probes], dim=1)
    route_stat = route_metrics(route[:, rows], expected_route)
    return {
        "status": "passed"
        if float(normalized.max()) <= 1
        and float(ln.max()) <= 1
        and route_stat["mismatched_query_heads"] == 0
        else "failed",
        "output_status": "passed" if float(normalized.max()) <= 1 else "failed",
        "lse_status": "passed" if float(ln.max()) <= 1 else "failed",
        "route_status": "passed"
        if route_stat["mismatched_query_heads"] == 0
        else "failed",
        "checked_queries": rows.numel(),
        "total_queries": out.shape[0],
        "output_max_abs": float(diff.max()),
        "output_rms": float(diff.square().mean().sqrt()),
        "output_max_tolerance_ratio": float(normalized.max()),
        "output_query_head_max_abs_quantiles": torch.quantile(
            diff.amax(-1).flatten(), torch.tensor([0.5, 0.9, 0.99, 1.0], device="cuda")
        ).tolist(),
        "lse_max_abs": float(ld.max()),
        "lse_max_tolerance_ratio": float(ln.max()),
        "route": route_stat,
    }


def append(path, value):
    with path.open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


class BackendCaseMixin:
    def inputs(self, qn, kn):
        torch.manual_seed(17)

        def rand(shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        return (
            rand((qn, 4, 128)),
            rand((kn, 1, 128)),
            rand((qn, 32, 128)),
            rand((kn, 4, 128)),
            rand((kn, 4, 128)),
        )

    def compare(self, actual, expected):
        out, lse, route = actual
        eo, el, er = expected
        # Python topk and the kernels may order equal scores differently.
        torch.testing.assert_close(
            route.sort(-1).values, er.sort(-1).values, rtol=0, atol=0
        )
        torch.testing.assert_close(out, eo, rtol=0.03, atol=0.03)
        torch.testing.assert_close(lse, el, rtol=0.002, atol=0.01)
