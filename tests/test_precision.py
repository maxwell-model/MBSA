#!/usr/bin/env python3
"""CPU/GPU contracts and full-query comparison against pinned HF Attention.

The shared config owns every shape and gate. Decode reserves scheduler budget;
only prefill enters the operator. A strict mismatch is a failure, including ties.
"""

from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import importlib.util
import itertools
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import torch
import torch.nn.functional as F
from mbsa import prepare_cutedsl_pipeline, reference_pipeline
from mbsa.reference import MaxwellBlockSparseAttentionReference

CUTEDSL_ARCHS = ((9, 0), (10, 3))
CONFIG_PATH = Path(__file__).with_name("config.json")


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
    original = next(
        c
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "CompassMoeBlockSparseGQAAttention"
    )
    names = {
        "_query_positions",
        "_allowed_keys",
        "_select_blocks",
        "_build_block_aligned_route",
        "_sparse_attention",
    }
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


class UpstreamReferenceTest(unittest.TestCase):
    def test_raw_negative_scores_select_distinct_group_routes(self):
        # ReLU would turn both eligible remote scores into ties. The upstream
        # oracle selects each group's least negative block independently.
        index_q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        index_k = torch.tensor(
            [
                [-10.0, -10.0],
                [-10.0, -10.0],
                [-4.0, -1.0],
                [-4.0, -1.0],
                [-2.0, -3.0],
                [-2.0, -3.0],
                [100.0, 100.0],
                [100.0, 100.0],
            ]
        )
        q = torch.zeros(1, 4, 2)
        k = torch.zeros(8, 2, 2)
        v = torch.arange(8 * 2 * 2).reshape(8, 2, 2).float()
        output, lse, route = reference_pipeline(
            index_q,
            index_k,
            q,
            k,
            v,
            block_size=2,
            dynamic_topk=1,
            prefix_blocks=1,
            local_blocks=1,
        )
        self.assertEqual(route.tolist(), [[[2]], [[1]]])
        for head in range(4):
            group = head // 2
            selected = [0, 1, 4, 5, 6, 7] if group == 0 else [0, 1, 2, 3, 6, 7]
            torch.testing.assert_close(output[0, head], v[selected, group].mean(0))
        torch.testing.assert_close(lse, torch.full((1, 4), 6.0).log())

    def test_partial_block_cached_chunk_matches_dense_masked_gqa(self):
        torch.manual_seed(11)
        queries, tokens, groups, heads, dim = 3, 11, 2, 4, 4
        index_q = torch.randn(queries, groups, dim)
        index_k = torch.randn(tokens, 1, dim)
        q = torch.randn(queries, heads, dim)
        k = torch.randn(tokens, groups, dim)
        v = torch.randn_like(k)
        # Top-K covers every remote block, so complete support is dense causal.
        output, lse, route = reference_pipeline(
            index_q,
            index_k,
            q,
            k,
            v,
            block_size=3,
            dynamic_topk=8,
            prefix_blocks=1,
            local_blocks=1,
            query_offset=7,
        )
        self.assertEqual(route.shape, (groups, queries, 8))
        self.assertTrue(bool((route[..., 3:] == -1).all()))
        for qi in range(queries):
            for head in range(heads):
                group = head // (heads // groups)
                scores = k[: qi + 8, group] @ q[qi, head] * dim**-0.5
                expected = scores.softmax(-1) @ v[: qi + 8, group]
                torch.testing.assert_close(output[qi, head], expected)
                torch.testing.assert_close(lse[qi, head], scores.logsumexp(-1))

    def test_fixed_prefix_local_support_has_no_duplicates(self):
        oracle = MaxwellBlockSparseAttentionReference(
            index_head_dim=2,
            head_dim=2,
            block_size=4,
            prefix_blocks=2,
            local_blocks=3,
            dynamic_topk=1,
        )
        positions = torch.arange(32).unsqueeze(0)
        remote = torch.full((1, 32, 2, 1), -1)
        route = oracle._build_block_aligned_route(remote, positions)
        for query, position in enumerate(positions[0].tolist()):
            current = position // 4
            expected = set(range(min(2, current + 1)))
            expected.update(range(max(2, current - 2), current + 1))
            for group in range(2):
                actual = [b for b in route[0, query, group].tolist() if b >= 0]
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(actual), len(set(actual)))

    def test_bf16_probability_cast_matches_upstream(self):
        torch.manual_seed(42)
        q = torch.randn(1, 2, 4).bfloat16()
        k = torch.randn(7, 1, 4).bfloat16()
        v = torch.randn_like(k)
        output, lse, _ = reference_pipeline(
            torch.ones(1, 1, 4),
            torch.ones(7, 1, 4),
            q,
            k,
            v,
            block_size=2,
            dynamic_topk=4,
            prefix_blocks=1,
            local_blocks=1,
        )
        for head in range(2):
            scores = k[:, 0].float() @ q[0, head].float() * 0.5
            expected = scores.softmax(-1).bfloat16() @ v[:, 0]
            torch.testing.assert_close(output[0, head], expected, rtol=0, atol=0)
            torch.testing.assert_close(lse[0, head], scores.logsumexp(-1))

    def test_input_contract_rejects_group_specific_index_keys(self):
        with self.assertRaisesRegex(ValueError, "shared index key"):
            reference_pipeline(
                torch.zeros(2, 2, 4),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 4, 4),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 2, 4),
            )


class ImportBoundaryTest(unittest.TestCase):
    def test_partial_precision_switch_and_explicit_override(self):
        import torch
        from mbsa.cutedsl_backend import _resolve_partial_dtype

        for mode, expected in (("0", torch.bfloat16), ("1", torch.float8_e4m3fn)):
            with (
                self.subTest(mode=mode),
                patch.dict(os.environ, {"MBSA_FP8_PARTIAL_OUT": mode}),
            ):
                self.assertEqual(_resolve_partial_dtype(None), expected)
                self.assertEqual(_resolve_partial_dtype(torch.bfloat16), torch.bfloat16)
        with patch.dict(os.environ, {"MBSA_FP8_PARTIAL_OUT": "invalid"}):
            with self.assertRaisesRegex(ValueError, "MBSA_FP8_PARTIAL_OUT"):
                _resolve_partial_dtype(None)
        with self.assertRaisesRegex(ValueError, "partial_dtype"):
            _resolve_partial_dtype(torch.float32)

    def test_public_import_is_lazy(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "python")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import mbsa; "
                "assert mbsa.__all__ == ['reference_pipeline', 'triton_pipeline', 'prepare_cutedsl_pipeline']; "
                "assert not {'torch','triton','cutlass','fmha_sm100'}.intersection(sys.modules)",
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cutedsl_rejects_cpu_before_loading_runtime(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "python")
        script = r"""
import sys, torch
from mbsa import prepare_cutedsl_pipeline
q = torch.zeros((1, 32, 128))
try:
    prepare_cutedsl_pipeline(q, q, q, q, q)
except ValueError as error:
    assert 'CUDA' in str(error), str(error)
else:
    raise AssertionError('CPU input must be rejected')
assert 'fmha_sm100' not in sys.modules
assert 'cutlass' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_bundled_runtime_does_not_take_over_global_modules(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "python")
        script = r"""
import importlib, sys, types
from mbsa.cutedsl_backend import _load_runtime
globals_before = {}
for name in ('src', 'interface', 'sparse_index_utils', 'fmha_sm100'):
    globals_before[name] = sys.modules[name] = types.ModuleType(name)
base = 'mbsa._vendor.msa.python.fmha_sm100'
sentinel = object()
for suffix, names in (
    ('.cute.interface', ('SparseForwardPhase', 'sparse_atten_func')),
    ('.cute.sparse_index_utils', ('build_k2q_csr',)),
    ('.cute.src.sm100.prepare_k2q_csr', ('SparseK2qCsrBuilderSm100', 'allocate_k2q_csr_scratch')),
):
    module = types.ModuleType(base + suffix)
    for name in names:
        setattr(module, name, sentinel)
    sys.modules[module.__name__] = module
path_before = sys.path[:]
_load_runtime()
owned = importlib.import_module(base)
assert owned.sparse_atten_func is sentinel
assert sys.path == path_before
assert all(sys.modules[name] is value for name, value in globals_before.items())
"""
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BackendAlignment(unittest.TestCase):
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

    def test_triton(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton is required")
        from mbsa import triton_pipeline

        for qn, kn in ((128, 4096), (17, 4113), (128, 128)):
            with self.subTest(q=qn, kv=kn):
                operands = self.inputs(qn, kn)
                self.compare(triton_pipeline(*operands), reference_pipeline(*operands))

    def test_triton_rejects_unaligned_offset(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton is required")
        from mbsa import triton_pipeline

        with self.assertRaisesRegex(ValueError, "block-aligned"):
            triton_pipeline(*self.inputs(16, 4096))

    def test_cutedsl(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("qualified CuTe DSL runtime requires SM90 or SM103")
        from mbsa import prepare_cutedsl_pipeline

        for qn, kn in ((16, 4096), (16, 4113), (128, 128)):
            with self.subTest(q=qn, kv=kn):
                operands = self.inputs(qn, kn)
                self.compare(
                    prepare_cutedsl_pipeline(*operands).run(),
                    reference_pipeline(*operands),
                )

    def test_cutedsl_fp8_partial_storage_matches_reference(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("qualified CuTe DSL runtime requires SM90 or SM103")
        from mbsa import prepare_cutedsl_pipeline

        for qn, kn in ((128, 8192), (17, 4113)):
            with self.subTest(q=qn, kv=kn):
                operands = self.inputs(qn, kn)
                pipeline = prepare_cutedsl_pipeline(
                    *operands, partial_dtype=torch.float8_e4m3fn
                )
                self.assertEqual(
                    pipeline.plan.attention.dynamic_workspace[0].dtype,
                    torch.float8_e4m3fn,
                )
                self.compare(pipeline.run(), reference_pipeline(*operands))

    def test_cutedsl_packed_permuted_index_pages(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("qualified CuTe DSL runtime requires SM90 or SM103")
        from mbsa import prepare_cutedsl_pipeline

        lengths = ((17, 4113), (23, 2195))
        operands = [self.inputs(qn, kn) for qn, kn in lengths]
        iq, q, k, v = (
            torch.cat([request[i] for request in operands]) for i in (0, 2, 3, 4)
        )
        pages = []
        for request in operands:
            ik = request[1]
            padded = torch.zeros(
                ((ik.shape[0] + 127) // 128 * 128, 1, 128),
                device="cuda",
                dtype=torch.bfloat16,
            )
            padded[: ik.shape[0]].copy_(ik)
            pages.append(padded.view(-1, 1, 128, 128))
        logical_ik = torch.cat(pages)
        permutation = torch.arange(logical_ik.shape[0] - 1, -1, -1, device="cuda")
        physical_ik = torch.empty_like(logical_ik)
        physical_ik.index_copy_(0, permutation, logical_ik)
        cu_q = torch.tensor([0, 17, 40], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, 4113, 6308], dtype=torch.int32, device="cuda")
        actual = prepare_cutedsl_pipeline(
            iq,
            physical_ik,
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            index_page_indices=permutation.to(torch.int32),
        ).run()
        start = 0
        for request, (qn, _) in zip(operands, lengths):
            qs = slice(start, start + qn)
            self.compare(
                (actual[0][qs], actual[1][qs], actual[2][:, qs]),
                reference_pipeline(*request),
            )
            start += qn

    def test_cutedsl_permuted_kv_pages_with_tail(self):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("qualified CuTe DSL runtime requires SM90 or SM103")
        from mbsa import prepare_cutedsl_pipeline

        operands = self.inputs(17, 4113)
        iq, ik, q, k, v = operands
        blocks = (k.shape[0] + 127) // 128
        permutation = torch.arange(blocks - 1, -1, -1, device="cuda")

        def paged(tensor):
            padded = torch.zeros(
                (blocks * 128, 4, 128), dtype=tensor.dtype, device=tensor.device
            )
            padded[: tensor.shape[0]].copy_(tensor)
            logical = padded.view(blocks, 128, 4, 128).permute(0, 2, 1, 3).contiguous()
            physical = torch.empty_like(logical)
            physical.index_copy_(0, permutation, logical)
            return physical

        actual = prepare_cutedsl_pipeline(
            iq,
            ik,
            q,
            paged(k),
            paged(v),
            page_table=permutation.to(torch.int32).unsqueeze(0),
            logical_k_tokens=k.shape[0],
        ).run()
        self.compare(actual, reference_pipeline(*operands))

    def test_cutedsl_vllm_interleaved_cache_and_single_query_chunk(self):
        """Use the actual split/permuted cache strides, including a KV tail."""
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("qualified CuTe DSL runtime requires SM90 or SM103")
        from mbsa import prepare_cutedsl_pipeline

        for qn, kn in (
            (1, 4113),
            (17, 4113),
            (1, 16384),
            (17, 16384),
            (16, 16384),
            (16, 16385),
        ):
            operands = self.inputs(qn, kn)
            iq, ik, q, k, v = operands
            blocks = (k.shape[0] + 127) // 128
            pool = torch.zeros((blocks, 128, 4, 2, 128), device="cuda", dtype=q.dtype)
            permutation = torch.arange(blocks - 1, -1, -1, device="cuda")
            for half, tensor in enumerate((k, v)):
                padded = torch.zeros(
                    (blocks * 128, 4, 128), device="cuda", dtype=q.dtype
                )
                padded[: tensor.shape[0]].copy_(tensor)
                pool[..., half, :].index_copy_(
                    0, permutation, padded.view(blocks, 128, 4, 128)
                )
            kp, vp = (pool[..., half, :].permute(0, 2, 1, 3) for half in (0, 1))
            self.assertEqual(kp.permute(0, 2, 1, 3).stride(), (131072, 1024, 256, 1))
            self.assertEqual(vp.data_ptr() - kp.data_ptr(), 128 * q.element_size())
            actual = prepare_cutedsl_pipeline(
                iq,
                ik,
                q,
                kp,
                vp,
                page_table=permutation.to(torch.int32).unsqueeze(0),
                logical_k_tokens=k.shape[0],
            ).run()
            self.compare(actual, reference_pipeline(*operands))

    def test_cutedsl_rebind_refreshes_index_pages_and_fused_output(self):
        """A reused plan must read the new layer and identity-to-permuted map."""
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("qualified CuTe DSL runtime requires SM90 or SM103")
        from mbsa import prepare_cutedsl_pipeline

        first = self.inputs(17, 8192)
        second = tuple(tensor.roll(1, 0).add(0.125) for tensor in first)
        blocks = 64

        def physical(operands, permute):
            iq, ik, q, k, v = operands
            pages = torch.arange(blocks, dtype=torch.int32, device="cuda")
            if permute:
                pages = pages.flip(0)
            pool = torch.empty((blocks, 128, 4, 2, 128), device="cuda", dtype=q.dtype)
            for half, tensor in enumerate((k, v)):
                pool[..., half, :].index_copy_(
                    0, pages.long(), tensor.view(blocks, 128, 4, 128)
                )
            ip = torch.empty((blocks, 1, 128, 128), device="cuda", dtype=q.dtype)
            ip.index_copy_(0, pages.long(), ik.view(blocks, 1, 128, 128))
            return (
                iq,
                ip,
                q,
                pool[..., 0, :].permute(0, 2, 1, 3),
                pool[..., 1, :].permute(0, 2, 1, 3),
                pages,
            )

        iq, ik, q, k, v, pages = physical(first, False)
        pipeline = prepare_cutedsl_pipeline(
            iq,
            ik,
            q,
            k,
            v,
            index_page_indices=pages,
            page_table=pages.unsqueeze(0),
            logical_k_tokens=8192,
        )
        self.compare(pipeline.run(), reference_pipeline(*first))
        iq, ik, q, k, v, pages = physical(second, True)
        out = torch.empty_like(q)
        pipeline.rebind(
            iq,
            ik,
            q,
            k,
            v,
            index_page_indices=pages,
            page_table=pages.unsqueeze(0),
            out=out,
        )
        actual = pipeline.run()
        self.assertEqual(actual[0].data_ptr(), out.data_ptr())
        self.compare(actual, reference_pipeline(*second))

    def test_vllm_adapter_reuses_packed_plan_with_new_layer_and_page_maps(self):
        """Packed routing and output ownership survive a real layer rebind."""
        self._check_vllm_adapter_rebind(use_graphs=False)

    def test_vllm_graph_adapter_replays_updated_queries_cache_and_page_maps(self):
        """Captured kernels must consume live values through stable staging."""
        self._check_vllm_adapter_rebind(use_graphs=True)

    def _check_vllm_adapter_rebind(self, use_graphs):
        if torch.cuda.get_device_capability() not in CUTEDSL_ARCHS:
            self.skipTest("qualified CuTe DSL runtime requires SM90 or SM103")
        from mbsa.vllm import PagedPrefill

        lengths = ((1, 130), (1, 4098), (17, 8193))
        counts = [(kn + 127) // 128 for _, kn in lengths]
        adapter = PagedPrefill(max_cached_plans=1, use_graphs=use_graphs)
        rounds = 3 if use_graphs else 2
        for iteration in range(rounds):
            operands = [
                tuple(t.add(iteration * 0.125) for t in self.inputs(qn, kn))
                for qn, kn in lengths
            ]
            iq, q = (torch.cat([row[i] for row in operands]) for i in (0, 2))
            if iteration != 2:
                pool = torch.zeros(
                    (sum(counts), 128, 4, 2, 128), device="cuda", dtype=q.dtype
                )
                ik = torch.zeros(
                    (sum(counts), 1, 128, 128), device="cuda", dtype=q.dtype
                )
            pages = torch.arange(sum(counts), device="cuda", dtype=torch.int32)
            if iteration % 2:
                pages = pages.flip(0)
            table = torch.zeros(
                (len(lengths), max(counts)), device="cuda", dtype=torch.int32
            )
            page_start = 0
            for slot, (row, count) in enumerate(zip(operands, counts)):
                ids = pages[page_start : page_start + count]
                table[slot, :count].copy_(ids)
                for half, tensor in enumerate(row[3:5]):
                    padded = torch.zeros(
                        (count * 128, 4, 128), device="cuda", dtype=q.dtype
                    )
                    padded[: tensor.shape[0]].copy_(tensor)
                    pool[..., half, :].index_copy_(
                        0, ids.long(), padded.view(count, 128, 4, 128)
                    )
                padded = torch.zeros(
                    (count * 128, 1, 128), device="cuda", dtype=q.dtype
                )
                padded[: row[1].shape[0]].copy_(row[1])
                ik.index_copy_(0, ids.long(), padded.view(count, 1, 128, 128))
                page_start += count
            out = torch.empty_like(q)
            actual = adapter(
                q,
                pool[..., 0, :],
                pool[..., 1, :],
                index_q=iq,
                index_k_cache=ik[:, 0],
                index_page_indices=pages,
                page_table=table,
                request_lens=lengths,
                out=out,
                partial_dtype=torch.bfloat16,
            )
            self.assertEqual(actual.data_ptr(), out.data_ptr())
            pipeline = next(iter(adapter.plans.values()))
            lse = pipeline.plan.attention.out_lse
            route = pipeline.plan.attention.tile_plan.dynamic_route
            start = 0
            for row, (qn, _) in zip(operands, lengths):
                qs = slice(start, start + qn)
                self.compare(
                    (actual[qs], lse[qs], route[:, qs]), reference_pipeline(*row)
                )
                start += qn
        self.assertEqual((adapter.hits, adapter.misses), (rounds - 1, 1))
        if use_graphs:
            self.assertEqual(len(pipeline._vllm_graphs), 2)


PACKAGE = ROOT / "python/mbsa"

MSA = PACKAGE / "_vendor/msa/python/fmha_sm100"

LOCAL_ROOTS = {
    "mbsa": ROOT / "python/mbsa",
}

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
        PACKAGE / "_vendor/msa/LICENSE",
        PACKAGE / "_vendor/msa/NOTICE",
        PACKAGE / "_vendor/anthony_bsa_backend/LICENSE",
        PACKAGE / "_vendor/anthony_bsa_backend/NOTICE",
        MSA / "cute/src/sm100/build_k2q_csr/build_k2q_csr.cu",
    )
    for path in required:
        if not path.is_file():
            errors.append(f"missing runtime/license file: {path.relative_to(ROOT)}")
    for name in ("kernels", "compass_bsa", "maxwell_bsa", "mbsa", "bench"):
        if (ROOT / name).exists():
            errors.append(f"obsolete source directory: {name}")
    for path in (
        ROOT / "README.md",
        ROOT / "PROVENANCE.md",
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
        "mbsa/_vendor/msa/LICENSE",
        "mbsa/_vendor/msa/NOTICE",
        "mbsa/_vendor/anthony_bsa_backend/LICENSE",
        "mbsa/_vendor/anthony_bsa_backend/NOTICE",
        "mbsa/_vendor/msa/python/fmha_sm100/cute/src/sm100/build_k2q_csr/build_k2q_csr.cu",
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
import mbsa._vendor.msa.python.fmha_sm100 as bundled
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


class ConfigurationContracts(unittest.TestCase):
    def test_prefill_only_selected_matrix(self):
        c = configuration()
        self.assertEqual(len(c["cases"]), 416)
        self.assertEqual(len({x["config_id"] for x in c["cases"]}), 416)
        self.assertEqual({len(x["requests"]) for x in c["cases"]}, set(range(1, 33)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FixedWindowRegression(unittest.TestCase):
    def test_graph_capture_waits_for_queued_eager_work(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("SM103 capture stream regression")
        torch.manual_seed(17)
        qn, kn = 128, 4096
        operands = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            for shape in (
                (qn, 4, 128),
                (kn, 1, 128),
                (qn, 32, 128),
                (kn, 4, 128),
                (kn, 4, 128),
            )
        ]
        p = prepare_cutedsl_pipeline(*operands, overlap_fixed_score=True)
        p.run()
        torch.cuda.synchronize()
        torch.cuda._sleep(100_000_000)
        operands[2].add_(0.125)
        p.run()
        # No host synchronization between the eager launch and capture.
        p.plan.capture_mega_graph()
        p.plan.run_mega()
        graph_out = p.plan.attention.out.clone()
        eager_out = p.run()[0]
        torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)

    def test_fp8_shape_sequence_capture_and_live_query_updates(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("SM103 FP8 shape sequence regression")
        case = select_cases(ids=["p1-d31-kv8192-q4096-boundary"])[0]
        # Reuse compiled kernels across small/tail/full Q shapes and return to
        # the failing shape after releasing each plan and its graph workspace.
        for qn, overlap in itertools.product((127, 129, 4096, 257, 4096), (False, True)):
            with self.subTest(q=qn, overlap=overlap):
                case = {**case, "requests": [{"phase": "prefill", "q": qn, "kv": 8192}]}
                data = make_inputs(case, "permuted")
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
                    partial_dtype=torch.float8_e4m3fn,
                    overlap_fixed_score=overlap,
                )
                pipeline.run()
                torch.cuda.synchronize()
                pipeline.plan.capture_mega_graph()
                for shift in (0.125, -0.25, 0.125):
                    data["q"].add_(shift)
                    pipeline.plan.run_mega()
                    graph_out = pipeline.plan.attention.out.clone()
                    graph_lse = pipeline.plan.attention.out_lse.clone()
                    eager_out, eager_lse, _ = pipeline.run()
                    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)
                    torch.testing.assert_close(graph_lse, eager_lse, atol=0, rtol=0)
                del pipeline, data
                gc.collect()
                torch.cuda.empty_cache()

    def test_unaligned_prefix_does_not_include_old_swa_page(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("SM103 TMEM regression")
        qn, kn = 8160, 65536
        q = torch.zeros((qn, 32, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((kn, 4, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        v[:, :, 0] = (torch.arange(kn, device="cuda") // 128)[:, None] / 512
        p = prepare_cutedsl_pipeline(
            q[:, :4].contiguous(),
            k[:, :1].contiguous(),
            q,
            k,
            v,
            overlap_fixed_score=False,
        )
        a = p.plan.attention
        a.run_fixed()
        position = torch.arange(kn - qn, kn, device="cuda")
        block = position // 128
        live = position % 128 + 1
        count = 1024 + 7 * 128 + live
        local_sum = (7 * block - 28) * 128 + block * live
        expected = (128 * 28 + local_sum).float() / count / 512
        actual = a.fixed_out
        if actual.ndim == 4:
            actual = actual.permute(1, 0, 2, 3).reshape(qn, 32, 128)
        torch.testing.assert_close(
            actual[:, :, 0].float(),
            expected[:, None].expand(qn, 32),
            atol=0.002,
            rtol=0.002,
        )
        torch.testing.assert_close(
            a.fixed_lse[0].T,
            count.float().log()[:, None].expand(qn, 32),
            atol=1e-5,
            rtol=1e-6,
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
            data = make_inputs(case, mapping)
            probes = oracle_checks(case, data, hf)
            for dtype, overlap in itertools.product(
                config["options"]["partial_dtype"], config["options"]["overlap"]
            ):
                p = prepare_cutedsl_pipeline(
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
            del data, probes
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps({"case": case["config_id"], "status": counts}), flush=True)
    (args.output / "complete.json").write_text(json.dumps(counts, indent=2))
    if counts["failed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument(
        "--wheel",
        type=Path,
        help="verify isolated wheel imports and bundled runtime sources",
    )
    parser.add_argument("--hf-source", type=Path)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output", type=Path, default=Path("/tmp/mbsa-precision"))
    args = parser.parse_args()
    if args.wheel:
        check_wheel(args.wheel)
    if args.matrix:
        if not args.hf_source:
            parser.error("--matrix requires --hf-source")
        matrix(configuration(args.config), args)
    else:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        )
        raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
