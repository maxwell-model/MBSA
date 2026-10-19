# SPDX-License-Identifier: Apache-2.0
"""Fixed/Router overlap, paged KV tails and state merge."""

import math
import unittest
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PublicSm100Regression(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("SM100/SM103 is required")

    def test_fixed_router_overlap_stream_rebind_and_graph(self):
        from mbsa import prepare_cutedsl_pipeline
        from mbsa.api import prepare_sm100
        from tests.common import BackendCaseMixin

        fixtures = BackendCaseMixin()
        for qn, kn in ((1024, 4096), (1025, 4096), (17, 4113)):
            with self.subTest(q=qn, kv=kn):
                inputs = fixtures.inputs(qn, kn)
                caller = torch.cuda.Stream()
                fixed = torch.cuda.Stream()
                caller.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(caller):
                    serial = prepare_cutedsl_pipeline(
                        *inputs, overlap_fixed_score=False
                    )
                    parallel = prepare_sm100(
                        *inputs, overlap_fixed_score=True, fixed_stream=fixed
                    )
                    expected = tuple(t.clone() for t in serial.run())
                    fixtures.compare(parallel.run(), expected)
                    # Tail Index-K is padded during preparation; bind_layer
                    # requires that prepared page shape rather than raw NHD.
                    inputs = (inputs[0], serial.plan.index_k, *inputs[2:])
                    # New addresses and caller-queued writes must be visible to
                    # the side stream after rebind; outputs stay caller-owned.
                    inputs = tuple(t.clone().add_(0.0625) for t in inputs)
                    serial.plan.bind_layer(*inputs, torch.empty_like(inputs[2]))
                    parallel.plan.bind_layer(*inputs, torch.empty_like(inputs[2]))
                    expected = tuple(t.clone() for t in serial.run())
                    fixtures.compare(parallel.run(), expected)
                    parallel.plan.capture_mega_graph()
                    for _ in range(2):
                        inputs[0].add_(0.03125)
                        inputs[2].mul_(0.9375)
                        expected = tuple(t.clone() for t in serial.run())
                        out, lse = parallel.plan.run_mega()
                        fixtures.compare(
                            (out, lse, parallel.plan.attention.tile_plan.dynamic_route),
                            expected,
                        )
                    # Explicitly using the caller stream remains a safe serial
                    # path and must not create a self-dependent event cycle.
                    same_stream = prepare_sm100(
                        *inputs, overlap_fixed_score=True, fixed_stream=caller
                    )
                    fixtures.compare(same_stream.run(), expected)
                caller.synchronize()

    def test_empty_and_inplace_state_merge(self):
        from mbsa.cute.src.sm100.merge_vectorized import (
            _merge_two_attention_states_vectorized as _merge_two_attention_states_cutedsl,
        )

        a = torch.full((3, 32, 128), torch.nan, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        la = torch.full((1, 32, 3), -torch.inf, device="cuda")
        lb = torch.randn(3, 32, device="cuda")
        original = b.clone(), lb.clone()
        _merge_two_attention_states_cutedsl(a, la, b, lb, b, lb)
        torch.testing.assert_close(b, original[0], atol=0, rtol=0)
        torch.testing.assert_close(lb, original[1], atol=0, rtol=0)
        a.fill_(1)
        la.fill_(math.log(2))
        b.fill_(3)
        lb.zero_()
        _merge_two_attention_states_cutedsl(a, la, b, lb, b, lb)
        torch.testing.assert_close(b, torch.full_like(b, 5 / 3), atol=0.01, rtol=0)
        torch.testing.assert_close(
            lb, torch.full_like(lb, math.log(3)), atol=0.001, rtol=0
        )
        la.fill_(-torch.inf)
        lb.fill_(-torch.inf)
        a.fill_(torch.nan)
        b.fill_(torch.nan)
        _merge_two_attention_states_cutedsl(a, la, b, lb, b, lb)
        self.assertTrue(bool(torch.all(b == 0)))
        self.assertTrue(bool(torch.all(lb == -torch.inf)))
