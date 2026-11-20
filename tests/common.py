# SPDX-License-Identifier: Apache-2.0
"""Shared GPU fixtures and comparisons for the retained core regressions."""

import itertools

import torch


def index_dispatch(**modes):
    """Context manager binding optimized Index plans with `modes` (IndexDispatch fields).

    Unnamed fields keep the enclosing index_dispatch block's value, else the baseline selection. Plans
    prepared or rebound inside the block use it on any device, independently of MBSA_INDEX_DISPATCH.
    """
    import dataclasses
    from mbsa.backends.optimized import index_policy
    base = index_policy.index_dispatch_request()[0] or index_policy.BASELINE_INDEX_DISPATCH
    return index_policy.index_dispatch_override(dataclasses.replace(base, **modes))


def baseline_index_dispatch():
    """Context manager binding optimized Index plans with the baseline selection."""
    from mbsa.backends.optimized import index_policy
    return index_policy.index_dispatch_override(index_policy.BASELINE_INDEX_DISPATCH)


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
