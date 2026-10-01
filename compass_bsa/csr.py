"""Independent KV-centric CSR and work-record construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CsrSchedule:
    indptr: torch.Tensor
    query_indices: torch.Tensor
    work_records: torch.Tensor
    lpt_order: torch.Tensor


def build_kv_centric_csr(
    route: torch.Tensor,
    *,
    num_kv_blocks: int,
    queries_per_record: int = 128,
) -> CsrSchedule:
    """Transpose ``[H,Q,K]`` query routes into head-major KV rows."""
    if route.ndim != 3:
        raise ValueError("route must be [kv_heads, queries, topk]")
    if num_kv_blocks <= 0 or queries_per_record <= 0:
        raise ValueError("num_kv_blocks and queries_per_record must be positive")
    heads, queries, topk = route.shape
    device = route.device
    route64 = route.to(torch.long)
    q_ids = (
        torch.arange(queries, device=device)
        .view(1, queries, 1)
        .expand(heads, queries, topk)
    )
    h_ids = (
        torch.arange(heads, device=device)
        .view(heads, 1, 1)
        .expand(heads, queries, topk)
    )
    valid = (route64 >= 0) & (route64 < num_kv_blocks)
    rows = h_ids[valid] * num_kv_blocks + route64[valid]
    values = q_ids[valid]
    total_rows = heads * num_kv_blocks
    if rows.numel():
        order = torch.argsort(rows)
        rows = rows[order]
        values = values[order]
        counts = torch.bincount(rows, minlength=total_rows)
    else:
        counts = torch.zeros(total_rows, dtype=torch.long, device=device)
    indptr = torch.empty(total_rows + 1, dtype=torch.long, device=device)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(counts, dim=0)

    host_counts = counts.cpu().tolist()
    records: list[tuple[int, int, int, int]] = []
    for row, count in enumerate(host_counts):
        begin = 0
        while begin < count:
            size = min(queries_per_record, count - begin)
            records.append((row // num_kv_blocks, row % num_kv_blocks, begin, size))
            begin += size
    if records:
        work_records = torch.tensor(records, dtype=torch.int32, device=device)
        lpt = sorted(range(len(records)), key=lambda idx: (-records[idx][3], idx))
        lpt_order = torch.tensor(lpt, dtype=torch.int32, device=device)
    else:
        work_records = torch.empty((0, 4), dtype=torch.int32, device=device)
        lpt_order = torch.empty((0,), dtype=torch.int32, device=device)
    return CsrSchedule(
        indptr=indptr.to(torch.int32),
        query_indices=values.to(torch.int32),
        work_records=work_records,
        lpt_order=lpt_order,
    )

