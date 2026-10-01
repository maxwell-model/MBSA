"""Executable fixed/dynamic attention-state specification."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .router import RouterConfig


@dataclass
class AttentionResult:
    output: torch.Tensor
    lse: torch.Tensor
    fixed_output: torch.Tensor
    fixed_lse: torch.Tensor
    dynamic_output: torch.Tensor
    dynamic_lse: torch.Tensor


def _fixed_blocks(position: int, num_blocks: int, config: RouterConfig) -> list[int]:
    q_block = position // config.block_size
    prefix = range(0, min(config.prefix_blocks, num_blocks, q_block + 1))
    local_begin = max(config.prefix_blocks, q_block - config.local_blocks + 1)
    local_end = min(q_block + 1, num_blocks)
    return sorted(set(prefix).union(range(local_begin, local_end)))


def _single_state(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blocks: list[int],
    *,
    query_position: int,
    block_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids: list[int] = []
    for block in blocks:
        begin = block * block_size
        end = min(begin + block_size, key.shape[0], query_position + 1)
        if begin < end:
            token_ids.extend(range(begin, end))
    if not token_ids:
        return torch.zeros_like(query), torch.tensor(
            -torch.inf, device=query.device, dtype=torch.float32
        )
    ids = torch.tensor(token_ids, device=query.device, dtype=torch.long)
    k_selected = key.index_select(0, ids).float()
    v_selected = value.index_select(0, ids).float()
    scores = torch.mv(k_selected, query.float()) * scale
    maximum = scores.max()
    weights = torch.exp(scores - maximum)
    denominator = weights.sum()
    output = torch.mv(v_selected.transpose(0, 1), weights) / denominator
    return output, maximum + torch.log(denominator)


def merge_states(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two normalized attention outputs using natural-log LSE states."""
    maximum = torch.maximum(first_lse, second_lse)
    both_empty = ~torch.isfinite(maximum)
    safe_maximum = torch.where(both_empty, torch.zeros_like(maximum), maximum)
    first_weight = torch.where(
        torch.isfinite(first_lse), torch.exp(first_lse - safe_maximum), 0.0
    )
    second_weight = torch.where(
        torch.isfinite(second_lse), torch.exp(second_lse - safe_maximum), 0.0
    )
    denominator = first_weight + second_weight
    safe_denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    output = (
        first_output.float() * first_weight.unsqueeze(-1)
        + second_output.float() * second_weight.unsqueeze(-1)
    ) / safe_denominator.unsqueeze(-1)
    lse = safe_maximum + torch.log(safe_denominator)
    lse = torch.where(both_empty, torch.full_like(lse, -torch.inf), lse)
    return output, lse


def dual_path_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dynamic_route: torch.Tensor,
    *,
    query_positions: torch.Tensor | None = None,
    config: RouterConfig = RouterConfig(),
    softmax_scale: float | None = None,
) -> AttentionResult:
    """Evaluate fixed and dynamic supports independently, then merge states."""
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("q must be [Q,Hq,D] and k/v must be [K,Hkv,D]")
    queries, q_heads, dim = q.shape
    kv_tokens, kv_heads, kv_dim = k.shape
    if dim != kv_dim or q_heads % kv_heads:
        raise ValueError("head dimensions must match and Hq must divide by Hkv")
    if dynamic_route.shape[:2] != (kv_heads, queries):
        raise ValueError("dynamic_route must be [Hkv,Q,K]")
    if query_positions is None:
        offset = max(kv_tokens - queries, 0)
        query_positions = torch.arange(
            offset, offset + queries, device=q.device, dtype=torch.long
        )
    scale = float(softmax_scale if softmax_scale is not None else dim ** -0.5)
    group = q_heads // kv_heads
    shape = (queries, q_heads, dim)
    fixed_o = torch.zeros(shape, device=q.device, dtype=torch.float32)
    dynamic_o = torch.zeros_like(fixed_o)
    fixed_lse = torch.full((queries, q_heads), -torch.inf, device=q.device)
    dynamic_lse = torch.full_like(fixed_lse, -torch.inf)
    num_blocks = (kv_tokens + config.block_size - 1) // config.block_size
    route_cpu = dynamic_route.to(torch.long).cpu()
    positions_cpu = query_positions.to(torch.long).cpu()
    for qi in range(queries):
        position = int(positions_cpu[qi])
        fixed = _fixed_blocks(position, num_blocks, config)
        fixed_set = set(fixed)
        for qh in range(q_heads):
            kvh = qh // group
            fo, fl = _single_state(
                q[qi, qh], k[:, kvh], v[:, kvh], fixed,
                query_position=position, block_size=config.block_size, scale=scale,
            )
            dynamic = [
                int(block)
                for block in route_cpu[kvh, qi].tolist()
                if 0 <= int(block) < num_blocks and int(block) not in fixed_set
            ]
            dynamic = list(dict.fromkeys(dynamic))
            do, dl = _single_state(
                q[qi, qh], k[:, kvh], v[:, kvh], dynamic,
                query_position=position, block_size=config.block_size, scale=scale,
            )
            fixed_o[qi, qh], fixed_lse[qi, qh] = fo, fl
            dynamic_o[qi, qh], dynamic_lse[qi, qh] = do, dl
    output, lse = merge_states(fixed_o, fixed_lse, dynamic_o, dynamic_lse)
    return AttentionResult(
        output=output.to(q.dtype),
        lse=lse,
        fixed_output=fixed_o.to(q.dtype),
        fixed_lse=fixed_lse,
        dynamic_output=dynamic_o.to(q.dtype),
        dynamic_lse=dynamic_lse,
    )
