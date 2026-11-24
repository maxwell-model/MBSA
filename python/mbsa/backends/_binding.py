# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side staging primitives; kernels and plan policy stay backend-owned."""

import copy
from types import MethodType
import torch


def explicit_device(device) -> torch.device:
    """Return torch.device(device) with a CUDA index: a bare "cuda" names the device current at the call.

    Runner and warmup caches key on the result, so that one bare name never stands for two GPUs. The call
    does not initialize CUDA: until it is initialized the current device is 0.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_initialized() else 0)
    return device


def _copy_object(value):
    result = copy.copy(value)
    for name, item in vars(value).items():
        if isinstance(item, (list, dict)):
            setattr(result, name, copy.copy(item))
        elif isinstance(item, MethodType) and item.__self__ is value:
            setattr(result, name, MethodType(item.__func__, result))
    return result


def _copy_maps(value, storages):
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        key = (value.device, storage.data_ptr(), storage.nbytes())
        if key not in storages:
            raw = torch.empty(0, dtype=torch.uint8, device=value.device)
            raw.set_(storage, 0, (storage.nbytes(),), (1,))
            storages[key] = raw.clone().untyped_storage()
        result = torch.empty(0, dtype=value.dtype, device=value.device)
        return result.set_(storages[key], value.storage_offset(), value.shape, value.stride())
    if isinstance(value, list):
        return [_copy_maps(item, storages) for item in value]
    if isinstance(value, tuple):
        items = [_copy_maps(item, storages) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, dict):
        return {key: _copy_maps(item, storages) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        result = copy.copy(value)
        for name, item in vars(value).items():
            setattr(result, name, _copy_maps(item, storages))
        return result
    return value


def stage_plan(plan, *, page_table=None):
    """Copy binding objects and mutable maps before validating a proposed update.

    Args:
        plan: Currently published plan; its original objects and tensor contents are not mutated.
        page_table: Optional replacement attention map used to recognize table sharing; None stages
            the current mapping.

    Returns:
        Staged shallow plan copy with private mutable binding/map objects; immutable data/storage
        may still be shared.
    """
    result = _copy_object(plan)
    attention = result.attention = _copy_object(plan.attention)
    attention.tile_plan = _copy_object(plan.attention.tile_plan)
    for name in ("fixed_attention", "dynamic_kvouter", "contiguous_cache"):
        helper = getattr(attention, name, None)
        if helper is not None:
            setattr(attention, name, _copy_object(helper))
    if getattr(plan, "mxfp6_router", None) is not None:
        result.mxfp6_router = _copy_object(plan.mxfp6_router)
    if page_table is not None:
        fixed = getattr(attention, "fixed_attention", None)
        if fixed is not None:
            # Map views and packed metadata must keep their storage aliases.
            # Otherwise refresh writes a different array from the compiled input.
            storages = {}
            for name in ("mappings", "metadata", "packed_metadata", "inverse_maps"):
                if hasattr(fixed, name):
                    setattr(fixed, name, _copy_maps(getattr(fixed, name), storages))
            if fixed._prepared_forward is not None:
                fixed._bind_prepared_forward()
        pages = getattr(plan, "cutedsl_packed_pages", None)
        if pages is not None:
            result.cutedsl_packed_pages = pages.clone()
    return result


def publish_plan(plan, staged):
    """Publish a successfully validated staged binding while preserving plan identity.

    Args:
        plan: Existing compatibility plan object to update in place.
        staged: Validated staged copy whose binding attributes replace plan state.

    Returns:
        None. The caller is responsible for retaining old storage through queued work.
    """
    attention = plan.attention
    state = dict(staged.__dict__)
    state["attention"] = attention
    for name, value in state.items():
        if isinstance(value, MethodType) and value.__self__ is staged:
            state[name] = MethodType(value.__func__, plan)
    attention.__dict__ = dict(staged.attention.__dict__)
    plan.__dict__ = state


def validate_output_aliases(out, *operands):
    """Reject any byte-span overlap between output and read-only operands.

    Args:
        out: Optional output tensor; None skips the check. Empty/non-tensor operands are ignored.
        operands: Read-only tensors or nested tuple/list operand collections. Read-only operands
            may share storage with one another.

    Returns:
        None; raises ValueError for overlapping output/input spans.
    """
    if out is None:
        return

    def span(tensor):
        if tensor.numel() == 0:
            return (0, 0)
        start = tensor.data_ptr()
        size = 1 + sum((n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride()))
        return start, start + size * tensor.element_size()

    begin, end = span(out)
    for operand in operands:
        if operand is None or operand.device != out.device:
            continue
        first, last = span(operand)
        if begin < last and first < end:
            raise ValueError("out must have a byte span disjoint from input operands")
