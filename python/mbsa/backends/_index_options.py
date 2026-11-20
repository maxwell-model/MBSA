# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Preparation-only controls for existing Indexer implementations."""


def validate_lean192(mode: str, *, backend: str | None = None) -> str:
    """Validate a requested policy without importing a GPU backend."""
    if not isinstance(mode, str) or mode not in ("auto", "on", "off"):
        raise ValueError('lean192 must be "auto", "on", or "off"')
    if backend is not None and backend != "optimized" and mode != "auto":
        raise ValueError('lean192="on"/"off" requires backend="optimized"')
    return mode


def resolve_lean192(mode: str, eligible: bool) -> bool:
    """Resolve an existing eligibility result; forced ON never silently falls back."""
    validate_lean192(mode)
    if mode == "on" and not eligible:
        raise ValueError(
            'lean192="on" requires the SM103 BF16 direct-addressing Index policy: '
            "one nonpacked request, Q=8192, whole 8192-token KV increments, "
            "PB1024, implicit identity Index pages (no supplied mutable map), "
            "contiguous dense attention K/V, "
            "and BF16 attention operands/partials. Use auto/off for other inputs."
        )
    return bool(eligible and mode != "off")
