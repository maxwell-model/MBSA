# Copyright 2026 Maxwell Team.
# SPDX-License-Identifier: Apache-2.0
"""Public Fireworks KV-outer attention; CUDA dependencies load on preparation."""


def __getattr__(name):
    if name == "PreparedKVOuter":
        from .prepared import PreparedKVOuter
        return PreparedKVOuter
    raise AttributeError(name)


__all__ = ["PreparedKVOuter"]
