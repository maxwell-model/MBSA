# SPDX-License-Identifier: Apache-2.0
"""Minimal stand-ins for the three ``vllm.*`` symbols the two vendored MiniMax
M3 Triton files (``minimax_index_topk_triton.py``, ``minimax_sparse_attn_triton.py``)
import. This lets those files run with only torch+triton installed -- no vLLM
in this venv.

Real symbols replaced:
  vllm.platforms.current_platform.{is_rocm, is_arch_support_pdl}
  vllm.triton_utils.{tl, triton}          (plain re-exports upstream too)
  vllm.utils.math_utils.round_up
"""

import triton
import triton.language as tl
import torch


class _CurrentPlatform:
    @staticmethod
    def is_rocm() -> bool:
        return False

    @staticmethod
    def is_arch_support_pdl() -> bool:
        # Programmatic Dependent Launch needs CUDA SM90+; this project only
        # targets NVIDIA B300 (SM100/103), which qualifies.
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


current_platform = _CurrentPlatform()


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple
