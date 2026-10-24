# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Zero-overhead local substitute for FlashInfer's optional API logger.

The unified backend intentionally does not import FlashInfer's process-wide
logging and CUDA Graph monkey-patching infrastructure.  Production profiling
uses explicit NVTX ranges in the caller.
"""

from __future__ import annotations

from typing import Callable, TypeVar


F = TypeVar("F", bound=Callable)


def flashinfer_api(function: F) -> F:
    return function


__all__ = ["flashinfer_api"]
