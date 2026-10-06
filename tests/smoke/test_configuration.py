#!/usr/bin/env python3
"""CPU/GPU contracts and full-query comparison against pinned HF Attention.

The shared config owns every shape and gate. Decode reserves scheduler budget;
only prefill enters the operator. A strict mismatch is a failure, including ties.
"""

from __future__ import annotations
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

CUTEDSL_ARCHS = ((9, 0), (10, 0), (10, 3))


from tests.common import configuration


class ConfigurationContracts(unittest.TestCase):
    def test_prefill_only_selected_matrix(self):
        c = configuration()
        self.assertEqual(len(c["cases"]), 416)
        self.assertEqual(len({x["config_id"] for x in c["cases"]}), 416)
        self.assertEqual({len(x["requests"]) for x in c["cases"]}, set(range(1, 33)))
