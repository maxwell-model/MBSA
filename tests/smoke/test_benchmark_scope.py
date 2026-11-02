# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for benchmark selection and cold-flush event ordering."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


def benchmark_module():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/bench_prefill.py'
    spec = importlib.util.spec_from_file_location('bench_prefill_scope_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkScope(unittest.TestCase):
    def test_explicit_backend_and_unchanged_default(self):
        bench = benchmark_module()
        with tempfile.TemporaryDirectory() as folder:
            argv = ['bench', '--lengths', '8192', '--no-plot', '--output', str(Path(folder) / 'out.json')]
            with patch('sys.argv', argv):
                self.assertEqual(bench.arguments().backend, 'default')
            with patch('sys.argv', argv + ['--backend', 'optimized']):
                self.assertEqual(bench.arguments().backend, 'optimized')

    def test_cold_flush_is_drained_before_start_event(self):
        calls = []

        class Event:
            def __init__(self, **kwargs):
                self.index = sum(x.startswith('create') for x in calls)
                calls.append(f'create{self.index}')

            def record(self):
                calls.append(f'event{self.index}')

            def synchronize(self):
                calls.append('end_sync')

            def elapsed_time(self, other):
                return 1.0

        stream = types.SimpleNamespace(synchronize=lambda: calls.append('flush_sync'))
        cuda = types.SimpleNamespace(synchronize=lambda: calls.append('initial_sync'),
            current_stream=lambda: stream, Event=Event)
        flush = types.SimpleNamespace(add_=lambda _: calls.append('flush'))
        result = benchmark_module().measure(types.SimpleNamespace(cuda=cuda),
            lambda: calls.append('run'), flush, warmups=0, samples=1)
        self.assertEqual(result['samples_ms'], [1.0])
        self.assertLess(calls.index('flush'), calls.index('flush_sync'))
        self.assertLess(calls.index('flush_sync'), calls.index('event0'))
        self.assertLess(calls.index('event0'), calls.index('run'))
        self.assertLess(calls.index('run'), calls.index('event1'))
