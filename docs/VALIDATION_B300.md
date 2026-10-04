# B300 validation — 2026-10-03

本文记录历史执行快照。旧 benchmark/diagnostic 命令的源码已归档到
[legacy-test-sources.tar.gz](validation/legacy-test-sources.tar.gz)；当前入口为
`tests/test_precision.py`、`tests/benchmark_performance.py` 和 `tests/config.json`。


The refactored `mbsa` package was validated through `smc toc 10.251.209.5` in Docker container `maxwell-refactor-serve-min`. Source tests and an independently installed wheel exercise the Python reference, Triton baseline and BF16 CuTe DSL runtime.

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA B300 SXM6 AC, SM103 |
| Python | 3.12.14 |
| NVIDIA driver / CUDA toolkit | 580.159.04 / 13.0.88 |
| PyTorch / CUDA runtime | 2.11.0+cu130 / 13.0 |
| Triton | 3.6.0 |
| CuTe DSL | 4.6.0 |
| cuda-python | 13.4.1 |
| apache-tvm-ffi | 0.1.10 |
| quack-kernels | 0.6.3 |

The GPU-tested wheel is `mbsa-0.1.0-py3-none-any.whl`, SHA-256 `41489765619f5d6c43939f41ed293905ad8e9bfc9f6435ad6a03e2bc781cb640`. Its runtime Python files match the repository implementation validated here. The container workspace and logs are under `/tmp/mbsa-validation-20261003/`.

## Correctness and packaging

- All 13 tests pass on B300, without skips: eight CPU/import tests and five GPU tests.
- Triton covers remote routing, partial tiles, fixed-only support and rejection of unaligned cached offsets.
- CuTe DSL covers fixed-only support, remote attention, logical KV tails, packed variable-length requests, permuted Index-K pages and permuted main K/V pages with a logical tail.
- Wheel validation checks isolated imports, the CPU oracle, bundled CUDA sources and license files. GPU checks run from an isolated wheel installation outside the source checkout, with both accelerator backends matching the reference.
- Remote block sets match exactly after sorting. Output checks use `rtol=0.03, atol=0.03`; natural-log LSE uses `rtol=0.002, atol=0.01`.

The recorded Triton oracle probes have maximum output error `0.015625` in the fixed-only case and maximum LSE error `9.54e-7`. Recorded CuTe benchmark and installed-wheel probes have maximum output error `0.0009765625` and maximum LSE error `1.91e-6`. These maxima describe the recorded probes, rather than an exhaustive bound for all inputs. Benchmark checks sample the first and final 16 query positions per request.

## Performance smoke checks

BF16, 32 Q heads, four KV heads, head dimension 128, two warmups and five CUDA-event samples after compilation; medians below exclude preparation/JIT.

| Backend | Q lengths | KV lengths | Index pages | Cache policy | End-to-end median |
|---|---|---|---|---|---|
| Triton | 8192 | 32768 | Contiguous | No explicit L2 flush | 4.900 ms |
| CuTe DSL | 8192 | 32768 | Identity | 256 MiB L2 flush | 1.003 ms |
| CuTe DSL | 777, 1153 | 33007, 16641 | Permuted | 256 MiB L2 flush | 0.852 ms |

CuTe reuses a prepared plan; Triton includes per-call allocations and scheduling. Cache policy also differs, so these runs establish execution and representative timings rather than a controlled backend speedup. They do not remeasure the supplied performance figure or qualify its 1M comparison. Raw logs and parsed measurements are in [validation/b300-20261003](validation/b300-20261003).

## Fixes validated

Dynamic MSA now receives the explicit `1/sqrt(head_dim)` softmax scale. K2's runtime tile guard calls a separate JIT scope compatible with CuTe DSL 4.6; the original guarded arithmetic, masks, synchronization and stores are preserved. Import-boundary checks run in a separate process so earlier GPU tests cannot affect their module state. The permuted main K/V page test guards the public paged-KV contract.

## Current test entry points

The measurements above are historical. Their benchmark, check and installed-wheel
scripts are in [the source archive](validation/legacy-test-sources.tar.gz), with
member paths and SHA-256 in [the archive index](validation/legacy-test-sources.json).
Use the current consolidated entry points for a new run on an empty GPU:

```bash
export PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 MAX_JOBS=2
python3 tests/test_precision.py
python3 tests/test_precision.py --wheel /path/to/mbsa-0.1.0-py3-none-any.whl
python3 tests/test_precision.py --matrix --hf-source /path/to/modeling_compass_moe.py \
  --output /tmp/mbsa-precision
python3 tests/benchmark_performance.py --hf-source /path/to/modeling_compass_moe.py \
  --output /tmp/mbsa-performance
```

All selected shapes, precision gates and performance options come from
[tests/config.json](../tests/config.json). Mixed decode requests reserve scheduler
budget only; the executed tensors contain prefill requests.
