# CompassV4 BSA Opt4：独立 Triton Baseline

## 1. 版本定位

本版本是 CompassV4 Block Sparse Attention 的独立可执行 baseline，目标是
在不依赖既有稀疏注意力源码的前提下，建立完整且可验证的 GPU 实现。GPU
计算阶段使用独立编写的 Triton kernels；PyTorch 仅用于张量管理、CUDA
stream/event 编排，以及当前 KV-major 调度表中的 `argsort` 操作。

仓库中的 attention kernels 均为独立编写，不包含复制的外部 kernel 源码，
也不依赖其他稀疏注意力实现。该版本可作为后续原生 CUDA/PTX 优化的源码
独立基线，但当前仍是多 kernel 执行流程，而不是单一物理 MegaKernel。

## 2. 输入与执行约束

当前 B300 性能路径面向下列固定配置：

| 参数 | 当前配置 |
| --- | --- |
| 数据类型 | BF16 |
| Batch | 1 |
| Query heads / KV heads | 32 / 4 |
| GQA group size | 8 |
| Head dimension | 128 |
| KV block size | 128 tokens |
| 固定支持集 | Prefix-8 + Sliding-Window-8 blocks |
| 动态支持集 | Exact Top-16 blocks |
| Query chunk | 不超过 8192 tokens，性能测试固定为 8192 |
| Query offset | 128-token block 对齐 |
| Index heads | 4 个 Index-Q heads，共享 1 个 Index-K head |

## 3. 实现设计

### 3.1 Index Score 与两级精确 Top-K

`_partition_score_kernel` 以 64 个 queries 为 query tile，并将候选 KV
空间划分为互不重叠的 16-block partitions。每个 query/block 的分数定义为
该 query 与 block 内所有 Index-K tokens 点积的最大值。固定 Prefix、
Sliding Window 和因果不可见区域在 score kernel 内直接屏蔽。

每个 partition 恰好包含 16 个 blocks，因此第一级精确 Top-16 等价于发布
该 partition 中的全部有效候选。随后 `_global_topk_kernel` 在所有 partition
候选的并集上执行第二级精确 Top-16，得到动态路由。该实现不物化 token
级 score tensor，但当前仍会写出完整的 block-score candidates；因此它是
精确层次化选择的正确性 baseline，尚未实现大 partition 下的 epilogue
候选压缩。

### 3.2 固定注意力路径

`_fixed_attention_tiled_kernel` 直接根据逻辑位置推导 Prefix-8 和
Sliding-Window-8，不构造固定稀疏索引。每个 program 处理同一 query head
中的 128-query tile，并在寄存器中维护 FP32 online-softmax 状态。固定路径
输出归一化结果和自然对数域 LSE，供最终状态归并使用。

### 3.3 KV-major 动态路径

动态 Top-16 路由首先按照 `(KV head, KV block)` 排序。连续选择同一 KV
block 的 route entries 被组织为 16-query groups；
`_dynamic_grouped_kernel` 在一个 CTA 中复用该 block 的 K/V 数据，并同时
计算 16 个 queries、每个 query 对应的 8 个 GQA heads。跨越 KV row
边界的少量分组由 `_dynamic_partial_kernel` 精确处理。

每条动态 route entry 产生一个 BF16 partial output 和 FP32 LSE。对于
Q=8192、Hkv=4、Top-K=16、GQA group=8、D=128 的配置，partial output
workspace 为 1 GiB；这是当前 baseline 的主要中间存储开销之一。

### 3.4 Online-softmax 状态归并与执行编排

`_merge_fixed_dynamic_kernel` 使用 LSE 对固定状态与 16 个动态 partial
states 进行数值稳定的 online-softmax 归并，直接生成最终 O/LSE。

`gpu_pipeline_v2` 使用独立 CUDA streams 启动固定路径和动态路由生成，并在
最终归并前建立显式事件依赖。由于当前 fixed 与 dynamic kernels 均具有较高
Tensor Core 占用，实测跨流执行主要体现为资源竞争，尚未获得显著重叠收益。

整体执行流程如下：

```text
                  ┌─ Fixed Prefix/SWA Triton ───────────────┐
Inputs ───────────┤                                          ├─ LSE merge ── O/LSE
                  └─ Score → exact Top-16 → KV-major Dynamic┘
                                │
                                └─ PyTorch argsort + boundary extraction
```

## 4. 正确性

路由结果与物化 dense block-score oracle 对比；最终注意力结果与固定、动态
支持集并集上的直接 attention 对比。

| 检查项 | B300 结果 |
| --- | ---: |
| Top-K route indices | 完全一致 |
| Router score 最大绝对误差 | `1.1444e-5` |
| Output 最大绝对误差 | `4.8089e-4` |
| LSE 最大绝对误差 | `9.5367e-7` |

本地语义测试共 3 项，覆盖两级 Top-K、KV-centric CSR 语义以及固定/动态
online-softmax 归并，均已通过。

## 5. B300 性能

测试环境：B300 SXM6 AC GPU 2，PyTorch
`2.12.0a0+0291f960b6.nv26.04.48445190`，Triton `3.6.0`，BF16，
Q=8192。计时使用 CUDA events，3 次 warmup，7 次采样取中位数。

| KV 长度 | Router | KV order | Fixed | Dynamic | Merge | 端到端 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 0.547 ms | 0.449 ms | 1.402 ms | 1.431 ms | 0.414 ms | **4.098 ms** |
| 32K | 1.045 ms | 0.439 ms | 1.415 ms | 1.621 ms | 0.416 ms | **4.787 ms** |
| 64K | 1.666 ms | 0.431 ms | 1.414 ms | 1.856 ms | 0.416 ms | **5.641 ms** |

相较早期独立 Triton 原型，当前端到端性能分别提升 3.50 倍、3.44 倍和
3.65 倍。本版本适合作为源码独立的功能与优化 baseline，不应作为最终
性能结果。

## 6. 代码结构

| 路径 | 作用 |
| --- | --- |
| `compass_bsa/gpu.py` | 精确 GPU router 与第一版 attention reference |
| `compass_bsa/gpu_v2.py` | 当前 tiled fixed、KV-major dynamic 和状态归并 |
| `compass_bsa/router.py` | PyTorch 精确路由语义参考 |
| `compass_bsa/attention.py` | PyTorch attention 与状态归并参考 |
| `compass_bsa/csr.py` | KV-centric CSR/LPT 语义参考 |
| `bench/bench_triton_baseline.py` | 稳定的 B300 测试入口 |
| `tools/check_independence.py` | 禁止依赖与来源标记检查 |

## 7. 测试与复现

本地检查：

```bash
python3 tools/check_independence.py
python3 -m unittest discover -s tests -v
```

B300 正确性与性能测试：

```bash
CUDA_VISIBLE_DEVICES=2 \
TRITON_CACHE_DIR=/tmp/compass_bsa_opt4/triton \
python3 bench/bench_triton_baseline.py \
  --revision 2 \
  --kv 8192,32768,65536 \
  --warmups 3 \
  --samples 7
```

## 8. 已知限制与后续方向

1. KV-major schedule 仍依赖 PyTorch `argsort`，并非纯 Triton route builder。
2. 16-block score partition 不压缩 block-score candidates。
3. 动态路径需要 1 GiB partial-output workspace，并额外执行状态归并 kernel。
4. 当前实现没有 TMA、warp-group specialization、LPT 或 CLC。
5. Triton mainloop 的 Tensor Core 利用率明显低于专用 SM100/SM103 kernel。

后续优化应保持当前数学语义与测试接口不变，依次替换 GPU CSR builder、
动态 partial-state 链路和 attention mainloop，并在每一步与本 baseline 做
正确性及性能回归。
