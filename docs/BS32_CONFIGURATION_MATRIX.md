# BS32 配置列表：KV 8K–256K，总 Chunk 8192，Block 128

本文件定义配置域与候选样例。[修复后完整选定矩阵测试](validation/bs32-refactor-20261004/index.html)单独报告实际覆盖率、精度和性能；[此前报告](validation/bs32-20261004/index.html)保留历史失败。Chunk 已确认是**每步总预算**，32 个请求共享
8192 个 query token。8K–256K 按本次 Attention 调用的逻辑 KV 总长度定义，
包含本步 query 对应的 token；1K=1024。配置列举不代表这些 shape 已通过验证。

## 固定参数

| 参数 | 配置 |
|---|---|
| 负载 / 调度请求上限 | BS=32，`max_num_seqs=32` |
| 每步总 query 预算 | `max_num_batched_tokens=8192` |
| Block Size | 128 |
| Q heads / KV heads / Head dim | 32 / 4 / 128 |
| Index-Q heads / Index-K heads / Index dim | 4 / 1 / 128 |
| 固定支持与路由 | Prefix-8 + SWA-8；Remote Top-16 |
| 主 Q/K/V、Index-Q/K、最终输出 | BF16 |
| Partial output | BF16 / FP8 E4M3FN |

KV 主档位（单位 K）全部列出：

```text
8, 16, 24, 32, 40, 48, 56, 64,
72, 80, 88, 96, 104, 112, 120, 128,
136, 144, 152, 160, 168, 176, 184, 192,
200, 208, 216, 224, 232, 240, 248, 256
```

完整 KV 域是整数 `[8192,262144]`，不限于上述档位；KV/block 尾部的
128 种余数均在该域中。主档位用于展开清单，随机 varlen 配置也包含非对齐 KV。

## BS32 mixed batch 的预算公式

设 D 个 decode，P 个 prefill，`D+P=32`。每个 decode 消耗一个 query token。
先给每个活跃请求预留一个 token，剩余预算就是：

\[
E=8192-32=8160,\qquad
Q_j^{\mathrm{prefill}}=1+e_j,\qquad
e_j\ge0,\quad\sum_{j=1}^{P}e_j\le8160.
\]

因此 `total_q=32+Σe_j≤8192`。随机生成时同时随机化剩余预算的使用量和
它在 P 个 prefill 之间的分配；不分别独立采样 Q 后直接拼接，避免超预算。
Q=1 的 prefill 作为缓存命中后的尾部 chunk 保留，并用 phase 标签区别于 decode。

本次只测试 prefill。D 个 decode 只用于预留调度预算；实际送入 MBSA 的
是 P 个 prefill 的 Q/KV，满预算时 `T_prefill=8192-D`，未满预算时更小。
decode 的 Q/KV 不参与算子执行、计时、HF 校验或路由统计。纯 decode 不在测试范围内。

| 配置族 | D / P | Prefill Q 范围与总预算 | 满预算例子 |
|---|---|---|---|
| Decode + 一个 prefill | 31 / 1 | Prefill Q 为 1–8161；extra 为 0–8160 | `8161 + [1]×31` |
| Decode + 两个 prefill | 30 / 2 | `Q₁+Q₂≤8162` | `6000+2162 + [1]×30` |
| Decode + 四个 prefill | 28 / 4 | `ΣQprefill≤8164` | `4096+2048+1024+996 + [1]×28` |
| Decode + 八个 prefill | 24 / 8 | `ΣQprefill≤8168` | `4096+2048+1024+512+256+128+64+40 + [1]×24` |
| Decode + 十六个 prefill | 16 / 16 | `ΣQprefill≤8176` | `[511]×16 + [1]×16` |
| Decode + 多个 prefill 完整族 | `32-P` / P，P=2–31 | `ΣQprefill≤8160+P` | 每个 P 都覆盖均分、倾斜、随机分配 |
| 纯 prefill，等长 Q | 0 / 32 | 每请求 Q=1–256 | `[256]×32` |
| 纯 prefill，varlen Q | 0 / 32 | 每请求 Q≥1，`ΣQ≤8192` | `[4096]+[128]×30+[256]` |

单 prefill 明确覆盖边界 Q：
`1,2,16,17,127,128,129,255,256,257,4095,4096,4097,8159,8160,8161`。
8160 是可分配的额外 token 上限；加上预留的一个 token，单 prefill 满预算 Q 是 8161。

BS32 表示整个请求负载。实际调度子步可能只有 A=1–32 个活跃请求，其余排队
或已经结束；此时 `ΣQ≤8192`、`max Qᵢ≤8192-(A-1)`。例如 A=1 时可以执行
Q=8192 的完整 chunk。32 个请求同时各执行 Q=8192 不符合这个总预算配置。

## 所有合法 Q/KV shape 的完整定义

对 32 个活跃请求，完整的有序 varlen shape 集合为：

\[
\mathcal S_{32}=\left\{((Q_i,K_i))_{i=1}^{32}:
1\le Q_i,\quad8192\le K_i\le262144,\quad
Q_i\le K_i,\quad\sum_iQ_i\le8192\right\}.
\]

每个请求的 cached prefix 是 `Kᵢ-Qᵢ`。Q 的所有整数值、KV 的所有整数值、
所有请求排列、不同 Q/KV 长度的组合都由这个集合覆盖；decode/prefill 的 phase
标签还要满足上面的配置族规则。整个 mixed 集合用约束表达，不声称有限样例已穷举。

等长子集是 `(Q,K)×32`，其中 Q 为 1–256。32 个 KV 主档位对应 **8192 个**
不同等长 shape；如果 KV 也逐整数展开，则是 **65,011,968 个**等长 shape。

实际 packed 张量只包含 prefill，令 `T=Σprefill Qᵢ`、`N=Σprefill ceil(Kᵢ/128)`，
下表 A 为实际执行的 prefill 请求数 P：

| 张量 | Shape |
|---|---|
| Q | `[T,32,128]` |
| Index-Q | `[T,4,128]` |
| 原生 interleaved KV pool | `[N,128,4,2,128]` |
| Index-K pages | `[N,1,128,128]` |
| Q/KV prefix sums | `[A+1]` |
| Page table | `[A,≥max ceil(Kᵢ/128)]` |

## 选项组合

算子矩阵按下面四个维度组合，每个 shape 有 16 个基础配置：

| 维度 | 取值 |
|---|---|
| Partial output | BF16 / FP8 E4M3FN |
| 执行方式 | Eager / Prepared CUDA Graph |
| 物理 page 映射 | Identity / Permuted |
| L2 策略 | Warm / Cold |

附加调度诊断选项为 `overlap_fixed_score=auto/off/on`。三种都列入待验证配置，
与上述维度组合为 48 个算子配置。BS32 的 Fixed Attention 使用静态 scheduler
和普通 TMEM load；不把内部只为 batch-one 资格化的优化随意开启。

vLLM 矩阵单独组合以下四个维度，共 16 个配置：

| 维度 | 取值 |
|---|---|
| MBSA adapter graph | `MBSA_VLLM_PREFILL_GRAPH=0/1` |
| Prefill partial 精度 | `MAXWELL_BSA_PREFILL_FP8_PARTIALS=0/1` |
| Prefix caching | Off / On；On 时分别记录 cold 与 hit 阶段 |
| Engine execution | Enforce eager / PIECEWISE prefill + FULL decode |

MBSA 接入开关为 `MBSA_VLLM_PREFILL=1`。直接算子使用
`MBSA_FP8_PARTIAL_OUT=0/1` 或显式 `partial_dtype`；显式 dtype 优先。vLLM adapter
按调用者的 partial dtype 执行，不把这两个环境变量当独立的精度维度重复组合。

每个 mixed 族还要覆盖：满预算／未满预算、KV 等长／随机 varlen、decode-first／
prefill-first／interleaved 顺序，以及 graph replay 后更换 query、层 cache 和 page map。
随机种子列为 `17,42,2026`。decode 顺序保留为调度元数据；算子输入仅保留
对应顺序的 prefill，不实现或测试独立 decoding。

对比路径包括 Python oracle、保留的 Triton 精度基线、目标容器和参考容器的
sparse GQA / unified prefill。有效性约束是：

- Triton prefill 要求 `(Kᵢ-Qᵢ)%128==0`；当前公开接口逐请求运行，不能冒充 packed BS32 性能。
- Python/Triton 不提供此 CuTeDSL 的 FP8 partial 开关；参考容器默认列 BF16。
- 原生 unified 的 Score plan 要求 `max_qo_len≥129`，包含 index-head 打包，不等于逻辑 Q。API 拒绝时记录实际使用的 legacy prefill fallback；fallback 使用 BF16 Attention，FP8 partial 开关不适用。
- 当前 MBSA 输入固定为 BF16，不能进入外层 CUDA graph capture；adapter 自己的 staged graph 是独立选项。

测量协议列为 5 次 warmup、19 次交替采样，CUDA-event 和同步 wall time 同时记录。
启动、准备计划和 capture 时间单独记录。BS32、KV=256K 时，每层原生主 KV pool 为
16 GiB，Index-K 为 2 GiB；18 GiB 仅是 cache，不包含权重、workspace 和 graph。

## 当前配置入口

[统一配置文件](../tests/config.json)保存当前选定的 416 个 prefill 配置、两种 page map、
BF16/FP8 partial、eager/graph、warm/cold L2、overlap auto/off/on，以及 HF 来源和精度门限。
等长请求使用 `uniform_prefill` 紧凑存储，加载后展开为实际请求列表。
`scheduler_phases` 仅描述调度顺序；decode 不提供被测 KV 张量。

完整候选域的定义仍由本文的约束给出。此前 8192 个等长候选和 3136 个 mixed 样例的
生成清单、旧测试和诊断脚本存放在[历史源码归档](validation/legacy-test-sources.tar.gz)，
不作为额外的测试入口。实际覆盖率以具体运行报告为准。
