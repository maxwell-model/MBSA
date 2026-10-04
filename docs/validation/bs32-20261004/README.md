# BS32 prefill 精度与性能测试

[中文 HTML 报告](index.html)包含选定配置的所有算子结果、输出/LSE/路由门限、
同 GPU 的原生 vLLM 对比、失败路由分数诊断，以及此前真实层的路由差异数量。
最终计数以 `summary.json` 和压缩 JSONL 为准。

测试只执行 prefill。mixed 调度负载中的 decode 请求仅预留预算，
不进入 MBSA 的输入、缓存、计时或精度统计。满预算时 `T_prefill=8192-D`。
Q=1 的 prefill 边界仍保留，其 phase 是 prefill。

## 实测结果

- 选定的 416 个配置完整展开：MBSA 19,968 条，原生 6,656 条；缺项和重复均为 0。
- MBSA：15,958 条 strict passed、4,010 条 failed，无执行异常或 OOM。
- 原生：5,336 条 passed、1,312 条 failed、8 条 error，无 OOM。异常涉及连续序列中的单 prefill Q=1/KV=64K/permuted pages；正常异步独立进程重放两种请求档位各 8 项通过，主矩阵仍保留 error。
- identity/eager/warm/auto，每个配置计一次：BF16 和 FP8 各有 103/7,926,276 个 query/head 与 HF 路由不同（约 1.30×10⁻⁵），每处替换 1 个 remote block。
- 167 个配置/page-map 对的 192 个差异诊断样本，FP32 选择分数差绝对值至多 1.192×10⁻⁶；固定 MBSA route 后输出均通过。仍有路由一致时的数值失败，LSE 失败涉及 3 个 Q=8160 边界配置。
- BF16、prepared graph、warm L2、native unified、双方严格精度通过的 382 个同卡配对：原生/MBSA 时间比中位数 0.784，即 MBSA 耗时约为原生的 1.275 倍。KV>128K 时的 Router 路径切换是明显瓶颈。

上述结果是 Attention 算子测试；当前实现尚未通过完整严格 HF 精度校验，也不能声称全矩阵性能领先。

## 范围与方法

- 416 个配置、413 个不同 prefill Q/KV geometry：224 个等长配置、192 个由 mixed 预算构造的配置。
- 等长 Q=16/17/127/128/129/255/256，KV=8K/16K/…/256K。
- P=1–32 的均分、倾斜、随机 varlen；单 prefill 的 16 个 Q 边界覆盖 KV=8K/64K/128K/256K。
- MBSA 每配置 48 项，合计 19,968 项；原生每配置 16 项，合计 6,656 项。
- BF16/FP8 E4M3FN partial、两种 page map、eager/prepared graph、warm/cold L2、MBSA overlap auto/off/on。
- 每项预热 5 次、测量 19 次，保留全部 CUDA event 与同步墙钟样本。
- 对每个配置的全部 prefill query 执行原始 HF 的未修改 Attention 方法；保留 Python 的输出与 HF bitwise 对齐，附加校验 LSE。
- 图表与时间比中的严格精度失败点仍保留，但不能计为通过精度的加速结果。

HF revision：`d97a6c52967a5ed332e2196a5b985f0b4a475549`。
[源文件与五个方法的哈希](hf-source.json)提供来源证明。
完整模型、投影/RoPE、MoE FFN、vLLM 调度和 engine graph 组合不属于此次算子矩阵。

## 复现

在独占空闲的 SM103 GPU、与报告相同的镜像和源码快照中运行：

```bash
python benchmarks/bench_bs32_matrix.py \
  --hf-source /path/to/pinned/modeling_compass_moe.py \
  --output /tmp/mbsa-matrix --save-oracles /tmp/mbsa-oracles

python tools/run_bs32_native.py --cases /tmp/mbsa-matrix/selected-cases.json \
  --hf-source /path/to/pinned/modeling_compass_moe.py \
  --output /tmp/mbsa-native --partial-dtype bf16 \
  --oracle-cache /tmp/mbsa-oracles

python tools/run_bs32_native.py --cases /tmp/mbsa-matrix/selected-cases.json \
  --hf-source /path/to/pinned/modeling_compass_moe.py \
  --output /tmp/mbsa-native --partial-dtype fp8 \
  --oracle-cache /tmp/mbsa-oracles

python benchmarks/diagnose_bs32_routes.py \
  --results /tmp/mbsa-matrix/results.jsonl \
  --oracles /tmp/mbsa-oracles \
  --hf-source /path/to/pinned/modeling_compass_moe.py \
  --output docs/validation/bs32-20261004/route-diagnostics.jsonl

python tools/render_bs32_report.py \
  --results /tmp/mbsa-matrix --native /tmp/mbsa-native \
  --output docs/validation/bs32-20261004/index.html
```

分片使用 `--case-list <配置 ID 的 JSON 列表>`，同一配置的 MBSA/native 必须在同一张卡。
脚本退出码 0 表示矩阵执行完成，精度是否通过以每条记录的 gate 为准。
`--resume` 用于断点续跑；最终按完整选项键去重，并检查缺项、全部 query、19 个样本和 GPU 配对。
`--boundary-oracle` 仅供探针，不能用于本报告的全 query 结论。

原生路径来自镜像内已安装的 vLLM。unified Score plan 的 `max_qo_len` 包含 index-head 打包；按实际 API 拒绝的结果记录 legacy sparse-GQA
prefill fallback；此 fallback 的 Attention 使用 BF16，FP8 partial 开关不适用。
不引入独立 decode benchmark。原生路径遇到 CUDA context 污染时由外层 runner
重启进程，继续未记录的组合；已记录的异常保留为 error，不改成通过。

## 证据

- `summary.json`：状态、各精度门限、覆盖率、时间比、GPU/运行环境及源码哈希。
- `results.csv`：可直接分析的完整记录索引。
- `results.jsonl.gz`、`native-bf16.jsonl.gz`、`native-fp8.jsonl.gz`：原始误差和全部计时样本。
- `route-diagnostics.jsonl`：差异 block 的原始 HF FP32 分数、FP64 dot、强制路由后的 Attention 误差。
- `gpu-monitor.jsonl`、`gpu6-monitor.jsonl`：每 30 秒采样的进程归属、GPU 占用、频率、温度和功耗。
- `unit-tests.log`：空闲 GPU 上 19 项测试通过，无 skip，耗时 232.482 秒。
- `full-real-route-counts.json`：历史真实层的完整 chunk 统计及捕获哈希，独立于此次 BS32 合成测试。
- `artifact-hashes.json`：交付压缩数据、执行快照和日志的 SHA-256。
- [执行快照、异常与恢复日志](evidence/README.md)。
- [测试结束后的 GPU/容器状态](post-test-cleanup.json)：两张测试 GPU 为 0 MiB/0%，测试容器已停止。

此前包含 decode token 的 mixed 试跑以及边界 pilot 不计入本报告。
等长纯 prefill 结果按完全相同的输入、shape 和选项复用；mixed 全部按 prefill-only 口径重跑。
