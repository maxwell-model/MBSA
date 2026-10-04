# 执行证据

最终资格化为 `final-v7`，两张空 B300 分片，每卡先运行 MBSA，再顺序运行原生 BF16 和 FP8。contract 固定所有运行时 Python 文件、两测试入口、配置、HF、GPU UUID 与 shape 清单；合并时逐文件对照本地源码，SM103、CUDA 元数据代码和测试入口全部一致，没有混用不同修复版本。3 个本地 SM90/H100 文件有独立更新，已在源码审计中保留两组哈希，属于本次 B300 结果之外的范围。原生 failed/error 原样保留。监控包含边界精修实验与最终资格化全过程，30 秒采样；不主张采样能排除采样间隙的任意瞬时活动。

最终默认 FP8 使用 cp.async Combine。先前 TMA 版本的失败及关闭 TMA 对照以压缩 JSONL 单独保留，不覆盖失败、不计入最终矩阵。短序列成功并不能确认 TMA 底层根因；本次通过限制未资格化路径修复默认行为。`final-v6-updates.tar.gz`、`final-v7-updates.tar.gz` 保存对应源码与测试快照。

`v6-smoke-evidence.tar.gz` 保存 96 项精修重点精度检查、144 项早期 smoke 性能、Fixed mask 修复及 HF dot 顺序实验；这些均为补充证据，不替代最终矩阵。最终单元测试见 `../unit-tests.log`，CPU、wheel 和构建日志在本目录。H100 历史报告与用户提供的结构/性能图保留，未计入此次 B300 资格化。

`post-test-cleanup.json` 中容器退出码 137 来自测试完成后停止常驻 sleep 容器；测试进程的退出码单独保存（MBSA=0，原生=1 表示严格 gate 失败），两容器 OOMKilled=false，测试 GPU 已无计算进程。

另外两个独立入口 `python/mbsa/sparse.py`、`python/mbsa/cute/sm90_sparse.py` 未在本次测试快照中；连同后续 SM90 改动保留在工作区，不纳入本次资格化 commit。原始测试快照及差异哈希均已保存。

独立新增的 `tests/test_sparse_api.py` 也保留在工作区；此次提交的测试面仍严格为两份 Python 入口和一份 JSON 配置。后续 H100 同步脚本更新亦未纳入本次提交。
