# 执行快照与日志

以下文件来自两张测试 GPU 的实际执行容器。主矩阵原始数值与样本保存在报告目录的压缩 JSONL。
每个 GPU 的旧 Python 执行快照已收进[历史源码归档](../../legacy-test-sources.tar.gz)，
原成员路径与 SHA-256 见[归档索引](../../legacy-test-sources.json)。以下 `.py` 链接下载该归档；
原生 fallback 的历史标签按实际 packed IndexPlan 含义规范化，原始标签保留在合并记录中。

GPU 1 的初始原生 BF16 队列退出码为 1，FP8 恢复队列也曾因 CUDA context 异常退出。
`native-prefill/recovery-history-*.json` 和相应日志记录重启进程后补跑的范围；
主矩阵中的 8 条 error 没有被覆盖。正常异步执行的独立 BF16/FP8 请求档位重放各 8 条通过，
该 Q=1 配置实际使用 BF16 prefill fallback。

测试容器已经停止，原始 oracle 保留在远端容器内；[释放后的 GPU 与容器状态](../post-test-cleanup.json)。

- [gpu1/bench_bs32_matrix.py](../../legacy-test-sources.tar.gz#docs/validation/bs32-20261004/evidence/gpu1/bench_bs32_matrix.py)
- [gpu1/bench_bs32_native.py](../../legacy-test-sources.tar.gz#docs/validation/bs32-20261004/evidence/gpu1/bench_bs32_native.py)
- [gpu1/diagnose_bs32_routes.py](../../legacy-test-sources.tar.gz#docs/validation/bs32-20261004/evidence/gpu1/diagnose_bs32_routes.py)
- [gpu1/matrix-prefill/complete.json](gpu1/matrix-prefill/complete.json)
- [gpu1/matrix-prefill/environment.json](gpu1/matrix-prefill/environment.json)
- [gpu1/matrix-prefill/selected-cases.json](gpu1/matrix-prefill/selected-cases.json)
- [gpu1/matrix-prefill.exit](gpu1/matrix-prefill.exit)
- [gpu1/matrix-prefill.log](gpu1/matrix-prefill.log)
- [gpu1/native-boundary-normal-fp8.log](gpu1/native-boundary-normal-fp8.log)
- [gpu1/native-boundary-normal.exit](gpu1/native-boundary-normal.exit)
- [gpu1/native-boundary-normal.log](gpu1/native-boundary-normal.log)
- [gpu1/native-prefill/complete-bf16.json](gpu1/native-prefill/complete-bf16.json)
- [gpu1/native-prefill/complete-fp8.json](gpu1/native-prefill/complete-fp8.json)
- [gpu1/native-prefill/environment-bf16.json](gpu1/native-prefill/environment-bf16.json)
- [gpu1/native-prefill/environment-fp8.json](gpu1/native-prefill/environment-fp8.json)
- [gpu1/native-prefill/recovery-bf16-001.log](gpu1/native-prefill/recovery-bf16-001.log)
- [gpu1/native-prefill/recovery-cases-bf16.json](gpu1/native-prefill/recovery-cases-bf16.json)
- [gpu1/native-prefill/recovery-cases-fp8.json](gpu1/native-prefill/recovery-cases-fp8.json)
- [gpu1/native-prefill/recovery-fp8-001.log](gpu1/native-prefill/recovery-fp8-001.log)
- [gpu1/native-prefill/recovery-fp8-002.log](gpu1/native-prefill/recovery-fp8-002.log)
- [gpu1/native-prefill/recovery-history-bf16.json](gpu1/native-prefill/recovery-history-bf16.json)
- [gpu1/native-prefill/recovery-history-fp8.json](gpu1/native-prefill/recovery-history-fp8.json)
- [gpu1/native-prefill-bf16.exit](gpu1/native-prefill-bf16.exit)
- [gpu1/native-prefill-bf16.log](gpu1/native-prefill-bf16.log)
- [gpu1/native-prefill-fp8.exit](gpu1/native-prefill-fp8.exit)
- [gpu1/native-recovery.log](gpu1/native-recovery.log)
- [gpu1/route-diagnostics-prefill.exit](gpu1/route-diagnostics-prefill.exit)
- [gpu1/route-diagnostics-prefill.log](gpu1/route-diagnostics-prefill.log)
- [gpu1/run_bs32_native.py](../../legacy-test-sources.tar.gz#docs/validation/bs32-20261004/evidence/gpu1/run_bs32_native.py)
- [gpu1/runtime-versions.json](gpu1/runtime-versions.json)
- [gpu1/uniform-reuse-environment.json](gpu1/uniform-reuse-environment.json)
- [gpu1/unit-tests.log](gpu1/unit-tests.log)
- [gpu6/bench_bs32_matrix.py](../../legacy-test-sources.tar.gz#docs/validation/bs32-20261004/evidence/gpu6/bench_bs32_matrix.py)
- [gpu6/bench_bs32_native.py](../../legacy-test-sources.tar.gz#docs/validation/bs32-20261004/evidence/gpu6/bench_bs32_native.py)
- [gpu6/diagnose_bs32_routes.py](../../legacy-test-sources.tar.gz#docs/validation/bs32-20261004/evidence/gpu6/diagnose_bs32_routes.py)
- [gpu6/matrix-prefill/complete.json](gpu6/matrix-prefill/complete.json)
- [gpu6/matrix-prefill/environment.json](gpu6/matrix-prefill/environment.json)
- [gpu6/matrix-prefill/selected-cases.json](gpu6/matrix-prefill/selected-cases.json)
- [gpu6/matrix-prefill.exit](gpu6/matrix-prefill.exit)
- [gpu6/matrix-prefill.log](gpu6/matrix-prefill.log)
- [gpu6/native-prefill/complete-bf16.json](gpu6/native-prefill/complete-bf16.json)
- [gpu6/native-prefill/complete-fp8.json](gpu6/native-prefill/complete-fp8.json)
- [gpu6/native-prefill/environment-bf16.json](gpu6/native-prefill/environment-bf16.json)
- [gpu6/native-prefill/environment-fp8.json](gpu6/native-prefill/environment-fp8.json)
- [gpu6/native-prefill-bf16.exit](gpu6/native-prefill-bf16.exit)
- [gpu6/native-prefill-bf16.log](gpu6/native-prefill-bf16.log)
- [gpu6/native-prefill-fp8.exit](gpu6/native-prefill-fp8.exit)
- [gpu6/native-prefill-fp8.log](gpu6/native-prefill-fp8.log)
- [gpu6/route-diagnostics-prefill.exit](gpu6/route-diagnostics-prefill.exit)
- [gpu6/route-diagnostics-prefill.log](gpu6/route-diagnostics-prefill.log)
- [gpu6/runtime-versions.json](gpu6/runtime-versions.json)
- [gpu6/uniform-reuse-environment.json](gpu6/uniform-reuse-environment.json)
