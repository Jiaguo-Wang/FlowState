# T2-G2 GPU1独立正式批次报告

**BLOCKED：首个Gate在Engine初始化时发生新增编排入口错误，已按协议立即停止；未启动正式60臂。**

## 设备绑定与冻结身份

- 冻结C/K3配置为Qwen3.5-9B、SGLang v0.5.17、`tp_size=1`、28槽；没有改为双卡或调整并行配置。
- GPU1：H100 PCIe，UUID `GPU-051c9abc-864a-9f31-c4f7-a737aed997ff`。启动前17MiB且无计算进程。
- Docker仅暴露该UUID；CUDA_VISIBLE_DEVICES同样指定该UUID；所有宿主门禁及清理只查询该UUID。容器内真实检查通过：仅一个CUDA设备，逻辑0，UUID与GPU1一致，见`g_CB_FS/device_binding.json`。
- 独立容器前缀`flowstate-t2g2-gpu1-`、控制端口50087和输出目录。没有调用GPU0控制入口，没有干预GPU0已有任务。
- 63臂协议、策略选集、K=3、输入、模型镜像和原Runner/Bridge/Control保持不变。镜像中16项Runtime源码与版本校验通过。逐文件变更Diff见`takeover/*.diff`。
- GPU0旧目录全部顶层原始文件在结束后复核摘要一致；旧Gate与主臂未混入本批次统计。

## 预检遗漏及实际失败

CPU静态检查和冻结身份检查通过，但没有覆盖Python multiprocessing spawn重新导入主入口的行为，因此预检READY不足以保证新编排入口可用。

本次新增`device_entry.py`在顶层校验设备并独占写入`/b5out/device_binding.json`，随后通过runpy进入原Runner。SGLang创建子进程时，`multiprocessing.spawn._fixup_main_from_path`以`__mp_main__`重新执行入口；该文件缺少`if __name__ == '__main__'`保护，重复写入同一证据文件，触发`FileExistsError`。日志随后记录Rank 0 Scheduler初始化失败。直接栈见`g_CB_FS.runtime.log`第4–25行及后续异常。

这是本次新增编排代码的缺陷，不是已证实的GPU1不兼容、TP配置错误、Strict Compatibility错误或Native恢复计算错误。日志中其他可选音频依赖警告不能替代上述直接根因。

未修改失败入口、未删除设备证据、未放宽独占写入规则、未自动重试。下一次若获准恢复，最小必要修正为主进程入口保护，并先补充CPU spawn重入回归；不能只删除已存在JSON来绕过错误。需使用新的独立批次，保留当前失败现场。

## 执行与清理

|范围|尝试|通过|未运行|
|---|---:|---:|---:|
|GPU1 Gate|1|0|2|
|GPU1正式主矩阵|0|0|60|

`g_CB_FS_process.json`记录退出码1；`run/engine_cleanup.json`记录engine_created=false、shutdown_returned=true。三次宿主收尾检查均17MiB、无计算进程，专用容器不存在，`g_CB_FS_cleanup.json`为PASS。完整63臂状态见`execution_ledger.csv`。

## 三层结果边界

- 模型化J及冻结选集只是沿用的输入，没有在本轮重新优化，不是GPU测量。
- 本轮没有完成历史状态物化，不能报告当前Engine的正式E/G/Prefill或Native恢复认证。
- 没有正式TTFT、CUDA、控制成本配对数据。没有使用GPU0、Smoke或旧Gate计时代填；无法形成五轮性能结论。
- Marconi-Evict Adaptation与KVFlow-inspired仍是策略适配，不是原论文完整系统。

## 证据索引

`migration_cpu_precheck.json`、`container_cpu_precheck.json`、`gpu1_execution_lock.json`、`implementation_lock_gate.json`：身份与预检。

`device_binding_audit.md`、`g_CB_FS/device_binding.json`：静态和实际设备隔离证据。

`launch_gate_failure.json`、`g_CB_FS.runtime.log`、`g_CB_FS/run/failure.json`、`batch_host.log`：失败原因和原始日志。

`g_CB_FS_cleanup.json`、`g_CB_FS/run/engine_cleanup.json`、`batch_final_status.json`：清理及最终门禁。

本轮停止，未启动其他Gate或60臂主矩阵，未修改生产代码、Phi、策略、冻结输入或GPU0历史结果。
