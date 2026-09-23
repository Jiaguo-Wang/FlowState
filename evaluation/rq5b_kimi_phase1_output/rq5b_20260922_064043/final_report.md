# RQ5-B Kimi Phase 1：运行时与组件隔离

最终状态：RQ5B_KIMI_PHASE1_READY。初始 smoke 和随后 3/3 个独立引擎生命周期均通过。

模型：`/data1/models/Kimi-Linear-48B-A3B-Instruct`；SGLang `0.5.17`，构建提交 `29481685462732237d80d86076d6563e1f658102`；两张 H100 PCIe，TP=2。模型配置、分词器、索引和全部 20 个权重分片的张量映射及长度一致。该完整性核对不包含全部权重内容的哈希。

启动参数由本地 [CLI 帮助](launch_server_help.txt) 与 SGLang 源码确认。实际使用 [Docker 启动命令](launch_command.txt) 和 [引擎参数](runs/smoke/engine_args.json)：32K 上下文、Triton 注意力和 KDA 后端、`extra_buffer` 缓存策略、256 令牌跟踪间隔、关闭 overlap 与 CUDA graph。HiCache、Mooncake、PD disaggregation、推测解码和 torch.compile 均未启用。

两个 rank 均加载 `KimiLinearForCausalLM`，建立含 FULL 与 MAMBA 组件的 `UnifiedRadixCache`。真实请求在 8K 与 16K 形成 KDA 检查点，MLA/attention 侧驻留到 16K。逻辑到物理节点映射在两个 rank 上均为 c1→21、c2→25。

| 检查阶段 | H | E | G | c1 KDA | c2 KDA | MLA 到 16K |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 驱逐前 | 16384 | 16384 | 0 | 驻留 | 驻留 | 驻留 |
| 仅驱逐 c2 后 | 16384 | 8192 | 8192 | 驻留 | 缺失 | 驻留 |

四次运行中，两个 rank 的 FULL 树摘要、分配器计数和非目标 c1 KDA 槽位都保持不变。全树循环状态差分只包含 c2 节点；没有原生循环状态驱逐、意外重物化、attention 侧级联或跨运行污染。每次关闭后两卡均回到 14 MiB，且没有 compute 进程。

原始证据见 `runs/smoke`、`runs/rep01`、`runs/rep02`、`runs/rep03`，各目录含运行时门禁、驱逐前后视图、逐 rank 驱逐证明和最终记录。详细汇总见 [summary.json](summary.json)，关键文件摘要见 [manifest.json](manifest.json)。

本轮止于 Phase 1。未运行 FlowState 分配、批量协调泛化、恢复成本、基线比较、RQ4 或 RQ6 实验，也未改动冻结产物。
