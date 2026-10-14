# 缺失 Primitive 与最小开发范围

本轮为只读源码审查，未运行 Agent、Engine、GPU、trajectory replay 或策略比较。日期：2026-10-09。

路径约定：`S/` = `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`；`B/` = `/tmp/audit_sglang/sglang-v0517-tag/`；其余相对路径以 `/home/wjg/code/FlowState/` 为根。源码引用均为本轮读取版本。16 个关键 SGLang/runtime/工作流文件已与停止的正式控制实验容器 `flowstate-control-profile` 逐字节对照，见 `source_verification.json`。早期 `/tmp/sglang0517_src` 的 Mamba 文件摘要不同，本报告不沿用其行号或新增行为。

**PARTIAL_RUNTIME_OR_AGENT_GAP**。已找到明确可复用路径，尚无开箱即用的联合原生事件证书。以下仅为接口设计，不是本轮实施记录。

| 缺口 | 最小实现位置 | 验证要求与范围 |
|---|---|---|
| 原生 ancestry 与稳定 invocation ID | 包装 `StreamExecutor.fork` 和 RuntimeEndpoint 的 commit/generate/stream 请求入口；复用 Tree-of-Thought | 记录实际父子事件、token identity；每次调用唯一 rid；包括 lazy commit，不只记录最终 gen |
| Ready 与 admission gate | 独立 Agent backend/coordinator，依赖解决后、网络发送前 | 至少两个真实输入已就绪；所有发送都受 gate 管理；取消/重试/异常关闭可核验。不能仅加 sleep |
| 原子范围明确的只读 capture | 独立 Engine 子类与现有 idle 控制 transport | 一次回调包含 C、所有 pending R、accounting、scope、非干扰摘要，解除当前四请求硬编码仅限新入口；无需改旧文件 |
| 原生 checkpoint 创建、COW、owner/generation | recurrent allocator、Mamba insert/cleanup、model forward COW 的审计 hook | 覆盖真实执行路径，完成 fence 后才能确认；事件丢失失败。只记录不改变 allocator/optimizer |
| 跨 rank 的观察屏障 | 复用 RQ5-B rank RPC，加共同 barrier 和全部 ACK | 不是 RPC 并行即原子。首个 TP=1 Pilot 可不开发该扩展，不能据此声明 TP=2 已合格 |
| 物理容量合同 | 独立预注册配置与逐 rank 字节/slot 账目 | 区分长期 retained K、工作状态、tracking、COW 与可用 staging；不能事后按 K_min 设 K |

最小开发可收敛为四个模块：①现有前端事件/backend 包装；②Agent admission coordinator；③独立 runtime capture 与 lifecycle 审计 hook；④CPU 证书校验器/预注册配置。没有依据给出精确人日或代码行数，不把它称作已经完成的少量日志改动。

无需重写 Agent framework，也无需修改 FlowState optimizer。必须扩展 runtime 观测/关联入口，当前正式 runner 本身不足以直接验证。是否必须修改 SGLang 核心仓库：**不必然**；已有 Engine 子类和 scheduler 包装机制提供隔离入口（`tests/runtime/rq6_batched_control_transport.py:256`、`:291`）。若关键 GPU 完成或 allocator 事件无法在包装层完整覆盖，才需要新增最小只读核心 hook，不能跳过 generation 或完成证明。首次运行前须 CPU 验证 hook 覆盖及门禁，但本轮没有执行这些开发或测试。

## 不在最小范围内

不新增 optimizer/baseline，不新增持久 pin，不开发跨 rank transaction/rollback，不启用 session radix/HiCache/int8/Unified Memory，不重建 AgentX trajectory。`S/srt/session/session_controller.py:218` 的 streaming session 拒绝重叠 inflight，`:254` 续接要求父请求已完成；session API 不能自动解决并行 fork，与当前 adapter 拒绝 session radix 的范围也有差异。不要为复用 session 偷换正式控制语义。

## 容量上的硬边界

若 K 被定义为捕获时所有可驻留 checkpoint 的硬物理最大数量，并且所有 C 已驻留，则 |C|≤K，因而 K_min_frontier≤|C|≤K。该定义下，要求所有候选真实驻留同时又 K<K_min 是不可能的；不能只调小配置数字掩盖矛盾。

要研究 admission/retention 竞争，必须有预先成立的资源合同：在边界前确实容纳了 C，边界后为即将执行的工作状态/跟踪/COW 预留空间，从而长期 retained K 小于候选总数。若使用临时 staging，须明确其真实来源、字节和归还时点，不能把额外内存隐藏起来。现有控制层读得到 free list，但没有通用的 retained-vs-staging 配额隔离 primitive；因此目前只能设计这种合同，不能声称已存在资源强制机制。没有真实合同则 Pilot 可以验证共享物理状态，不能验证有效容量竞争。
