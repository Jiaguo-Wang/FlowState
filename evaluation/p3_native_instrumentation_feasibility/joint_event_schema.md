# 最小联合事件关联协议（设计，未实现）

本轮为只读源码审查，未运行 Agent、Engine、GPU、trajectory replay 或策略比较。日期：2026-10-09。

路径约定：`S/` = `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`；`B/` = `/tmp/audit_sglang/sglang-v0517-tag/`；其余相对路径以 `/home/wjg/code/FlowState/` 为根。源码引用均为本轮读取版本。16 个关键 SGLang/runtime/工作流文件已与停止的正式控制实验容器 `flowstate-control-profile` 逐字节对照，见 `source_verification.json`。早期 `/tmp/sglang0517_src` 的 Mamba 文件摘要不同，本报告不沿用其行号或新增行为。

## 身份和事件

公共 envelope：schema_version、run_uuid、engine_uuid、root_workflow_id、branch_id、request_id（可空但须说明事件类型）、event_id=(producer_uuid,local_seq)、event_type、monotonic_ns、clock_domain、causal_predecessors、barrier_id、rank、payload_digest。所有 ID 在生成时固定；重试有新 attempt_id 并关联原 request_id。rank-local slot 不能脱离 pool/engine UUID；同名 slot 在不同 rank 不是同一物理存储。

| 事件 | 必需 payload | 采集边界 |
|---|---|---|
| WORKFLOW_START | 任务摘要、runner/模型/tokenizer/template/runtime 版本 | 真正执行入口 |
| FORK | 父 branch、父 invocation、子 branch 列表、继承上下文摘要、字符位置、token 边界证明及尚未确认字段 | `StreamExecutor.fork` 的实际上下文复制处，`S/lang/interpreter.py:370`；无来源证明不得补造 |
| REQUEST_READY | 已确定的 input token identity、A、依赖事件、sampling/extra_key | 依赖解决后，backend 发请求前；入表达式队列本身不够 |
| AGENT_DISPATCH / RUNTIME_ACCEPT / AGENT_COMPLETE | 稳定 rid、发送/接收/响应事件 ID、状态 | backend HTTP 或 Engine 适配层与 scheduler 分别记录，不能合并为一个时间戳 |
| SLOT_ALLOC / SLOT_FREE / SLOT_DONATE | rank/pool/slot、generation、旧/新 owner、关联 rid | allocator 与 tracking 转交处；generation 在实际分配时递增，转交 owner 有单独事件，不伪称重新分配 |
| CHECKPOINT_COMMIT | canonical checkpoint_id、完整 token prefix 摘要、L、node_id、各组件 slot/generation、来源 invocation、提交序号 | 插入完成且状态计算就绪后；不能在请求开始/提交前写 MATERIALIZED |
| COW_READY | source checkpoint、source/destination slot 及 generation、rid、conv/temporal 复制路径与完成 fence | copy 发起与完成分开，`model_runner.py:1505` 路径 |
| CHECKPOINT_DETACH / NATIVE_EVICT | node/prefix/组件、generation、reason、调用来源 | recurrent detach/free 路径，用于识别历史候选失效与自动替换 |
| EPOCH_CAPTURE / RELEASE | P、C、compatibility 证据、R/T/L、所有 rank ACK、pool 数据、前后语义摘要 | 下述唯一边界；RELEASE 之后的状态不是同一 snapshot |

最小 runtime 事件 hook 的位置见 runtime_state_capability.md。新增 hook 仅记录真实事件，不能更改 match/evict/LRU、给 checkpoint 加保护锁、强制创建节点或改变 eligibility。事件先存内存缓冲，关键 capture 后落盘；缓冲溢出或事件缺失必须使关联失败，不能继续猜测。写盘失败同样标记不可审计。

## 跨进程顺序：使用因果关系，不比较不同 clock_domain 的纳秒值

1. Coordinator 关闭独占 Engine 的 admission gate，生成 barrier_id，等待全部此前已 dispatch 的 rid 收到完成且无在途提交。所有候选 Ready 请求保留在 Agent 层；读取 P 时确认 ready 且无 dispatch/cancel。这不是把尚未完成依赖的未来请求算 pending。
2. 向 scheduler 发送 CAPTURE(barrier_id, ready_manifest_digest, candidate_provenance_digest)。scheduler 在原有 idle 安全点确认所有队列/异步插入/传输已清空，确认此前 GPU 操作完成。TP=1 一次处理；TP>1 每 rank ACK，同一 barrier 内不释放 admission。
3. 同一次 scheduler 控制函数读取 checkpoint 路径、全局组件映射、FA frontier、allocator/headroom、状态事件序号。读取前后比较树、recency、LRU、引用、free list；不调用 match_prefix 预热或 materialize。必要同步 fence 是证据开销，不能说零开销；只保证无 cache 决策状态改变。
4. Coordinator 校验全部 ACK 与 P digest，依据 FORK 因果链 + 实际 prefix/model identity 生成 compatibility；T=min(A,R)，每边 L≤T。保留 checkpoint 的物化证明必须 happened-before capture，detach/free/代际改变不得夹在其间。
5. 持久化该边界证书后发送 RELEASE。Ready 请求按预注册的调度协议真实发出，记录后续 COW/执行/complete 作为验证；不能把后续才物化的 checkpoint 倒填到 C。

可验证顺序示例：parent dispatch → parent runtime accept → checkpoint commit → parent complete → fork → branch ready；这不是唯一合法顺序。另一种是 fork 先发生、继承前缀后来才在某个 child 请求中首次物化。必须按实际事件记录先后，并把该 checkpoint 限制为 commit 之后 epoch 的候选；不能仅用 ancestry 把它视作 fork 时已存在。

没有 wall-clock 校准时，独立生产者之间没有消息/ACK 连接的两个事件只具有偏序，顺序记 UNKNOWN。barrier 的作用是建立可验证 happens-before，不是为所有历史事件随意生成一个全局时间。物化 hook 必须传播源 rid/因果 ID，不能事后仅依靠相同 token 哈希关联不同工作流。

## generation 的最小边界

现有 request_pool.req_generation 不覆盖 recurrent allocator。Pilot 使用独立审计 sidecar 为 recurrent alloc/free/donate/commit 建立 generation/owner 账本，不改变 allocator 选择；可用实例包装或独立 Engine 子类挂钩，不能拿 UUID epoch_token 冒充 slot generation。若无法覆盖所有实际 alloc/free 路径，generation 标为 UNKNOWN，Pilot 身份门禁失败。

观察非干扰校验必须比较原系统语义状态，不能因为新增审计计数器递增判定 cache 被扰动；反之不能把真实 LRU/reference 改动当成允许的日志计数。所有必要状态只读，审计侧日志与 runtime 数据结构分离。
