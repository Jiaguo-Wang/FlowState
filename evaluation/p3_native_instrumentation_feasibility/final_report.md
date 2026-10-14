# P3-G0 原生联合事件能力审查

本轮为只读源码审查，未运行 Agent、Engine、GPU、trajectory replay 或策略比较。日期：2026-10-09。

路径约定：`S/` = `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`；`B/` = `/tmp/audit_sglang/sglang-v0517-tag/`；其余相对路径以 `/home/wjg/code/FlowState/` 为根。源码引用均为本轮读取版本。16 个关键 SGLang/runtime/工作流文件已与停止的正式控制实验容器 `flowstate-control-profile` 逐字节对照，见 `source_verification.json`。早期 `/tmp/sglang0517_src` 的 Mamba 文件摘要不同，本报告不沿用其行号或新增行为。

**PARTIAL_RUNTIME_OR_AGENT_GAP**

可以复用 SGLang 已有真实 fork 执行设施和 FlowState 的只读状态/control substrate，未发现“共享祖先 + 独立分支状态”在限定模式下原理上不可实现。但当前没有完整的 native ancestry → rid → checkpoint/owner/generation → 同一观察边界证书；也没有经核实的物理资源合同证明该工作流自然出现紧预算竞争。不能直接批准现成 runner 开跑，更不能据此承诺新的 cohort 可区分 FlowState 与强基线。

## A. 已读取的前提

- `evaluation/p2_shared_cohort_feasibility_audit/final_report.md` 与 `proposed_cohort_protocol.md`：旧 raw 缺 fork/ready/residency；23 snapshots × 3 budgets 的 K<K_min 为 0/69，其他派生 epoch UNKNOWN。本轮不重算或改变这些结果。
- `evaluation/p1_strong_baseline_validation/final_report.md`：573 cases 上 PCB/LDR/WFC/FlowState 选择和 modeled objective 相同；不是改 baseline 的理由。
- `evaluation/agentx_runtime_snapshot_collection.py:397,710` 和 replay 构建路径：实际串行重放，不能当原生 Agent runner。
- 当前 StateCatalog、PendingContinuation、adapter、FA 只读查询、batched control/优化 transport、Kimi 双 rank transport，以及已停止正式实验容器对应的 SGLang 核心源码。

来源核查很重要：旧 `/tmp/sglang0517_src` 已与停止容器中的 Mamba 文件不同。本轮以逐字节核对通过的 `S/` 版本为依据，而不是复制历史报告里的行号。16 个关键文件对照见 source_verification.json；其他辅助源码路径也明确列出，不将其当作本轮 GPU 验证结果。

## B. 能力判定

| 问题 | 判定 | 直接证据与限制 |
|---|---|---|
| 实际可执行 Agent workflow | 可复用，非现成完整方案 | Tree-of-Thought `B/benchmark/tree_of_thought_deep/bench_sglang.py:44,82,104` 有模型规划/求解/反思入口；AgentX collector 是串行 replay；本轮未验证依赖启动成功 |
| 原生 fork / 并发分支 | 源码具备 | `S/lang/interpreter.py:370` 显式复制父历史、`:329` 独立工作线程；fork 字符位置不是 recurrent token L |
| 多个 Ready / identity 贯通 | 缺原生联接与门控 | RuntimeEndpoint `:165` 未传 rid；Engine 与 GenerateReqInput 已支持 rid；无持久 ready/admission 事件协议 |
| 祖先复用与分支独立状态 | 机制成立，实际工作流待证 | MambaComponent `:173` COW、model_runner `:1505` 真正复制、memory_pool `:939` conv+temporal；需两个分支真实命中与独立 checkpoint 记录 |
| 无扰动读取 C/R/映射/池 | 基础接口已有 | barrier_fa_frontier_control.py `:26,96` 只读遍历；targeted_probe.py `:174,237` 映射和计数；当前四请求批量入口及分离 RPC 需独立 capture 扩展 |
| 同一 Allocation Boundary | 限定独占空闲模式可补齐 | fully_idle 包含 waiting_queue 为空（scheduler `:3967,3984`）；需 Agent admission gate、GPU 完成及同步 capture，batch 不是全局原子性 |
| generation/ownership | 缺完整 recurrent 事件账本 | request pool 有 req_generation，但 recurrent allocator `mamba.py:81,88` 无 slot generation；现有 Handle 无该字段 |
| TP=2 | 部分接口已具备 | RQ5-B rank-local 观测/比较，不是 transaction/全 rank 持续 barrier；最小 Pilot 先 TP=1 |
| recurrent-only 保留集合 | 有条件可用 | adapter `:332,399,416,533` 校验身份/引用、无 cascade；batch `:194` 验证集合；当前无长期 selected pin，不支持 active/session/HiCache/int8 的泛化驱逐 |
| 自然紧预算竞争 | UNKNOWN | 无真实新运行证据与资源合同；不能把多线程、共享 prompt、分支数或减小 K 当作证明 |

## C. A / B / C 必须分开

**A 当前应用自然可能产生：** 已有 Tree-of-Thought 代码可产生多条真实继承分支，后续规划/求解请求提供形成更深状态的机会。未证实它在当前 Qwen 配置中自然同时等待、保有共享/专属 checkpoint 并受限；两个分支可能生成相同文本，也可能因 tracking 或 native eviction 缺状态。Captain-Agent 所查 groupchat 是逐 speaker，Magentic-One 外层 async 不能证明同源并发。不能将 A 判为已验证自然 workload。

**B 预先固定的受控工作负载：** 复用真实 tree_search 阶段和 fork，在下一阶段模型调用发出前设两个 Ready 请求共同 admission gate，使用独占 TP=1 engine 一次读取。此协议可以在不捏造 lineage、不重放旧 trajectory、不运行策略的前提下检验真实状态；需要小范围事件/捕获开发，执行结果目前 UNKNOWN。

**C 当前模式不支持：** 现有 AgentX 串行 replay 直接提供原生 fork/ready；已把请求送进 runtime waiting_queue 仍要求 fully_idle；将 HiMamba/session streaming 当作现有 FlowState 控制对象；用跨 RPC/跨 rank 时间接近声称原子；以没有实际 slot 的文本前缀作为 checkpoint。这些用法不成立，但不构成整个系统的不可补救障碍。

容量还有硬约束：若 K 是同一时刻全部驻留 checkpoint 的硬物理上限，则 C 全驻留意味着 K_min≤|C|≤K，不可能同时 K<K_min。必须事前定义真实 retention/admission 资源合同，证明边界前候选与边界后 working/tracking/COW 的空间竞争，临时 staging 不得隐瞒。现有代码没有自动提供这种配额隔离证明。详见 missing_primitives.md。

## D. 五个最终回答

1. **是否重开发 Agent Runner：不需要重建框架。** 优先复用现有 SGLang Tree-of-Thought/StreamExecutor；新增薄层事件/backend 和 admission coordinator。若坚持使用原 AgentX collector 而拒绝改变执行入口，则它不能完成原生证据目标。
2. **是否必须改 SGLang Runtime：必须扩展观测与关联能力；不必然修改核心算法/源码。** 可沿用独立 Engine 子类、scheduler/control 和 allocator/insert/COW 审计 hook；不能只读旧日志宣称完成。hook 无法完整覆盖时才加最小只读核心接口，不改 eviction/optimizer 语义。
3. **最小开发量：四个模块。** 前端 native fork/rid/ready 日志；独占 admission coordinator；单次联合 capture 与 lifecycle/generation 审计；CPU 证书校验和冻结配置。先限 TP=1。具体人日 UNKNOWN，当前没有实现，也没有跑测试。
4. **能否真实形成竞争：共享与专属物理状态有源码支持；实际形成仍待 Pilot。** 是否存在合理 K<K_min 尚不能保证；无资源合同或状态并存失败时必须报告未建立，不事后缩预算/改 prompt 修结果。
5. **是否值得一次最小 Pilot：有条件值得。** 它能直接验证目前最关键的原生身份与状态缺口。先补接口并冻结配置，再另获执行授权；不以 FlowState 获胜为门禁，不自动采集 cohort。

## E. 交付及停止

- agent_event_capability.md：实际执行入口、九个字段和自然/受控并发边界。
- runtime_state_capability.md：实际 cache 路由、物理状态路径、只读和 TP/容量限制。
- joint_event_schema.md：事件字段、因果关系、admission/barrier 与 generation 设计。
- missing_primitives.md：最小接口缺口及物理预算矛盾的适用边界。
- minimal_pilot_protocol.md：仅一个真实工作流、一个预注册边界，不执行策略/性能实验的设计。

只新增本目录的说明与来源摘要；没有启动停止容器，没有执行 GPU/Engine/Agent/replay，没有修改 P0/P1/P1-B/P2、策略、正式结果、论文或 PPT。本轮到设计审查为止。

完整性复核：与 P2 保存的既有来源摘要比较，1817 个文件一致，差异 0；六项要求交付齐全。见 source_verification.json。未重跑 P0/P1/P1-B 或 P2 分析。
