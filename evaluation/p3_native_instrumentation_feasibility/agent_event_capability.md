# Agent 原生事件能力

本轮为只读源码审查，未运行 Agent、Engine、GPU、trajectory replay 或策略比较。日期：2026-10-09。

路径约定：`S/` = `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`；`B/` = `/tmp/audit_sglang/sglang-v0517-tag/`；其余相对路径以 `/home/wjg/code/FlowState/` 为根。源码引用均为本轮读取版本。16 个关键 SGLang/runtime/工作流文件已与停止的正式控制实验容器 `flowstate-control-profile` 逐字节对照，见 `source_verification.json`。早期 `/tmp/sglang0517_src` 的 Mamba 文件摘要不同，本报告不沿用其行号或新增行为。

## 实际入口与适用范围

| 入口 | 已有执行行为及证据 | 本任务适用性 |
|---|---|---|
| AgentX collector | `evaluation/agentx_runtime_snapshot_collection.py:397` 的 `_SglangRuntimeAdapter.execute`；`:710` 在 for 循环中同步调用 execute，返回后才执行下一条；`:327` 构建旧 current-context pending | 是可执行的串行重放器，不是原生 Agent 决策、fork、ready 事件来源；不能改名后当成新证据 |
| Captain-Agent | `/home/wjg/code/TraceElephant/code/agent_system/Captain-Agent/scripts/run_assistantbench.py:102` 调用 `initiate_chat`；同目录 `autogen/agentchat/groupchat.py:589` 的 `run_chat`，`:626` 选一个 speaker、`:628` 同步生成；异步路径 `:706`/`:708` 也依次等待 | 已有真实 Agent 入口，但所查 groupchat 路径逐 speaker 执行；不能将多个 agent 对象解释成同源并发分支。未证明全部嵌套工具路径都串行 |
| Magentic-One | `/home/wjg/code/TraceElephant/code/agent_system/Magentic-One/run_assistant_bench.py:499` 创建 `MagenticOneGroupChat`，`:509` 消费 `run_stream` | 有真实应用入口；所读外层未暴露继承前缀 fork 或 ready 队列；本地外层源码不足以确认依赖内的全部调度语义，标记 UNKNOWN，不因 async 判定并发继承 |
| SGLang Tree-of-Thought | `B/benchmark/tree_of_thought_deep/bench_sglang.py:44` 的 propose_plan、`:54` 的 execute_plan、`:63` 的 reflect_solution、`:82` 的 tree_search、`:104` 的 main；`:123` run_batch。各阶段真实调用模型，不从已有轨迹伪造输出 | 最小 Pilot 推荐复用的实际可执行工作流入口。具有显式 fork 与多阶段分支，运行依赖与 Qwen 实际输出未在本轮验证；不是 AgentX 原应用自然产生的 cohort |

`B/examples/frontend_language/usage/parallel_sample.py:22` 也是显式多分支示例，但其 Tool 是生成文本，并没有实际执行 calculator/browser；不能把这份示例包装成真实工具 Agent。Tree-of-Thought 的规划—求解—反思属于实际模型驱动工作流，适合作为受控接口 Pilot，不据此证明生产应用的分布。

## 原生 fork 的精确边界

`S/lang/interpreter.py:370` 的 `StreamExecutor.fork` 在 size>1 且文本非空时先提交 `SglCommitLazy`，`:378` 同步父队列；`:381` 创建独立 StreamExecutor；`:392`–`:398` 复制当前变量、文本、messages、role 和图片元信息。每个 executor 在 `:289` 有新 sid、`:329` 有工作线程，`submit(:342)` 入队，工作线程 `:422` 依次执行各自表达式。`ProgramState.fork(:888)` 暴露该路径，`ProgramStateGroup.__iadd__(:1084)` 向各分支提交表达式。因此支持分支并发执行，不只是相同 prompt 的偶合。

但 fork 复制的是当前工作流上下文，不是已证明复制/引用了物理 recurrent checkpoint。`fork_start_text_pos` 在 `:397` 是字符长度，不是 token 位置，不能直接写入 L 或 fork_position_tokens。必须记录父事件及实际分词后的共同继承 token 边界；token LCP 可以校验真实继承是否满足物理复用，不能用来发现/捏造 parent。tokenizer、chat template、模型、额外缓存 key 与生成输出的 token 级处理必须固定。文本重新分词发生边界变化时，允许可复用前缀短于继承文本长度，不能伪造对齐 checkpoint。

`S/lang/backend/runtime_endpoint.py:105` 的 commit_lazy_operations 会真实发送 max_new_tokens=0 的请求。它属于工作流已有执行动作，必须分配独立 request_id、记录完成及实际物化位置；它不是只读观测。该请求完成不保证任意字符末端恰有 recurrent checkpoint，仍需读取真实 cache 节点。

## 字段能力表

| 字段 | 当前能力 | 最小补充 |
|---|---|---|
| root_workflow_id | 入口 arguments/根 executor 存在；没有稳定跨层事件字段 | 在一次真实 tree_search 入口生成，所有子分支继承 |
| request_id | Engine.generate 的 rid 已支持；`evaluation/controlled_multiworkflow_v1/runtime_gate.py:152`–`:162` 已实际透传 | 每次 commit/generate 单独分配，不复用 branch sid；自定义前端 backend 在请求体写 rid |
| parent_request_id | fork 处能直接取得父 executor，但当前不保存父 invocation 因果字段 | 保存最后父 invocation 与 fork_event_id；多依赖另存 depends_on_event_ids，不能把所有依赖挤入单一 parent |
| branch_id | `StreamExecutor.sid` 可作当前进程内分支标识 | 加 run UUID 与根 ID，记录父子映射和重启身份 |
| fork_event | 显式 `fork(:370)` 调用真实存在，无持久事件日志 | 在实际上下文复制处记事件，不从 completion 记录反推 |
| fork_position | 当前只有字符位置 | 记录继承 token identity、模型/模板摘要以及可证明 token 边界，另存字符长度；不能将其等同 checkpoint L |
| ready_timestamp | 工作线程队列存在，但 expr 入队不一定意味着输入依赖已解决 | 在 backend 实际输入完全确定、允许发往服务端但尚未发送时产生 READY |
| dispatch_timestamp | HTTP 调用 `runtime_endpoint.py:186` 可挂钩，当前没有关联日志 | 在发送前记 AGENT_DISPATCH；另记 runtime receive，二者不能混用 |
| completion_timestamp | HTTP 响应 `:194` 可挂钩 | 记 AGENT_COMPLETE；额外关联 scheduler 执行/缓存提交事件，响应到达不是物化时间证明 |

`runtime_endpoint.py:165` 的 generate 请求体只提供 text/sampling 等，没有自动把 executor.sid 作为 rid；commit_lazy(:106) 同样缺 rid。`S/srt/managers/io_struct.py:160`–`:167` 支持 rid 与 session_id，省略 rid 时 runtime 自建，无法事后可靠关联父分支。推荐只扩展 backend 透传 rid，不启用 session_params；原生 session 与 FlowState 当前控制范围并不等价。

## 同时 ready 的结论

同一 fork group 的多个工作线程可以走到 backend dispatch 前；真实 Ready 的证据必须是两个输入和依赖均已确定、均无 dispatch/cancel 的事件。现有代码会直接发送，未暴露 admission barrier。因此“自然同时 ready 的持续窗口已存在于正式数据”是 UNKNOWN。

Pilot 可以预先固定一个阶段边界的两分支 admission gate，使两个真正已就绪的请求在 Agent 队列等待，只在共同观察完成后发送。这是明确披露的受控工作负载 B，不声称自然调度 A；不能先发送到 SGLang waiting_queue 再使用 fully-idle 控制点。最小实现是现有执行器的事件与请求 backend 包装，不需要重建 Agent framework。
