# 最小联合事件 Pilot 协议（仅设计，未执行）

本轮为只读源码审查，未运行 Agent、Engine、GPU、trajectory replay 或策略比较。日期：2026-10-09。

路径约定：`S/` = `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`；`B/` = `/tmp/audit_sglang/sglang-v0517-tag/`；其余相对路径以 `/home/wjg/code/FlowState/` 为根。源码引用均为本轮读取版本。16 个关键 SGLang/runtime/工作流文件已与停止的正式控制实验容器 `flowstate-control-profile` 逐字节对照，见 `source_verification.json`。早期 `/tmp/sglang0517_src` 的 Mamba 文件摘要不同，本报告不沿用其行号或新增行为。

## 目标与准入前置条件

只验证一次真实两分支工作流是否可产生可审计的联合 Allocation Epoch。不跑策略、不测性能、不创建正式 cohort。推荐复用 `B/benchmark/tree_of_thought_deep/bench_sglang.py` 的规划—求解—反思和 SGLang fork 执行设施，而不是 AgentX 合成 token replay。属于预先固定的受控工作负载 B；不是自然生产 cohort A，也不宣称工具执行，因为该入口没有外部工具调用。

先完成 missing_primitives.md 中 TP=1 所需事件、admission、capture、generation/owner 与证书校验。未完成则不得启动 Pilot。只计划 Qwen3.5-9B、SGLang 正式 0.5.17 对应来源、TP=1、单独 Engine；本文件不是运行授权。Kimi TP=2 另需全部 rank 的屏障确认，不加入本次最小计划。

执行前冻结一次配置文件并哈希：runner/source commit 与 hook 摘要、模型/tokenizer/chat template、任务来源/准确任务文本、最大各阶段生成长度、seed/sampling、branch 数=2、tracking interval/cache strategy、path cap、recurrent/FA pool 配置、working/COW/临时 headroom 合同、capture 规则、超时和终止条件。当前任务文本、生成配置与资源合同尚未选定，不填虚构数值；这是待绑定设计，不能直接作为已注册 protocol 开跑。

任务选择：从已存在、可用的真实问题按与输出无关的确定性来源序号取一题，或由用户预先指定一项实际规划任务；先确认本地题目存在与许可。本轮没有下载题目。不能运行后换题、增大输入、修改 sampling 或重试直到产生不同分支；两个分支输出相同是合法失败结果。

## 最小工作流的确定范围

不直接运行原 benchmark 默认的 2×2×2×2 扩展。计划复用其前端与阶段表达式，只在根执行一次真实 fork(2)，各分支依次完成实际规划/求解，并在下一次反思调用前观察；阶段内不再扩增分支。这个受控两分支入口需作为新 Pilot 薄层实现事前冻结，不能声称原 benchmark 无改动直接可用。

admission gate 只截住预注册的目标阶段请求；此前 lazy commit 和前序依赖正常完成并留档。必须先向两个独立 executor 提交目标阶段表达式，再等待双方 READY；禁止在注册第二个分支前对第一个被 gate 拦住的请求同步等待，也不能 gate 住 fork 内部 commit 导致父线程死锁。所有目标请求到齐后才关闭其他 ingress 并 drain。该顺序需要 CPU 状态机验证，不能仅靠两个线程证明无死锁。

## 执行步骤

1. 一个 root workflow 真正执行父上下文准备与原生 fork；记录默认 lazy commit 请求，不额外发送预热/重放以补造祖先。记录 fork 的父子身份及确实继承的历史。父文本末端与 checkpoint 边界不等同。
2. 两个分支真实执行规划/求解阶段，使用各自实际模型输出形成下一阶段反思输入。保留所有实际物化 checkpoint，包括失败/自动替换记录，不预选“看起来有利”的 c0/c1/c2。若框架需做阶段同步，必须预先固定并标为受控调度。
3. 在两分支下一次模型请求的输入及依赖都已确定时产生 REQUEST_READY，并在实际发送前等待同一 admission gate。至少两个 Ready 记录；此时请求尚未进入 SGLang waiting_queue。若某分支已结束、取消或仍依赖未来信息，不满足门禁。
4. 等待此前真正派发的工作完成，关闭全部其他 ingress，确认无在途请求。在 scheduler idle 回调单次捕获所有候选路径、FA frontier、pool、generation/owner 和语义前后摘要。GPU 已完成证明与 source request 因果链必须存在；不通过发请求观测 R。
5. CPU 校验证书：至少一个共同祖先 c0 被两个真实分支继承且当前驻留；每分支至少一个更深专属 c1/c2 当前驻留且身份不同、无非法 slot alias；conv/temporal 完整；P、C、A/R/T/L/compatibility 同一 barrier；无未来候选。实际 prefix 不满足继承或没有 checkpoint 时失败，不回填虚拟节点。
6. 解除 gate 后让两个已 Ready 请求按预注册原工作流完成，记录实际匹配/COW 源与目的、执行完成及退出。复用命中必须有事件证据；若更深 checkpoint 命中，应如实记录，不能为了“证明共享”驱逐深状态强迫选祖先。祖先的真实双分支复用证据可来自更早的首次分支执行，但必须已被原生事件捕获；只有 compatibility 仍不够。
7. 保存日志、只读证书和完整退出/cleanup 状态。仅一个预注册 workflow、一个指定阶段边界；超时、OOM、异常自动驱逐、身份/非干扰失败即停止并保留失败原因，不追加案例修结果。

## 分别判定两个问题

**联合事件/状态 Pilot 成功**：真实父子继承、至少两个已 Ready 请求、实际共享祖先复用证据、两个更深专属 checkpoint 与同一边界完整状态证据均通过。若只证明祖先可兼容但从未实际复用，单列“复用未证实”，不得整体通过该条件。

**有效容量竞争证书成立**：仅在预先冻结的真实资源合同下，计算 E_p(C)，对达到所有最大前沿的唯一 checkpoint 求精确 K_min；检查 K<K_min。该证书是结构检查，不是运行 baseline。要求候选 materialization 和 headroom 账目同时可实现，不能以满池 native eviction 后已丢失状态为候选。没有预算合同或 K≥K_min 时如实报告“容量竞争未建立”，即使第一项成功也不升级为有区分力 cohort。

容量必须满足每 rank 的真实峰值账目：捕获前 C+工作/临时状态不超过 pool；捕获后约定的 retained 上限 K 加工作/跟踪/COW 预留不超过 pool。可用公式 K=min_r floor((B_r-W_r-H_r)/b_r) 仅在 B/W/H 的定义、互斥关系和强制/审计方式提前成立时使用，不能先求 K_min 再指定 W/H。若要捕获超过长期 K 的候选，必须解释此前可用的工作区/临时 staging，不能双重占用同一内存。

不输出 latency 胜负、统计显著性或任何 FlowState 性能结论。不把本 Pilot 计入原 573 cases；不自动进入后续 cohort 或 GPU 比较阶段。
