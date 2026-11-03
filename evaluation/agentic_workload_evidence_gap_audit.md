# AgentX 与现有 OpenHands 实验的证据缺口审计

审查日期：2026-10-29。范围：现存来源、冻结结果、原始 JSON/CSV 及实际采集代码；未启动 GPU、Docker、模型或 Agent，未重跑策略、分词或正式实验。本轮仅新增本报告，不修改冻结数据、生产代码、Runner 或论文。

## 结论与推荐路线

**推荐继续以现有 OpenHands 的严格轨迹重放作为论文的真实来源证据，不优先重做 A/B，也不尝试用旧 AgentX 六例补充 GPU 性能结论。** 这不是因为 OpenHands 更容易使用，而是它已经具有完整消息、目标模型 Token、当前物理状态认证、真实协调和 Native Resume/Cold 对照的连续证据；AgentX 六例没有同等的输入与兼容关系基础。

- **OpenHands A/B 的科学证据增量有限。** 它直接复用 B5-C2 的 A/B 来源；在包含 C/D 的既有边界中，A/B 共享状态已物化，其中 B 已真实恢复、完成 COW 并通过 Cold 输出比较。缩为 A/B-only 可以测试更小边界与显式 Prepared-Pending 接线，但不是新的来源，也不能证明原生 Qwen Agent 工具闭环或自然并发。
- **现有本地 AgentX 归档不能恢复缺失的原始执行输入。** 六例还存在兼容矩阵与物理预算问题，不能仅靠补文件或重命名 Node 修复。原始采集方是否另有未公开完整历史为 **UNKNOWN**。
- **现在不必为了已有的受控恢复主张另建工作负载。** 若论文坚持主张原生 Qwen 工具执行、自然 Ready/Fork 或在线真实 Agent 决策，则需要新的实际执行采集；必须先有真实事件，再观察是否形成共享，不能预造有利关系。
- 已验证的 GPU 恢复计算改善不等于稳定的完整在线净收益；增加另一条轨迹本身不能补齐控制成本与计时边界的缺口。

以下判定针对各证据链，不能合并成整个系统的一次全覆盖认证。

## 1. OpenHands 已经完成的证据

|证据链及来源|已完成范围与结果|可以支持|仍不能支持|
|---|---|---|---|
|历史 Formal 分配：168 个 OpenHands 快照、504 个预算条件；另有 AgentX 23 个快照、69 个条件|573 条中 FlowState/PCB/LDR/WFC 同集、同成本；OpenHands 正收益支持可按 Pending 分离|给定旧逻辑矩阵下的结构与退化情形解释|不能统称 Strict Runtime 认证，也不能据此主张联合分配普遍优于简单策略|
|旧 OpenHands Runtime：24 快照×3 策略×3 重复|216 个生命周期、864 条请求记录齐全；逐快照相对 TTFT 改善均值：对 LRU 43.057902%，对 Marconi Adaptation 24.264401%；胜/平/负为24/0/0与22/0/2|附旧协议限制的历史受控请求级比较，审计结论 QUALIFIED|不是完整官方 Marconi 比较；未全覆盖当前 Strict/逐请求 Cold 合同；不含控制路径净收益|
|B5-C2/C3：两个真实任务、每任务两条独立执行 A/B/C/D|完整12份输入；8条历史实际回放；同一 Barrier 7个真实候选、4个受控 Pending；28边=13准入/15拒绝/0未知；支持域含 ABCD、AB、CD|实际 Token 共享、多个同时驻留的合法 Recurrent 状态及完整严格矩阵|C3 未提交 Pending；准入不是逐边 Resume 验证；不是原生 Fork/自然并发|
|B5-C4：上述7候选、K=1–4|统一成本下 Exact 与 Greedy/简单规则存在真实目标差异；K2 Exact J=48.229 ms，Greedy J=130.803 ms（模型值）|有限共享分支实例中的集合替代和 Greedy 局限|不是 GPU 时间；不是一般工作流分布结论|
|B5-D3：同一 A/B/C/D 来源，K2 Exact|新生命周期保留逻辑 N18/N27，5项真实驱逐；目标 B 从 L=3712 恢复，实际 H/E/G=4096/3712/384；COW 完成；8 Token 与独立 Cold 完全一致|共享状态的真实协调、Native Match、COW、输出和保留组件保护|不认证全部兼容边；不认证张量跨 Workflow 创建血缘|
|E2：w1–w4 四组独立来源组合|4/4物化成功，各7候选；准入边12/12/14/13，UNKNOWN均0；16个预算条件 Exact最优16，Greedy/WFC各4；Greedy/WFC 16/16持平|跨四组固定来源的严格共享与模型化分配差异|E2本身未执行 Pending，不是16个独立工作负载|
|E4-1A：E2 w1/K2|4 Pending×2策略共8/8 Resume认证，4个独立Cold；每臂独立重建、5项驱逐、7→2驻留；真实来源匹配、COW、8 Token、FA保护及12份清理记录通过|不同选集确实落实，并在独立真实来源上支持全部预定请求正确恢复|不能外推 w2–w4 的全部请求已恢复；不是性能实验|
|E4-1B：w1/K2轻量路径|4个Smoke；12个正式臂=2 Pending×2策略×3重复；A 的CUDA Prefill均值73.026→42.618 ms、TTFT102.601→74.258 ms；C同前沿负对照CUDA37.062→37.460 ms|特定请求恢复前沿变化的真实计算效果，保留负对照|3组重复不能证明生产分位数、吞吐或稳定控制后净收益|
|E5-Step1：w1/K3 Exact 对固定首改善1-Swap|8个Smoke、24个正式臂、4个Cold；三组四请求独立恢复CUDA总和差值为−13.399、+1.938、−3.102 ms|联合取舍可实际执行；强局部基线比较与反向观测均存在|方向混合，未证明 Exact 稳定优于1-Swap；独立恢复总和不是并发完成时间|

主要来源：[历史证据核验](legacy_runtime_evidence_p0_audit/final_report.md)、[T1结构审计](paper_audits/t1_structure_budget/t1_final_report.md)、[B5-C3原始严格快照](multishare_runtime_gate_b5c3/strict_epoch_snapshot.json)、[B5-C4](multishare_allocation_gate_b5c4/final_report.md)、[B5-D3](exact_reconciliation_correctness_gate_b5d3/final_report.md)、[E2](independent_runtime_allocation_e2/final_report.md)、[E4-1A](native_resume_correctness_e4_phase1a_run01/final_report.md)、[E4-1B](native_resume_performance_e4_phase1b/final_report.md)、[E5](exact_swap_runtime_e5_step1/final_report.md)。

本轮另直接检查 C3 快照的7候选/4 Pending/28边及未派发标志、D3 的8个输出 Token 逐项一致记录、E4 的8行真实恢复认证；读取 E2 的112行兼容矩阵和96行策略结果、E4-1B 的12行原始计时、E5 的24行原始计时。上述行数是产物粒度，不增加独立样本数。未重跑旧完整审计。

### 物理映射与恢复的准确含义

既有严格路径以完整 Token Prefix 对齐逻辑候选，在每个新 Engine 重新读取 Node、Slot、Conv/Temporal 切片身份、引用和槽位账本，经过 Bind、Preflight、Apply-time 与逐驱逐 Postcheck；**历史 Node 编号不是跨生命周期物理绑定**。E4 中同名 `B5C3_NODE_*` 是沿用的逻辑命名，不代表仍使用 B5-C3 的任务或旧物理对象。

协调阶段验证 FA 索引及内容不变；请求执行后允许正常的新节点、工作状态和访问元数据变化，不要求整棵树机械不变。Native Match/COW/Cold 的认证只覆盖实际执行的 Arm；不能把矩阵中全部 EPOCH_ADMISSIBLE 边升级为 RESUME_VALIDATED。所有相关案例仍保留 **PROVENANCE_UNVERIFIED**。

现有受控性能证据也保留负面结果：B5-D4b 五对中，Pending C 的 CUDA Prefill均值78.348→46.099 ms，五对均改善，但客户端 TTFT1039.121→1078.325 ms，五对均恶化；完整控制至完成13219.006→13135.119 ms仅3/5对改善。来源：[原始五对报告与CSV](exact_greedy_paired_gpu_gate_b5d4b/final_report.md)。不能以模型分数或抽掉估算审计成本声称生产净收益。

## 2. 拟议 A/B 到底新增什么

对照 [P1-B可行性报告](agentic_executable_workload_feasibility.md)，A/B 来自同一任务 `0b01001001__spectree-64` 的两次独立 OpenHands 执行：

- A：`chatcmpl-71420ffc5b410547d43ac67939b0441c`，Parquet零基行10918，原消息97条。
- B：`chatcmpl-71b073b89097665d14501919b224c26b`，行12519，原消息99条。
- A/B首请求均3728 Tokens；第二请求4182/4150；第三次受控 Pending4862/4860。第三次输入只使用索引0–5的历史消息。实际原生生产模型、墙钟 Ready/Fork事件仍 UNKNOWN。

|拟议任务内容|现有覆盖|真正增量与边界|
|---|---|---|
|原消息→Qwen输入的来源及因果截断|B5-C2保存原历史、配置锁、完整数组与工具调用顺序；P1-B已逐条对原Parquet复核|重复确认可用于接线回归，但不构成新来源或新模型运行|
|A/B真实Token共享|A1/B1全数组相同；第三输入LCP=3765；C3中AB状态实际L=3712|不能把LCP直接当Checkpoint；缩小到A/B-only后的新物化结构尚未证明|
|物理映射、严格边、驱逐与B恢复|C3已有完整状态；D3已经执行同一B第三次输入的Native Resume/COW/Cold|A/B-only是新边界，需要新认证，但属于最小化工程复现，不补充新的机制|
|Prepared→分配→协调→发送事件账本|旧输入已冻结，受控输入就绪和执行分离已有；P1-B提出更显式统一账本|可增强可审计接入，不能恢复原生产自然Ready时间，也不变成原生Agent执行|
|非平凡预算竞争或策略收益|四Session及w1已有实测；仅AB的候选域和竞争未认证|不能保证AB-only有更强区别，不应以得到区别为通过标准|
|原生Qwen工具闭环、真实分支/同时Ready|现有重放不具备|A/B重放仍不具备；必须真实接入执行与采集|

截至本轮，`agentic_executable_workload_cpu_gate/` 仅有未完成适配脚本，尚无完成的准备产物、测试结果或最终CPU门禁报告；不得将此前请求或代码文件存在描述为已通过。当前任务已暂停该实现，不继续补做。

**判断：A/B不是完全无价值，但它的主要增量是缩小原型和补接入事件，不是论文目前缺少的独立 Agentic 科学证据。** 已有 P1-B 的 PARTIAL 指可进入适配的潜力，不代表必须追加此GPU实验。

## 3. AgentX 六例：原始数据能否找回

### 3.1 可恢复与不可恢复的信息

本地原始来源 `/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl` 包含393条trace，块大小64、hash作用域local，保存调用时间、长度、块身份和部分subagent记录；不含完整 Prompt、消息或真实 Token 数组。完整文件身份和六例的原始行见 [P1-A](agentx_real_trace_gpu_feasibility_audit.md)。

本轮检查该来源目录、AgentX外层旧脚本/日志及六个 `per_snapshot/.../snapshot.json`：对六份JSON递归检查 `token_ids/input_ids/prefix_token_ids/messages/prompt/input_token_ids`，均未找到相应完整输入字段。旧快照有长度、摘要、Node与驻留信息，但不具备当前Strict所需完整输入及组件切片绑定。这个结论限定于已检查的现存归档；未声称其他机器或原始采集方绝无备份。

实际调用链：

1. `agentx_structure_audit.py` 与 `agentx_chain_detection.py` 从完整trace的hash、时间及seam推导逻辑链，部分继承/分支关系为事后派生，并非原生Fork记录。
2. `agentx_qwen_replay_preflight.py::_compose_prompt_tokens`（194行）将块身份映射成确定性Token并裁剪/补尾；能复现**合成器输入**不等于找回原始输入。
3. `_request_index_for_target`（228行）可能指向最后完成请求，不是已认证的下一次Prepared Continuation。
4. `agentx_runtime_snapshot_collection.py::_build_replay_plan`（275行）及 `_collect_snapshot`（654行）确实曾执行合成历史并观察物理状态；Raw边取实际合成Token前缀。
5. `agentx_runtime_final_correctness_audit.py::logical_compatible_pending_ids` 和 `rq3_agentx_formal_policy_evaluation.py::encode_usable_compatibility/build_agentx_formal_snapshots` 将逻辑祖先兼容用于Formal分配，不能等同Raw前缀关系。

因此原始调用元数据、旧Formal成本和部分旧物理观测可核查；原Prompt、原Token、当时自然Pending与完整物理创建血缘不能从hash、摘要或模型化收益无损推回。重新下载相同来源并不会自动恢复它未包含的内容。

### 3.2 六例的具体阻断

六例各为一个不同trace的K1条件，均K_min=2；Greedy/Exact同集。下列J为旧Formal模型值，单位ms，**不是GPU测量**。

|快照|候选/不同Node|J_WFC / J_Exact|兼容与物理障碍|
|---|---|---|---|
|g005-epoch8|8/8|2573.278 / 620.027|WFC两条、Exact一条选中Formal正边均Raw假|
|g006-epoch35|35/34|10305.793 / 9166.410|选中正边Raw真，但Exact深状态还Raw支持另一Pending；Formal漏计，完整域有39条Raw-only边；另有Node别名|
|g010-epoch38|38/36|6044.830 / 5611.522|WFC/Exact选中目标边均Raw假，另有别名|
|g017-epoch12|12/11|6501.556 / 4655.433|WFC共享边及Exact目标边Raw假，另有别名|
|g018-epoch30|30/28|10436.817 / 9021.082|WFC共享边Raw真，Exact深层目标边Raw假，另有别名|
|g021-epoch149|149/149|12519.081 / 7254.040|Exact目标边Raw真，WFC共享边Raw假；149候选无法原样套入近期28槽配置|

来源：[冻结184条件扫描](paper_workload_structure_final/agentx_budget_scan.csv)、[Formal/Raw差异表](gate1_step3_compatibility_retention_contract_audit/compatibility_diff.csv)、P1-A第2–4节及其逐例原始快照索引。Raw只是当次合成输入的前缀检查，不是逐请求Native恢复认证。

全184条件Greedy最优184、WFC178、StandaloneTopK157，可作为该旧Formal矩阵内结构分析；不能将六例的收益差包装为真实执行收益。K1下Greedy就是独立总收益最大化，这六例也不能证明动态边际更新优于StandaloneTopK。

**找回数据后的资格也不是自动恢复。** 即使原采集方提供完整消息，仍须重新确定目标模型Token与Pending因果边界、按实际物理前缀重建全部关系、处理多逻辑ID共Node及容量/COW余量。`flowstate/strict_control.py::bind`（141行）拒绝重复Node/槽组件等不安全映射。旧池为192槽，近期配置为28槽。去重、改边或重新物化都可能改变候选、K_min及胜负，必须作为新问题，不能维持旧六例结论不变。

**当前判定：旧六例忠实Strict GPU复现为 BLOCKED；能否获得外部未归档原始输入为 UNKNOWN。** 不建议把它当作短期只需修补映射的有限接线任务。

## 4. 三条路线的证据价值与成本

|路线|必要新增工作|能补充的主张|不能补充/风险|推荐|
|---|---|---|---|---|
|使用已有OpenHands严格实验|整理来源、认证等级、有限样本和负面结果；无需新执行|真实来源输入在Hybrid Runtime中物化、严格集合分配、真实协调与恢复；有限请求计算对照|自然并发、原生Qwen Agent、全域逐边认证、稳定在线净收益仍无证据|**当前论文优先**；已存在可验证证据链，成本最低|
|修复旧AgentX六例|向原始方取得无损历史/配置/边界；重新认证Token、完整矩阵、物理域与预算，再另行审查GPU资格|若成功可增加来源多样性|现有本地数据不足；原差异可能消失；工作量与外部数据可得性UNKNOWN|不作为当前投稿证据的关键依赖；仅保留限定的Formal分析|
|新实际Agent执行采集|接通模型—工具真实循环，保存全消息/Token和准备事件，观察真实共享；之后才认证与比较|仅此路线有机会补原生Agent工具闭环、自然Ready/真实Fork证据|框架源码存在不等于环境已就绪；未必自然出现共享竞争；不保证算法收益|仅当论文必须保留这些新增主张时必要；本轮不启动|

另建人工共享分支只能增加受控结构样本，无法填补原生Agent证据；把AgentX hash重新渲染成合成输入也属于新的受控问题。两者都不能作为“恢复了原始历史”来报告。

## 5. 已充分与仍缺失的证据

|论文层面问题|当前资格|依据/限制|
|---|---|---|
|真实来源历史是否具有完整可审计消息和目标Token|已有，限定于固定OpenHands截断历史|C2来源与配置锁；不含原生产模型及自然Ready时间认证|
|Token共享是否可变成同时驻留的Recurrent共享|已有，限定于已采集Barrier|C3、E2物理Census和严格完整矩阵；不是仅LCP推断|
|不同选集能否安全落实并正确恢复|已有，限定于实际通过的策略—Pending臂|D3、E4、E5的真实驱逐、FA保护、Native Match/COW及Cold；不推及未执行边|
|简单规则何时不足、Exact何时改善模型目标|已有有限实例及反例|Strict20、E3；Greedy/WFC大量持平，1-Swap18/20最优不能省略|
|分配改变是否有真实计算影响|已有有限请求级证据|E4的正/负对照；E5总体方向混合；不宣称通用性能优势|
|三策略完整在线控制周期是否稳定净获益|本审查所列OpenHands证据不足|请求TTFT、CUDA、审计完整窗口不同；新A/B或更多来源本身不能解决计时与控制成本问题|
|原生Qwen Agent、自然Ready、原生Fork|缺失|既有历史Assistant/Tool为源轨迹内容，不是Qwen真实工具闭环|
|共享张量的跨Workflow物理创建血缘|仍未认证|保持PROVENANCE_UNVERIFIED；Token相同、首次观察与Native Match不等于血缘认证|
|AgentX六例的完整输入与正确物理问题|缺失且有明确矛盾|非可逆hash来源、Formal/Raw冲突、别名与容量条件|

**唯一建议的当前动作：按来源及认证等级整合已有 OpenHands 证据并收窄论文主张，不追加 A/B-only GPU 重复来填补并不存在的基础恢复缺口。** 若坚持原生Agent主张，应明确承认现有证据不够，再单独决定真实执行采集；不以构造共享或重命名历史关系替代它。本轮未安排或执行后续任务。

## 6. 核验范围与证据索引

- [P1-A详细来源、六例表、逐文件摘要及调用链](agentx_real_trace_gpu_feasibility_audit.md)；[P1-B完整消息与环境核验](agentic_executable_workload_feasibility.md)。原始来源分别为 `/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl` 与 `/home/wjg/data/agentic_coding_trajectories/sessions.parquet`。本轮复用其既有全文件摘要核验，不重复扫描两个完整来源计算摘要。
- [C2输入目录](multishare_cpu_input_gate_b5c2/)、[C3历史回放账本](multishare_runtime_gate_b5c3/historical_replay_log.json)、[C3物理Census](multishare_runtime_gate_b5c3/physical_state_census.json)。本轮直接读取的 `strict_epoch_snapshot.json` SHA-256：`3b41f232b4ee4d79aab081643c6c2adad1671d0364841617e849f52b6c16ac7a`。
- [D3实际输出](exact_reconciliation_correctness_gate_b5d3/output_correctness.json)，本轮SHA-256：`095e64f3419635b289512d739fb950041771303442b21be4a015e6e8ddf56b35`；[E4逐臂认证](native_resume_correctness_e4_phase1a_run01/resume_correctness.json)，SHA-256：`a8b1577817729df066e2a6384ebbdb21ccb3ffb60d0f6eae6b82276be89c7ab4`。这些是产物身份，不认证未来GPU物理状态。
- [E2物化汇总](independent_runtime_allocation_e2/runtime_materialization_summary.csv)、[E2完整比较](independent_runtime_allocation_e2/allocation_comparison.csv)、[E4原始计时](native_resume_performance_e4_phase1b/gpu_raw_timings.csv)、[E5原始计时](exact_swap_runtime_e5_step1/gpu_raw_timings.csv)。
- 主要代码：[OpenHands消息截断入口](rq3_openhands_neutral_collector.py:236)、[AgentX合成输入](agentx_qwen_replay_preflight.py:194)、[AgentX目标请求选取](agentx_qwen_replay_preflight.py:228)、[AgentX采集](agentx_runtime_snapshot_collection.py:654)、[严格物理绑定](../flowstate/strict_control.py:141)。
- Synthetic Sequential/Multi-Workflow/C-K3 与 T2 三策略实验可用于其各自受控机制及性能问题，**不能计为额外OpenHands/AgentX真实来源样本**；本轮未接触其他智能体的P0-4执行路径，也未以其设计或计划作为完成证据。

结论到此为止：保留已有严格真实来源重放链，限定旧Formal与历史TTFT的资格；不自动新增工作负载或GPU实验。
