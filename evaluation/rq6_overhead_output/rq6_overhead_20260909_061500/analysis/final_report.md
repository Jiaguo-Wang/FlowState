# RQ6 FlowState 控制面开销评估

## 技术摘要

P0 已确认 RQ2 canonical recovery model，RQ6 随后通过全部 runtime 与 instrumentation 门禁。191 个冻结 allocation snapshots 形成 573 个唯一 snapshot-budget case，每个 case 计时 3 次。CPU 状态构造加全局分配的直接总计时平均为 0.313745 ms，P50 为 0.291330 ms，P95 为 0.514072 ms。

24 个 RQ4 runtime snapshots 各进行 3 次独立 Engine 生命周期运行，共 72 次。完整控制路径平均为 8398.846825 ms，P50 为 5617.014251 ms，P95 为 26811.051527 ms，最大值为 55185.557138 ms。均值中 reconciliation 占 77.19%，introspection 占 22.78%；纯状态构造与分配仍为亚毫秒级。

## 冻结工作负载与计时定义

- CPU：OpenHands 168 个快照、AgentX 23 个快照，budgets 为 25% / 50% / 75%，按 RQ3 规则去重相同 K。
- CPU：573 个唯一 case × 3 次 = 1719 行。
- Runtime：RQ4 冻结 24 个 OpenHands 快照，25% budget，24 × 3 = 72 行。
- 环境：Qwen3.5-9B、SGLang 0.5.17、单张 NVIDIA H100 PCIe。
- 计时器：`perf_counter_ns`；原始单位 ns，汇总单位 ms。
- `T_total_control` 围绕连续控制路径直接计时，不由四阶段均值相加得到。
- 明确排除 prefill、decode、recurrent recovery computation 与正常请求 TTFT。
- scaling 仅使用冻结 population 的自然范围，未使用 synthetic case。

四阶段分别为：只读 runtime introspection；executable-state construction；冻结 set-dependent marginal global allocation；recurrent-only runtime reconciliation 与操作后 residency 验证。

## Runtime 阶段分解

| 阶段 | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| Introspection | 1913.301282 | 1213.562999 | 6788.978964 | 14259.581384 |
| State construction | 0.204297 | 0.208156 | 0.348426 | 0.373182 |
| Allocation | 0.747152 | 0.523442 | 1.608744 | 1.805219 |
| Reconciliation | 6483.255894 | 4118.309375 | 20303.287132 | 40922.380499 |
| Direct total | 8398.846825 | 5617.014251 | 26811.051527 | 55185.557138 |

直接 total 在 72/72 runs 中均不小于同一 run 的四阶段之和。差值平均为 1.338200 ms，范围为 0.431975–2.078943 ms，证明 total 保留了阶段间控制逻辑，而非事后相加的 microbenchmark 估计。

## CPU 决策路径

| Population | 记录数 | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| OpenHands | 1512 | 0.312508 | 0.298857 | 0.510705 | 0.636200 |
| AgentX | 207 | 0.322781 | 0.121063 | 1.399516 | 2.377367 |
| 合计 | 1719 | 0.313745 | 0.291330 | 0.514072 | 2.377367 |

合并 CPU 样本中，construction 均值为 0.035797 ms、P95 为 0.050574 ms；allocation 均值为 0.277133 ms、P95 为 0.465128 ms。因此 runtime 秒级总开销并非来自优化计算本身。

## 自然 workload scaling

观测范围为 |P_t|=2–8、|C_t|=6–149、K=1–111。CPU allocation 与 |C_t| 的 Spearman ρ=0.872736，与 K 的 ρ=0.609025，与 |P_t| 的 ρ=0.052224。Runtime 的 |P_t| 固定为 4，无法识别 total 与 pending count 的关系；runtime total 与 |C_t| 的 ρ=0.810536。

| Runtime candidate count | 记录数 | Mean total (ms) | P95 (ms) |
|---:|---:|---:|---:|
| 8 | 18 | 2312.841293 | 4078.458086 |
| 12 | 18 | 4403.421200 | 7252.609199 |
| 16 | 18 | 9547.776654 | 20065.250142 |
| 20 | 18 | 17331.348151 | 45115.351790 |

这些相关性只描述所测自然 population，不证明算法复杂度或候选数的独立因果效应。

## 正确性、测试与限制

- 72/72 runtime runs 为 PASS；handle mapping、recurrent residency、FA preserved、fresh-engine empty 均为 PASS。
- native recurrent eviction contamination、unexpected rematerialization、FA cascade、OOM、truncation、future leakage 均为 0。
- cross-run contamination 为 NO；每次 worker 退出后 GPU 稳定回到 14 MiB，且无 compute process。
- instrumentation 与 reference 的 selected set 为 1791/1791 一致；H/E/G、snapshot immutability 与 online boundary 均为 PASS。
- 正式 runtime source before/after SHA256 为 PASS；六个冻结输入文件的 SHA256 无变化。
- 复合记录键重复数为 0；四次诊断 preflight 未进入 1791 行正式 JSONL。
- RQ6 相关测试：147 passed。
- full CPU suite：866 passed、2 skipped、2 warnings；未新增 skip、xfail 或删除测试。

第一次 full suite 在真实 AgentX audits 只读挂载下有 2 个与 RQ6 无关的写权限失败；第二次空写层复验隐藏了下游输入并产生 3 个环境失败。两份诊断日志均保留。最终 suite 使用包含完整冻结输入副本的 `/tmp` 可写隔离层，真实数据继续只读，结果为 866 passed。测试临时改写的历史生成报告已逐字节恢复。

Runtime total 呈明显右偏且三个重复间存在波动，因此论文必须同时报告 mean、P50、P95 与 max。Runtime 的 pending count 恒为 4，不能外推 runtime pending-count scaling。结果只覆盖 Qwen3.5-9B、SGLang 0.5.17、H100 PCIe 与当前冻结 population。

## 论文冻结结果

RQ6_SYSTEM_OVERHEAD_READY

formal workload = RQ3 OpenHands 168 + AgentX 23；RQ4 runtime 24
CPU snapshots = 191
runtime snapshots = 24
runtime repetitions = 3

Mean/P50/P95 total control overhead = 8398.846825 / 5617.014251 / 26811.051527 ms
Mean/P95 introspection overhead = 1913.301282 / 6788.978964 ms
Mean/P95 state-construction overhead = 0.204297 / 0.348426 ms
Mean/P95 allocation overhead = 0.747152 / 1.608744 ms
Mean/P95 reconciliation overhead = 6483.255894 / 20303.287132 ms

largest observed |P_t| = 8
largest observed |C_t| = 149
allocation scaling observation = CPU allocation 与 candidate count 呈强正秩相关（ρ=0.872736），与 K 呈中等正秩相关（ρ=0.609025）；runtime total 与 candidate count 的 ρ=0.810536。

runtime correctness = PASS
instrumentation equivalence = PASS
frozen artifacts modified = NO
tests = RQ6 related 147 passed
full CPU suite = 866 passed, 2 skipped, 2 warnings

paper-ready conclusion = “在 191 个冻结 allocation snapshots 上，FlowState 的状态构造与全局分配平均仅需 0.314 ms（P95 0.514 ms）。在 Qwen3.5-9B / SGLang 0.5.17 / H100 上的 24 个 runtime snapshots、每个 3 次独立重复中，包含只读 introspection 与 recurrent-only reconciliation 的完整控制路径平均为 8.399 s（P50 5.617 s，P95 26.811 s），其中 reconciliation 与 introspection 分别占均值的 77.19% 和 22.78%。全部 72 次运行通过 selected-set、H/E/G、residency、FA-preservation、future-leakage 与跨运行污染门禁。自然 workload 中 allocation 时间主要随 candidate 数增长（Spearman ρ=0.873）；runtime pending 数固定为 4，因此不对 runtime pending-count scaling 作外推。”

本任务未启动 RQ5-B、Kimi 或其它实验。
