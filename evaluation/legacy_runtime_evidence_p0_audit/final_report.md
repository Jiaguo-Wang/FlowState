# 旧版强基线证据核验

## 总结论：QUALIFIED

旧版两项 TTFT 改善可保留为**明确限定范围的历史受控比较**，不能直接作为当前严格兼容控制闭环、完整官方 Marconi 系统或控制成本后净收益的证据。数值可复现与执行兼容认证是两个不同判断。

本轮仅读取源码和冻结产物，新增独立核验脚本及报告；未启动 GPU、Engine 或策略实验，未修改生产代码和历史产物。

| 审计对象 | 结论 | 论文可用范围 |
|---|---|---|
| 216 个生命周期、864 条请求及逐快照 TTFT 汇总 | KEEP | 原始记录完整性与描述性统计 |
| FlowState 对 LRU 的 43.06%、对 Marconi Adaptation 的 24.26% | QUALIFIED | 旧版协议、协调完成后、单输出 Token 的受控请求级 TTFT |
| Marconi Adaptation 的具体冻结启发式 | QUALIFIED | 明示增量 Token 跨度代理、元数据限制、α=1 的受控适配 |
| 胜过完整官方 Marconi 系统或一般 SOTA 的主张 | NOT USABLE | 未执行等价官方系统，不能由本适配推出 |
| 当前 Strict Epoch 全覆盖认证、逐请求恢复正确性和完整来源血缘 | NOT USABLE | 历史证据未满足这些认证条件 |
| 控制成本后端到端净收益、生产延迟或吞吐收益 | NOT USABLE | 两项百分比的计时不包含控制路径 |
| 573 条冻结 Formal 条件上的集合等价和结构统计 | KEEP | 给定旧逻辑矩阵下的模型化结果 |
| 旧 Formal 的策略成本差异 | QUALIFIED | 明示矩阵并非严格 Runtime 认证，不作为实际性能 |

## 一、原始证据及完整性

权威实验根目录为 `rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652/`，不是后续 E 系列或其他控制实验。

本轮独立核对 `frozen_protocol.json`、`manifest.json`、`run_plan.json`、`runs.jsonl`、`source_integrity.json`、各生命周期目录及原分析入口。逐项结果见 [lifecycle_inventory.csv](lifecycle_inventory.csv)。

- 24 个快照 × 3 策略 × 3 次重复 = **216 个不同生命周期**，计划、总账和运行目录的身份集合一致。
- 每个生命周期有 4 条 Pending，共 **864 条请求**；逐运行结果与总账一致。
- 每个运行的 `correctness.json`、`selection.json`、`request_telemetry.jsonl`、`replay.json`、`runtime.log`、`result.json`、`pending_requests.jsonl`、`handle_mapping.json`、`reconcile.json` 均存在且非空；216 份 worker 日志均存在。
- worker 日志记录使用容器内 `/output/...` 路径，本轮按同一运行 ID 映射到本地实验根目录；不将容器绝对路径在宿主机不存在误报为数据缺失。
- 所有运行记录通过旧版正确性门禁，退出码为 0、无 shutdown_error，运行前后资源稳定检查通过。此处是核对历史记录，不是重新检查 GPU。
- 冻结快照 canonical 内容摘要、选集前后快照摘要、预算、选中 ID 及同快照跨策略请求顺序和输入签名一致。

冻结抽样按 round 2/3/4/5 各 6 个、目标长度分层及固定散列规则选择，不根据策略胜负挑选；策略执行顺序使用冻结交错规则。样本单位是 24 个快照，不能把 216 个生命周期或 864 条请求视为独立工作负载。

**版本溯源限制：**历史清单记录 7 个源文件执行前后摘要相同；当前 6 个仍相同，`flowstate/optimizer.py` 已不同。后续生产修改不能反推历史被篡改，但完整旧优化器源码重建在本轮为 UNKNOWN。该清单也未覆盖全部 Adapter、SGLang 安装源码及策略模块，故不能宣称完整历史可执行环境已由内容锁闭合。本轮验证了冻结快照及已读产物摘要；未重新对来源 Parquet 全文件做独立摘要认证。具体已核验文件及未变检查见 [read_integrity.json](read_integrity.json)。

## 二、TTFT 独立复算

本轮直接从逐请求 `ttft_ms` 计算：

1. 每次运行对 4 个 Pending 求均值；
2. 每个快照、策略对 3 次重复求均值；
3. 对 24 个快照等权汇总。

改善百分比是 **24 个快照各自相对改善率的平均**：

`mean_s[(TTFT_baseline,s − TTFT_FlowState,s) / TTFT_baseline,s]`。

不能将其写成三个总均值之间的比值。逐快照结果见 [snapshot_ttft.csv](snapshot_ttft.csv)。

| 比较 | 基线平均 TTFT（ms） | FlowState 平均 TTFT（ms） | 平均绝对降低（ms） | 平均逐快照相对降低 | 快照胜/平/负 |
|---|---:|---:|---:|---:|---:|
| LRU | 164.705002 | 94.024580 | 70.680422 | **43.057902%** | 24/0/0 |
| Marconi Adaptation | 129.097170 | 94.024580 | 35.072590 | **24.264401%** | 22/0/2 |

按原冻结分析函数复算的分层快照 bootstrap（种子 20260907，10,000 次）95% 区间：

- 对 LRU：绝对降低 [63.806243, 78.300269] ms；相对降低 [39.940314%, 46.066374%]。
- 对 Marconi Adaptation：绝对降低 [27.670502, 43.174819] ms；相对降低 [17.740919%, 31.321582%]。

点估计为本轮独立汇总；区间复算复用了原分析函数，不宣称是另一套独立统计实现。区间仅对应该分层样本设计。

两项负面结果均保留，未筛掉：

| Snapshot | FlowState TTFT（ms） | Marconi Adaptation TTFT（ms） | FlowState 相对改善 |
|---|---:|---:|---:|
| rq3-openhands-main-g015-round5 | 54.200754 | 42.840265 | −26.518252% |
| rq3-openhands-main-g184-round2 | 212.166711 | 191.886575 | −10.568815% |

原分析报告还保留运行漂移警告，以及 72 个快照—策略组合中 2 个 CV>0.20 的组合；没有据此删除样本。可复现百分比不能消除有限重复和运行漂移限制。

**计时边界：**`evaluation/sota_latency_runtime.py:137` 的 `measure_streaming_request` 从调用 `engine.generate` 前计时，到首次返回输出 Token；该实验生成 1 Token。`evaluation/rq4_unified_runtime_harness.py:680` 之后才逐项提交 Pending，选择、协调和前置状态检查已经完成。因而这是请求接口可见 TTFT，包含该调用内的传输、排队和执行，**不包含历史重建、观察、选集求解、Reconciliation 及前置检查**，也不是 CUDA Prefill 时间。不能用这两个百分比表述完整控制路径净收益。

## 三、公平性与真实控制能力

真实调用链：

`冻结 snapshot → select_frozen_policy → 新 Engine 历史回放 → validate_rebuilt_snapshot → 当前 handle 映射 → FrozenSelectedSetOptimizer → StateController.reconcile → 映射/驻留检查 → 四个串行 Pending → 请求 telemetry`

主要源码依据：

| 源码及位置 | 实际作用与边界 |
|---|---|
| `evaluation/rq4_unified_runtime_harness.py:146`，`select_frozen_policy` | 重复选择一致、冻结摘要不变、合法 ID 和预算检查 |
| 同文件 `:201`，`validate_rebuilt_snapshot`；`:525–550` | 核对重建逻辑快照，重新建立当前生命周期物理 handles |
| 同文件 `:600–675` | 通过同一 Controller/Adapter 路径执行候选补集驱逐，保存前后观察 |
| `evaluation/openhands_policy_to_actuator_mapping_gate.py:272`，`evaluate_mapping_invariants` | 保留/驱逐集合、Node 路径、槽位差额、FA/树结构及实际操作集合核验 |
| `evaluation/rq4_unified_runtime_harness.py:359`，`_executable_checkpoint` | 按同 Workflow、位置及驻留信息推断来源；不是直接 Native 来源追踪 |
| 同文件 `:964`，`_run_correctness` | 汇总旧版正确性条件；没有 Cold 输出一致性合同 |

三种策略使用同一冻结候选域、Pending、模型配置、逻辑预算及物理协调入口。候选数 8/12/16/20，`K=max(1,floor(0.25n))`，即 2/3/4/5。每个运行从独立空缓存 Engine 重建历史，当前映射重新定位，不要求跨生命周期 Node/Slot 编号相同。资源为 Qwen3.5-9B、SGLang v0.5.17、H100 PCIe、TP=1、28 个 Recurrent Slots。

**可以确认的等价性：**逻辑快照、输入、初始候选驻留和可观察缓存结构按旧门禁等价；选集补集确实通过真实 Recurrent-only 控制路径执行，不是仅在 CPU 上删 ID。无物理容量压力时也按选集执行控制；不能把 K 当作 HBM 耗尽证明。

**不能确认的等价性：**逐组件字节内容、全部存储切片身份、完整跨操作隔离及当前 Strict Apply-time 新鲜度合同未被旧日志完整认证，标记 UNKNOWN。缺证据不等于已经发现控制错误。

四个 Pending 在每个生命周期串行执行。后续请求可能受先前请求合法物化的新状态影响；旧检查主要核对当前 Workflow 候选，不等价于每个 Pending 均从独立原始 Barrier 恢复。三策略固定顺序提供相同实验程序，但不能把四个测量解释为互不污染、只由原始 S 决定的独立恢复实验。

## 四、LRU 与 Marconi Adaptation 的确切范围

LRU：`evaluation/controlled_multiworkflow_v1/policies.py:20`，按 last_access_order 降序、creation_order 降序、ID 升序保留预算内候选。它是受控候选域的保留策略，不是完整服务系统的原生逐次 LRU 管理器。

Marconi：`evaluation/sota_policies.py:101,216` 和 `evaluation/rq3_frozen_snapshot_evaluator.py:836` 实际使用：

`utility(c)=minmax(last_access_order(c)) + α·minmax(flop_saved(c)/memory_bytes(c))`，

按 utility 降序、ID 升序选 K 项。24 个快照的 α 均为 **1.0**。本轮独立从冻结元数据重算所有 **72/72** 个 Marconi 运行选集，与原结果一致；LRU 同样逐运行按其原规则重算。

`evaluation/sota_metadata.py:106` 的 `build_marconi_flop_saved` 使用同 Workflow/Lineage 中最近较浅候选到当前候选的 Token 跨度。本轮逐项核验此代理与冻结值一致。它不是针对 Qwen3.5 GatedDeltaNet 的实际省 FLOPs 测量。

`evaluation/rq3_openhands_neutral_collector.py:480–510` 的访问元数据依据创建序及可观测的同 Workflow 恢复位置更新，不是完整原生 Cache 的跨 Workflow 访问、插入和分裂历史。实际路径使用原始 last_access_order 数值，不应误写为另一 recency 排名转换函数。

因此：

- 实现、输入、α 和同分规则明确，可复核**该启发式**的公平受控比较。
- α=1 的独立调参或 bootstrap 依据、全部原生访问元数据闭合性为 UNKNOWN。
- 官方系统的混合缓存资源管理、原生节点驱逐资格和计算效率模型未被完整复现；已有 `evaluation/sota_baseline_gate_b5b0/marconi_input_gate.md` 也记录这些适配差异。
- 可称“本文定义的 Marconi-inspired/Marconi Adaptation 启发式”，必须给出公式及代理定义；不可省略“Adaptation”而写成击败官方 Marconi。

## 五、旧恢复合同与当前 Strict Epoch

| 条件 | 旧实验已有证据 | 当前认证缺口 |
|---|---|---|
| 输入和快照 | 冻结请求重建、签名、逻辑摘要与因果边界检查 | 每条原始 Runtime 完整 Token 数组及完整执行配置内容锁未按当前合同保存 |
| Compatibility | Workflow/Lineage 逻辑关系 | 不能自动升级为完整逐边 EPOCH_ADMISSIBLE |
| 物理控制 | 当次 handles、选集驻留、补集释放、槽位与 FA 结构检查 | 完整组件切片身份、内容不变、连续 Strict 控制窗口 |
| 实际恢复 | Runtime H/E/G、请求完成、推断候选来源 | 直接 Request-to-Node/Handle 及 Native COW 完成链 |
| 输出正确性 | 1 Token 请求成功、无记录异常 | 独立 Cold 逐 Token 对照与历史源张量保护 |
| 血缘 | 逻辑 Workflow 标签 | 物理张量来源为 UNKNOWN，不能宣称已认证跨 Workflow 血缘 |

本轮发现 E>0 且旧推断来源为空的请求数为 0；这只是旧辅助字段完整，**不构成直接来源认证**。后来的 B1/R2/E4 个案不能追溯性认证旧 864 条请求。

## 六、旧 Formal allocation 仍能保留什么

引用 `evaluation/greedy_swap_structure_e3_step1/legacy_573_structure.csv`、`legacy_summary.json` 及已有等价审计，不启动旧大实例 Exact 枚举。

| 数据集 | 冻结条件数 | 共享正收益条件数 | K<K_min 条件数 | 可保留结构解释 |
|---|---:|---:|---:|---|
| OpenHands | 504（168×3） | 0 | 90 | 正收益按 Pending 可分；同链重叠不等于跨 Pending 联合优势 |
| AgentX | 69（23×3） | 66 | 0 | 存在逻辑共享，但预算均覆盖全候选最优前沿 |

FlowState、PCB、LDR、WFC 在 **573/573** 条件最终集合、E/G 和模型化成本一致。这是保留负面结果的重要结论，不能宣称旧正式矩阵已经证明 Global Allocation 对强简单策略的独立优势。

`evaluation/p4_evaluation_restructuring/overall_allocation_matrix.csv` 中可作为条件性模型化证据保留的平均恢复成本如下，单位为模型预测 ms，绝非实测 TTFT：

| 数据集 | 预算比例 | FlowState | LRU | Marconi Adaptation |
|---|---:|---:|---:|---:|
| OpenHands | 25% | 129.504564 | 428.870335 | 321.141758 |
| OpenHands | 50% | 3.370698 | 146.387968 | 121.429326 |
| OpenHands | 75% | 3.370698 | 53.833812 | 29.739345 |
| AgentX | 25% | 7044.031074 | 8009.159326 | 7261.507014 |
| AgentX | 50% | 7044.031074 | 7238.294567 | 7057.545877 |
| AgentX | 75% | 7044.031074 | 7130.113983 | 7044.571595 |

上述均来自旧逻辑矩阵。AgentX 已知 70 条 Formal-only 问题未由本轮修复；OpenHands 原始严格 Token 证据覆盖缺口也未消失。可保留为旧抽象模型的计算结果，不能作为真实可执行恢复收益或当前严格矩阵认证。24 个 Runtime 快照与 168 个 Formal 快照的平均值也不能混算。

## 七、论文写法与唯一补充动作判断

可直接使用的限定表述：

> 在 24 个预先选定的 OpenHands 受控回放快照、三次重复和统一逻辑候选预算下，旧版 FlowState 在完成状态协调后的单 Token 请求级 TTFT，相对全局 LRU 及本文明确实现的 Marconi Adaptation，平均逐快照降低 43.06% 和 24.26%。对应胜/平/负为 24/0/0 与 22/0/2。该测量不计入控制成本，适配基线并非完整官方系统，历史恢复验证不等同于当前 Strict Epoch 和逐请求 Cold 正确性认证。

**是否必须补实验：**保留以上 QUALIFIED 历史结论不必新增 GPU 实验；应先准确限定论文范围。不能以一个新 Smoke 将整个旧实验升级为严格认证。

若论文必须将这两项百分比作为“当前严格控制闭环下的强基线性能”主证据，唯一必要的补充方向是：**对原固定 24 快照、三种冻结策略做同范围严格正确性及计时复核**，补齐原始输入身份、当次物理绑定、直接恢复来源/COW、Cold 输出及串行请求状态边界。新复核的结果须独立报告，不能借其通过给旧百分比追授认证。单个来源个案或只补一个负例不足以闭合此范围；完整官方 Marconi 的公平性也不能由这项恢复实验自动解决。

本轮不安排、不执行该补充实验。没有依据建议新工作负载或新算法。

## 复核入口

`/home/wjg/uv312/bin/python -B -m evaluation.legacy_runtime_evidence_p0_audit.audit`

脚本只读取历史记录，在本目录输出统计、账本和读取摘要。点估计独立计算，Marconi/LRU 规则独立重算；旧 Formal 结构与模型均值明确复用已有冻结审计，不冒称重新执行全部策略。
