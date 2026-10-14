# Evaluation 重组大纲

本轮交付是使用冻结证据的章节设计与图表初稿，不修改论文。最新论文正文及 P4-G0 在已搜索的仓库、`/home/wjg/code` 与附件中未定位到；现有 Section 6 的逐段对应关系为 **UNKNOWN**。以下编号是建议编号。证据路径统一见 `evidence_index.md`，逐案例来源见 `per_case_evidence.csv`。

## 6.1 实验设置与证据层次

- 研究问题：各实验实际评价哪个对象，统计单位、输入和信息边界是否可比较？
- 假设：冻结同一 snapshot、候选与预算可以作策略配对；模型成本与 GPU 延迟必须独立报告。
- 工作负载：OpenHands 168 snapshots、AgentX 23 snapshots，各 25%/50%/75%；Runtime 为既定 24 个 OpenHands snapshots、25%、每策略三次独立生命周期、四个串行单令牌请求。
- 对照：主矩阵八种方法；Exact OPT 仅可穷举子集；Runtime 只有 LRU、Marconi Adaptation、FlowState，不虚构强基线 GPU 结果。
- 指标：C(S)=ΣPhi(G,T)，单位 modeled ms；Full Coverage、E/G、集合等价；TTFT 与 epoch 单独定义。
- 图表：设置表列数据来源、模型、GPU、并发、版本、预算、重复数、计时区间和限制；指标定义表。
- 已有结果：573 个 snapshot–budget 条件、191 个 snapshots；每批 Runtime 216 生命周期、864 请求。
- 支持：输入固定条件下的描述性比较。三预算不能合并为 573 个独立样本；Runtime 先生命周期内请求均值，再三重复，再 24 个 snapshot 配对。
- 未解决：AgentX 原生 fork/ready 证据不足；逻辑 K 不能自动解释成独立物理池限制。最新论文与 P4-G0 尚未核对。

## 6.2 恢复代价刻画与模型准确性：主实验的测量基础

- 研究问题：attention 命中时 recurrent loss 是否仍产生恢复代价；Phi 能否预测此代价？
- 假设：可执行 gap 有真实代价；position-aware 模型在校准域内具有较小预测误差。
- 工作负载：RQ1 固定 32K FA-KV、六个 gap、每档五次；RQ2 冻结 held-out 预测集。Qwen3.5-9B、SGLang 0.5.17、单 H100；两者不与主分配 snapshots 混作一批。
- 对照：RQ1 G=0 与各 gap；RQ2 冻结预测与实测，不重新拟合。
- 指标：服务端 TTFT 中位数/IQR、held-out MAPE/MAE、有效域检查。
- 图表：复用 RQ1 gap–latency 图与 RQ2 prediction/error 图；附模型系数及 0≤G≤T≤131072 的适用域。
- 已有结果：RQ1 30 runs，32.374→1531.882 ms；RQ2 MAPE 2.68248%、MAE 31.17958 ms，147/147 网格通过。此处数字引用冻结审计；本轮没有重跑拟合或重绘这些图。
- 支持：恢复成本真实存在，模型在已测域准确，为随后 modeled allocation 提供依据。
- 未解决：准确估值不等于最终选择收益；跨硬件、并发、冷启动不能外推。RQ1 服务端 TTFT 与 RQ4 客户端流式 TTFT 边界不同。

## 6.3 总体分配质量：主实验

- 研究问题：固定正式输入下，八种方法的最终可执行恢复成本和覆盖程度如何？
- 假设：识别可执行前沿的保留方法可减少旧静态保留的冗余，但未预设 FlowState 优于强基线。
- 工作负载：全部 191 snapshots × 三预算，不筛选胜例。
- 对照：LRU、LFU、Marconi Adaptation、StandaloneTopK、PCB、LDR、WFC、FlowState。
- 指标：每预算均值/中位数、baseline−FlowState 配对差、FlowState W/T/L、Full Coverage、集合等价、K 与 K_min_frontier。
- 图表：Figure A `overall_allocation_quality.pdf`；完整 48 行矩阵；Exact OPT 独立小表 `exact_opt_subset_quality.csv`。
- 已有结果：四种 workflow-aware 方法全量持平；对 StandaloneTopK 严格改善为 OpenHands 144/504、AgentX 2/69；Exact 子集 482/482 objective 匹配，其中 434/434 与 48/48。
- 支持：既有正式分布上相对旧保留策略的 modeled 改善及可求解子集质量。
- 未解决：不能以 Exact 匹配证明任意实例全局最优；不能将 modeled 改善称为 GPU speedup；强基线没有额外收益。

## 6.4 强基线验证与工作负载区分力：主验证与解释性分析

- 研究问题：更简单的 workflow-aware 规则能否解释主结果？
- 假设：无共享收益支持或预算足够时，简单方法可与 FlowState 等价。
- 工作负载：同一 573 个条件；不新增样本，不降低预算。
- 对照：PCB、LDR、WFC 与 FlowState。
- 指标：最终目标、集合、逐 pending E/G 一致性；选择顺序差异；K_min_frontier 与共享正收益覆盖。
- 图表：Figure B `strong_baseline_validation.pdf`；Figure C `budget_workload_discriminativeness.pdf`；结构明细 `workload_structure.csv`。
- 已有结果：各强基线 573/573 目标、集合、E/G 相同；顺序差异 PCB 9、LDR 12、WFC 21。OpenHands 90 个受限条件仍可分；AgentX 0/69 存在 K<K_min_frontier。
- 支持：当前强方法的最终效果无法区分，部分机制确实改变中间排序。
- 未解决：缺少共享收益重叠与有效预算同时迫使取舍的正式证据；AgentX 22/23 个 snapshot 有正收益共享结构，不等于原生共享竞争。

## 6.5 机制归因与消融：不承担总体优势证明

- 研究问题：哪些收益来自去冗余、哪些来自共享估值或位置感知？
- 假设：边际更新可避免静态 TopK 重复覆盖；位置与共享可改变排序，但是否改变终态取决于预算。
- 工作负载：P0 全量分布和冻结 Case C/F；P1-B 的 g051、g005 作为必要补充，不重新选“最漂亮”的案例。
- 对照：StandaloneTopK、去位置感知消融、PCB/LDR/WFC；Marconi 仅按其实际适配定义。
- 指标：逐步 marginal、零边际冗余、最终 E/G/C、顺序与集合差异分别列示。
- 图表：案例 C 的候选/选择步骤表、案例 F 的共享收益表；全量消融表。详见 `mechanism_attribution_plan.md`。
- 已有结果：AgentX 66/69 正收益共享、48/69 实际多 continuation 新增收益、9/69 联合排序反转；仅 2/69 优于 StandaloneTopK，不能隔离为同时多分支增益；强基线增益为零。
- 支持：机制在已有记录中执行且能解释具体结果。
- 未解决：无 position-aware 估值带来最终额外收益的证据；无一般 global allocation 独特增益证明。

## 6.6 请求级 Runtime 验证

- 研究问题：既定 reconciliation 之后，真实请求恢复路径是否受保留集合影响？
- 假设：减少可执行 gap 可降低该批次客户端首令牌延迟。
- 工作负载：RQ4 的 24 snapshots、216 生命周期、864 串行单令牌请求，Qwen3.5-9B、单 H100、TP=1。
- 对照：LRU、Marconi Adaptation、FlowState。
- 指标：提交调用前至首个非空 output_ids 的 TTFT；24 个 snapshot 配对差、CI、W/T/L、round 分层与重复 CV。
- 图表：三方法 TTFT 主表；按 snapshot 的配对差图。直接复用冻结 CI，不与 E2E 拼接。
- 已有结果：164.705、129.097、94.025 ms；FlowState 对 LRU 24/0/0，对 Marconi 22/0/2。
- 支持：指定 replay 配置下的请求级改善。
- 未解决：不含观测/选择/协调，不是系统总体吞吐、完整多令牌 continuation 或强基线优势。

## 6.7 控制成本与控制包含式恢复 epoch：Runtime 系统证据与负面结果

- 研究问题：请求侧收益在真实控制、同步与正式检查加入后是否仍存在？
- 假设：批处理减少控制开销；净端到端收益必须由同批完整计时验证，不能从分段下降推导。
- 工作负载：RQ6 独立 72 runs；初始及优化后 E2E 各 216 runs，分别报告，不混池。
- 对照：RQ6 原/批处理路径；E2E 三策略分别在本批次内配对。
- 指标：完整 t0→c4、mean barrier-to-completion、mean request execution、控制分段、无法归因余量、CI/W/T/L/CV。
- 图表：批次×策略矩阵；各批 E2E 配对森林图建议；余量单独用斜线填充，禁止标成纯生产开销。
- 已有结果：优化后 FlowState 控制均值 131.904 ms，请求均值 91.152 ms；E2E 对两基线 CI 都跨零。Serving-path 尚未执行，0/216，不得填零延迟。
- 支持：可降低当前实现的控制开销；正式含探针 E2E 尚无稳定净收益。
- 未解决：余量细分 UNRESOLVED；去探针同时保留完整正确性证明尚未完成。

## 6.8 限制与后续工作

- 研究问题：结果能推广到哪些执行条件，哪些关键主张尚无证据？
- 假设：显式限制比将可分结构、派生 lineage、串行单令牌外推到真实并发系统更可靠。
- 工作负载/对照：总结上述全部冻结证据；Kimi 仅作跨架构可执行恢复的补充，不与 Qwen 性能合并。
- 指标/表：Claim–Evidence 表、未知项清单；不生成新的性能统计。
- 已有结果：原生共享竞争、强基线最终收益、位置感知最终增益、稳定生产 E2E 均未建立。
- 支持：对既有分布和实现条件作有限结论。
- 未解决：先完成数学区分条件与物理预算合同，再验证原生事件 Pilot；是否值得采集新 cohort 由准入证据决定，不按策略胜负决定。
