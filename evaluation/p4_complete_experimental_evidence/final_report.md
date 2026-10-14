# P4-G2 完整实验过程与结果整理

已生成[FlowState完整实验手册](complete_experiment_handbook.md)，按九项实验分别记录目的、环境、准备与执行过程、计时/统计定义、结果、异常和来源。本轮没有调整论文结构，没有修改论文或PPT。

## 已恢复的主要内容

| 实验 | 恢复状态 | 交付内容 |
|---|---|---|
| 1 Hybrid正确性 | 主要限定范围证据可恢复 | 原生gap/replay、直接组件驱逐、OpenHands causal gate、16-snapshot/69-request正确性 |
| 2恢复刻画 | 正式数值与协议已恢复 | Qwen30-run、128K126-trial、位置矩阵280-trial、失败前序尝试、Kimi五对 |
| 3成本模型 | 校准/留出及选择过程已恢复 | 20校准点、15留出条件210raw、12非零评价点、M0/M1/M2全部误差与状态 |
| 4总体分配 | 完整复用P4-G1 | 48行八方法矩阵、Full Coverage、W/T/L；200→168的采集与32失败来源 |
| 5强基线/机制 | 正式结果与A–G详表已恢复 | 全候选、步骤、最终集合/成本；573全平、K_min、482Exact子集 |
| 6请求TTFT | 原始读数与汇总已恢复 | 864请求、抽样规则、216生命周期、真实计时边界、配对CI与负例 |
| 7控制成本 | 各批主要数据与执行路径已恢复 | fine-grained、6-run诊断、72-run批处理、72-run profiling、optimized transport；历史内部归因缺口明确 |
| 8 E2E | 两正式批次完整数据已恢复 | 两批各216runs、1728条请求时间戳汇总、epoch/控制/余量、CI与全部负例；serving0-run阻塞保留 |
| 9辅助与审查 | 主要辅助结果归档，设计单列 | state大小/容量、Kimi三个phase、P2/P3、早期H100/TraceLab/Profiler版本边界 |

“恢复”指读取已有证据并编成可阅读手册，不表示新运行实验或所有旧日志逐行复审。完整shell历史、WP1原启动probe日志、一些内部独立时钟及未执行serving/Pilot仍缺失（state大小由另存原始fingerprint交叉支持）；[missing_evidence.md](missing_evidence.md)逐项说明。

## 文件

- [实验手册](complete_experiment_handbook.md)：本轮主交付，包含完整数值表与P0案例表。
- [结果索引](experiment_results_index.csv)：实验/指标、规模、验证状态和数据/图路径。
- [证据路径](experiment_evidence_paths.md)、[源码函数索引](source_code_index.md)、[历史命令](historical_commands.md)。
- [图表浏览](figures/index.md)，`figures/`：原图归档及既有数据重绘，PNG/PDF均有；来源在figure_sources.csv。
- `tables/`：完整Allocation复用表、恢复/模型数据、864条RQ4原读数、1728条E2E原时间戳、全部epoch负例等。
- [复现脚本](assemble_evidence.py)、[核验摘要](verification.json)、[输入完整性](source_integrity.json)。

## 必须保留的解释边界

模型预测、真实TTFT、内部resume和epoch不能互换；旧模型/受控样本不得混入OpenHands/AgentX正式总体。边界价值原图是按WP2斜率估计的恢复价值，并非每个边界独立时延实测。P1-B全平、两个E2E批次的非显著结果、serving资格失败及P2/P3设计缺口均保留。

仅新增本目录的文档、表格、归档图及CPU整理脚本；源文件只读。没有新GPU/引擎/Agent/replay，没有新baseline、workload、P3开发或测试。本轮完成后停止。
