# 缺失、未执行与无法唯一验证的证据

本文件列的是本轮查到的证据边界，不把未找到等同于从未存在。目录分类自动核对见evidence_categories.csv，完整文件存在性见artifact_inventory.csv。

| 实验 | 状态 | 缺口或限制 | 对结果的影响 |
|---|---|---|---|
| 全部批次 | PARTIAL | 有些实验仅留脚本/config/plan，没有完整原始shell历史 | 有命令文件的逐字转录；其他操作细节标RECONSTRUCTED_FROM_CODE，不凭空补启动命令 |
| 全部批次 | PARTIAL | 当前源码不必与所有历史版本逐字节相同；容器内路径不能直接当宿主机路径 | 运行身份优先采用冻结source manifest、config和记录；未启动容器查证 |
| 1组件正确性 | UNKNOWN | 没有一般并发、slot-generation生命周期、跨rank事务/回滚的完整证明 | PASS仅限定正式空闲安全点与实际gate；不能无限扩张 |
| 1 FA preservation | PARTIAL | 多数门禁比对FA索引、tree/allocator摘要，非全部KV tensor逐字节hash | 能证明所观测映射/驻留非干扰，不冒称全tensor数值等价 |
| 2Qwen恢复 | UNKNOWN | restore/COW、replay kernel、采样、隐式同步未全部独立计时 | TTFT为实际路径总时间，不伪拆成纯kernel |
| 2Kimi | NOT_MEASURED | 未有完整gap曲线、重拟合Phi、Kimi多策略TTFT/E2E | 仅5对固定状态sanity；Phase2的Qwen模型数不能称Kimi实测预测 |
| 3模型 | MISSING | 模型选择之后的另一个独立最终test set未见 | heldout未参与拟合，但参与模型选择，必须如实说明 |
| 4AgentX | MISSING | 原生fork/ready/dispatch/state generation联合事件 | frozen formal compatibility CPU比较成立；自然共享pending证据不成立 |
| 4OpenHands | RECORDED_FAIL | 32/200指定groups因non_monotonic_inputs失败 | 主总体168，未补位；不是隐藏失败 |
| 5强基线 | NOT_OBSERVED | 相对PCB/LDR/WFC的终态收益、位置感知带来的最终额外收益 | 573/573同目标同集合，不能以排序差异替代 |
| 5Exact | NOT_RUN | 91/573条件超过既定穷举阈值，无Exact目标 | 分母482，未知项不是失败或匹配 |
| 6RQ4 | MISSING | raw只留ttft_ms/request_latency_ms，没有绝对submit/first/completion时间戳 | 时长可恢复，绝对事件无法事后重建 |
| 6RQ4 | NOT_MEASURED | 控制包含式TTFT、强基线GPU、多token与并发吞吐 | 不从该批请求级结果推导系统净收益 |
| 7细粒度控制 | PARTIAL | 冻结72-run内部transport/线程子段未全打点 | 新diagnostic的87.79%不能回填历史8398.847ms |
| 7旧E2E frontier | UNRESOLVED | 历史293.093ms无完整内部时间戳 | profiling299.434ms来自新批次，不能充作历史完整归因 |
| 7诊断 | UNSUPPORTED | tensor.cpu/tolist隐式设备同步与free异步完成未单独测量 | 主机0.090111ms不是GPU操作完成总时间 |
| 8E2E | UNRESOLVED | residual混合同步、请求前后formal验证、dispatch与等待，无法唯一拆分 | 不能全部称production开销，不能全部扣除制造serving结果 |
| 8Serving-path | BLOCKED_BY_CORRECTNESS_GATE | 被动变更日志无法替代完整逐请求门禁 | GPU runs0/216、请求0/864；性能UNKNOWN而非零 |
| 9WP1启动probe | MISSING / PARTIAL | 当前目录没有原报告所列server_probe.log；findings与patch存在，外部既有strict_L2048.json的真实payload fingerprint可读 | 49.125MiB由该原始JSON交叉支持；不能声称重新读取了缺失启动日志 |
| 9物理预算 | DESIGN_ONLY | 独立retention quota/lease/admission合同未实现 | 旧K是逻辑预算，不能声称已强制长期物理限制 |
| 9P3 Pilot | NOT_RUN | 原生事件联合证据与真实共享/专属状态竞争未执行验证 | 设计配置不计GPU样本或结果 |
| 9早期辅助目录 | PARTIAL | 已恢复配置/汇总/代表raw，但未逐行重审所有早期日志与全部尝试 | 不声称完整认证仓库所有历史研究分支；保留独立来源与版本边界 |

## 明确保留的失败、负面与排除记录

1. `recovery_profiler_128k_20260826_133234_448476/config.json`：E不匹配，1/126后FAIL；正式PASS在另一目录。
2. WP2首次冷gate和不平衡pilot：原报告明确不计入30个正式measured runs；原记录仍保留。冷慢的具体硬件/JIT原因未独立定位。
3. position audit：gap-only假设FAIL、legacy复现WEAK、差异归因MIXED。
4. 模型选择：M0 FAIL、M1 WEAK、M2 PASS；旧Profiler v2的piecewise并非全部误差指标优于Old Phi。
5. Fine-grained RQ6早期CPU suite有写权限/输入环境失败；最终通过不覆盖或抹去旧失败日志。
6. RQ6-D初步诊断及probe安装失败独立保留；最终6-run不可与前序尝试混算。
7. RQ4相对Marconi两个负例与drift warning保留。
8. 两批probe-inclusive E2E所有负例、重复波动与跨零CI保留；性能净收益未建立。
9. Serving-path资格阻塞不是已执行GPU失败；P2/P3是审查/设计，未运行Pilot。

本轮没有为弥补上述缺口启动任何新实验、重建population或修改预算。
