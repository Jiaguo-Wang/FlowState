# Runtime、控制与 E2E 的组织方案

三个问题依次回答：保留集合改变请求恢复吗；控制实现能否降低开销；控制和正式检查一起计时后是否有稳定净收益。三者不能互相替代。自动恢复的数值见 `runtime_summary.csv`、`runtime_ttft_summary.csv`；配对 CI 和负例完整保存在 `runtime_paired_frozen.json`。

## 实验身份与计时范围

| 批次 | 配置与规模 | 起点—终点 | 排除项与限制 |
|---|---|---|---|
| RQ4 正式 TTFT，20260907 | Qwen3.5-9B，SGLang 0.5.17，单 H100 PCIe，TP=1；24 snapshots×3策略×3重复；每次4个串行单令牌请求 | `sota_latency_runtime.py:145` 调用 generate 前的单调时钟→`:159` 首个非空 output_ids 流块 | reconciliation 和请求前检查不在 TTFT；含客户端调用/传输/排队，非纯 kernel 时间 |
| RQ6 批处理控制，20260911 | 同类 Qwen 单卡配置，24 snapshots×3重复=72 runs，控制实现对照 | 原/批量 introspection、allocation、reconciliation 的旧专用计时 | 统一后验验证不计入报告的 2 RPC 口径；不是三策略216-run E2E |
| 初始 E2E，20260928 | Qwen3.5-9B、0.5.17、单 H100、TP=1；24×3×3=216；4串行单令牌；25%逻辑预算 | history 完成且 pending 就绪后的 t0→第4个请求流结束 c4 | 含正式同步探针；不含 history、引擎建立、最后请求结束后的审核/cleanup |
| 优化后 E2E，20260929 | 同一冻结 protocol 及规模，但独立 GPU 批次、不同控制实现 | 同上，详见 `control_path_optimized_e2e.py:164,218` | 不得与旧 TTFT 或旧控制计时拼接；余量未细分 |
| Serving-path 资格检查，20260929 | 仅 CPU 资格检查；0/216 GPU runs、0/864请求 | 未执行 | BLOCKED_BY_CORRECTNESS_GATE；延迟为 UNKNOWN，不是0 |

## 指标定义与聚合

RQ4 的 `measure_streaming_request` 在 `evaluation/sota_latency_runtime.py:137` 定义；harness 在 reconciliation 后串行调用（`evaluation/rq4_unified_runtime_harness.py:677,710`）。raw 保存 ttft_ms/request_latency_ms，没有保存可独立检查的绝对起止时间戳。本轮从864个冻结请求读数恢复三种均值。正式分析 `evaluation/rq4_snapshot_level_analysis.py:274,313`：先四请求均值、三重复均值，再24个 snapshot 等权；bootstrap 在 round 2/3/4/5 内分别抽样，不把864请求作为独立样本。

E2E 的 `stream_request` 在 `evaluation/control_path_optimized_e2e.py:73`（初始版本同函数）：s_i 在调用 generate 之前，首 token 为第一个非空 output_ids；c_i 在流迭代器耗尽后。字段 submit_ts/first_token_ts/completion_ts 已转为同 worker 单调时钟的**毫秒**，不能再乘1000。请求执行时间是客户端提交调用到流完成，包含输入构造实参、传输及 runtime 等待，不是纯模型执行时间。只能称单令牌请求完成。

`epoch_metrics:99` 定义完整epoch=c4−t0；平均屏障到完成=mean(c_i−t0)；平均请求执行=mean(c_i−s_i)。本轮对每个batch的216个record重新汇总这些已有数值，与冻结summary逐指标在1e-6 ms内一致。所有配对推断均先三次重复聚合后以24个snapshot为单位；冻结round分层95% bootstrap CI直接引用，不重新拟合或重抽样。

## 应放入论文的主表

均为 ms；请求为每请求均值，epoch/控制为每epoch均值。

| 批次 | 方法 | 请求侧指标 | 总控制 | epoch |
|---|---|---:|---:|---:|
| RQ4，TTFT | LRU / Marconi / FlowState | 164.705 / 129.097 / 94.025 | 不在计时内 | 未测 |
| 初始 E2E，请求完成 | LRU | 163.375 | 232.892 | 2130.623 |
| 初始 E2E，请求完成 | Marconi | 128.157 | 267.758 | 1888.372 |
| 初始 E2E，请求完成 | FlowState | 92.537 | 656.771 | 2078.838 |
| 优化后 E2E，请求完成 | LRU | 163.191 | 149.455 | 2952.253 |
| 优化后 E2E，请求完成 | Marconi | 127.987 | 97.957 | 1875.991 |
| 优化后 E2E，请求完成 | FlowState | 91.152 | 131.904 | 2425.641 |

94.025与91.152来自不同批次、不同控制版本、不同终点；不能把差值解释为完成比首令牌更快，或将单令牌当作二者等价的充分条件。各批次内部可以比较策略；跨批次仅并列说明，不提供跨批次因果净收益估计。

| 批次/指标 | 对照 | mean baseline−FlowState | 95% CI | FlowState W/T/L |
|---|---|---:|---|---|
| RQ4 TTFT | LRU | 70.680 | [63.806,78.300] | 24/0/0 |
| RQ4 TTFT | Marconi | 35.073 | [27.671,43.175] | 22/0/2 |
| 初始 E2E epoch | LRU | 51.785 | [-226.758,384.206] | 8/0/16 |
| 初始 E2E epoch | Marconi | -190.466 | [-384.911,35.209] | 11/0/13 |
| 优化后 E2E epoch | LRU | 526.612 | [-1057.440,2214.943] | 15/0/9 |
| 优化后 E2E epoch | Marconi | -549.649 | [-1784.913,699.396] | 8/0/16 |

后两批对两基线的CI都跨零；正确性PASS不是净收益PASS。保留各批 `analysis/summary.json` 中 rounds、cv_summary 和全部 non_winning_cases；附件 `runtime_batch_diagnostics.json` 完整带出。RQ4 CV有drift warning，优化后FlowState epoch重复CV中位数约0.364、P95约1.280；不删除高波动样本。

## 控制分段与无法归因的剩余时间

| FlowState 批次 | 公共观测 | 专属观测 | 输入构造 | allocation | reconciliation |
|---|---:|---:|---:|---:|---:|
| 初始 E2E | 150.926 | 293.093 | 0.299 | 1.115 | 211.337 |
| 优化后 E2E | 66.225 | 24.985 | 0.240 | 0.929 | 39.525 |

优化包括批量 frontier 查询、减少重复同步与验证；不能据此直接推断epoch下降。RQ6另有旧控制8398.847→629.827 ms、51.5→2 RPC/epoch的证据，必须独立成表，不能与本表相减；其验证范围见旧报告，不能把2RPC当成系统总同步操作数。

优化后代码 `:203–217`：每请求前精确驻留检查，请求后metrics/H/E/G查询和census。首请求前检查与前三请求后的检查位于epoch内；第四请求的结束后检查在c4外。请求函数自身的流块处理在s_i到c_i内；写盘与最终objective核对在epoch外。关键路径中的selected-set冻结参考核对也不等于生产所必需算法工作。

严格分解为：已计时 policy control + Σ请求执行 + runtime synchronization + evaluation-only validation/diagnostics + unresolved。后三项未独立打点，不能唯一拆开。数值余量 `epoch−control−4×mean_request`：LRU 2150.033、Marconi 1266.087、FlowState 1929.129 ms（优化批）。它混合请求前后探针、同步、dispatch及未归因等待；必须标成 **UNRESOLVED**，不能全部扣掉再宣称serving净收益，也不能全部称为FlowState生产开销。

## 推荐图表

1. Runtime TTFT表及同批snapshot配对差图，保留Marconi两个负例。
2. 控制优化按批次分别列五阶段表，RQ6另列；不堆叠不同批次测量。
3. E2E两批独立森林图，零线和跨零CI清晰保留；主表同时列request、control、epoch。
4. 计时边界示意图表示t0/control_end/s1/c1/…/c4；未拆分余量用斜线标示UNKNOWN贡献。图示是代码边界，不是假想性能。
5. 附录列round分层、CV、全部负例、correctness/cleanup与serving资格阻塞。

本轮只完成主Allocation图A/B/C；上述Runtime图是组织建议，现成CSV/冻结分析可供后续排版，不启动新测量。
