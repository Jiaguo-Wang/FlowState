# E4/E5 统一性能与 E2E 计时补强——最终报告

**Gate：READY。** 现有 E4-1B / E5-1 冻结日志已足以在**单次合法 Engine 生命周期内**同时还原五项统一指标（Control Time、CUDA Prefill Time、Request TTFT、实际恢复前沿 E/G、Control-inclusive Epoch Time），因此按任务 Step 2 第一分支**未修改任何 Runner**，只新增离线统一汇总并通过 CPU 回归与 54 个真实生命周期复核（0 一致性问题）。READY 不表示 FlowState 具有性能优势。

## 1. 现有 Runner 已支持 / 缺少的时间点

两套实验同构（E5 明确"完全复用E4-1B边界和持久化"），每个运行目录均已有：

| 统一时刻 | 已有证据（文件·字段） | 状态 |
|---|---|---|
| T0/T1 控制RPC发起/返回 | `summary.json` control_rpc_wall_ms（客户端单调钟，含证据持久化）；`reconciliation.json` timing 七阶段拆分（观察/Strict认证/矩阵/求解/Plan验证/Apply-time/协调后验） | 已有 |
| T0a/T1a 前置重型参考审计 | `summary.json` reference_observation_rpc_wall_ms（客户端）＋`reference_audit.json` wall_ms（服务端自测） | 已有（独立区间） |
| T2/T3/T4 请求提交/首Token/流结束 | `stream_chunks.json` start_ns/first_token_ns/end_ns **原始单调钟落盘**＋summary 派生 ttft_ms/request_wall_ms | 已有 |
| CUDA Prefill / COW / Decode | `post_request.json` trace.compute（is_extend 与非 extend 的 CUDA 事件）、trace.copies（COW CUDA）、trace.execution（外层 ns 区间＋extend_token_count） | 已有（设备事件，含提交空隙） |
| 实际恢复前沿 | `summary.json` native_match：executable_prefix(E)/replay_gap(G)/physical_fa_hit(H) | 已有 |
| Control-inclusive Epoch | control_through_completion_ms（T4−T0）、audit_inclusive_through_completion_ms（T4−T0a） | 已有（派生） |

缺少且**判定为非本轮五指标必需**（未改 Runner，如实登记）：
- T0/T1/T0a 的原始 ns 未落盘（只有派生 ms）；T2–T4 有原始 ns，统一脚本已做 ns↔ms 对账与端点代数复核，全部通过。
- 服务端控制入口/出口、证据写盘 fsync、首 Decode 内部细分计时（`e4_control_timing_audit` §4 已列）：属"拆出纯生产控制成本"的需求，超出本轮五指标口径，保留原样、不扣除任何区间。
- Cold 基线臂按设计**无控制窗口**：T0/T1 不适用，统一表标记 `control_window=ABSENT(cold)` 并留空，不推算、不置零。
- Runner 每 Epoch 只执行一个正式 Pending（设计如此）；E5 的"四 Pending"是四次独立生命周期，**不存在也不伪造多 Pending 连续执行的单 Epoch 加总**。

## 2. 实际修改的文件（最小差异）

**未修改任何既有文件**（flowstate/、E4/E5 冻结目录、两套 finalize 均原样）。新增仅 2 个代码文件＋3 个生成产物，全部在本目录：

- `unified_timing.py` —— 只读离线汇总：按上述口径从各运行目录抽取统一指标，输出逐生命周期 `unified_timing_rows.csv`（37 列：五项指标＋七阶段＋未归因余额＋两个端点间隙＋首Decode外层/内层＋E/G/H＋实际Prefill独立复算）、`unified_timing_stats.csv`（仅正式测量臂的描述性统计，单生命周期重复，无跨生命周期 Epoch 加总）、`unified_timing_verification.json`（代数/顺序/ns对账/计数检查门禁）。
- `test_unified_timing.py` —— CPU 回归：合成 Cold/正式/失败三臂，验证抽取、端点代数、缺失标记、失败处理、统计边界，并注入矛盾数据确认顺序与对账检查必报错。

## 3. 真实历史记录汇总示例（全部为冻结日志实测值）

| 运行（单生命周期） | Control Time (T1−T0) | TTFT (T3−T2) | CUDA Prefill | E / G (H) | 实际Prefill | Epoch=T4−T0 | 含审计=T4−T0a |
|---|---:|---:|---:|---|---:|---:|---:|
| E5-1 r1_A_Exact | 814.161 | 74.511 | 42.602 | 3392/256 (3648) | 858 | 7734.566 | 10240.034 |
| E5-1 r1_A_Swap | 826.764 | 87.321 | 52.249 | 3648/0 (3648) | 602 | 7705.512 | 9813.516 |
| E4-1B r1_A_Greedy | 703.828 | 101.587 | 72.545 | 2432/1216 (3648) | 1818 | 7137.142 | 8965.804 |
| E5-1 A_Cold | **空(ABSENT)** | 25550.640 | 24163.783 | 0/0 (0) | 4250 | **空(不适用)** | **空(不适用)** |

单位 ms。E5 r1_A_Exact 分解：七阶段和 323.868，RPC 未归因余额 490.294（端点差额，不臆测构成）；控制返回→请求发起间隙 170.057；参考审计→控制发起间隙 108.754；请求 Wall 6750.3 中首 Decode 外层区间 6546.321、内层 Eager CUDA 1559.758（嵌套，不相加，原因 UNKNOWN、不扣除）。Cold 臂控制窗口缺席按设计标记为空，其 TTFT/Wall/Prefill 与控制臂同用 T2→T4 请求边界。

描述性统计示例（3 次独立生命周期均值，`unified_timing_stats.csv` 共 72 组）：E4-1B A：Exact/Greedy Control 824.074/775.516、TTFT 74.258/102.601、Prefill CUDA 42.618/73.026、Epoch 7352.864/7527.624；E5-1 A：Exact/Swap Control 821.498/822.224、TTFT 75.253/80.796、Prefill CUDA 43.126/47.832、Epoch 7644.727/7721.896。这些数值与两套实验各自 finalize 输出及 `e4_control_timing_audit` 逐项一致（如 A-Exact Control 均值 824.074、r1_A_Greedy 首Decode 6058.531），实现独立复算交叉验证。

## 4. 后续正式 GPU 实验复用入口

1. 新实验沿用 E4-1B/E5-1 模式建目录（`launch.py`/`runner.py` 冻结复用、`experiment_protocol.json` 同构、每臂独立 Engine 生命周期、独立 Triton 目录与清理检查）；容器命令模板见各目录 `*_command.json`。
2. GPU 侧不改 Runner：五项指标随每次运行自动落盘（summary/reconciliation/reference_audit/stream_chunks/post_request）。
3. 主机统一汇总：`/home/wjg/uv312/bin/python -I evaluation/e45_unified_timing/unified_timing.py [新实验目录 ...]`（目录需含 `experiment_protocol.json`，标签取目录名；默认已含 E4-1B、E5-1）。
4. 修改汇总逻辑前先跑 CPU 回归：`/home/wjg/uv312/bin/python -I evaluation/e45_unified_timing/test_unified_timing.py`。

## 5. 口径与边界（随行声明）

- Control Time 含客户端证据持久化（call_saved 双写盘＋fsync），不是纯服务端控制成本；不扣除、不单侧改善。
- CUDA 事件为设备时间线区间（含提交空隙），不用客户端 Wall 替代；首 Decode 长区间（约 5.7–7.6 s）保留并单列，不删除。
- 每行=一个独立生命周期；任何跨行加总只是描述性统计，不是吞吐，更不是一个 Epoch。
- 前置重型参考审计、请求后内容检查与清理在 Epoch 端点之外/之内按上表口径如实归属。
- 本轮零 GPU、零策略运行、零冻结文件修改；`unified_timing_verification.json` gate=PASS（54/54 生命周期、0 一致性问题）。

**完成即停止，等待正式 GPU 实验任务。**
