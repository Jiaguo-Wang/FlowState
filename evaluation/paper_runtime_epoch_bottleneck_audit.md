# FlowState P0-3：完整 Epoch 额外耗时定位与最小优化可行性审查报告

日期：2026-10-28。判定：**NO_CHANGE_NEEDED**（不补 GPU；两项协议级优化候选仅登记供未来 harness 迭代，不回溯任何冻结结果）。全程 CPU-only 只读：未启动 GPU、未修改 Runtime/Runner/冻结实验、未关闭正确性门禁、未将未分类 Residual 当作可删除开销、未跨 Harness 拼接。

产物目录 [paper_runtime_epoch_bottleneck_audit/](paper_runtime_epoch_bottleneck_audit/)：
[recompute.py](paper_runtime_epoch_bottleneck_audit/recompute.py)（一键复算 <30s，10 检查全 PASS）、[audit_data.json](paper_runtime_epoch_bottleneck_audit/audit_data.json)、[epochs_new.csv](paper_runtime_epoch_bottleneck_audit/epochs_new.csv)（216 epoch 明细）、[profiling_epochs.csv](paper_runtime_epoch_bottleneck_audit/profiling_epochs.csv)（72 profiling epoch 拆解）。

数据源：新 Control-inclusive（transport 优化后，[optimized_formal](control_path_optimization_output/control_20260929/optimized_formal/)，216 生命周期）+ 旧基线（[resume_epoch_final_20260928](resume_epoch_e2e_output/resume_epoch_final_20260928/)，216）+ FlowState 控制路径 [profiling](control_path_optimization_output/control_20260929/profiling/)（冻结协议仅加计时探针，72）。Runner 源码：[resume_epoch_e2e.py](resume_epoch_e2e.py) / [control_path_optimized_e2e.py](control_path_optimized_e2e.py)（两者请求验证循环逐字节相同，[control_path_optimized_transport.py](tests/runtime/control_path_optimized_transport.py) 只替换三个控制入口并"沿用冻结调度器安全点、队列和服务器"）。

## 1. 计时口径与恒等闭合（Task A 基础）

每 epoch 计时边界（[resume_epoch_e2e.py:99-112](resume_epoch_e2e.py#L99-L112)）：t0=公共观测开始，control_end=协调+compact_census 完成，随后 4 个串行单 Token 请求，**epoch 结束=第 4 请求流结束**——第 4 请求后的 metrics/census 审计按协议天然在窗口外（"第四请求后审计不计入"），故最后请求后剩余时间恒为 0。逐 epoch 恒等式：

**Epoch = Control + 请求执行和(4×) + Residual**，且 Residual ≡ g1 + g2 + g3 + g4，其中 g1=control_end→请求1提交（首个请求前），g2..g4=前一请求完成→下一请求提交（相邻请求之间）。

432/432 epoch（两数据集）闭合误差 <1e-3ms（时间戳量级 ~5e9ms 的双精度减法固有噪声 ~1e-6ms）；Residual 与字段 `between_request_validation_and_dispatch_ms` 逐位一致。**两数据集 epoch/residual 均值与冻结 comparison.json 逐位一致；spec 表格四数字（Epoch 2425.6/1876.0/2952.3、FS−MC 残余差 663.0、请求执行节省 −147.3ms）全部复现**（检查 A3–A5）。

## 2. 三策略同协议配对比较（24 快照，新数据集）

配对口径沿用正式协议：快照内三重复先取均值，再 24 快照配对（重复不作为独立样本）：

| 指标（ms） | FS−MC 均值 | FS−MC 中位 | 范围 | FS−LRU 均值 |
|---|---:|---:|---|---:|
| Epoch | +549.6 | +134.4 | [−8059, +10562] | −526.6 |
| Control | +33.9 | +28.8 | [−62, +200] | −17.6 |
| 请求执行和 | **−147.3** | −180.1 | [−383, +8] | −288.2 |
| **Residual** | **+663.0** | **+226.0** | **[−7933, +10573]** | −220.9 |
| g1（首请求前） | −34.3 | +63.4 | [−2728, +818] | −112.3 |
| g2+g3+g4（请求间） | **+697.3** | +195.4 | [−5204, +10041] | −108.6 |

要点：①FS 完整 Epoch 高于 Marconi 的 549.6ms = 请求执行节省 147.3ms 抵消后残余多 663.0ms——**差异全部落在请求间隙（g2–g4），不在首请求前也不在控制段**；②但配对差范围 [−7933, +10573] 跨零且极宽，均值被少数极端快照主导（见 §5）；③LRU 同协议下残余反而低于 FS（−220.9），三策略残余中位数（715–798ms）接近。

## 3. FlowState 残余区间的精确构成（Task B：profiling 72 epoch，同窗口原始时间戳）

窗口内每 epoch 的调用构成（代码 [control_path_optimized_e2e.py:203-217](control_path_optimized_e2e.py#L203-L217)）：**14 次 inspect RPC**（每请求前逐候选读取相关 workflow 候选状态，[openhands_policy_to_actuator_mapping_gate.py:241-269](openhands_policy_to_actuator_mapping_gate.py#L241-L269)）+ **3 次 flowstate_runtime_metrics**（请求 1–3 后 H/E/G 读取）+ **3 次 census**（请求 1–3 后全树+分配器快照，[targeted_probe.py:464-472](motivation/artifacts/wp3b_gate_20260820/targeted_probe.py#L464-L472)）+ 调度纯间隙。

| 构成（每 epoch 均值，ms） | 值 | 占比 |
|---|---:|---:|
| 残余区间（profiling 会话实测） | 1126.2（中位 768.5，范围 [98, 8023]） | 100% |
| **窗口内 RPC 墙钟合计** | **1101.5** | **97.8%** |
| ├ inspect（14 次 × 67.8ms） | 949.0 | 86.1% |
| ├ census（3 次 × 33.7ms） | 101.0 | 9.2% |
| └ metrics（3 次 × 17.2ms） | 51.5 | 4.7% |
| 调度纯间隙（gap − RPC 覆盖） | 24.7 | 2.2% |

**RPC 墙钟内部**（同主机单调时钟五段闭合，沿用 profiling 口径）：

| 段 | 每 epoch 均值(ms) | 占墙钟 | 逐调用 P50/P95/max（inspect） |
|---|---:|---:|---|
| **before_submit**（客户端连接/编码/发送 + 服务线程读取解码入队） | 783.1 | 71.1% | inspect 42.6 / 189.3 / **1873.7** |
| **event_handoff**（调度器完成→服务线程唤醒返回） | 263.1 | 23.9% | census 10.1 / 132.5 / 1084.8 |
| **server**（调度器线程真实执行：快照/哈希/校验） | 43.4 | **3.9%** | metrics 2.1 / 37.7 / 1554.7 |
| after_submit（响应传输+客户端解码） | 10.7 | 1.0% | — |
| **queue_wait**（等待调度器空闲安全点 drain） | 1.2 | **0.1%** | — |

三个决定性事实：①**区间 97.8% 是控制 RPC 墙钟，调度纯间隙仅 24.7ms**；②**安全点排队（queue_wait）可忽略（0.1%）**——控制请求一旦入队几乎立即被 drain，调度器并非忙于计算；③**服务端状态管理计算仅 3.9%**——metrics 服务端工作 0.075ms/次（纯字典读取）却花了 2–1555ms 墙钟，inspect 服务端 2.9ms/次（before+after 双快照+逐候选身份断言），census 服务端 0.74ms/次。**墙钟的 95% 消耗在 worker↔调度器进程服务线程的交接段（before_submit+event_handoff），与状态管理语义无关**。重尾显著：inspect P95=189ms、单次最大 1874ms；残余 epoch 级 [98, 8023]ms。

GPU 同步：服务端子操作实测（`_global_maps`/`_path_snapshot`/tensor_sha256 全树摘要合计 <6%/epoch）；profiling 未新增任何 GPU synchronize 屏障，tensor.cpu/tolist 的潜在设备等待已含在服务端函数耗时内（[measurement_scope.md](control_path_optimization_output/control_20260929/profiling/analysis/measurement_scope.md) 明确边界：不能宣称测得每次设备同步时间）。日志：服务端每个控制操作一行 print 含于 server 段（量级可忽略）；关键路径不写 artifact（协议声明，证据持久化在计时窗口外）。

## 4. `between_request_validation_and_dispatch` 调用链（Task B 清单逐项）

| # | 检查项 | 结论（函数/位置/证据） |
|---|---|---|
| 1 | 实际起止计时点 | 端点差额：control_end→第 4 请求 completion，无独立起止打点；本审计用 profiling 每 RPC 时间戳填充 |
| 2 | 每 epoch 调用次数 | inspect 14（每请求相关候选数之和，策略无关）+ metrics 3 + census 3（第 4 次在窗口外）——profiling 72/72 恒定 |
| 3 | 重型状态验证 | census 服务端全树快照 0.74ms/次、inspect 双快照 2.9ms/次——**非耗时来源** |
| 4 | GPU Tensor 内容读取/哈希 | 存在（tensor_ids/sha256），全部含于 server 段（43.4ms/epoch） |
| 5 | torch.cuda.synchronize | 未新增；无显式同步屏障（measurement_scope 声明） |
| 6 | 重复状态观察/树遍历/census | inspect 每候选读 before+after 两份快照（冻结安全设计）；树遍历在 server 段，量级毫秒 |
| 7 | RPC/日志/持久化 | 每调用一连接（TCP+JSON）；响应 8–30KB；print 一行/操作；持久化在窗口外 |
| 8 | 必要的请求调度等待 | queue_wait 0.1%——不是等待；"dispatch"本体=stream_request 的 submit 已计入请求窗口 |
| 9 | 策略相关操作量差异 | 调用数/载荷形状完全相同；树节点差 FS−MC +1.15（[0,3]，96 格中 72 格非零，源于 mamba 分支点），census 服务端斜率 0.029ms/节点 ⇒ 树差影响 **<0.1ms** |
| 10 | 能否用现有日志拆子阶段 | 已拆（§3 五段+服务端子操作）；before_submit/event_handoff 内部构成（GIL vs OS 调度 vs 同机竞争）**不可从现有时间戳区分，不武断归因** |

## 5. FS−MC 663ms 差异归因（Task C，分级）

**结构性证据**（三重）：

1. **停滞主导**：残余 >5s 的停滞 epoch——新会话 FS 3 / LRU 4 / MC 2（旧会话仅 LRU 1）。三个 FS 停滞 rep（g122 rep3=29260ms、g050 rep2=24336ms、g167 rep2=10959ms）把配对均值推高；**剔除 >5s 停滞后 FS−MC 均值 663.0→199.8（中位 209.9）**，三策略停滞剔除后均值 1077.4/893.2/971.0——约 70% 的 663ms 是 3–5 个停滞 epoch 的算术后果。
2. **同代码跨会话符号翻转**：请求验证路径两轮逐字节相同，请求执行时间跨会话稳定（摆动 ≤1.4ms），但残余区间 FS +877.2 / LRU +905.8 / MC +158.1 摆动——旧会话 FS−MC = **−56.1ms（FS 更低）**，新会话 +663.0。若 663ms 是 FS 内生控制成本，不可能跨会话变号。
3. **调用与计算完全对称**：三策略每 epoch 相同的 14+3+3 次调用、相同载荷形状、服务端工作 <6% 且树差影响 <0.1ms——不存在 FS 特有的状态管理计算可解释 663ms。

**归因表**：

| 差异阶段 | FlowState | Marconi | 已证实原因 | 是否必要 | 能否安全优化 |
|---|---|---|---|---|---|
| 请求间隙验证 RPC 构成 | inspect 14+census 3+metrics 3 | 相同 | CONFIRMED：代码与 profiling 调用数恒等 | 研究取证（A 类）| 可（协议级，§6） |
| 交接段耗时（before_submit+event_handoff） | 墙钟 95% | 同构成 | **CONFIRMED**：交接段主导（71.1%+23.9%），服务端 3.9%、排队 0.1% | 非状态管理必需 | harness 级可 |
| 停滞重尾（单 rep 11–29s） | 新会话 3 次 | 2 次（g078 rep2=23396ms 使该快照反转为 FS 胜） | **CONFIRMED**：停滞两边都发生，均值被算术放大 | 非代表性行为 | 不可（环境） |
| FS−MC 均值 +663.0 | — | — | **CONFIRMED**：~70% 来自停滞；剔除后 +199.8 | — | — |
| 剔除停滞后残余 ~200ms 差 | 1077.4 | 893.2 | **LIKELY**：交接段环境差异（新旧会话符号相反 −56→+200，不构成稳定策略内生成本） | — | 不可（未定位） |
| 交接段内部成因（GIL/OS/同租户） | — | — | **UNKNOWN**：现有时间戳不可区分，profiling 报告明确禁止武断归因 | — | 不适用 |

快照级相关性 FS↔MC = −0.056、FS↔LRU = −0.029：残余差异无结构化共同成分，呈无序噪声。

## 6. 最小安全优化建议（Task D：仅方案，不改代码）

**热点 1（量级最大）：每请求前逐候选 inspect 的 14 次串行一连接 RPC**（949ms/epoch，86%）。建议：合并为每请求 1 次批量只读 RPC——**复用已在 RQ6-E/transport 优化中验证语义等价的 `flowstate_batch_introspection` 模式**（单安全点一致视图+digest 合同）。必须保留：每请求 pre-residency+identity 断言（在批量响应上逐候选执行）、非干扰验证。不改候选准入/严格 Epoch/驱逐语义；预计交接次数 20→7，墙钟近似按调用数缩减。验证：现有 13 项语义等价门禁（选集/H-E-G/FA/无污染）+ 一轮小规模 GPU 对照即可。

**热点 2（协议级）：请求 1–3 后的 census+metrics（152.5ms/epoch）移出请求间隙**（统一在 epoch 末尾批量执行一次）。语义代价：失去逐请求污染归因（仍能检测 epoch 内任何污染/驱逐）；属**正式协议变更，只适用未来采集，不得回溯用于任何冻结结果**。

**禁止触碰**（全部保留）：Strict Epoch Authentication、物理映射复核、Recurrent State Ownership、FA-KV Preservation、Reconciliation Postconditions、Native Resume Correctness Gate。必要的在线安全检查（inspect 的身份与驻留断言）与重型内容审计已区分：前者保留在请求前，后者本就不在本窗口。

**结论**：两项候选都是实验 harness 的取证传输优化（与本轮已完成的 control_path 优化同类），**不改变系统生产路径，不修正任何已发表数字**——FlowState 完整 Epoch 高于 Marconi 的原因不是可优化的生产控制，而是取证 RPC 的交接开销与环境停滞（§5），因此不构成投稿前必须实施或验证的修复。

## 7. 判定与 GPU 决策（Task E）：**NO_CHANGE_NEEDED**

按 spec 三选一的判据逐条核对：额外时间主要来自**实验验证仪表**（每请求 census/metrics+逐候选 inspect = 研究取证 RPC，占区间 ~98%，其中交接段 95%、服务端计算 3.9%）与**既有协议的非代表性部分**（一调用一连接的 TCP/JSON 交接、同机竞争导致的 5–30s 停滞重尾）；**现有轻量控制路径（C 类生产核心，transport 优化后 131.9ms/epoch）未表现出值得进一步优化的明确热点**（求解 <1ms、映射亚毫秒、reconciliation 39.5ms）。

**不建议补最小 GPU 验证**，理由：①待验证的量（FS−MC 残余差）剔除停滞后 ~200ms 且跨会话符号相反（−56→+200），配对 bootstrap 95% CI 含零——新一轮三策略轻量控制路径对照预期复测的是环境噪声，不产生可写进论文的因果结论；②Signal/噪声比：交接段单次 P95 189ms、停滞 epoch 数秒级，均远大于策略间信号；③论文现有证据边界（T2-G2 请求级收益、控制优化语义等价、完整 Epoch CI 含零如实报告）已自洽，P0-2 已裁决 NO_NEW_GPU_NEEDED，本轮证据进一步支持。**合规声明**：旧 Control-inclusive 的 "Marconi Adaptation" 与 T2-G2 的 Marconi-Evict（动态驱逐增强版）是不同 Harness/实现，本报告未将两者拼接或互相引用性能数字。

## 8. 六个问题的直接回答

- **Q1 为什么 FS 完整 Epoch 比 Marconi 慢？** 请求执行 FS 省 147.3ms，但请求间隙残余多 663.0ms（均值）；后者 ~70% 由 3–5 个停滞 epoch 驱动，剔除后 +199.8ms 且旧会话为 −56.1ms——不是稳定的 FS 内生控制成本。
- **Q2 663ms 出现在什么阶段？** 全部在相邻请求之间（g2–g4 配对 +697.3；首请求前 −34.3；最后请求后恒 0）；构成为 inspect 86.1%+census 9.2%+metrics 4.7% 的 RPC 墙钟+2.2% 调度间隙；RPC 墙钟 95% 是 worker↔服务线程交接段，服务端计算 3.9%、安全点排队 0.1%；交接段内部成因 UNKNOWN。
- **Q3 是否由 FS 特有状态选择触发？** 否：调用数/载荷/服务端工作跨策略相同；树差 +1.15 节点影响 <0.1ms；测量噪声与编排差异（停滞）才是主导，已按 §5 分级。
- **Q4 为什么 FS 残余比 Marconi 高？** 可证实部分只有"停滞 epoch 在新会话更多落在 FS rep"+交接段环境差异；两者皆非策略内生；剩余 ~200ms 来源 UNKNOWN（跨会话不稳定）。
- **Q5 是否存在无需改核心 Runtime 的最小优化？** 存在两条协议级候选（inspect 批量化、census/metrics 移出窗口，§6），语义等价路径与门禁齐备；但因不修正任何论文数字、不改变生产路径，判定为未来 harness 迭代项而非本轮必做。
- **Q6 是否值得补 GPU？** 否（§7）。

**最终判定：NO_CHANGE_NEEDED。**
