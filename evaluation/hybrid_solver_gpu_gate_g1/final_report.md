# G1 最终报告：Hybrid Solver Exact/Greedy 双路径 GPU 闭环验收（2026-10-25）

## 总判定：**PASS**（两路径 8/8 臂全部通过，9 行核验全 PASS）

单受控 SGLang Scheduler 生命周期内完整证明链：**真实物理状态观察 → StrictAllocationEpoch 构造与验证 → plan_strict() 在线求解（hybrid 自动分支）→ StrictAllocationPlan 校验 → reconcile_strict() 物理协调（使用本轮 Plan）→ Native Resume → Cold Reference 逐 Token 一致**。

## 一、实验A：w1/K2 Exact 分支（NEW——不同于 Spec 8 的 B5-C4）

- 输入：E4-1A 冻结 w1 8 历史＋4 Pending（E2/w1 快照：7 候选、12 容许边、K=2）；Cold 参照=E4-1A {A,B,C,D}_Cold（经 validated_resumes 逐 Token 验证）。
- 运行模块：`evaluation.hybrid_solver_gpu_gate_g1.pkg_exact.runner`（fix_a 复制＋3 处手术改动，见 `exact/prepare.py` 与 `input_provenance.json`）。
- 4 臂：`e1..e4_w1k2_exact_{A,B,C,D}`（G1_PENDING 逐臂参数化），全部独立 docker 生命周期、清理 PASS、returncode 0。
- 结果（4/4 一致）：hybrid 门内自动选 **Exact**（7 候选/4 Pending/K2/29 子集，gate_allowed=True），在线选集 **{B5C3_NODE_18, B5C3_NODE_21}**，J=**53.77075886917578**（容器内 oracle 亦 PASS），驱逐 5 keep 2；逐臂 E/G=(3392,256)/(3392,320)/(3712,384)/(3712,448)==e3 冻结值；每臂 8/8 Token==Cold，零回撤。
- 交叉核对：与 e3 `condition_results.csv` w1/K2 Exact 行逐位一致；宿主 CPU 冒烟（装配时）亦逐位一致。

## 二、实验B：C/K3 Marginal Greedy 分支

- 输入：c_formal g1_CB_FS 冻结 16 历史×2 轮＋4 Pending（K=3）；4×Cold==m1{A,B,C,D}_FS 正式臂冻结输出。
- 运行模块：`evaluation.hybrid_solver_gpu_gate_g1.pkg_greedy.runner`（fix_b 复制＋2 处最小修复，见 `pkg_greedy/build_record.json`；E>0 断言面与 fix_b 逐字节一致）。
- 4 臂：`g1..g4_ck3_greedy_{A,B,C,D}_r3`，全部 PASS、清理 PASS。
- 结果（4/4 一致）：16 候选超门禁（>12）→ 自动 **Marginal Greedy**（gate_allowed=False），census 16=16，在线选集（CPU 映射）**{C_B_P05120, C_C_P06144, C_D_P07168}**==冻结 FS 三元集，J=**154.34895561599998**，驱逐 13 keep 3；逐臂 E/G=(0,4096)/(5120,0)/(6144,0)/(7168,0)、Prefill=G+32==冻结合同；每臂 8/8 Token==Cold。policy_plans.json 仅在求解完成后读入作事后对照（oracle_match=True），未参与选择。

## 三、9 行核验表（两路径）

| # | 核验项 | 实验A（Exact） | 实验B（Greedy） |
|---|---|---|---|
| 1 | 真实 GPU Strict Epoch | PASS | PASS |
| 2 | 实际调用生产 plan_strict() | PASS | PASS |
| 3 | 自动求解分支正确 | PASS（gate_allowed=True→Exact） | PASS（gate_allowed=False→Greedy） |
| 4 | 选集及 J 与冻结参考一致 | PASS（==e3 Exact 行） | PASS（==冻结 FS，oracle 仅事后） |
| 5 | Strict Plan 验证通过 | PASS | PASS |
| 6 | 实际物理协调完成 | PASS（5 驱逐全 returned） | PASS（13 驱逐全 returned） |
| 7 | Attention-side State 正确 | PASS | PASS |
| 8 | Native E/G/Prefill 正确 | PASS | PASS |
| 9 | Cold Reference 逐 Token 一致 | PASS（4/4 臂） | PASS（4/4 臂） |

明细证据指针见 `checks.json`；逐臂检查见 `exact/verify.json`（4×10 项）与 `greedy/verify.json`（4×13 项＋全局 pkg 收敛核验）。

## 四、实际执行了什么

- 8 个 GPU 臂（两路径×4 Pending），每臂一个全新 docker 生命周期（冻结镜像 sha256:1dfce…，GPU0 H100 PCIe，独占窗口内逐臂执行），完整证明链如上。
- 每臂证据：`{RUN}/`（initial/history/reconciliation/stream_chunks/output/post_request/summary 等 30-60 个文件）、`{RUN}.runtime.log`、`{RUN}_command/gpu_before/process/cleanup.json`、`execution_lock_{RUN}.json`。
- 离线核验脚本：`exact/verify.py`、`greedy/verify.py`（只读冻结产物）。

## 五、哪些结果直接来自 GPU

8 臂的 Strict Epoch 观察、在线求解决策、物理协调（COW/驱逐）、Native Resume 流式输出、E/G/Prefill 实测、FA/保留状态请求后复核，全部来自 GPU 容器内真实执行；宿主仅做装配（冻结复制+CPU 冒烟）与事后只读核验。计时字段仅执行记录，不构成性能口径。

## 六、发现的问题（全部保留证据并修复/豁免）

1. **fix_b runner E=0 无源分支缺失**：g1 首跑（C_A，E=0）在第 94 行 StopIteration——runner 假定目标必有绑定源。gate_b 当时只验证过 C_B(E>0)，路径未覆盖。修复=pkg_greedy 对齐 c_formal 生产语义（E>0 断言不变；E=0 正向断言无源/无COW）。属测试程序缺口，非核心语义问题。
2. **target_request.json 根目录 x 写跨臂冲突**：g1 首败遗留 + g1_r3 成功后均在 /b5out 根留档，致 g2-g4 首批与 r2 全部 FileExistsError（该文件无任何读取方，纯留证）。修复=v2 写入本臂 run 目录。共 7 次失败尝试全部保留（`greedy/verify.json` 台账）。
3. **e1 执行锁中 launch.py 为守卫修复前版本**：e1 与 e2 之间修复了多臂重跑守卫（原守卫误挡后续臂）。verify 按证据零漂移口径豁免并记录。
4. **GPU 协调使用本轮 Plan**：两路径控制流均为「观察→在线 plan_strict→validate→用该 Plan 协调」；冻结选集只在求解完成后作对照（greedy）或作为容器内 Exact 参考 oracle（exact，独立重算比对）。未通过外部参数指定 Exact/Greedy。

## 七、后续

- 未修改 Φ、Exact/Greedy 规则、Strict Compatibility 语义、预算与候选集合、冻结产物；未降低任何安全门禁；无需改核心语义。
- **允许启动 B5-C4 正式性能实验**（本验收即其正确性前置；Exact 分支已在 NEW 场景 w1/K2 四 Pending 验证）。

## 附：目录

- `exact/`（装配 prepare.py、launch.py、verify.py、4 臂证据与锁）、`greedy/`（同构＋失败台账与 amendment）、`pkg_exact/`、`pkg_greedy/`（复用程序与改动记录）、`audit_readonly.md`、`test_protocol.json`、`implementation_lock.json`、`checks.json`、本报告。
