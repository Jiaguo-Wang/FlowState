# T2-G1：三策略 GPU 接入与最小验证报告

日期：2026-10-27。**CPU 统一接入完成（13/13 合同检查 + 5/5 测试）；GPU Smoke Test PASS（3/3 臂，24/24 收尾验证）；正式多轮测量未启动。** 未修改三种策略的评分与驱逐算法，未修改任何既有文件（smoke 复用 pilot runner/control/bridge 零代码改动，仅以 T2-G1 版 policy_plans.json 替换 case 目录计划文件）；新增仅在本目录。

## 1. 三种策略是否完成统一接入

**完成。** [adapter.py](adapter.py) 在 C/K3 冻结域（16 候选 / 四 Pending C_A..C_D 与 CPU ready 调用逐 Token 全等 / K=3）上统一产出三策略选集与 pilot 同格式的 [t2g1_plans.json](t2g1_plans.json)（GPU runner 经 `PILOT_POLICY` 消费）：

| 策略 | 选集（CPU id） | 四 Pending E | 当前域 J (ms) | 选择耗时(诊断) |
|---|---|---|---|---|
| FlowState-Greedy | C_D_P07168 / C_C_P06144 / C_B_P05120 | 0/5120/6144/7168 | 154.349 | ~0.05ms |
| Marconi-Evict（T2-Mini） | C_JOIN_P04032 / C_JOIN_P04096 / C_ROOT_P01024 | 1024×4 | 703.455 | ~0.1ms |
| KVFlow-inspired | C_AB_P01984 / C_AB_P02048 / C_A_P04032 | 4032/2048/0/0 | 627.615 | ~0.1ms |

关键交叉核对：**FS 经统一接口逐位复现已通过 GPU 验证的 pilot 计划**（选集与四 Pending E/G 预期全等，`test_fs_reproduces_pilot`）——统一接口本身已被 GPU 证据锚定；KVFlow == 冻结 CPU 行；Marconi-Evict == T2-Mini 产物。三策略选集两两不同、覆盖结构互补。

## 2. Marconi 与 KVFlow 的合法输入是否满足

**满足。** 决策时刻输入全部为请求前冻结声明：Marconi 所需访问历史 = `last_access_order`（16 候选全有限，源自 DECLARED_SYNTHETIC_ACCESS 声明事件，经 T2-R `validate_declared_case` 复核 Token/访问/前缀合同）；KVFlow 所需 ASG/STE = 冻结 graph/declarations，STE 无 UNKNOWN。评分域恰为四 GPU Pending；C_JOIN（CPU 独有 future）不进入决策（`test_decision_inputs_legal` 断言评分域 == {C_A,C_B,C_C,C_D}）。

**Recurrent-only 落实性**：Marconi 的逻辑父子驱逐只输出最终保留集；三个选集 ⊆ 16 个已物化物理状态（逐候选独立 slots，T2-R Census 证据），控制路径按 `logical_eviction_ids` 真实驱逐其余状态并保护 FA 索引/内容——**不需要也不执行原 Marconi 的 Attention KV 合并操作**。

## 3. GPU Smoke Test 是否通过

**PASS（3/3 臂，2026-10-27，[smoke/](smoke/)）。** GPU 转入空闲后执行：三策略各 1 臂、同一 Pending C_B（唯一使三策略 E 全>0 的 Pending，恢复/COW 路径全覆盖）、每臂独立 Docker/Engine 生命周期、复用 pilot runner/control/bridge **零代码改动**（smoke_lock.json 锁定摘要；FS 臂同时为正对照——该组合已在 pilot 通过 GPU 验证）。GPU 礼仪偏离如实记录：GPU0 有 1.5G 他人常驻进程（0% 利用率），门禁放宽为空闲≥40G+利用率 0%（[gpu_gate.json](gpu_gate.json)）；正式 60 臂仍须空卡。

[verify_smoke.py](verify_smoke.py) 24/24 核对通过（[smoke_results.json](smoke/smoke_results.json)）：
- **保留集合**：三臂 census 16=16、选集==t2g1_plans、13 项驱逐全部真实落实（returned）；
- **恢复前沿**：E 实测 == 计划预期（FS 5120 / Marconi-Evict 1024 / KVFlow 2048）；prefill==len(pt)−E 逐臂成立（32/4128/3104）；FA 命中==冻结边界 5120 且内容受保护；
- **生成结果**：三臂输出与 pilot 冻结 cold_C_B（独立空缓存生命周期）逐 Token 一致；
- **初步性能（指示性，非测量口径）**：TTFT 3548/2364/3016 ms、CUDA Prefill 356/155/119 ms、控制 RPC ~11s——无形状预热/测量模式，已知污染在场（FS 32 Token 的 CUDA Prefill 反而最大=首形状编译污染的直接证据）；**不做任何方向性结论**，正式测量以 [formal_plan.md](formal_plan.md) 60 臂协议（形状预热+测量模式+配对口径）为准。

## 4. 各策略的保留集合、恢复前沿和初步性能

CPU 层见 §1 表。GPU smoke 实测层（正确性合同，三策略全过）：E 5120/1024/2048 ↔ Prefill Token 32/4128/3104 ↔ 输出与独立 Cold 一致；计时观测见 §3（指示性）。

## 5. 正式 GPU 对照是否 READY

**方案 READY、通道已验证、执行待批。** Smoke 已证明 runner 消费 t2g1_plans.json 的完整路径（含 Marconi-Evict/KVFlow 两新策略臂）。[formal_plan.md](formal_plan.md)：3 策略 × 4 Pending × 5 轮 = 60 臂，沿用 C/K3 冻结五轮独立生命周期协议 + `c_formal_perf_gpu` 测量合同（三层口径、请求窗口零审计零新增编译、形状预热、Cold 逐 Token 一致）+ G2 调度实践（schedule 预固定、策略顺序轮换、拒绝跳序）；统计口径与负向观测保留规则同 G2。启动前置条件：(1) GPU 空卡（严格礼仪）；(2) 正式多轮测量属下一任务，本轮未启动。

**本轮到此为止：接入 + 最小 GPU 验证（smoke 3/3 PASS）完成。**
