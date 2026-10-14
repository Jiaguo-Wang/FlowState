# Executable Compatibility 准入合同（Task A1）

制定日期：2026-10-12 ｜ 状态：定义完成，等待最小样本（A2）与 Runtime 认证（A3）回填证据
适用范围：FlowState 投稿冲刺任务 A（ICDE 2026-11，投稿截止 2026-10-15）；本合同是对 Gate1/Step3 审计结论「唯一最高优先级下一步」的直接回答。

---

## 1. 为什么需要这份合同

Gate1/Step3 审计（`evaluation/gate1_step3_compatibility_retention_contract_audit/audit_report.md`）发现：

- **Formal 矩阵**（`rq3_agentx_formal_policy_evaluation.py`）用 **B1 逻辑祖先**（lineage path 前缀 + is_inherited + token_position ≤ active target）加人工 laminar-cluster lineage 编码判定兼容，23 个 snapshot 共 114 条 Formal 正边；
- **Raw 矩阵**（`agentx_runtime_snapshot_collection.py:884-895`）用**实际回放 token 前缀** `_prefix_match(token_ids[:recurrent_prefix_length], pending_tokens)` 判定；
- 其中 **70 条 Formal-only 边在 B2 实际 Token Prefix 谓词下失败**。

根因（A4 详述）：`_compose_prompt_tokens`（AIPerf compose 语义）对**每个请求独立**从其自身 hash_ids 合成 token，即使逻辑上 p 是 c 的后代，两者的 token 序列也不前缀连续。

**本合同的目的**：给出唯一的、可机器执行的 `ExecutableCompatible(c, p)` 判定标准，使「逻辑 DAG 上有共享祖先」永远不能替代「真实 Recurrent State 被合法恢复」。新矩阵的每条边必须通过本合同认证；既有 573 个正式结果不被覆盖、不被修改。

## 2. 四层证据模型（层次不可互相替代）

| 层 | 名称 | 证据来源 | 在合同中的角色 |
|---|---|---|---|
| L1 | 逻辑 Workflow 祖先 | B1 lineage（`agentx_runtime_final_correctness_audit.py:87-115`） | **仅作候选生成器**（必要不充分），不进入谓词 |
| L2 | Token 前缀一致 | 逐 token 比较（B2 `_prefix_match` 是其在 pending token 层的特例） | 条件 C2 |
| L3 | 物化 Checkpoint 合法 | runtime inspect 通道（节点、摘要、驻留、保护） | 条件 C1/C3–C6 |
| L4 | 实际恢复正确 | 真实 Resume 与全量重算的输出一致性 + 物理命中计量 | 条件 C7（动态认证） |

**核心原则**：L1 通过 + L2 通过仍不构成可执行兼容；只有 L1–L4 全部通过，一条边才能进入正式 Executable Compatibility Matrix。

## 3. 形式定义

设：

- `c`：一个**真实物化**的 checkpoint（radix 树中存在对应节点，持有 FA-KV 与 Recurrent/Mamba 状态）；
- `p`：一个待恢复的 continuation（完整 token 输入序列）；
- `L`：checkpoint 的**状态位置**（其 token 历史的物化深度）；
- `T`：continuation 在查询时刻的可恢复前沿（≤ `len(Tokens(p))`）。

```
Admissible(c, p)   ≡ C1–C6 均无 FAIL            （准入：可送 runtime 认证）
Certified(c, p)    ≡ C1–C7 全部 PASS            （唯一可写入正式矩阵的边）
ExecutableCompatible(c, p) 在本文档中即 Certified(c, p)
```

## 4. 七个条件

| # | 条件 | 判定内容 | PASS | FAIL | UNKNOWN |
|---|---|---|---|---|---|
| C1 | 身份一致 | `model(c) == model(p)` 且 `tokenizer(c) == tokenizer(p)`（含权重/tokenizer 配置摘要） | 完全一致 | 任一不匹配 | 身份信息缺失 |
| C2 | Token 历史精确一致 | `Tokens(c)[:L] == Tokens(p)[:L]` **逐 token** 比较 | 全位一致 | 任一位分歧（记录首个分歧位置） | 不可比（长度信息缺失） |
| C3 | 位置合法 | `0 < L ≤ T` | 合法 | L≤0 或 L>T | T 未知 |
| C4 | 恢复条件满足 | checkpoint 同时持有 FA-KV 与 Recurrent 状态且驻留设备（inspect 实测） | 双驻留 | 任一缺失/仅 host | inspect 不可达 |
| C5 | 物理身份与生命周期 | radix node id + `sha256(prefix_tokens)` 摘要（+ slot generation 若账本可得） | 齐备且与物化时一致 | 摘要不一致（节点被复用/改写） | node/digest/generation 任一缺失 |
| C6 | 保护检查 | 读取时点 `lock_ref == host_lock_ref == session_ref == 0`（无 Active/COW/Reference 保护） | 全零 | 任一非零 | 引用计数不可读 |
| C7 | 动态认证（实际 Resume 正确性） | 真实 runtime 从 c 恢复 p：物理命中/可执行前缀/恢复间隔计量符合预期，且确定性输出与全量重算一致，两个 continuation 不共享可写 recurrent slot | 全部实测通过 | 任一不符 | 未执行 runtime 阶段（离线恒为 UNKNOWN） |

**C7 的具体计量判据**（由 WP3B 传输层 `instrumented_match_prefix` 记录）：
- `physical_fa_hit ≈ L`（±1，容差来自 mamba track 边界取整）；
- `executable_prefix ≈ L`（±1）——这是「实际用了 checkpoint 而非重 prefill」的直接证据；
- `replay_gap = physical_fa_hit − executable_prefix ≤ 1`；
- `mamba_branching_seqlen ≈ L`（±1）——Recurrent 状态分支点即 checkpoint 位置；
- 输出 token 与 Full Recompute 参考路径**完全一致**（temperature=0、max_new_tokens=1、禁 CUDA graph/overlap 的确定性设置）。

**负例判据**（P_bad，历史在 L 之前分歧）：C2 必然 FAIL；runtime 侧 `physical_fa_hit` 停在首个分歧位置， corroborate 非法前缀无法恢复到 L。

## 5. 组合语义与状态分类

| 状态 | 定义 | 允许的用途 |
|---|---|---|
| REJECTED | 任一条件 FAIL | 不得进入任何矩阵；记录失败条件与证据 |
| ADMISSIBLE | C1–C6 无 FAIL（允许 UNKNOWN），C7 UNKNOWN | 可送 runtime 认证；不得计入「已证实可执行」 |
| CERTIFIED | C1–C7 全 PASS | 唯一可写入正式 Executable Compatibility Matrix 的边 |

**诚实性规则**：
1. UNKNOWN 不因「大概率成立」而升级为 PASS；C5/C6 有 UNKNOWN 的边即使 C7 通过也只能标 ADMISSIBLE+运行时注记，不得计入 CERTIFIED（本轮最小样本中 slot generation 账本缺失即属此类，如实标注）。
2. 禁止以任何人工 lineage 重编码使 C2 失配的边「看起来」连续（Formal 的人工 laminar-cluster 编码正是被审计否定的做法）。
3. Token Prefix 匹配（C2 通过）本身**不等于**已完成真实 Recurrent State 恢复——没有 C7 就没有 CERTIFIED。

## 6. 与既有 B1/B2/Raw/Formal 的关系

- **不修改** B1、B2、Formal 数据、既有 573 个正式结果、正式 Compatibility Matrix、Recovery Cost Model、FlowState/PCB/LDR/WFC 算法逻辑；
- B1 逻辑祖先降级为候选生成器；B2 `_prefix_match` 语义被 C2 吸收（C2 是其在「checkpoint 物化历史 vs continuation 全序列」上的推广）；
- 新矩阵生成流程（A4 提案）：对每个 B1 候选边 → 运行时物化祖先 → 用本合同谓词逐条件判定 → 仅 CERTIFIED 边进入新矩阵。

## 7. 实现

谓词实现：`evaluation/submission_p0_executable_recovery/executable_compatibility.py`
（只读、无副作用；`evaluate_executable_compatibility(checkpoint, continuation, ...) -> CompatibilityVerdict`，逐条件返回 PASS/FAIL/UNKNOWN 与证据描述；C7 由 runtime 阶段以 `runtime_evidence` 参数回填）。

样本构造（A2）：`build_minimal_cases.py` → `compatibility_cases.json`
Runtime 认证（A3）：`minimal_recovery_smoke.py` → `runtime_evidence.json`（由 `run_minimal_recovery_smoke.sh` 在 docker 内可复现执行）

## 8. 完成标准对照（任务 A 的 PASS 门）

> ≥1 个真实物化 checkpoint，在同一模型下以合法 token 前缀恢复**两个**不同 continuation，经真实 runtime 正确性验证（C7），且非法前缀 continuation 被谓词拒绝。

- 全部达成 → **PASS**；
- 仅前缀/兼容性验证（C1–C6）达成、runtime resume 未认证 → **PARTIAL**；
- runtime 机制无法执行定义的合法恢复 → **BLOCKED**（记录缺失能力，不伪造 checkpoint、不跳过 runtime 路径）。
