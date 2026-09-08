# Step 13G-B0.1 · AgentX Weka Corpus 在线安全兼容性修复审计 最终报告

**状态：** `AGENTX_ONLINE_COMPATIBILITY_AUDIT_READY`  
**报告生成时间：** 2026-09-05 11:03:45  
**Canonical Audit Root：** `/home/wjg/data/agentx/audits/agentx_online_compatibility_20260905_110116`  
**消费对象：** 已冻结的 AgentX Weka corpus (`semianalysisai/cc-traces-weka-062126`)

---

## 1. 执行摘要

本报告修复 Step 13G-B0 中 d(c)=734 与 max pending=32 的语义矛盾，引入 online-safe、epoch-scoped 的 compatibility degree d_t(c)。
审计严格区分 physical prefix sharing 与 logical inherited ancestry，禁止 future leakage，并强制验证 d_t(c) <= |P_t|。

| 需求 | Gate | 结果 | 证据 |
|---|---|---|---|
| 继承上下文分支 | inherited_context_branching | PASS | 84/393 traces have online-safe d_t(c)>=2 |
| 同时已知 pending continuation | simultaneously_known_pending_continuations | PASS | 312/393 traces show >=2 simultaneous pending |
| 跨 pending 共享候选兼容性 | cross_pending_shared_checkpoint_compatibility | PASS | max d_t(c) = 16, epochs with d_t(c)>=2 = 14979, candidate-epoch pairs with d>=2 = 179054 |
| d_t(c) <= |P_t| invariant | per-epoch invariant | PASS | max d_t(c)=16, max |P_t|=32 |
| 上下文长度可行性 | context_length_feasibility | WARN | max input = 989824 tokens, OpenHands limit = 131072 |

### 1.1 关键指标（online-safe）

- **总 trace 数**：393
- **含 subagent 的 trace**：175 / 393
- **最大 |P_t|**：32
- **最大 d_t(c)**：16
- **d_t(c) <= |P_t|  invariant**：PASS
- **存在 online-safe d_t(c)≥2 的 trace 数**：84 / 393
- **存在 d_t(c)≥2 的 epoch 数**：14979
- **最大单个请求输入长度**：989,824 tokens
- **请求 <= 131200 的比例**：0.4699

---

## 2. 审计范围与约束

| 项目 | 值 |
|---|---|
| 输入 | 已冻结 AgentX Weka corpus |
| 修改权限 | **只读**；未修改 FlowState 核心、selector、恢复模型 |
| 计算资源 | CPU only；未使用 GPU/SGLang |
| 新增代码 | `evaluation/agentx_structure_audit.py` |
| 新增测试 | `tests/test_agentx_structure_audit.py` |

---

## 3. Allocation Epoch 定义

Allocation epoch 由以下三类**在线可见**事件触发：

1. 任意 conversation 的 `start_seconds`；
2. 任意 hash-bearing request 的完成时刻（产生新的 candidate 或增大 pending target）；
3. 任意 conversation 的 `end_seconds`。

同一时刻的事件处理顺序：`start` -> `target` -> snapshot -> `end`，从而保证 pending set P_t 采用半开区间 `[start, end)`。
所有 snapshot 仅使用截至该时刻已 materialized 的 candidate 与 pending target，不使用任何未来 trajectory。

---

## 4. 旧 d(c)=734 的 Root Cause

- 旧版最大 d(c) = **734**
- 出现在 trace = `c7f957fd16feff9c2426ddb164994f410ebb`
- candidate = `c7f957fd16feff9c2426ddb164994f410ebb@256`
- 734 代表的对象数 = **733**
- 这些对象包括：未来才产生的 descendant、已经完成的 descendant、spawn（非继承）descendant。
- **根本原因**：旧实现把整个 trace 的 descendant conversation 集合当作同时已知的 P_t，忽略了 epoch scope、active 状态、completion 状态以及 fork/spawn 语义。

详见 `OLD_MAX_COMPATIBILITY_DIAGNOSTIC.json`。

---

## 5. SPAWN/FORK 精确计数

- SPAWN child 总数 = 8482
- FORK child 总数 = 968
- UNKNOWN child 总数 = 0
- traces with SPAWN = 268
- traces with FORK = 196
- traces with both = 146
- traces with neither = 75

fork_depth>0 的示例见 `SPAWN_FORK_AUDIT.json` 中的 `deterministic_fork_examples`。

---

## 6. Pending Concurrency（修复后）

- max |P_t| = 32
- mean |P_t| = 3.26
- median |P_t| = 2.0
- P75 |P_t| = 4.0
- P95 |P_t| = 10.0
- epochs with |P_t| >= 2 = 60564
- epochs with |P_t| >= 4 = 33584
- epochs with |P_t| >= 8 = 11796
- epochs with |P_t| >= 16 = 1058

---

## 7. Epoch-Scoped Compatibility Degree

- max d_t(c) = 16
- max |P_t| = 32
- invariant max d_t(c) <= max |P_t| = PASS
- all per-epoch invariants = PASS

| d_t(c) | candidate-epoch 对数 |
|---|---|
| 0 | 9,769 |
| 1 | 12,018 |
| 2 | 6,242 |
| 3 | 4,246 |
| 4 | 5,824 |
| >=5 | 162,742 |

---

## 8. Online-Safe Shared Coverage

- traces with online-safe d_t(c)>=2 = 84
- epochs with online-safe d_t(c)>=2 = 14979
- ONLINE_SAFE_SHARED_COVERAGE = YES

10 个确定性示例见 `ONLINE_COMPATIBILITY_DEGREE.json` 中的 `shared_coverage_examples`。

---

## 9. Context-Length 可行性

- OpenHands replay 输入上限：131,072 tokens
- AgentX 最大请求输入长度：989,824 tokens
- 请求 <= 131200: 46442 / 98827 (0.4699)
- 完全可 replay 的 trace 数（所有请求 <= 131200）: 85 / 393

**结论**：上下文长度仍是 AgentX 作为 RQ3 workload 的主要 blocker。

---

## 10. 诚实科学结论

### 10.1 旧 d(c)=734 为什么错误

它把整个 trace 的 descendant conversation 集合当作同一 epoch 的 P_t，违反了 FlowState 的正式定义 d_t(c) = |{p ∈ P_t : c compatible with p}|。

### 10.2 online-safe d_t(c) 的真实 max

修复后 max d_t(c) = **16**，且始终 <= max |P_t| = **32**。

### 10.3 AgentX 是否仍存在 shared coverage

是。有 84 个 trace 在至少一个 epoch 出现 d_t(c)≥2。

### 10.4 是否补足 OpenHands 结构缺口

是的。OpenHands Main Population 的 max d=1 且无 branching；AgentX 在 online-safe 语义下仍提供 branching 与 cross-pending shared coverage。

### 10.5 是否值得进入 formal snapshot construction

**值得，但需先解决 context-length 限制**。建议下一步：
1. 设计长度过滤/采样协议；2. 在受控 runtime 下验证 candidate residency 与 FA frontier。

---

## 11. Artifact 清单

见 canonical audit root 目录下的 JSON/MD 文件。

---

*报告生成于 2026-09-05 · FlowState RQ3 Step 13G-B0.1 审计*
