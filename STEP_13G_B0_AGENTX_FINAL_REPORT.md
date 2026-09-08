# Step 13G-B0 · AgentX Weka Corpus RQ3 Feasibility Audit 最终报告

**状态：** `AGENTX_RQ3_FEASIBILITY_AUDIT_READY`  
**报告生成时间：** 2026-09-04 17:53:25  
**Audit Root：** `/tmp/pytest-of-wjg/pytest-102/test_run_audit_integration0/agentx_audit`  
**消费对象：** 已冻结的 AgentX Weka corpus (`semianalysisai/cc-traces-weka-062126`)

---

## 1. 执行摘要

本报告对 AgentX Weka trace corpus 执行只读、CPU-only 的结构与可行性审计，
判断其能否作为 FlowState RQ3 的第二个真实 workload。
审计严格区分 **physical prefix sharing**（hash_ids LCP）与 **logical inherited ancestry**。

| 需求 | Gate | 结果 | 证据 |
|---|---|---|---|
| 继承上下文分支 | inherited_context_branching | PASS | 296/393 traces have candidate d(c)>=2 |
| 同时已知 pending continuation | simultaneously_known_pending_continuations | PASS | 312/393 traces show >=2 simultaneous pending |
| 跨 pending 共享候选兼容性 | cross_pending_shared_checkpoint_compatibility | PASS | max d(c) = 734, total candidates with d>=2 = 21749 |
| 上下文长度可行性 | context_length_feasibility | WARN | max input = 989824 tokens, OpenHands limit = 131072 |

### 1.1 关键指标

- **总 trace 数**：393
- **含 subagent 的 trace**：175 / 393
- **候选 checkpoint 总数**：98,827
- **最大 compatibility degree d(c)**：734
- **存在 d(c)≥2 的 trace 数**：296 / 393
- **最大同时 pending 数**：32
- **最大单个请求输入长度**：989,824 tokens

---

## 2. 审计范围与约束

| 项目 | 值 |
|---|---|
| 输入 | 已冻结 AgentX Weka corpus |
| 修改权限 | **只读**；未修改 FlowState 核心、selector、恢复模型 |
| 计算资源 | CPU only；未使用 GPU/SGLang |
| 新增代码 | `evaluation/agentx_structure_audit.py`, `evaluation/agentx_chain_detection.py` |
| 新增测试 | `tests/test_agentx_structure_audit.py` |

---

## 3. Schema 与官方语义

- 顶层字段固定：`id`, `models`, `block_size`, `hash_id_scope`, `requests`，可选 `totals`。
- 请求类型：`n`（normal）、`s`（streaming）、`subagent`（嵌套 inner requests）。
- 当前 corpus 所有 trace 的 `hash_id_scope='local'`，即 hash ID 仅在本 trace 内有效。
- 官方解析器通过两阶段 hash_id LCP 检测将扁平请求重建成 per-agent chain；
  SPAWN vs FORK 由 `fork_depth` 是否为 0 区分。

详见 `SCHEMA_INVENTORY.json`、`SCHEMA_NOTES.md`、`SEMANTICS_EVIDENCE.json`。

---

## 4. SPAWN/FORK 与前缀共享审计

详见 `SPAWN_FORK_AUDIT.json` 与 `PREFIX_SHARING_AUDIT.json`（合并于 SPAWN_FORK_AUDIT.json）。

---

## 5. Logical Workflow 与 Compatibility Degree

我们将每个 trace 视为一个 FlowState `workflow_id`，conversation id 层级构成 `lineage_path`。
root conversation 产生的 checkpoint 位于祖先 lineage，因此可被多个子 conversation 的 pending continuation 共享。

- 总候选数：98,827
- 每 trace 平均 max d(c)：19.0
- d(c)≥2 的候选数：21,749

---

## 6. Shared-Coverage Gate 与 Branching Audit

- Shared-coverage gate：PASS
- Branching audit：312 / 393 traces 存在 ≥2 个同时 pending continuation

---

## 7. 与 OpenHands Main Population 对比

| 维度 | OpenHands Main | AgentX |
|---|---|---|
| snapshots / traces | 168 | 393 |
| pending per snapshot/trace | 4 | ~25.05 (max 737) |
| max d(c) | 1 | 734 |
| cross-pending candidates | 0 | 21,749 |
| branching | 无 | 有（max concurrent pending = 32） |

---

## 8. Context-Length 可行性

- OpenHands replay 输入上限：131,072 tokens
- AgentX 最大请求输入长度：989,824 tokens
- 超出比例：7.55x

**结论**：AgentX 的上下文长度显著超出当前 OpenHands RQ3 引擎配置。若要用作 RQ3 workload，
需要更大的 max context length、按长度过滤 trace，或仅使用其子结构。

---

## 9. 诚实科学结论

### 9.1 可行性判定

AgentX corpus **在结构上满足** FlowState RQ3 same-snapshot 评估所需的核心特征：

1. **继承上下文分支**：subagent 与 flat-chain 检测均发现大量 fork（fork_depth > 0）。
2. **同时已知 pending continuation**：多个子 conversation 的时间区间重叠。
3. **跨 pending 共享候选兼容性**：存在 d(c)≥2 的 checkpoint。

### 9.2 主要限制

1. **上下文长度**：单个请求可达 ~990k tokens，远超 OpenHands 的 131072 限制。
2. **规模与复杂度**：平均每个 trace 约 25 个 conversation，最大 737 个，给选择器/Exact OPT 带来组合压力。
3. **未经过 runtime 验证**：本审计仅基于 trace 元数据；实际 checkpoint 驻留、radix tree 行为、FA frontier 需额外 runtime 采集。

### 9.3 建议

- AgentX 适合作为 **second controlled workload** 用于验证 FlowState 在分支/并发场景下的优势，
  但应设计长度过滤与采样协议，避免直接复用 OpenHands 的 128k 引擎配置。
- 正式使用前需完成一次类似 Step 13E 的 neutral runtime 采集，确认 candidate residency、
  FA frontier、checkpoint 唯一性等 runtime gate。

---

## 10. Artifact 清单

见 audit root 目录下的 JSON/MD 文件。

---

*报告生成于 2026-09-04 · FlowState RQ3 Step 13G-B0 审计*
