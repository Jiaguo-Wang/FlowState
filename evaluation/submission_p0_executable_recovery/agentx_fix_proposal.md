# AgentX Formal-only 边最小修复提案（Task A4）

日期：2026-10-12（2026-10-13 更新：补入最终运行结论与第三条 runtime 事实）｜ 性质：**提案，不实施**（任务 A 明确：不自动重建工作负载、不覆盖 573 个正式结果、不修改正式矩阵）

---

## 1. 问题定位：70 条 Formal-only 边为什么会存在

### 1.1 两侧判定语义（已冻结的事实）

- **Raw（B2 实际回放）**：`agentx_runtime_snapshot_collection.py:884-895` 用
  `_prefix_match(req.token_ids[:cp.recurrent_prefix_length], pending_toks)` —— 拿
  **实际回放构造出的 token 序列**做逐 token 前缀比较；
- **Formal**：`rq3_agentx_formal_policy_evaluation.py:172-234`（`encode_usable_compatibility`）
  用 **B1 逻辑祖先**（lineage path 前缀 + is_inherited + token_position ≤ active target）
  + L≤T 过滤，再用人工 laminar-cluster lineage（"兼容簇：<digest>"）重编码进矩阵。

审计结论（`gate1_step3_compatibility_retention_contract_audit/audit_report.md`）：
23 snapshot 中 Formal 正边 114 = 70 Formal-only + 44 双侧；Raw 正边 457 = 413 Raw-only + 44。
70 条 Formal-only 边全部满足 L>0、L≤T、正式正收益，但对其对应 pending token 序列未通过
Raw 完整前缀谓词（例 g000/req0000：L=37760，T=110527，Raw 匹配为**空**——连第一个
hash 块都不一致）。

### 1.2 断裂的精确机制（根因链）

1. **AgentX 语料无文本，token 靠逐请求独立合成**。回放计划
   （`agentx_runtime_snapshot_collection.py::_build_replay_plan`，:309-314 materialized、
   :334-339 pending）对**每个请求**分别调用
   `_compose_prompt_tokens(conv.request_hash_ids[ridx], conv.request_input_lengths[ridx], …)`：
   从**该请求自己的 hash_ids** 展开 64-token 块、截断/按
   `seed=compose-{该请求前3个hash_id}` 做 tail-fill，凑到**该请求自己的精确输入长度**。
2. **前缀连续性从来不是构造不变量**。两个请求的合成 token 前缀连续，当且仅当
   后者的 hash_ids 逐块延伸前者的 hash_ids **且** 输入长度对齐。同会话内的累积请求
   通常满足；**跨会话 fork/spawn 边不保证**——语料的逻辑谱系（FORK inherited 结构元数据）
   与块级内容指纹（hash_id 前缀）是两套证据，`hash_id_scope: "local"` 下内容级身份
   不跨会话保证。
3. **Formal 用逻辑谱系替代了内容约束**。`encode_usable_compatibility` 的 laminar-cluster
   重编码把「逻辑上是祖先」直接当作「可执行兼容」写进矩阵，跳过了 token 前缀与物理
   恢复两层证据——这正是审计判定为语义冲突、Gate1/Step3 BLOCKED 的原因。

一句话：**70 条边不是 runtime 拒绝了合法恢复，而是回放构造从未产生过连续 token 前缀，
Formal 的逻辑重编码把这一缺口编码成了正边。**

### 1.3 与最小样本（A2/A3）的关系

A2/A3 证明了断裂是**采集器构造问题，不是 runtime 能力问题**：只要 continuation 的
token 前缀在 L 处与物化 checkpoint 精确一致（A2 的 H 即用与正式采集**完全相同**的
`_compose_prompt_tokens` 语义合成），冻结 SGLang v0.5.17 就能完成
exact-node 物化 + FA/Mamba 双驻留 + 前缀命中恢复 + 确定性输出一致（见
`runtime_evidence.json`；2026-10-13 最终运行完整通过：P1/P2 恢复认证
physical=executable=8192、gap=0、输出==全量重算，P_bad 在分歧点 4096 被拒，
驱逐反事实 executable 8192→4096）。因此修复应落在**回放构造与矩阵生成**两侧，
不需要先改 SGLang 核心。

A3 冒烟迭代还产出三个对 §2.3 直接有用的 runtime 事实：

1. **生成 token 不进入 radix 树路径**：materialize(H) 后，exact 节点只存在于
   H 输入边界（8192），对 H+生成token（8193）的 inspect 报
   `exact prefix path missing after 8192/8193`——这解释了正式采集为何以
   请求输入边界（token_pos）为 checkpoint 位置；
2. **分支终止 Mamba 状态只在 track 边界落盘**：63+1 token 的分支终止在 8256
   （非 256 边界）时 census 无新 Mamba 行；后缀对齐到 8448（33×256）后终止
   状态才物化为独立节点。矩阵重生成的"每分支独立状态"观测必须把这一点计入
   采样设计（分支长度需跨 track 边界；最终运行证实：256-token 后缀对齐
   8448=33×256 后，两分支终止状态物化为独立节点 68/69）。
3. **`mamba_branching_seqlen` 只在 FA 前沿与可用 Mamba 状态不一致时填充**：
   最终运行三个数据点——P1/P2 全前缀恢复 = None、P_bad 部分命中 = 4096
   （= 其 FA 前沿/分歧点）、驱逐 C0 Mamba 后的 P3 = 8192（FA 命中全长但可用
   Mamba 路径变短）。确切语义需 runtime 侧确认；C7 的 Recurrent 依赖主证据
   采用驱逐反事实（primitive proof：only_target_mamba_changed=true）。

---

## 2. 最小修复方案（consistency replay）

### 2.1 修复点 1（首选提升点）：`_build_replay_plan` 的祖先绑定合成

**位置**：`evaluation/agentx_runtime_snapshot_collection.py::_build_replay_plan`

**现状**：每个 materialized/pending 请求独立合成 token（§1.2）。

**修改**（对 pending 侧）：pending p 不再从自身 hash_ids 全量合成，而是——

1. 取 p 的 B1 lineage 最深已物化祖先 c（B1 逻辑谱系只在这里用作**候选生成器**）；
2. `Tokens(p) := Tokens(c) + Δ(p)`，其中 Δ(p) 由 p 自身**超出 c 的块边界之后的
   新 hash 块**展开（仅最后一个不足块允许 per-request tail-fill，seed 绑定
   `c 的 digest + p 的新块序列`）；
3. 若 p 的 hash_ids **并不逐块延伸** c 的 hash_ids（即 §1.2 第 2 步条件不成立），
   如实记录 `first_divergent_block` 并把该 (c, p) 对标为**不可合成**——
   **不允许**用任何重编码把这类边变成正边。

**效果**：凡语料内容指纹支持共享的边，token 前缀连续成为**构造不变量**
（等价于把 A2 样本的构造规则推广到全量回放）；凡不支持的边（至少包括 g000/req0000
这类 Raw-empty 源），修复后的矩阵会**重现 Raw 的否定**而不是 Formal 的肯定。

**关键性质**：该修改让「逻辑祖先」与「可执行前缀」两种判定在数据构造层一致化——
不是放宽兼容标准，而是把 70 条边按证据分成「本可连续（修复后合法转正，需 C7 认证）」
与「内容本不连续（修复后合法转负）」两类。

### 2.2 修复点 2：`encode_usable_compatibility` 移除 laminar-cluster 重编码

**位置**：`evaluation/rq3_agentx_formal_policy_evaluation.py:172-234`

**修改**：矩阵边的唯一准入改为 A1 合同谓词
（`evaluation/submission_p0_executable_recovery/executable_compatibility.py`）：
- C1–C6 离线判定（输入来自修复点 1 的回放 + inspect 通道物化证据）；
- C7 由 runtime 探针认证（全量逐边或分层抽样认证，未认证边保持 ADMISSIBLE 不得计正）；
- 删除「兼容簇：<digest>」人工 lineage 编码路径——它正是绕过前缀失配的机制。

### 2.3 修复点 3：SGLang Compatibility API 缺口（按需，非前置）

最小修复**不依赖**任何 SGLang 改动（A3 已验证现有 API 足够认证 C1–C4/C6/C7 与
C5 的 node+digest 部分）。规模重生矩阵时建议补两项（需批准后才动 runtime）：

| 缺口 | 影响条件 | 提议 |
|---|---|---|
| slot generation 账本不可读 | C5 完整 PASS（现只能 node_id + prefix_sha256 + FA 内容摘要 + mamba slots，generation=UNKNOWN） | 在 inspect path 快照中暴露每 slot 的 generation 计数 |
| 逐请求恢复计量非稳定接口 | C7 规模化认证 | 把 WP3B 传输层的 `flowstate_runtime_metrics`（physical_fa_hit / executable_prefix / replay_gap）升格为正式 checkpoint_control action |
| `mamba_branching_seqlen` 只在 FA 前沿与可用 Mamba 状态不一致时填充（实测 None/4096/8192 三数据点，语义待确认） | C7 的 Recurrent 依赖直接证据 | A3 已用「evict 反事实」替代（驱逐 C0 Mamba 后 executable 8192→4096、gap 0→4096，proof: only_target_mamba_changed=true）；建议 runtime 文档化该字段语义 |
| 生成 token 不入树、分支状态仅在 track 边界落盘 | 矩阵重生成采样设计 | 采集器文档化这两条语义（A3 实测），checkpoint 位置约定对齐请求输入边界 |

### 2.4 新矩阵的生成流程（修复落地后的形态）

1. 按 snapshot 回放 materialized 请求（祖先绑定合成）；
2. 对每个 candidate 做 inspect（物化证据：node、digest、双驻留、lock refs）；
3. 逐 (candidate, pending) 对运行合同谓词 → 每条边带 C1–C7 逐条件状态；
4. 仅 CERTIFIED 边计入新 Executable Compatibility Matrix；ADMISSIBLE 边单列
   （含 first_divergent_block / 未认证原因）；
5. **新矩阵与旧矩阵并列存档**（版本化目录），不覆盖、不改写 573 个正式结果；
   对 70 条 Formal-only 边输出逐条处置：`repaired_legal`（修复后合法转正）/
   `confirmed_illegal`（内容不连续，转负）/ `pending_certification`。

## 3. 边界与禁止事项（重申）

- 本提案不实施任何代码修改；修复点 1/2 动的是**采集与矩阵生成**，不改 SGLang 核心、
  不改 Recovery Cost Model、不改 FlowState/PCB/LDR/WFC 算法逻辑；
- 不自动重跑 573 条件；新矩阵生成需人工批准后单独执行；
- 合成分支（CONTROLLED 后缀）永远标记 CONTROLLED，不得标成原生 AgentX fork；
- Token Prefix 匹配（C2）永远不等于已恢复（C7 缺失时最高 ADMISSIBLE）。
