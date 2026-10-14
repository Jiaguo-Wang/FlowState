# 最小 Runtime 恢复验证报告（Task A 总报告）

日期：2026-10-13 ｜ 目录：`evaluation/submission_p0_executable_recovery/`
输入任务：FlowState 投稿冲刺任务 A（Executable Compatibility 修复与最小 Runtime 恢复验证，ICDE 2026-11）

---

## 0. 结论（TL;DR）

**最终判定：PASS**（任务定义的完成门全部达成；合同层面 P1/P2 边为 ADMISSIBLE + C7 认证，
C5 的 slot generation UNKNOWN 如实保留，不冒充 CERTIFIED）。

最终运行：2026-10-13 01:07:07–01:07:56（本地，GPU 0，全程 49 秒，由共享机器 GPU
空闲观察器在两卡同时归零的窗口自动触发；执行后两卡归零，无冲突）。证据：
`runtime_evidence.json`（status=PASS）。

核心事实：

- 冻结 SGLang v0.5.17 上，一个源自**真实 AgentX 语料 hash 块**的 8192-token 共享前缀 H
  被物化为 exact radix checkpoint（FA-KV + Mamba 双驻留，node 67，slot [3]）后，
  **两个不同 continuation 都以 physical_fa_hit=8192、executable_prefix=8192、
  replay_gap=0 完成恢复**——8192 token 前缀全部命中缓存，各自仅需计算 256-token
  后缀 + 1 个生成 token；
- 恢复路径输出与全量重算参考**完全一致**（P1: 220，P2: 248046；temperature=0
  确定性设置，且 anchor 物化两次输出均为 220）；
- **驱逐反事实**：用正式 primitive 仅驱逐 C0 的 Mamba 状态（primitive 自带 proof：
  `only_target_mamba_changed=true, fa_unchanged=true, same_node=true`），第三个合法
  continuation P3 的 executable_prefix 从 8192 **塌缩至 4096**、replay_gap 从 0
  **增至 4096**，而 FA 命中保持 8192、输出仍与全量重算一致——证明 P1/P2 的恢复
  依赖的是 **Recurrent State**，不是仅 FA-KV；
- 非法前缀 continuation（token 历史在 L 之前 4096 处分歧）被谓词在 **C2 准入阶段
  拒绝**，runtime 物理命中也精确停在 4096（executable=0，输出仍与其自身全量重算
  一致，排除确定性破坏）。

## 1. 任务与完成标准

| 任务 | 交付物 | 状态 |
|---|---|---|
| A1 严格合同 | `compatibility_contract.md` + `executable_compatibility.py` | ✅ 完成 |
| A2 最小合法样本 | `build_minimal_cases.py` → `compatibility_cases.json` | ✅ 完成 |
| A3 最小真实 Runtime Resume | `minimal_recovery_smoke.py` + `run_minimal_recovery_smoke.sh` → `runtime_evidence.json` | ✅ **PASS**（最终运行全阶段通过） |
| A4 修复提案 | `agentx_fix_proposal.md` | ✅ 完成（提案，未实施） |
| 总报告 | 本文件 | 本文件 |

PASS 门（任务定义）：≥1 个真实物化 checkpoint 在同一模型下以合法 token 前缀恢复
两个不同 continuation，经真实 runtime 正确性验证 + 非法前缀被拒绝。
**未通过放宽兼容性标准获得任何结论**（UNKNOWN 条件如实保留，见 §9）。

## 2. 交付物清单

| 文件 | 说明 |
|---|---|
| `compatibility_contract.md` | ExecutableCompatible(c,p) 七条件合同（A1） |
| `executable_compatibility.py` | 谓词实现（只读、无副作用；C1–C6 准入 + C7 动态认证） |
| `build_minimal_cases.py` | 样本构造（真实 hash 前缀 + CONTROLLED 后缀） |
| `compatibility_cases.json` | 样本：H(8192) + P1/P2/P3/P_bad + 完整溯源 |
| `minimal_recovery_smoke.py` | 六阶段 GPU 冒烟（REF→MAT→RES→NEG→EVICT→REMAT） |
| `run_minimal_recovery_smoke.sh` | 独立可复现入口（all/build/smoke；自动选空闲 GPU，前后 nvidia-smi） |
| `runtime_evidence.json` | 最终运行全部原始证据（status=PASS） |
| `agentx_fix_proposal.md` | 70 条 Formal-only 边最小修复提案（A4） |
| 本文件 | 总报告 |

独立复现：`./evaluation/submission_p0_executable_recovery/run_minimal_recovery_smoke.sh`
（构造样本 + GPU 冒烟一步完成；要求空闲 GPU 与 docker 镜像
`lmsysorg/sglang:v0.5.17-cu129-runtime`）。

## 3. A1 合同摘要（详见 compatibility_contract.md）

四层证据模型（不可互相替代）：L1 逻辑祖先（仅候选生成器）→ L2 token 前缀逐位一致
（C2）→ L3 物化 checkpoint 合法（C1/C3–C6）→ L4 实际恢复正确（C7 动态认证）。

| 条件 | 判定 | 状态来源 |
|---|---|---|
| C1 身份一致 | model/tokenizer（含配置摘要） | 样本 identity + 引擎 |
| C2 Token 历史精确一致 | `Tokens(c)[:L] == Tokens(p)[:L]` 逐位 | 离线可判 |
| C3 位置合法 | `0 < L ≤ T` | 离线可判 |
| C4 恢复条件 | FA-KV 与 Recurrent 双驻留（inspect 实测） | runtime |
| C5 物理身份 | node_id + prefix_sha256 + FA 内容摘要 + mamba slots（slot generation 账本缺失→UNKNOWN） | runtime |
| C6 保护 | 四个 lock/session 引用计数全零（inspect 实测） | runtime |
| C7 动态认证 | physical/executable/gap 计量 + 输出一致 + 驱逐反事实 | runtime |

三态：REJECTED（任一 FAIL）/ ADMISSIBLE（C1–C6 无 FAIL）/ CERTIFIED（全 PASS，
本轮因 C5 的 slot generation UNKNOWN 而如实止步于 ADMISSIBLE+C7-PASS）。

## 4. A2 最小合法样本（compatibility_cases.json）

- **H（真实前缀）**：冻结语料 `traces.jsonl` trace `002001296e8a8c38ad9d7cc436d691afc602`
  request#1 的前 128 个 hash 块，经**与正式 B2 采集完全相同**的
  `_compose_prompt_tokens` 语义 + 真实 Qwen3.5-9B tokenizer 物化 = 8192 token
  （恰好 128 块 × 64，无 tail-fill）；
  `sha256(H) = 4b26742e57e97f09a99217594d440b64472d4524337d43ec7228708358e365d9`；
- **C0**：materialize(H) 后 radix 树在 H 输入边界的 exact 节点（L=8192）；
- **P1/P2**：H + 各自 256-token **CONTROLLED** 后缀（确定性 sha256 采样，显式标记，
  不冒充原生 AgentX fork；长度 256 使分支终止点对齐 mamba track 边界 8448=33×256，
  终止状态才落盘——见 §7 工程发现 2，最终运行证实：两分支各自物化新节点 68/69）；
- **P_bad**：H 在块对齐位置 4096 处翻转一个 token（64324→64325）+ P1 后缀
  ——分歧严格在 L 之前；
- **P3**：H + 第三个 CONTROLLED 后缀，仅用于驱逐反事实，不参与合同判定；
- 模型身份：`qwen3.5-9b@d0883072e01861ed`；tokenizer 身份
  `qwen3.5-9b-tokenizer@316230d6a809701f:vocab=248077`。

## 5. A3 Runtime 协议与结果（runtime_evidence.json）

引擎：`FormalEndToEndGateEngine`（WP3B 传输层 + instrumented match_prefix），
配置 = `runtime_gate.ENGINE_CONFIGURATION`（context 45056、mamba pool 16、
extra_buffer、track_interval 256、**禁 CUDA graph / overlap**，temperature=0、
max_new_tokens=1），模型挂载 `/model`，SGLang 0.5.17（镜像
`lmsysorg/sglang:v0.5.17-cu129-runtime`）。

六阶段协议与结果（全部断言通过）：

| 阶段 | 操作 | 结果 |
|---|---|---|
| REF | 全新引擎全量重算 P1/P2/P3/P_bad（互相 flush 隔离） | 输出 220 / 248046 / 220 / 220，physical_fa_hit 全部 = 0（零命中基线成立） |
| MAT | materialize(H) → inspect C0 | anchor 输出 220（physical=0 基线）；**node=67**，prefix_sha256 == token_digest(H)，FA+Mamba 双驻留（mamba 状态在位置 8192、slot [3]，path_mamba_positions 实测）；census mamba 节点数 = 1 |
| RES | P1、P2 从 C0 恢复 | P1: output=220==REF，P2: output=248046==REF；两者 **physical=8192、executable=8192、gap=0**；C0 node 67 / slot [3] / digest 全程稳定；census 1→3，**新增分支节点 [68, 69]**（各分支终止状态独立物化，不共享可写 slot） |
| NEG | P_bad | **physical_fa_hit=4096**（精确停在分歧点），executable=0，gap=4096；output=220==自身 REF（确定性无破坏） |
| EVICT | 正式 primitive 仅驱逐 C0 的 Mamba → P3 | primitive proof：`sanity_check=true, fa_unchanged=true, only_target_mamba_changed=true, same_node=true, tree_unchanged=true`；inspect 确认 C0 Mamba 消失、FA 保留；P3：**physical=8192（FA 不受影响）、executable=4096（塌缩）、gap=4096（增大）**，output=220==自身 REF |
| REMAT | flush 后重新 materialize(H) | anchor 输出 220==首次（确定性）；prefix_sha256 与 FA-KV 内容摘要（target_full_value_sha256）跨物化稳定；双驻留恢复（新节点 80——flush 后 node id 重新分配合法，内容摘要一致为准） |

隔离性证据：同一棵树内 C0 的 mamba slot（[3]）在两次分支恢复全程不变
（`within_tree_c0_mamba_slots_stable=true`）；两分支各自物化新的终止 Mamba 状态
节点 68/69——不共享可写 slot。

## 6. 谓词判定（最终运行，runtime 证据注入后）

| continuation | C1 | C2 | C3 | C4 | C5 | C6 | C7 | 结论 |
|---|---|---|---|---|---|---|---|---|
| P1 | PASS | PASS | PASS | PASS | UNKNOWN | PASS | **PASS** | ADMISSIBLE + C7 认证 |
| P2 | PASS | PASS | PASS | PASS | UNKNOWN | PASS | **PASS** | ADMISSIBLE + C7 认证 |
| P_bad | PASS | **FAIL** | PASS | PASS | UNKNOWN | PASS | UNKNOWN | **REJECTED**（准入阶段拒绝） |

- C5 UNKNOWN：runtime 未暴露 slot generation 账本（node id、token digest、FA-KV
  内容摘要、mamba slot 列表均实测一致），如实标注；
- P_bad 的 C7 UNKNOWN：被准入拒绝（C2 FAIL）后不再进入动态认证，符合合同分层。

## 7. 迭代记录：三次运行的两个工程发现（对 A4 直接有用）

1. **生成 token 不进入 radix 树路径**（运行 1 失败根因）：materialize(H) 后对
   H+生成token（8193）的 inspect 报 `exact prefix path missing after 8192/8193`
   ——exact 节点只存在于请求**输入边界**。这正是正式采集以 token_pos（请求输入
   末端）为 checkpoint 位置的根因；C0 定义随之对齐。
2. **分支终止 Mamba 状态只在 track 边界落盘**（运行 2 失败根因）：63+1 token 的
   分支终止在 8256（非 256 倍数）时 census 无新 Mamba 行；后缀对齐到 8448
   （33×256）后，最终运行证实各分支终止状态物化为独立节点 68/69。此语义已写入
   修复提案 §2.3，矩阵重生成的分支采样必须跨 track 边界。
3. **`mamba_branching_seqlen` 的实测行为**（最终运行三个数据点）：P1/P2 全前缀
   恢复路径 = None；P_bad 部分命中 = 4096（= 其 FA 前沿/分歧点）；驱逐后 P3 =
   8192（FA 命中全长但可用 Mamba 路径变短）。即该字段只在 FA 前沿与可用 Mamba
   状态不一致的情形下填充，确切语义需 runtime 侧确认（已列入修复提案 §2.3 的
   API 文档化项）。因此 C7 的 Recurrent 依赖主证据采用**驱逐反事实**（直接因果，
   见 §5 EVICT），不依赖该字段。

三次运行的全部中间证据均保存在 git 历史与本报告引用的 runtime_evidence.json
（每次运行覆盖写入，最终版为完整协议通过版；运行 1/2 的失败根因即上述发现 1/2）。

## 8. 七个必答问题

1. **是否存在已证实的合法可执行共享 Recurrent Checkpoint？**
   **是。** minimal_C0（真实物化 exact 节点 67，FA+Mamba 双驻留，digest
   4b26742e…e365d9）在同一模型内被两个不同 continuation（P1/P2）合法恢复并通过
   C7 动态认证（physical=executable=8192、gap=0、输出==全量重算、驱逐反事实证明
   Recurrent 依赖）。合同层状态为 ADMISSIBLE + C7 认证（C5 generation UNKNOWN，
   见 §9）。
2. **哪个模型 / checkpoint / continuations？**
   Qwen3.5-9B（/home/wjg/models/qwen3.5-9b，身份 d0883072e01861ed）；checkpoint
   minimal_C0 = 真实语料 hash 前缀 H（8192 token，trace 002001296e… request#1）
   的输入边界 exact 节点（node 67，mamba slot [3]，位置 8192）；P1/P2 = H + 互不
   相同的 256-token CONTROLLED 后缀。
3. **prefix 与物理身份是否一致？**
   是——inspect 的 prefix_sha256 == token_digest(H)，跨两次物化与分支恢复稳定
   （`digest_stable_across_materializations=true`）；FA-KV 内容摘要
   （target_full_value_sha256）跨物化稳定；node id（67）与 slot（[3]）在同一棵
   树内全程稳定（flush 后 node id 重分配为 80，属合法行为，以内容摘要为准）；
   slot generation 账本 runtime 不暴露 → C5 如实 UNKNOWN（见 §9）。
4. **两次 Resume 是否成功？**
   **是。** P1: output=220、P2: output=248046，均与全量重算参考完全一致；两者
   physical_fa_hit=8192、executable_prefix=8192、replay_gap=0；各自仅需计算
   256-token 后缀 + 1 个生成 token。
5. **为什么 Formal-only 边不可信？**
   70 条边来自 B1 逻辑祖先 + laminar-cluster 人工重编码（`encode_usable_compatibility`），
   绕过了 token 前缀与物理恢复两层证据；回放侧每个请求的 token 由自身 hash_ids
   独立合成（`_build_replay_plan` → `_compose_prompt_tokens`），前缀连续性从未是
   构造不变量——审计例 g000/req0000 的 Raw 匹配为空（连首块都不一致）即此类。
   逻辑 DAG 上的共享祖先 ≠ 可执行恢复，本报告的最小样本证明的是**后者可以在
   构造正确时成立**（含 Recurrent State 依赖的因果证明），而非 70 条边本身成立。
6. **最小修复方法？**
   见 `agentx_fix_proposal.md`：(1) `_build_replay_plan` 祖先绑定合成（前缀连续成为
   构造不变量；hash 不延伸的边如实转负）；(2) `encode_usable_compatibility` 移除
   laminar-cluster 重编码、以合同谓词为唯一准入；(3) 按需补 runtime API
   （slot generation 账本、稳定计量接口、分支采样跨 track 边界）。均为采集器/
   矩阵生成侧修改，不动 SGLang 核心，不覆盖 573 个正式结果。
7. **能否进入 Global Allocation 实验？**
   合同与谓词已就绪且经 runtime 验证，实验**设计**可以开始；正式对比结论应等
   Executable Compatibility Matrix 按合同重生成（CERTIFIED/ADMISSIBLE 边集）后
   进行。本轮遵守边界：未重跑 573 条件、未修改任何正式结果与 Recovery Cost
   Model。

## 9. 诚实性声明（未放宽任何标准）

1. **C5 slot generation UNKNOWN**：runtime 未暴露 slot generation 账本；node id、
   token digest、FA-KV 内容摘要、mamba slot 列表均实测，但 generation 缺失如实
   标注 → P1/P2 边保持 ADMISSIBLE+C7-PASS，不冒充 CERTIFIED；
2. **mamba_branching_seqlen 语义未定**：三个数据点（None/4096/8192）表明其只在
   FA 前沿与可用 Mamba 状态不一致时填充，不把它当作 C7 主证据；C7 的 Recurrent
   依赖由驱逐反事实（primitive 自带 only_target_mamba_changed proof）证明；
3. **CONTROLLED 后缀**：P1/P2/P3 的分支后缀为确定性合成，显式标记，永不标为
   原生 AgentX fork；真实部分是 H（语料 hash 块 + 正式 compose 语义 + 真实
   tokenizer）；
4. **最小规模**：单 checkpoint、两分支——证明机制与合同可执行，不声称 70 条边
   已修复或 573 结果已更新；
5. **失败迭代如实记录**：运行 1（8192/8193 边界）、运行 2（8256 非 track 边界）
   的失败根因即 §7 发现 1/2；证据文件无论成败均落盘。

## 10. 完成标准对照与判定

| PASS 门条件 | 证据 | 达成 |
|---|---|---|
| ≥1 真实物化 checkpoint | C0 exact 节点 67，FA+Mamba 双驻留，digest 校验一致 | ✅ |
| 同一模型、合法 token 前缀恢复两个 continuation | P1/P2 前缀与 C0 在 L=8192 逐位一致，physical=executable=8192 | ✅ |
| 真实 runtime 正确性验证 | 计量（gap=0）+ 输出==全量重算（220/248046）+ 驱逐反事实（executable 8192→4096，only_target_mamba_changed） | ✅ |
| 非法前缀被拒绝 | P_bad 谓词 C2 FAIL + runtime physical_fa_hit 精确停在 4096 | ✅ |

**最终判定：PASS。**（合同层 P1/P2 边 = ADMISSIBLE + C7 认证；C5 slot generation
UNKNOWN 诚实保留，按合同不升级为 CERTIFIED——这不妨碍任务 PASS 门，因其只要求
runtime 正确性验证，已由 §5 全部证据满足。）
