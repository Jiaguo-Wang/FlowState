# LMCache 真实消息 Runtime 恢复验证报告（Task A5，案例 lmcache_10）

日期：2026-10-13 ｜ 目录：`evaluation/submission_p0_lmcache_recovery/`
输入任务：FlowState 投稿冲刺任务 A5（LMCache 真实消息的 Executable Recovery 验证）

---

## 0. 结论（TL;DR）

**最终判定：PASS**（六步协议全部通过，完成门全达成；C5 slot generation UNKNOWN
如实保留，不冒充 CERTIFIED）。

唯一目标——"使用 LMCache 的完整多轮消息所生成的真实 Token 输入，在 SGLang
Hybrid Runtime 中物化一个共同 Recurrent Checkpoint，并验证它可以正确支持两个
具有不同后续输入的 Continuations"——**在 lmcache_10 上达成**：

- 冻结的完整多轮消息 Token 输入（deepseek 会话 16338 token / minimax 会话
  21520 token，共同前缀 10206，其中 [0,9984) 逐位一致）**不经任何重模板化、
  直接以 input_ids 提交**；
- 共同 Recurrent Checkpoint C0 = [0,9984)（39×256 track 边界）**真实物化**为
  exact radix 节点（FA-KV + Mamba 双驻留，inspect 实测，node 57，slot [3]）；
- **两条独立恢复**（每次从只含 C0 的干净状态出发）：P1/P2 均
  physical=executable=9984、gap=0，输出 token（9764 / 760）与各自隔离的
  Full Recompute **完全一致**；
- **Recurrent State 依赖的因果证明**：正式 primitive 仅驱逐 C0 的 Mamba 后，
  P1 的 executable 9984→0、gap 0→9984，而 physical 保持 9984、输出仍一致；
- 非法前缀（9984 之前的 4992 处翻转一个 token，CONTROLLED 标记）被谓词
  **C2 准入阶段拒绝**，runtime 物理命中精确停在 4992。

最终运行：2026-10-13 07:51:57–07:53:06 UTC（本地 15:51–15:53，GPU 1，全程
69 秒；运行时两卡状态：GPU0 被同事占用 53 GB 未触碰，GPU1 执行前后均 14 MiB）。
证据：`runtime_evidence.json`（status=PASS）。

## 1. 完成门对照

| PASS 门条件 | 证据 | 达成 |
|---|---|---|
| C0 真实物化 | exact 节点（首物化 node 57），FA+Mamba 双驻留，digest==token_digest([0,9984))，lock 引用全零 | ✅ |
| P1/P2 使用原始冻结完整 Token 输入 | 文件 sha256==manifest；长度/LCP/分歧点逐项复核；`generate` 以 input_ids 直提（runtime_gate.py:152-173，无 decode/re-template） | ✅ |
| 两条 Continuations 实际合法恢复 | RES1/RES2 独立环境：physical=executable=9984、gap=0；谓词 ADMISSIBLE | ✅ |
| 输出与隔离 Full Recompute 一致 | P1: 9764==9764，P2: 760==760（temp=0、max_new_tokens=1、禁 CUDA graph/overlap） | ✅ |
| 驱逐反事实证明 Recurrent State 影响 | executable 9984→0、gap 0→9984、physical 9984 不变、输出不变；proof: only_target_mamba_changed=true, fa_unchanged=true | ✅ |
| 非法 Prefix 被拒 | P_neg 谓词 C2 FAIL + physical_fa_hit 精确停 4992 | ✅ |

C5 slot generation UNKNOWN → 边保持 **ADMISSIBLE + C7 认证**，不冒充
CERTIFIED（沿用 Task A 合同，未放宽任何标准）。

## 2. 交付物清单

| 文件 | 说明 |
|---|---|
| 本文件 | 总报告（含八个必答问题与诚实性声明） |
| `runtime_evidence.json` | 最终运行全部原始证据（status=PASS，六阶段） |
| `input_identity.json` | Step 1 输入身份冻结（文件 sha、token digests、身份合同比对、负例派生记录） |
| `compatibility_results.json` | 谓词判定结果（C1–C7 逐条件） |
| `reproduce.sh` | 独立复现入口（docker 模板、自动选空闲 GPU、前后 nvidia-smi） |
| `lmcache_recovery_smoke.py` | 最小测试代码（INIT→REF→MAT→RES1/RES2→DUAL→NEG→EVICT→谓词） |

独立复现：`./evaluation/submission_p0_lmcache_recovery/reproduce.sh`
（要求一张空闲 GPU 与镜像 `lmsysorg/sglang:v0.5.17-cu129-runtime`；
不需要 agentx 数据挂载，输入为仓库内冻结文件）。

## 3. 案例与输入身份（Step 1：冻结与核验）

来源：`sammshen/lmcache-agentic-traces`，task `swebench__django__django-15731`，
两个不同模型供应商的完整会话（`__deepseek` / `__minimax`），Phase 2 以同一
Qwen3.5-9B tokenizer `apply_chat_template(tokenize=True, add_generation_prompt=True)`
冻结为 input_ids（无 tools/enable_thinking、无截断）。

逐项核验（全部通过，`input_identity.json`）：

| 声明 | 实测 |
|---|---|
| P1 长度 16338 / P2 长度 21520 | ✅ |
| LCP=10206（P1[10206]=40 vs P2[10206]=27） | ✅ |
| [0,9984) 逐位一致 | ✅ |
| 文件 sha256 == manifest input_sha256 | ✅（2eee5178… / 05f95958…） |
| token_digest([0,9984)) | fb6e08bbac6a5cefbb2ea9929f37499ca15728a7f1276fe80f2deeba327edd87 |
| token_digest(P1 全长) / (P2 全长) | c7951033e7665a58… / b013bfc3af293c28… |
| 引擎模型文件 sha == Phase 2 tokenizer_identity.json | ✅ 四个文件全一致（config d0883072e01861ed / tokenizer_config 316230d6a809701f / tokenizer 5f9e4d49… / chat_template a4aee8af…）——与 Task A run-3 引擎身份也一致（跨轮 C1 兼容） |

提交方式：**直接提交保存的 input_ids**（`runtime_gate.generate` 以
`input_ids=list(token_ids)` 发送，无 decode/re-template，代码引用
runtime_gate.py:152-173）；未重构后缀、未删除消息、未调整系统提示。

CONTROLLED 负例：P1 在位置 4992（=9984//2，严格在 C0 之前）翻转一个 token
（33955→33956），显式标记 CONTROLLED，不冒充原始 LMCache 数据。

## 4. 运行环境与协议

引擎：`FormalEndToEndGateEngine` + WP3B 测试传输层（控制端口 49939），
配置 = `runtime_gate.ENGINE_CONFIGURATION`（context 45056、
mem_fraction_static 0.40、max_mamba_cache_size 16、extra_buffer、
mamba_track_interval 256、禁 CUDA graph/overlap、temp=0、max_new_tokens=1），
模型挂载 `/model`，docker `lmsysorg/sglang:v0.5.17-cu129-runtime`（SGLang
0.5.17）。max_running_requests 被 mamba pool 上限压到 4（串行单请求，无影响）。

八阶段协议（全部断言通过；独立恢复用 flush→重新只物化 C0 构造"只含 C0 的
干净状态"）：

| 阶段 | 操作 | 结果 |
|---|---|---|
| INIT | 输入核验 + 身份冻结 | 全部声明复核通过；`input_identity.json` 落盘 |
| REF | 全新引擎全量重算 P1/P2/P_neg（互相 flush 隔离） | 输出 9764 / 760 / 40，physical_fa_hit 全部=0（零命中基线成立） |
| MAT1 | materialize(C0=[0,9984)) | anchor 输出 16944；**node 57**，prefix_sha256==digest，FA+Mamba 双驻留（mamba 位置 9984，slot [3]），lock 引用全零；census mamba 节点数=1 |
| RES1 | flush→重物化 C0→恢复 P1 | output=9764==REF；**physical=9984、executable=9984、gap=0**；C0 node/slot/digest 稳定 |
| RES2 | flush→重物化 C0→恢复 P2 | output=760==REF；**physical=9984、executable=9984、gap=0**；C0 稳定 |
| DUAL | flush→重物化 C0→同树先后恢复 P1、P2 | P1: 9984/9984/gap 0；P2: **physical=10206**（深于 C0，来源=P1 分支留下的合法 FA-KV 段 [9984,10206)，如实记录、不归功 C0），executable=9984，gap=222；C0 源状态全程未修改；新增 mamba 节点 [90,92]（census 1→3） |
| NEG | P_neg（同树，C0+双分支之后） | **physical_fa_hit 精确停 4992**（翻转位置），executable=0，gap=4992；output=40==自身 REF |
| EVICT | flush→重物化 C0→正式 primitive 仅驱逐 C0 Mamba→恢复 P1 | proof 全 true（sanity_check / fa_unchanged / fa_identity_unchanged / only_target_mamba_changed / same_node / path_unchanged / tree_unchanged / cascade_called=false）；P1：**physical=9984（FA 不变）、executable=0（塌缩）、gap=9984（增大）**，output=9764==REF |

跨物化确定性：5 次物化（mat1–mat5）anchor 输出全部 16944；FA-KV 内容摘要
（target_full_value_sha256，becc677cbc951295…）跨 flush 稳定；node id 跨
flush 重分配（57/67/78/89/103，合法行为，以内容摘要为准）。

## 5. 谓词判定（compatibility_results.json）

| continuation | C1 | C2 | C3 | C4 | C5 | C6 | C7 | 结论 |
|---|---|---|---|---|---|---|---|---|
| P1（deepseek 全长） | PASS | PASS | PASS | PASS | UNKNOWN | PASS | **PASS** | ADMISSIBLE + C7 认证 |
| P2（minimax 全长） | PASS | PASS | PASS | PASS | UNKNOWN | PASS | **PASS** | ADMISSIBLE + C7 认证 |
| P_neg（CONTROLLED） | PASS | **FAIL** | PASS | PASS | UNKNOWN | PASS | UNKNOWN | **REJECTED**（准入阶段拒绝） |

C7 的 runtime 证据来自**独立恢复**（RES1/RES2，各自只含 C0 的干净状态），
DUAL 仅作补充观测，不进入认证。

## 6. 八个必答问题

1. **C0 是否真实物化？**
   **是。** inspect 实测：exact 节点（首物化 node 57），prefix_sha256 ==
   token_digest([0,9984))，FA-KV（target_full_present）与 Mamba
   （target_mamba_present，位置 9984，slot [3]）双驻留，四个 lock/session
   引用全零。9984=39×256 的边界对齐只是候选理由，物化以 inspect 为准
   （实测成立）。若失败会记录阻断原因而非换位置——未发生。
2. **P1/P2 是否都从合法 Recurrent State 恢复？**
   **是。** 各自从"只含 C0 的干净状态"（flush→重新只物化 C0）出发：
   physical=9984（物理命中达 C0 全长）、executable=9984（Recurrent 可执行
   恢复起点达 9984，**未退化为全量 prefill**）、gap=0；谓词 ADMISSIBLE。
   独立环境中不存在比 C0 更深的缓存（不存在"更深命中"的可能）；DUAL 补充
   实验中 P2 获得的 10206 深命中已如实归因于 P1 分支的合法 FA-KV 段
   [9984,10206)，executable 仍为 9984（状态源仍是 C0）。
3. **两次恢复是否与 Full Recompute 一致？**
   **是。** P1: 9764==9764，P2: 760==760（隔离 REF 基线 physical=0）；
   温度 0、单 token、确定性配置；5 次物化 anchor 输出恒为 16944。
4. **Recurrent-only Eviction 是否改变实际 Executable Frontier？**
   **是。** 配对比较（同一 C0、同一 P1）：驱逐前 executable=9984/gap=0 →
   驱逐后 executable=0/gap=9984，physical 保持 9984、输出保持 9764。
   实际 Frontier 后退量=9984（实测值，非预设）；FA-KV 未动
   （fa_unchanged=true）。
5. **C5 和其他条件还有哪些 UNKNOWN？**
   - C5_physical_identity（三条边）：runtime 未暴露 slot generation 账本
     （node id、token digest、FA-KV 内容摘要、mamba slot 列表均实测）；
   - P_neg 的 C7：被 C2 准入拒绝后不进入动态认证（合同分层，预期行为）；
   - 案例级：native_fork_evidence / pending_event_evidence /
     physical_checkpoint_evidence（Phase 2 已标 UNKNOWN，本轮不改变——
     见问题 7）。
6. **相比上一轮 CONTROLLED 分支验证增加了什么新证据？**
   - **输入真实性升级**：Task A 的 H 是 AgentX hash 块+正式 compose 语义
     合成、后缀为 CONTROLLED 合成；本轮 P1/P2 是 LMCache 完整多轮消息经
     真实 tokenizer 冻结的**原始 input_ids 全长提交**（16338/21520），
     分歧后缀是两个真实 agent 会话的自然分歧，非人工构造；
   - **跨数据源身份链**：Phase 2 tokenizer_identity.json（数据侧）==
     本轮引擎模型文件 sha（runtime 侧）== Task A run-3 身份，三处一致；
   - **更深的真实 LCP 行为**：DUAL P2 展示了 LCP(10206)>C0(9984) 时物理
     命中合法超出 checkpoint 而可执行前缀仍由 C0 决定（gap=222）——
     Task A 的样本 LCB 恰在 C0 边界，未覆盖此情形；
   - **新 `mamba_branching_seqlen` 数据点**：DUAL P2=10176、NEG=4992，
     与 Task A 的 None/4096/8192 一起支持"仅在 FA 前沿≠可用 Mamba 状态时
     填充"的观测（语义仍待 runtime 确认，不作主证据）；
   - **分支内 track 边界物化观测**：DUAL 中两分支各物化 1 个新 mamba 节点
     （[90,92]）——分支长 6354/11536 token 跨越多个 256 边界，内部边界
     状态落盘；与 Task A 发现②（终止点非边界则**终止**状态不落盘）不矛盾，
     但说明长分支会物化中间边界状态（值得矩阵重生成采样设计注意）。
7. **仍不能证明什么？**
   - **非原生物叉**：lmcache_10 的两条会话是"实验回放"（同模型重模板化的
     跨模型供应商会话），不是 runtime 内原生 fork/spawn——
     native_fork/pending_event/physical_checkpoint 三类证据保持 UNKNOWN；
     本轮证明的是**冻结真实 Token 输入的物化与恢复正确性**，不声明语料
     具备原生物叉语义；
   - **C5 slot generation**：账本不暴露（同 Task A）；
   - **规模外推**：仅 lmcache_10 一例（按停止条件未运行 lmcache_16 等
     15 例）；无吞吐/策略性能结论（未开展 benchmark）；
   - **`mamba_branching_seqlen` 确切语义**：五个数据点（None/4096/4992/
     8192/10176）均为观测，runtime 侧未文档化。
8. **是否具备进入下一轮合法 Global Allocation 实验的基础？**
   **方法论上具备**：Task A 证明了合成前缀+合同谓词可执行，本轮把输入侧
   升级到真实多轮消息冻结 input_ids 仍成立（物化→双 continuation 恢复→
   正确性→驱逐因果→负例拒绝全链路）。但 Global Allocation 的正式对比结论
   仍应等待 Executable Compatibility Matrix 按合同重生成（Task A 修复提案
   §2.4 流程：祖先绑定合成 + 谓词唯一准入 + CERTIFIED/ADMISSIBLE 分层），
   且建议先把本协议扩到 lmcache 全部 4 个真实共享案例（10/12/13/16）作为
   数据侧前置。

## 7. 诚实性声明（未放宽任何标准）

1. **C5 UNKNOWN**：P1/P2 边为 ADMISSIBLE + C7 认证，不冒充 CERTIFIED；
2. **CONTROLLED 负例**：P_neg 为冻结输入派生（4992 处 33955→33956），
   显式标记，不冒充原始数据；其作用仅是验证非法前缀被拒；
3. **实验回放属性**：P1/P2 是跨模型供应商会话的同模型重模板化回放，
   原生物叉证据保持 UNKNOWN，报告措辞不越界；
4. **DUAL 深命中的归因**：P2 物理 10206 的来源是 P1 分支 FA 段，与 C0 无关
   （C0 只覆盖 [0,9984)）；认证只用独立恢复证据；
5. **单案例规模**：按停止条件只完成 lmcache_10；未自动扩展到 16 组；
6. **观测不等于语义**：mamba_branching_seqlen 新数据点与分支内边界物化
   观测均如实记录为实测，未作为任何 PASS 门依据；
7. **无性能声明**：本轮无吞吐/延迟对比结论（REF P1 e2e 27.4s 为冷启动
   首请求的 compile/预热耗时，P2 1.2s——不构成 benchmark，不引用）。

## 8. 失败尝试与未确认项

- 无失败运行（本轮一次通过；GPU 等待器未触发，GPU 1 直接空闲）；
- 日志中的 torchcodec/FFmpeg traceback 为镜像固有加载噪音（多模态可选
  组件），不影响文本引擎路径，历史运行同样存在；
- 未确认项见 §6 问题 5/7。

## 9. 最高优先级下一任务建议

**把本协议扩到 LMCache 全部 4 个真实共享案例（lmcache_10/12/13/16），
产出数据侧的"真实消息 Executable Recovery"证据集**（复用
`lmcache_recovery_smoke.py` 参数化 case_id/C0 位置，每例独立目录与证据
文件）。理由：Global Allocation 实验的共享 Checkpoint 供给方（数据侧）
目前只有单案例证据；4 例覆盖 LCP 2.7k–13k 的不同形态，可在一轮 GPU
窗口内完成，是把"方法论成立"升级为"数据侧就绪"的最短路径。
