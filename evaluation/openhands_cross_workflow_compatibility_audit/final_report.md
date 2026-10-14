# OpenHands 跨 Workflow 可执行兼容性审计

**BLOCKED：当前冻结资料不足以把重建出的跨 Workflow 前缀提升为原 Runtime 已认证的可执行兼容边。** 这不表示共享不存在。审计确实发现了一个强候选模式：168 个快照中每个都有一条位置为 2,432 的 recurrent resident 候选；用同一项目聊天模板和本地模型 tokenizer 重建后，它均与同组另外三个 pending 输入形成 token 前缀，共 504 条边。它们都止于共享 System Prompt/模板区域，没有跨越到不同的首条 User 消息。

## 核查范围与身份

逐一检查正式 Main Population 的 168 个 snapshot、168 个 Runtime Correctness JSON、168 个 barrier-final census 和四个源 session，共 672 sessions。采集协议为 `evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/protocol.json`：SGLang 0.5.17、Qwen3.5 9B、TP=1、group 独立 engine、round 2/3/4/5 全部完成后、下一轮执行前建立共同 barrier；pending 是 r+1 的四个当前输入，不读取输出或 r+2 历史。原 Runtime 记载 transformers 5.12.1、tokenizer/model 路径 `/model`。

输入 messages 从冻结 `sessions.parquet` 和 snapshot 指定 session 恢复，沿 `rq3_openhands_neutral_collector.py:250–300` 的停止规则只消费到 pending assistant 标记。重建调用与正式实现相同语义的 `apply_chat_template(tokenize=True, add_generation_prompt=True)`，并沿用 `normalize_message` 的 tool call 参数解析。

本机只有 transformers 5.13.0；原运行的 tokenizer 文件哈希未冻结，`/model` 与本地 tokenizer 的字节身份也没有历史摘要。因此这次 token arrays 是**重建值**，不是原始 Runtime 输入身份。重建出来的全部 checkpoint turn 输入长度与 168 份历史 request row 的 `offline_input_tokens` 一致，四个 pending 长度也与原 snapshot `anchor_pos` 一致（168/168）；长度相同仍不能证明 token IDs 相同。原 Runtime 没有留存 pending 完整 token IDs，也没有把候选 `prefix_sha256` 保存进正式 snapshot/census。

## 候选与驻留证据

正式 snapshot 每份含 8–20 个 candidate，均标为 FA 与 recurrent resident。Barrier census 保存当前 resident recurrent node、slot、token position。历史快照中有些位置重复，不能仅按位置关联所有 candidate；但本次 504 条跨 Workflow 重建边的 source candidate 均在 position=2,432，该位置在 146/168 个快照唯一对应一个 resident node/slot；另 22 个快照有同位置多节点歧义。collector 的 `locate_materialized_candidate`（`openhands_common_barrier_snapshot_gate.py:174–284`）在执行时把新 census node 与正在运行的请求输入前缀相连，并通过 Runtime inspect 比较 position 和 prefix digest；`validate_candidate_at_barrier`（`:288–317`）又在同一 barrier 重查 node、digest、FA 与 recurrent residency。snapshot 保留 resident flags 和位置，但未持久化该 digest/handle-node 关联。全体 candidate 位置到 census node 有重复歧义的共 428 个/124 个快照；上述重建跨 Workflow 边中，146 个 candidate 有唯一位置映射，22 个仅能依赖快照中的 resident 标记和 collector 当时的检查路径，无法从持久 census 唯一关联 slot。本审计没有把单独的 Radix 祖先关系当作兼容证明。

## Token 前缀重建结果

共枚举 6,828 条“候选 × 不同 Workflow pending”配对。其中 5,427 条候选位置不超过该 pending 的实测 resident FA frontier T；重建 token 序列显示 504 条 exact prefix 边，全部满足 L≤T。504 条分布在 168 个快照，每例 3 条：候选均是 Workflow B 的 `TURN_001`，物理 position=2,432，分别指向 A/C/D 的 pending。剩余 4,923 条 L≤T 的配对在 transformers 5.13 重建序列上不匹配；另有 1,401 条候选本身深于 T。逐边和逐快照证据见 `reconstructed_edges.csv`、`snapshot_identity_audit.csv`。

跨四条 session，首条 System Message 在 168/168 组内逐字段相同，首条 User Message 在 168/168 组内不同。LMCache 同目录 tokenizer 与模板复建时，System 内容和下一个 chat boundary 共 2,438 tokens，候选位置 2,432 位于其内；这解释了 504 条重建边为什么只覆盖系统/模板前缀。由于 Runtime 原执行 tokenizer 版本及文件哈希没有锁定，2,438 是重建验证，不作为原运行的 token 边界认证。没有发现完整用户/工具交互历史构成的跨 Workflow 边。

## 原兼容规则遗漏情况

Formal `is_compatible` 在 `flowstate/state_catalog.py:60–73` 要求 workflow_id 相同并且 lineage prefix。以上 504 条重建边的 source workflow B 与 target A/C/D 不同，故该规则确会排除这类系统前缀候选。分成结论等级：

- **原兼容实现可能遗漏的重建边：504 条**，属于仅共享 System Prompt/模板的内容候选。
- **关联到正式快照 recurrent-resident 标记的重建边：504 条**。其中 438 条在持久 census 中可由唯一位置关联到 node/slot；另外 66 条因 position 重复只能依赖 checkpoint resident 标记和 collector 当时 inspect 路径，持久 slot 身份有歧义。没有直接 cross-resume 验证。
- **具有原运行 token IDs/prefix digest 身份证据的跨 Workflow 边：0 条归档**。同理，直接证实该候选在另一 Workflow 恢复成功的边为 0。
- **无法取得完整原运行身份而未认证的重建边：504 条**。其余 6,324 条在本机重建下不匹配或超过 T；旧执行 tokenizer 未被身份锁定，不能将它们改称 runtime 不兼容。

## 代表案例与结论

`rq3-openhands-main-g000-round2`：候选 `RQ3_G000_B_TURN_001`，position 2,432，barrier census 的 node 20 / slot 6，recurrent 与 FA resident；重建后与 A/C/D pending 的前 2,432 个 token 完全一致。三个目标 T 分别为 4,352、4,160、4,032，均覆盖候选位置。四条 session 的 System Message 均相同，而首条 User Message 各异。这是本审计最有代表性的**重建候选**，不是原 token IDs 已认证的跨 Workflow restore 案例。

因此按任务门禁结论为 **BLOCKED**：存在 504 条可能被 lineage 排除的系统提示共享候选及真实驻留的来源 recurrent state，但原运行完整 token IDs、原 candidate prefix digest 和跨请求恢复试验都未归档；本次重建还使用了不同 transformers patch 版本。无法声称已经找到具有“充分执行证据”的合法跨 Workflow 边，也不能据此计算新收益或预算。BLOCKED 不代表共享不存在。

下一步若进入独立收益验证，最低身份门槛是取得原执行 tokenizer/template 文件身份，或在无 GPU 的原 Tokenizer 版本下重建后，再用归档的 Runtime 候选 prefix digest 对原候选和 pending token prefix 双向校验。若缺失的 digest 无法从现有日志恢复，则需新的只读 Runtime identity capture；之后还要单独认证 cross-request resume。以上只是证据缺口，不是本轮执行请求。

## 复现与完整性

- `extract_inputs.py`：流式提取 168×4 个已存在 session 的消息前缀，不触碰正式 artifact。
- `reconstruct_tokens.py`：本地 tokenizer 重建并比对 6,828 条边及原始记录长度。
- `snapshot_identity_audit.csv`：每个快照输入长度、tokenizer 和冻结文件摘要。
- `reconstructed_edges.csv`：每条跨 Workflow 边的 checkpoint、pending、T、重建 token 摘要和物理状态等级。
- `audit_summary.json`：审计计数与状态。
- `evidence_manifest.json`：来源 SHA-256。

只运行 CPU 文件读取与 tokenizer，不启动 SGLang/GPU，不修改旧 504 个分配条件或任何正式 artifact。
