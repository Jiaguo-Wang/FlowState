# Weka / AgentX 身份与回放语义审计

**PASS（身份和代码机制可复核），旧 Formal 的运行时语义冲突没有被本轮消除。**

## 原始身份

本地 `/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl` 全文件 SHA-256：
`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`。
与 Hugging Face 官方文件树的 LFS SHA-256 完全一致，文件大小 1847151435 bytes。`official/weka_tree.json` 保存远端元数据；`weka_identity.json` 保存本地全量扫描及 393 个 ID。不是仅靠计数判断，也没有重复下载 1.85GB 文件。

统计 393 traces、98827 调用、1697 subagent groups；逐字节摘要相同涵盖所有原始 ID、hash 数组和嵌套内容。历史 B1 身份记录 `/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614/INPUTS.json` 同一摘要。因此 Weka 是旧 AgentX 的同一原始语料，不能作为新增独立数据集。

代表 trace `02bc0afb13f7a2d9efa86c28511261d85c0e` 在原文件第 5 行：行摘要 `2c5f6abfed52a7db8e550ab3df872b1aa1bf79dd86591bc89a893f6f87217038`，requests 摘要 `b75b32b36e42dc3dcb4b0a3ec901b39992120ce8df1296342528358537505798`；hash_id_scope=local，block_size=64，包含 claude-haiku-4-5-20251001 与 claude-opus-4-8。全文件摘要一致提供原始 hash/嵌套身份保证；并未声称公开库从未更新，身份仅针对本次取得的 LFS 对象。

原语料相同不表示下游无处理：旧 B1/B2 选取正式子集，链检测、逻辑累计位置和 Formal laminar 编码均是派生处理。不是另外一份 Weka 解决了这些问题。

## 旧回放与官方机制

|环节|旧 FlowState 回放|所核对官方 AIPerf|
|---|---|---|
|块生成|`evaluation/agentx_qwen_replay_preflight.py:116–169`，SHA256(hash_id,counter,salt) 映射词表，跳过 special IDs|`official/weka_trace.py:1632`，corpus RNG 按 trace/hash 重播，固定作用域内确定性块|
|作用域|块缓存键/seed 无 trace_id；不同 trace 同整数可能伪共享|每 trace 设 scope 并清缓存；root/子 Agent/检测链共享同 trace 命名空间|
|输入构造|`:194–225` 直接拼 hash 块、裁至 input_length、确定性补尾|`official/weka_prompt_compose.py:10–48` 拼块/裁剪/确定性尾部；具体采样字节与旧实现不同|
|聊天边界|该 composer 不给每分支另套独立聊天模板|官方重建共享 setup 边界，取 declared/observed 较长者；最终消息渲染需遵守一致边界|
|链与模型|`agentx_chain_detection.py` 从 hash LCP/时间等重建链，不能当原生事件|官方文档 `official/weka-trace.md:270–309` 区分同模型续链及跨模型 spawn；也包含检测出的派生链|
|最终兼容|Raw 检查实际 token prefix；Formal 改用 B1 逻辑祖先集合|官方 hash 回放不是实际 recurrent checkpoint 的证书；最终目标输入仍要检查 token prefix|

在固定 tokenizer、词表范围、special 集合、盐和块大小下，相同有序 hash 前缀在旧 composer 中确实产生相同 token 前缀（限两输入共同包含的完整/部分已知块）。**没有证据支持“旧代码把同一 hash 块按分支随机化”或“重复套模板破坏共享”的解释。** 缺少原始文本、64-token hash 身份及合成尾部不能恢复原始 Claude token。hash 也不是数学无碰撞证明。

缺失 trace 命名空间主要造成跨 trace 的伪共享风险，不是在同 trace 内破坏相同块。隔离 engine 的历史实验不能仅凭这个缺口断言已被跨 trace 污染。尾部补齐不是未知原始历史的证据；不同模型的状态不能因为 hash 相同直接共享。

## 70 条 Formal 新增边为何仍有问题

`evaluation/agentx_runtime_snapshot_collection.py:647–651,884–895` 用真实合成 token 检查 prefix。`evaluation/agentx_runtime_final_correctness_audit.py:87–174` 用逻辑祖先、inherited 标记和累计位置建立 B1 集合；`evaluation/rq3_agentx_formal_policy_evaluation.py:327–425` 将 B1 集合与 B2 驻留/深度组合并重编码，不要求 Raw prefix 成立。该路径解释了实现选择；作者当时的主观动机 UNKNOWN。

沿用 `evaluation/gate1_step3_compatibility_retention_contract_audit/audit_report.md` 的正式审计：70 条正收益 Formal-only 边，其对应此次 Raw token predicate 为假；不能用祖先身份自动授予可执行恢复能力。此处未重新生成矩阵或重算策略。B1 的累计逻辑位置也不能等同某个独立回放输入的物理 prefix 位置。

## 最小后续修复方向（本轮未执行）

统一固定目标模型、trace-scoped 块映射、确定性尾部、统一渲染边界，并在最终 input_ids 上建立可执行 prefix 谓词；只允许实际前缀内已物化的位置作为恢复源。可以避免新的作用域/格式问题，不需改优化算法。但当前旧 composer 已保留同 trace 相同块，故**尚无证据说仅修复回放就能消除那 70 条边中的任意确定数量**。不可为保留这些边而强行给子请求拼接祖先历史。合法性的缺口可能必须通过纠正离线兼容定义或采用新可认证输入解决；本轮都不执行。

## 官方来源与版本边界

- [公开 Weka 数据](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)，内容身份以 LFS SHA-256 为准。
- [NVIDIA 官方回放说明](https://docs.nvidia.com/aiperf/latest/tutorials/datasets-inputs/replay-weka-agentic-coding-traces)。
- AIPerf 固定 commit `bfa5ab1d4bb5cd1292862d44e4f3a5441cf00981`，`official/` 归档源码；[源码入口](https://github.com/ai-dynamo/aiperf/blob/bfa5ab1d4bb5cd1292862d44e4f3a5441cf00981/src/aiperf/dataset/loader/weka_trace.py)。比较的是当前可取得的固定源码，不宣称这是历史 B2 当日使用的 AIPerf 版本。

未运行官方回放、旧回放或 GPU。本报告确认可构造一致受控目标输入的代码基础，不认证自动重建的链就是生产原生 fork，也不认证不同原模型共享物理状态。
