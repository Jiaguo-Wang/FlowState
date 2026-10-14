# LMCache Exact Token Prefix 数据准备

**PASS（内容输入门禁），不代表 Runtime 执行认证。** 日期：2026-10-13。

## 冻结与覆盖范围

复用 `/tmp/lmcache_check/0.parquet` 至 `4.parquet`，没有重新下载模型或采集请求。五个分片共 767 sessions、629 个按 session 后缀规则识别的 tasks、120 个异模型多会话 tasks。`all_120_structure.json` 保留全部 120 组结构元数据；任务分组来自 session 命名，不等同原生 workflow identity。

`selection_lock.json` 在 token 分析前写入：每 session 选消息数最多的已有请求，平手依文件编号升序、行号降序；每 task 选字典序最早的异模型会话对。按序列化消息的字符 LCP 排序切四层，每层按 SHA256(seed+task_id) 取三组，再补未选中的四个最长文本会话。seed=20261013。16/120 个任务进入验证，未替换任何失败样本，未按收益或策略胜负筛选。全部 120 组序列化字符 LCP 的最小值及排序第 30/60/90 项均为 13072，最大为 321656。文本四分层含大量相同 LCP，不能当成四种不同 token 深度。

## 目标输入合同

统一目标 `/home/wjg/models/qwen3.5-9b`，本地 Qwen2Tokenizer，transformers 5.13.0。文件身份见 `tokenizer_identity.json`。Hub revision UNKNOWN，以 tokenizer、配置及模板 SHA-256 固定版本。调用 `apply_chat_template(tokenize=True, add_generation_prompt=True)`，不传 tools、enable_thinking，不截断，不额外拼接祖先历史。模板默认生成思考前缀。

保留原始 system/user/assistant/tool、tool_call_id/name 及 tool_calls。工具 arguments 字符串按 JSON 解析为映射，以满足真实模板 `chat_template.jinja:109–129`；工具结果由模板 `:134–145` 格式化。结构化工具调用没有被丢弃。原数据未提供顶层 tools schema，不能声称恢复了原提供商的完整工具声明；本轮构造的是明确固定的目标模型输入。

项目实际格式化对照为 `evaluation/openhands_single_workflow_smoke.py:150–167`，同样使用 apply_chat_template 后提交 input_ids。LMCache 的原生 tool_calls 字段不能直接套用只识别 tool_calls_json 的旧 normalize_message。旧 GPU 环境的 transformers 版本等价性、HTTP 服务器重新格式化等价性 UNKNOWN；交接应使用已保存 input_ids，不再把它们 decode 后套另一层模板。

`/home/wjg/uv312` 的 transformers 存在 huggingface-hub 依赖版本冲突；未修改环境。提取使用 uv312/pyarrow，tokenize 使用已有 uvgood。仅 tokenizer CPU 操作，没有加载模型权重。

## 全量固定样本结果

|案例|P1 tokens|P2 tokens|Exact LCP tokens|完整相同消息数|前缀内候选位置|
|---|---:|---:|---:|---:|---|
|lmcache_01|13092|30283|2719|0|[2560]|
|lmcache_02|27261|56753|2719|0|[2560]|
|lmcache_03|23167|33379|2719|0|[2560]|
|lmcache_04|19263|46774|2719|0|[2560]|
|lmcache_05|17553|41728|2719|0|[2560]|
|lmcache_06|29580|22478|2719|0|[2560]|
|lmcache_07|27275|32346|2719|0|[2560]|
|lmcache_08|30340|59593|2719|0|[2560]|
|lmcache_09|34495|19939|2719|0|[2560]|
|lmcache_10|16338|21520|10206|3|[9984]|
|lmcache_11|34820|59690|2719|0|[2560]|
|lmcache_12|21036|46397|8834|3|[8704]|
|lmcache_13|84070|86784|84068|89|[83968]|
|lmcache_14|33355|62121|2719|0|[2560]|
|lmcache_15|34741|74519|2719|0|[2560]|
|lmcache_16|40340|69883|35650|32|[35584]|

16/16 模板转换成功，16/16 有非空共同 token 前缀，16/16 在两者结束前出现不同 token。最短 LCP=2719，最长=84068。没有完全相同的最终输入，也没有 token 层面一条严格包含另一条的情况。完整字段、比例、原始模型与 session ID 见 `prefix_statistics.csv`。首个不同位置采用 **0 基 token 索引**，checkpoint 位置采用 **前缀 token 数**。

12/16 在第一条 system 消息内部发生分歧，共享部分主要是系统指令；这不能证明共同任务执行历史。其余 lmcache_10、12、13、16 分别有 3、3、89、32 条完整相同消息。lmcache_13 的原始完整消息历史存在包含关系，短输入在生成提示符处与长历史分歧，不能把最长 LCP 当成最强分支证据。

消息边界采用对消息前缀调用同一模板，并验证结果确实是两个完整输入的共同 token 前缀。只有通过该验证的边界才记录。仅 system 的前缀不是该模板合法独立对话，会被跳过；模板对 reasoning 的上下文依赖也可能导致某些边界无法确认，未用各消息 token 数相加替代。公共前缀的其余部分可能位于消息或模板标记内部。

## 三个交接案例

- **lmcache_01**：`swebench__django__django-13794`，LCP 2719，候选 C0=2560；两输入长度 13092/30283。展示短 system 共享及结构化 tool call/result 格式，完整历史实际不同。不是共享执行历史证明。
- **lmcache_10**：`swebench__django__django-15731`，LCP 10206，C0=9984；长度 16338/21520。前三条消息相同，随后 assistant 消息不同；三消息边界已验证为 10203 tokens。适合作为首个最小内容分叉验证输入。
- **lmcache_16**：`swebench__sphinx-doc__sphinx-9658`，LCP 35650，C0=35584；长度 40340/69883。前 32 条消息相同，随后一边 user、一边 assistant，32 消息边界=35649。展示较深共同历史与明显角色/内容分歧。

案例用于结构覆盖，不代表算法收益筛选。全部 16 组输入仍交付，非代表样本未删除。候选 C0 为 floor(LCP/256)*256，仅是在 exact prefix 内的对齐位置；运行时 tracking 粒度、真实状态物化、owner/generation、FA frontier、容量和同步 Ready 全部待验证。尚未读取模型权重或测量资源；长输入能否在目标 engine 配置下 admission 为 UNKNOWN。

## 输入和复现

`inputs/lmcache_XX_messages.json` 保存来源分片与 0 基行号以及原始消息；`inputs/lmcache_XX_p1_tokens.json`、`p2_tokens.json` 保存规范化消息与实际 input_ids。`runtime_candidate_manifest.jsonl` 保存完整路径和摘要。所有样本是跨同源 session 的目标模型回放输入，不是生产原生 fork。native_fork_evidence、pending_event_evidence、physical_checkpoint_evidence 均为 UNKNOWN。

复现命令在 `evidence_manifest.md`。报告不打印原始完整内容；输入保留于本地交接目录。来源归属：[sammshen/lmcache-agentic-traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces)，许可证信息沿用 Phase 1 记录，外发前应核对适用声明。未声称取得新发布授权。

## 研究判断

当前最小 Executable Recovery 输入优先使用 lmcache_10：真实完整消息、明确相同目标 token 前缀及不同后续内容，规模小于长案例。Weka 更适合后续研究有时间结构的受控多 Agent 回放，但 hash 合成、逻辑链检测、真实 Ready 和容量竞争仍需单独认证。LMCache 的输入相同不证明同时 pending；Weka 的子 Agent 嵌套也不保证正好形成可执行容量竞争。两者现阶段均不足以宣称真实 Global Allocation 收益负载已建立。
