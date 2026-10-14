"""汇总离线输入验证结果，写入独立交接目录。"""
import csv, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def write(name,text): (ROOT/name).write_text(text)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
rows=list(csv.DictReader((ROOT/'prefix_statistics.csv').open()))
manifest=[json.loads(x) for x in (ROOT/'runtime_candidate_manifest.jsonl').read_text().splitlines()]
lock=json.loads((ROOT/'selection_lock.json').read_text())
identity=json.loads((ROOT/'weka_identity.json').read_text())
reps=['lmcache_01','lmcache_10','lmcache_16']
for m in manifest:
    inputs=[json.loads(Path(p).read_text()) for p in m['input_artifact_paths']]
    a,b=[v['input_ids'] for v in inputs]; n=m['prefix_tokens']
    assert a[:n]==b[:n] and a[n]!=b[n]
    assert all(0 < c <= n for c in m['checkpoint_candidate_positions'])
    assert [sha(Path(p)) for p in m['input_artifact_paths']]==m['input_sha256']
    common=next((i for i,(x,y) in enumerate(zip(inputs[0]['messages'],inputs[1]['messages'])) if x!=y),min(len(x['messages']) for x in inputs))
    m['shared_complete_messages']=common
    m['history_relationship']='一条完整消息历史是另一条的前缀' if common==min(len(x['messages']) for x in inputs) else '共有历史之后消息内容或角色不同'
    m['representative_case']=m['case_id'] in reps
    m['checkpoint_position_unit']='从输入起点计数的 token 数；按 256 对齐只是待验证位置，不证明原生物化'
    m['source_type_evidence']='同任务异模型已有会话作为统一模型的离线回放材料；原始采集事件与原生 fork 未认证'
with (ROOT/'runtime_candidate_manifest.jsonl').open('w') as f:
    for m in manifest: f.write(json.dumps(m,ensure_ascii=False)+'\n')
summary={'status':'PASS','fixed_cases':len(rows),'tokenized':len(manifest),'nonempty_prefix_and_distinct_token_suffix':len(manifest),'native_fork_certified':0,'physical_checkpoints_certified':0,'representative_cases':reps,'input_integrity':'PASS','gpu_executed':False}
write('validation_summary.json',json.dumps(summary,ensure_ascii=False,indent=2))
table='\n'.join('|'+ '|'.join(str(r[k]) for k in ['case_id','p1_total_tokens','p2_total_tokens','prefix_tokens','common_messages','checkpoint_candidate_positions'])+'|' for r in rows)
write('lmcache_exact_prefix_report.md',f'''# LMCache Exact Token Prefix 数据准备

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
{table}

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
''')
write('weka_agentx_identity_audit.md',f'''# Weka / AgentX 身份与回放语义审计

**PASS（身份和代码机制可复核），旧 Formal 的运行时语义冲突没有被本轮消除。**

## 原始身份

本地 `/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl` 全文件 SHA-256：
`{identity['local_sha256']}`。
与 Hugging Face 官方文件树的 LFS SHA-256 完全一致，文件大小 1847151435 bytes。`official/weka_tree.json` 保存远端元数据；`weka_identity.json` 保存本地全量扫描及 393 个 ID。不是仅靠计数判断，也没有重复下载 1.85GB 文件。

统计 393 traces、98827 调用、1697 subagent groups；逐字节摘要相同涵盖所有原始 ID、hash 数组和嵌套内容。历史 B1 身份记录 `/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614/INPUTS.json` 同一摘要。因此 Weka 是旧 AgentX 的同一原始语料，不能作为新增独立数据集。

代表 trace `02bc0afb13f7a2d9efa86c28511261d85c0e` 在原文件第 5 行：行摘要 `{identity['representative']['record_sha256']}`，requests 摘要 `{identity['representative']['requests_sha256']}`；hash_id_scope=local，block_size=64，包含 claude-haiku-4-5-20251001 与 claude-opus-4-8。全文件摘要一致提供原始 hash/嵌套身份保证；并未声称公开库从未更新，身份仅针对本次取得的 LFS 对象。

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
''')
# 清单包含原始输入摘要、外部证据和本轮产物，避免自引用摘要。
files=[p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['evidence_manifest.md','evidence_hashes.json']]
source_names=['evaluation/agentx_qwen_replay_preflight.py','evaluation/agentx_chain_detection.py','evaluation/agentx_runtime_snapshot_collection.py','evaluation/rq3_agentx_formal_policy_evaluation.py','evaluation/agentx_runtime_final_correctness_audit.py','evaluation/openhands_single_workflow_smoke.py','evaluation/gate1_step3_compatibility_retention_contract_audit/audit_report.md']
files += [REPO/x for x in source_names]
hashes={str(p):sha(p) for p in files}
hashes.update(lock['parquet_sha256'])
hashes[identity['source']]=identity['local_sha256']
write('evidence_hashes.json',json.dumps(hashes,ensure_ascii=False,indent=2))
write('evidence_manifest.md','''# 证据、版本与复现

本目录是独立离线交接产物。输入来源仅读取；没有修改旧正式数据/矩阵/模型/Runtime，没有 GPU、Engine、Replay、策略比较或预算调整。

## 复现入口

在 `/home/wjg/code/FlowState` 执行（仅写本交接目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/wjg/uv312/bin/python agentic_workflow_dataset_research/phase2_submission_inputs/audit_weka.py
PYTHONDONTWRITEBYTECODE=1 /home/wjg/uv312/bin/python agentic_workflow_dataset_research/phase2_submission_inputs/prepare_samples.py
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' /home/wjg/uvgood/bin/python agentic_workflow_dataset_research/phase2_submission_inputs/tokenize_samples.py
PYTHONDONTWRITEBYTECODE=1 /home/wjg/uv312/bin/python agentic_workflow_dataset_research/phase2_submission_inputs/build_reports.py
```

audit_weka 使用已归档的官方文件树；prepare_samples 只使用五个本地分片。若 /tmp 原始分片已移走，应以 selection_lock 的摘要找到相同内容，不得换版本。已保存 16 组完整消息，因而 tokenizer 验证与交接无需再次扫描分片。脚本没有网络请求、模型加载、策略执行。

## 文件身份

`evidence_hashes.json` 包含原始五分片、Weka 全文件、本轮脚本、报告、原始消息及生成 token 文件、所引用源码和官方代码的 SHA-256；清单排除自身及本索引以避免循环摘要。tokenizer 文件摘要另见 `tokenizer_identity.json`，对应本地 `/home/wjg/models/qwen3.5-9b/`。

`official/revision.txt` 保存 AIPerf commit；官方文件版权与许可证头保留原样，不是新增英文注释。`official/weka_tree.json` 保存公开源文件的 LFS 身份。本轮源码摘取供审计，不代表已运行该软件版本。

## UNKNOWN 和限制

- 原始 LMCache 顶层工具定义、原生 fork / ready / queued 事件、任务 ID 的原生调度含义未知。
- 本地 tokenizer Hub revision 未保存，文件 SHA 固定；旧 GPU 软件环境与当前 5.13.0 完全等价未知。
- target engine 的 context limit、tracking checkpoint 生成、owner/generation、FA frontier、容量和恢复正确性均未动态认证。
- Weka hash 不能还原原文本，检测链不是直接 fork 事件；旧 Formal 70 条边没有因本次数据准备得到修复。
- source_type“实验回放”描述本次交接材料的使用方式及已有同源会话结构，不声称已证明每条原始会话的采集来源。
- 输入内容仅本地保存，不在报告打印全文；不得将本目录发布行为推定为已经授权。

`representative_retokenization_check.json` 记录三个代表案例共六个输入重新模板化后逐 token 一致。

`validation_summary.json` 为本次 16 组输入的结构门禁结果，PASS 不包含任何 GPU 或算法胜负结论。
''')
print(json.dumps(summary,ensure_ascii=False))
