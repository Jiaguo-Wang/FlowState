# 来源清单与检查范围

本轮只读现有本地数据，未联网、下载、分词、构建快照或执行策略。具体文件存在性、大小、SHA-256见source_manifest.json。以下将本轮直接检查与历史审计分开，历史数字不冒充新运行结果。

|来源|真实路径与处理入口|本轮检查及已有证据|结论|
|---|---|---|---|
|OpenHands 原始归档|`/home/wjg/data/agentic_coding_trajectories/sessions.parquet`；`evaluation/rq3_openhands_population.py:32–40,184–224`|全部5000条`source_dataset=nebius-swe-rebench-openhands`记录，完整消息对象前缀；保留session_id/source_id/instance_id；不是仅检查旧168组|完整消息可读；每条是一条串行记录，多次运行同任务可追溯；原模型recorded_model全部为空，缺原生父子/Ready及Token身份|
|严格OpenHands受控证据|`evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/`；B5-C0 structural_contract.json、verification.json|本轮复用B5-C0的四组合同和结论，不重新运行矩阵评估|四组单共享，不支持多共享推断；新组合不能继承旧物理身份|
|AgentX/Weka|`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl`；agentx_chain_detection.py、agentx_runtime_snapshot_collection.py|本轮全文件SHA与P2记录一致，重读首条schema。历史P2全量393 traces、98827 invocations、1697 subagent容器|真实容器归属及hash形状可用；缺原始消息/Token、Ready、父请求继承事件。不能把旧Formal边改名Strict|
|FreeInference|`evaluation/new_dataset_suitability_audit/raw/fi_2026-08-09/`、`raw/fi_2026-08-30/`；sample_audit.py、session_metrics.jsonl|本轮读完已有464行逐会话指标，保留全部165个至少两个block前缀分叉的线索，其中25个含subagent|仅block结构；历史报告明确无原始消息且渲染非无损。多分叉不等于多个当前候选|
|LMCache|`/tmp/lmcache_check/0.parquet`至`4.parquet`存在；`agentic_workflow_dataset_research/phase2_submission_inputs/`|本轮不重扫五分片；读取selection_lock与既有报告。历史767 sessions、629 tasks、120多运行任务、16对冻结目标Token输入|完整消息/目标Token受控输入可用；16对旧合同每对仅一个共享C0，其他中点候选是PROPOSED；跨对不能直接合并成原生工作流|
|TraceElephant|P2 source_inventory.json列出的五个step_records.json|本轮重读五个固定首轨迹，共9/21/47/48/18步；input.messages可读，均有多个agent_id|标签和工具转交不等于继承前缀；未完成跨步骤输入、时间和执行配置关联。其余轨迹UNKNOWN，不宣称全库不适用|
|TraceLab|`/home/wjg/data/tracelab/v0.0.2/syfi_coding_trace.duckdb`|本轮核对路径；复用P2只读schema：rounds、timing_events、tool_calls、trace_source|轮次/工具时序并非模型Ready或共享状态；原始输入完整性待核验|
|InferenceX/Weka|`/home/wjg/data/inferencex_weka/traces_first30.jsonl`|路径存在；复用P2首条schema|同类hash输入，不补齐Exact Token|
|GATC|`evaluation/gatc_real_session_sample_audit/final_report.md`与更新的D1 final_report.md|本地没有已取得的实际样本证据；较新D1记录访问受限401，较旧D2记录网络阻断|BLOCKED；不重复下载，也不据卡片认定存在合格案例|

## 证据时效与边界

P2和早期真实结构报告早于B1/B4，不能用它们“OpenHands未证实共享”的旧结论否定后来真实跨Workflow恢复。这里复用的是原始数据字段限制；后来的B4四组单共享证据以B5-C0为准。

原始归档不是原服务端Token日志。OpenHands的消息可构成新目标模型受控输入，但不能恢复已丢失的原模型Tokenizer/模板身份。LMCache冻结数组也只认证对应目标重分词输入，不是原模型生产Token。所有来源的旧/新身份必须分别记录。
