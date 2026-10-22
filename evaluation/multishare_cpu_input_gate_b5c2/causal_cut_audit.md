# 因果截断与工作负载身份

## 固定来源

本轮身份为 **Trace-derived Controlled Composition**。按B5-C1 candidate_workload_plan.md已预定的instance_id、session_id字典序，而不是Parquet行序、Token长度或胜负选择。两任务分别为0b01001001__spectree-64与12rambau__sepal_ui-411，四Session固定，没有替换。

|标签|任务|Session|Parquet零基行号|第三次请求截断消息索引|
|---|---|---|---:|---:|
|A|0b01001001__spectree-64|`nebius-swe-rebench-openhands::chatcmpl-71420ffc5b410547d43ac67939b0441c`|10918|6|
|B|0b01001001__spectree-64|`nebius-swe-rebench-openhands::chatcmpl-71b073b89097665d14501919b224c26b`|12519|6|
|C|12rambau__sepal_ui-411|`nebius-swe-rebench-openhands::chatcmpl-6f6bc3d08ce275e9f059158e0a4513ef`|10472|6|
|D|12rambau__sepal_ui-411|`nebius-swe-rebench-openhands::chatcmpl-9d7d3d3ab3856419dec69a7d1a106c8b`|10378|6|

## 读取与时间证据

每个输入只消费原始索引0至5：System、User、Assistant1、Tool1、Assistant2、Tool2。索引6仅通过role定位第三次请求；其content不进入重建。准备脚本需要解析源JSON记录以读取列表，但分词脚本仅收到截断历史加一个只含role的assistant哨兵，无法访问第三次输出及后续消息。没有用后续内容选取Session或计算前缀。

八条Tool记录均有对应先前Assistant的tool_call_id：调用索引2→结果3、调用4→结果5。原始顺序证明结果已在第三次请求输入之前出现；没有原始wall-clock、Ready/Dispatch日志，绝不把顺序索引冒充真实时间戳或自然并发证明。工具输出是历史原记录，不是本轮Qwen生成或执行工具的结果。

完整调用ID、索引与核验保存在causal_checks.json。沿用neutral collector的五个未来信息检查均为false。四个来源均有完整两次历史与第三次输入，没有补位或截短输入。

## 与B4一致性及差异

原函数：evaluation/openhands_single_workflow_smoke.py:66–110、169–187；evaluation/rq3_openhands_neutral_collector.py:230–303。按B4执行锁逐文件SHA核对后，AST原样加载三个规范化函数、materialize_group_requests及依赖，仅避免模块顶层导入Runtime。函数体、消息规范化和apply_chat_template参数未变。

使用冻结Qwen3.5-9B Tokenizer及模板、Transformers 5.12.1，SGLang镜像版本0.5.17。tools参数未传入，原始System内工具说明、历史tool_calls和tool_call_id按B4规则保留；没有额外猜测或补写工具schema。enable_thinking未传入，使用相同模板默认值。未截断Token。

差异是来源组合和CPU范围：本轮有8个准备好的历史输入及4个Pending缓冲，没有实际回放、Runtime接收确认、FA观测或Strict Epoch。独立B5C2请求ID替换旧构造器的标签，Token保持相同；requests.json保留旧构造器返回ID供核对。新组合不属于B4冻结g000或g001。

复用工具按turn再A/B/C/D的顺序仅保存为未来回放计划；本轮没有执行任何历史请求。
