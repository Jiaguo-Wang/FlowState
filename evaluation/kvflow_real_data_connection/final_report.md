# KVFlow 真实数据接入结果

## 1. 是否找到了合法真实 ASG 与 Fixed Prefix

**没有找到同时满足两项要求、且能连接现有候选域的来源。已停止数据接入。** 没有改动现有策略，没有生成猜测Sidecar，也没有用未知输入运行CPU选集。

直接检查范围：旧24个Runtime快照及E2 w1–w4、B5-C4的上游，共29个快照、116个不同OpenHands Session；另对AgentX原始393条trace进行了全文件流式字段核查。不是仅复述上一轮结论，也不声称检查了所有其他OpenHands Session的完整语义。

OpenHands来源：`/home/wjg/data/agentic_coding_trajectories/sessions.parquet`。116条来源均命中，逐条在目标Assistant请求之前截断，未消费目标输出。截断内984条消息只有`role/content/tool_calls_json/tool_call_id`；376条工具调用全部可以解析，分别是think 99、str_replace_editor 236、execute_bash 41。参数只有thought、command、path、view_range，没有声明Agent invocation、Ready事件、前驱或AND/OR聚合。

每条来源都有一条System消息，故**角色提示文本是可得的**，不能把结果写成“没有Prompt”。然而没有逐Agent固定提示契约、固定末端Token边界及其与Invocation的映射。System角色标签不能独自决定整个Agent固定提示的范围，也不能据此将其余所有候选认定为动态后缀。所有116条recorded_model为空；ground_truth字段仅为instance_id、exit_status、resolved、patch_present，没有调度声明。

`evaluation/rq3_openhands_neutral_collector.py:236–302`按历史消息调用`apply_chat_template`，在当前Pending marker停止；`evaluation/openhands_single_workflow_smoke.py:95–110`仅规范化消息和工具字段。Tokenizer路径及新严格回放的模板身份可得，但这些代码没有额外携带ASG或Fixed Prefix声明。目标模型配置可追溯不等于源Agent计划可追溯。

AgentX来源：`/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl`，全量393条trace、98,827次请求及1,697个子Agent容器。请求字段为`t/model/in/out/hash_ids/api_time/type/ttft/think_time`；子容器另有agent_id、subagent_type、duration_ms、total_tokens、tool_use_count、status、requests、models。可追踪调用容器和事后时间，但没有决策时声明的AND/OR、Ready、固定提示Token或原始消息。

`evaluation/agentx_chain_detection.py:278–315`通过hash LCP和时间关系产生ChainFork；这是检测结果，不是源客户端当时声明的ASG。`evaluation/agentx_qwen_replay_preflight.py:193–225`由hash块构造确定性Token并补齐长度，`agentx_runtime_snapshot_collection.py:278–345`调用该逻辑。合成回放Token不能恢复原始Agent固定提示，不作为本次真实Fixed Prefix来源。

原始字段证据及SHA见`openhands_source_inspection.json`、`tool_call_inspection.json`、`agentx_source_inspection.json`与`source_identity.json`。Parquet保存了命中原始记录及截断摘要；本轮没有宣称重算整个Parquet文件SHA。AgentX摘要为全文件流式计算。

## 2. 哪些快照能够运行

**本次确认可忠实运行的快照为0。** `snapshot_results.csv`逐项列出旧24个OpenHands快照及5个严格快照的拒绝原因。AgentX原始来源的共同缺口同样阻止将其派生快照接入本策略；未把旧Formal关系转成ASG或Strict身份。

这不否认严格快照已有的Token兼容和Recurrent驻留证据；缺失的是KVFlow特有的执行依赖和固定提示语义。

## 3. 实际CPU对照结果

没有合法KVFlow选集，因此没有新增可报告的J、E/G或胜平负。CSV的选集和J留空，不能解释为选择空集、零收益或策略失败。上轮14项单测和既有Greedy/WFC数字不是本轮成功数据接入证据。

没有调用策略生成伪结果，没有重跑其他分配算法，没有运行GPU。现有`evaluation/kvflow_inspired_retention_cpu/policy.py`保持不变。

## 4. 无法从当前归档恢复的信息

- **OpenHands：**当时声明的Agent Invocation/Step身份与Ready状态、Predecessors及AND/OR语义；与该身份对应的固定提示声明、固定末端Token边界和版本可见性。消息中的过去工具调用和自然语言thought不能唯一恢复这些信息。受控客户端输入已准备也不能证明原生Ready事件。
- **AgentX：**上述声明，加上原始完整消息/Token及固定提示配置身份。block hash是内容标识而非可逆编码；合成回放无法恢复这些原始信息。

“无法恢复”限定为当前已保存归档与代码，不能据此断言源系统从未拥有相关信息。需要源客户端当时的声明或事件记录才能认证；本轮不补造、不采集新数据、不安排后续实验。检查结束。
