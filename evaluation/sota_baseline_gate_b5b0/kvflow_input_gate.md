# KVFlow 输入可行性：BLOCKED

当前严格快照没有可认证的Workflow Execution DAG和当前可见的Steps-to-execution。应暂缓主实验适配，不应由兼容图推造工作流步骤距离。

|必要输入|状态|证据及判断|
|---|---|---|
|当前已声明执行DAG|MISSING|四组snapshot没有声明图、图版本、边依赖或声明事件；B4-A final_report.md:30–34明确为受控已知输入回放、自然Ready/Fork未知|
|Agent Step节点及位置|PARTIAL|有Session、workflow_label、turn和request_ordinal，但turn只是回放消息位置，不是Agent Step Graph节点|
|AND/OR、条件依赖|MISSING|不存在结构化分支依赖及条件触发证据；消息中提及工具/Agent不等于运行图声明|
|Pending→实际Agent Step|MISSING|只有Pending→来源Session/下一请求映射，无真实Agent Step身份|
|Checkpoint→Agent|PARTIAL|可追溯创建请求/Session/Token，不能把Session名称或共享Token前缀等同KVFlow agent_id|
|决策时信息已可见|PARTIAL|当前四个Pending完整输入在受控客户端已准备；不能据此认证将来的DAG、条件分支或工具结果当时已声明|
|仅当前信息计算Steps|MISSING|缺上述DAG/事件，无法给出可审计的距离|
|是否需要未来真实轨迹|UNVERIFIED/禁止替代|按完整离线会话事后推导距离会改变在线信息边界；本轮不这样做|

## 四种不同关系

- Workflow Execution Dependency：声明哪些执行步骤依赖哪些步骤，以及AND/OR/条件语义。
- Workflow Lineage：同Session/祖先身份，不提供调度依赖距离。
- Token Prefix：执行输入的内容前缀关系，不提供Agent步骤。
- EPOCH_ADMISSIBLE：同边界、Token/配置/完整状态条件下的恢复准入，不提供未来执行计划。

四个已准备Pending都可以立即提交，不代表所有缓存Agent的steps均为1，更不代表未来Agent Step Graph已知。现有`evaluation/sota_metadata.py:64–77`把steps固定为1属于受控代理。不能以此把KVFlow驱逐机制纳入忠实SOTA主表。

B5-A adaptation_contract.md §2将“Pending图上候选距激活的纪元距离”视作现有steps输入，其信息等价性没有数据支持。本轮明确不采纳这一前提，不修改旧报告。公开论文所需图与当前缓存兼容图语义不同，即便二者都在当前内存中，也不等价。

## 版本边界继续保留

B5-A源码锁：7ef897eaedc16626a281ccfe0d453fae80251e8f，工作树核验干净。只沿用已有审计，不重新获取仓库。

- 论文描述的steps优先级驱逐：需要原生执行依赖输入；论文实验版本与当前仓库对应UNVERIFIED。
- 当前混合HEAD的分层保持：`python/sglang/srt/mem_cache/lora_hiradix_cache.py:269–295`按阈值保护，并非统一优先级排序。`AgentManager.update_agent_timestep`（agent_manager.py:24起）从外部声明更新距离。
- 默认非分层路径：B5-A已确认`radix_cache.py:141`、`lora_radix_cache.py:120`早返recency，当前LRU行为不能当作论文steps机制。

论文/代码版本归属不明是独立于数据缺失的第二层风险。不能把现有LRU简单改名为完整KVFlow，也不能用本轮矩阵检验其完整预取/分层系统性能。

## 最小必要补充

真正的来源工作流需要在决策前保存：声明式DAG及版本/时间、Agent/Step稳定ID、AND/OR/条件依赖语义、当前位置与已完成事件、Pending到Step映射、Checkpoint创建请求到Agent映射。步骤距离只能从当时声明的依赖和状态计算；不要求先知道未来实际工具输出，预先声明的计划也不等于未来泄漏，但必须有声明时刻证据。

当前来源未提供这些条件，故不推荐先写距离代理。暂缓KVFlow主实验实现；未来补齐真实元数据和版本适配边界后另行审查。本轮不新采集、不实现代理。
