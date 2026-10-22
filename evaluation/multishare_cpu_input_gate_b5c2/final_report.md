# Phase B5-C2 最终报告

## Gate：READY（仅CPU输入身份与Token内容结构）

固定四Session来源、因果截断、目标Tokenizer/模板身份和完整Token数组均通过。形成四请求公共前缀与两个任务内更深共享子集，未更换来源或修改消息。不代表新Strict Snapshot、Recurrent物化、实际恢复或分配收益。

## 输入规模与LCP

|标签|历史1 Token数|历史2 Token数|第三次Pending Token数|
|---|---:|---:|---:|
|A|3728|4182|4862|
|B|3728|4150|4860|
|C|3507|3922|4435|
|D|3507|3947|4486|

|LCP（tokens）|A|B|C|D|
|---|---:|---:|---:|---:|
|A|4862|3765|2438|2438|
|B|3765|4860|2438|2438|
|C|2438|2438|4435|3566|
|D|2438|2438|3566|4486|

- 四方共同最大前缀：**2438**；支持集合{A,B,C,D}。
- System完整渲染前缀：**2428**；四方共同LCP还包括用户消息头部的10个Token，不能把2438全称System。
- 任务一A/B：**3765**；任务二C/D：**3566**。两者均严格大于2438，支持分别恰为{A,B}、{C,D}。
- 所有跨任务配对LCP均为2438；Token分叉发生在User的工作区路径，位于完整System之后。
- A/B完整System＋User结束于3723；3765处分叉位于首条历史Assistant文字。C/D完整System＋User结束于3502；3566处分叉位于首条历史Assistant的think工具参数文字。不是两个任务共享同一原生Fork。

实际渲染全文及逐Token字符offset均保存；causal_checks.json记录每个消息边界与完整Token数组一致性，message_boundary_supports.json记录System及完整User边界支持。token_prefix_matrix.json保留所有配对、分叉Token及字符范围。

## 身份、因果性与配置

来源SHA与B5-C1一致；固定选择顺序在分词前写入source_selection_lock.json。四条记录均在第3个Assistant标记（索引6）之前截断；8条工具结果可追溯到更早tool_call_id。原始墙钟时间、自然Ready及跨Session同步关系UNKNOWN；READY只针对顺序可核验的受控已知输入，不声称原生在线因果事件认证。

原B4镜像ID为`sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926`；Transformers 5.12.1、Tokenizers 0.22.2、SGLang包0.5.17。Tokenizer文件及模型config摘要、实际加载模板摘要与B4一致。工具参数、规范化、生成提示参数与冻结入口相同。模型权重未加载；没有SGLang导入、Engine、GPU设备或网络。

当前主机uv312的Transformers 4.57.6不同于B4，因此没有用它生成正式Token。运行采用冻结镜像CPU容器、runc、无GPU挂载、网络禁用、只读源码/模型，仅本目录可写。容器已自动删除。

## 必须保留的限制

Token位置都在对应输入范围内；实际Planning Target与FA frontier尚未观测，L≤T只能留待Runtime确认。System/任务消息边界和LCP都不是已物化Checkpoint，也不是建议强制创建的位置。多个非单例内容支持仍为层次/不交结构，不证明一般困难性或FlowState收益。本轮没有计算任何收益、预算或策略。

来源中recorded_model缺失，不影响本轮明确指定Qwen的受控输入协议，但不能声称恢复了原模型的生产输入身份。历史Assistant和Tool均来自原轨迹，不冒称由本轮目标模型产生。

## 执行异常及完整性

四次前置/诊断失败均保留cpu_attempt1–4.log：AST注入节点缺少行号、容器UID写权限、System-only诊断被模板拒绝、offset的tuple/list序列化比较差异。前两次未生成Token；后两次发生在完整12份Token生成后。修复仅涉及本轮脚本/诊断，最终运行逐项复核此前Token和配置相同，没有修改冻结源码、输入、模板或换Session。最后cpu_run.log完整输出结构通过；门禁不单靠退出码。

独立finalize.py重新验证12份数组长度与SHA、全部LCP、工具因果索引、B5-C1所有产物摘要与原Parquet摘要。verification.json及artifact_manifest.json保存结果和交付身份。

主要交付：source_selection_lock.json、input_configuration_lock.json、token_prefix_matrix.json、causal_cut_audit.md、runtime_pilot_prerequisites.md，以及12份完整Token数组、原始截断历史、渲染文本和复核脚本。

**本轮完成后停止，不自动进入Runtime Pilot。**
