# 严格兼容重建可行性

|来源|历史内容|Token/配置身份|同边界Pending与物化|本轮资格|
|---|---|---|---|---|
|OpenHands全量原始记录|完整序列可读，重复任务来源明确|原始Runtime Token缺失；recorded_model全部空；可未来固定Qwen身份生成新的受控输入|新组合均未采集；自然Ready未知|PARTIAL：可准备新受控输入，不可补写旧Strict矩阵|
|B4 g001–g004|已有实际输入与严格准入证据|同配置身份和物理合同已有|已认证受控静止边界|已有单共享子类；不是本轮多共享达标样本|
|AgentX|hash块与容器结构|不能逆推出原Token或完整文本|旧replay是block形状，原生Ready缺失|无法直接重建严格多共享输入|
|FreeInference|block_ids和请求/子代理记录|非无损；无消息可重建|时间重叠不等于Ready；无Recurrent物化|只能结构研究|
|LMCache|真实消息、部分冻结目标模型Token数组|目标受控输入身份有据；原执行配置不等同目标配置|多共享候选及共同边界缺失|PARTIAL：旧单共享认证不能外推|
|TraceElephant/TraceLab|部分消息或轮次/工具事件|覆盖与模板完整性尚未证明|父子继承和Ready尚未闭合|UNKNOWN，需数据合同审查|
|GATC|未取得样本|UNKNOWN|UNKNOWN|BLOCKED|

## 最小证据链

新受控工作负载需要：来源Session/请求索引→当前准备好的完整目标Token→实际提交及接收摘要→历史请求真实完成→原生Checkpoint物化→同一静止边界全树Census→组件/所有权/引用/别名→实际FA与T→逐边准入。只用消息摘要相同不足；不能假设完整首次输入末尾恰好留下checkpoint，原生跟踪粒度、模板尾部和Radix分裂均需观测。

复用入口为rq3_openhands_neutral_collector.py:236–303的输入截断和既有strict_compatibility_b4b0/capture.py。准备Pending时不读取待生成assistant输出；先冻结Barrier证据，再做接收身份核对，不用之后生成的状态反填。历史工具输出只在受控已知输入回放中作为已提供记录使用，不能称自然闭环Agent执行。

新组合必须独立于B4已有矩阵和物理绑定。EPOCH_ADMISSIBLE仍不等于RESUME_VALIDATED；R2跨Workflow个例也不认证新组合每条边。是否存在候选外可恢复状态、是否能保持多个祖先与分支专属Checkpoint同时驻留、安全协调是否落实配额，均需后续独立Runtime证据。

目前可可靠认证的新多共享严格Snapshot数为0；这是尚未采集/证据不足，不是证明数据里绝无此类结构。
