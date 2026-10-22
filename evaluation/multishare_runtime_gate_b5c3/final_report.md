# Phase B5-C3 最终报告

## Gate：READY

固定四Session的八条历史请求完成，四个Pending均未派发。实际捕获7个同边界驻留Recurrent候选；准入矩阵28条：EPOCH_ADMISSIBLE=13、REJECTED=15、UNKNOWN=0。目标三个严格正收益支持集合是否全部出现：True。

## 实际驻留结构

|Checkpoint|实际L|当前Slot|Token内容支持|严格正收益支持|
|---|---:|---|---|---|
|B5C3_NODE_18|3712|[3]|A,B|A,B|
|B5C3_NODE_20|2432|[8]|A,B,C,D|A,B,C,D|
|B5C3_NODE_21|3456|[10]|C,D|C,D|
|B5C3_NODE_22|4160|[12]|A|A|
|B5C3_NODE_24|4096|[14]|B|B|
|B5C3_NODE_25|3904|[16]|C|C|
|B5C3_NODE_27|3520|[18]|C,D|C,D|

位置均来自真实Cache节点，不预设为CPU LCP 2438/3765/3566。本轮没有插入候选、移动分叉边界、恢复/驱逐状态或更换Session。完整Conv/Temporal摘要、组件切片、Slot/Node、FA路径、引用及同窗口不变性均存于physical_state_census.json。

实际结构包含两个支持C/D的共享节点，但没有仅支持D的专属候选；八条历史请求不对应八个独占候选。D2的原生记录显示恢复前沿3456、分支跟踪位置3520，最终Census中的node27位于3520。此处只报告原始Match记录与最终物化观察，不把逐请求Census当作精确插入事件。未为补齐D专属状态采取任何操作。

## Pending前沿

|Pending|完整Token数|实际FA frontier|T=min(输入长度,FA)|
|---|---:|---:|---:|
|A|4862|4160|4160|
|B|4860|4096|4096|
|C|4435|3904|3904|
|D|4486|3566|3566|

FA由已验证的只读树遍历取得，不调用Prefix Match，不提交Pending。完整Token关系与严格准入分列；只有EPOCH_ADMISSIBLE且冻结Phi(T,T)−Phi(T−L,T)>0才计入正支持，REJECTED为0，UNKNOWN不视为合法。本轮只为正支持核验单边收益，不计算任何策略选集、预算扫描或总分配目标。

## 固定回放账本

|请求|实际接收Token数|原生生成Token|完成后驻留节点数|本步消失节点|
|---|---:|---|---:|---|
|B5C2_A_TURN_001|3728|[9764]|1|[]|
|B5C2_B_TURN_001|3728|[9764]|1|[]|
|B5C2_C_TURN_001|3507|[9764]|2|[]|
|B5C2_D_TURN_001|3507|[9764]|3|[]|
|B5C2_A_TURN_002|4182|[9764]|4|[]|
|B5C2_B_TURN_002|4150|[9764]|5|[]|
|B5C2_C_TURN_002|3922|[6820]|6|[]|
|B5C2_D_TURN_002|3947|[6820]|7|[]|

实际提交与Native Match接收摘要逐请求一致；completion_tokens=1、无截断或回撤。原生生成输出逐项保存，只作为正常缓存副作用，不加入四个冻结Pending。历史Assistant和工具结果来自原始记录，不称目标模型原生闭环执行。没有请求级性能比较或TTFT改善结论。

## 合同与限制

Qwen3.5-9B / SGLang v0.5.17 / TP=1 / 单H100 PCIe / 28槽；完整输入与模板沿用B5-C2。全新独立生命周期，单次Scheduler同步窗口内冻结PhysicalView、FA、T和完整边矩阵。无分配器调用、无Reconciliation、无主动Eviction、无Pending执行。

本工作负载仍是Trace-derived Controlled Composition，不是自然同时Ready或原生Fork。EPOCH_ADMISSIBLE不等于逐边RESUME_VALIDATED；本轮不验证恢复输出。逻辑预算K未指定，不把当前池使用量解释为物理容量压力。多共享前缀支持仍可能满足树形精确可解结构，不证明任何策略优势。

## 完整性与结束

来源/既有冻结文件摘要：PASS；Engine shutdown返回：True；GPU清理：PASS。三次清理观测与专用容器退出证据见gpu_cleanup.json，未终止其他GPU进程。

退出日志保留一条CUDA IPC警告：生产进程结束时部分共享CUDA张量尚未释放。Engine shutdown已返回、进程返回码为0；此后专用容器已退出，三次独立资源检查均无计算进程、显存为14 MiB。清理PASS依据这些实际检查，而不是仅依据退出码。警告未被删除，也未据此推断执行期状态损坏。

已交付全部九项文件及原始逐请求日志、源码锁、组件摘要、完整原始观测。完成后停止，不自动进入策略评估或恢复验证。
