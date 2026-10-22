# 现有规划接口与绑定边界

源码路径相对仓库根目录。

|接口|位置|实际合同|
|---|---|---|
|StrictCandidate/StrictPending/StrictEdge|flowstate/strict_epoch.py:30、40、52|候选核心对象、完整Token及摘要、Epoch与配置；Pending目标；全配对三态及14项条件|
|StrictAllocationEpoch|flowstate/strict_epoch.py:87、101|唯一身份、Token摘要、同边界、等大小、完整配对、位置、拒绝UNKNOWN、内容摘要；构造和入口均验证|
|StrictAllocationPlan|flowstate/strict_epoch.py:72|Epoch/Snapshot/内容摘要、K、选集和数量、逐请求T/E/G/成本、总J/收益、逻辑驱逐补集；无句柄、无执行方法|
|CheckpointCandidate|flowstate/state_catalog.py:12|逻辑ID、来源Workflow、Lineage、位置、大小、驻留；本轮Workflow不参与严格边判定|
|frontiers|flowstate/strict_epoch.py:175|仅使用EPOCH_ADMISSIBLE边，不调用Legacy谓词|
|validate_plan|flowstate/strict_control.py:94|验证计划身份、预算、选集、补集和使用冻结模型重算的全部PendingPlan；不重新选择|
|bind/preflight|flowstate/strict_control.py:141、155|另需当前PhysicalView、控制窗口及当前节点/组件映射；本轮不调用|

严格Epoch本身不自动证明物理组件无别名，也不是当前GPU状态认证。新适配器先调用原validate，再核对可信原始快照字节、Node/Slot、引用与组件证据，限定为冻结快照纯规划。来源SHA必须来自可信外部Manifest；传入一个自签SHA不能把任意外部数据升级为认证。

B4来源直接调用原from_snapshot_bytes。B5-C3原格式缺少核心候选metadata字段，独立from_frozen从原组件切片求memory_bytes；候选workflow_id采用candidate_provenance中首次观察所对应历史请求的真实来源Session，仅作必填溯源字段，不宣称该请求是物理唯一Owner，Lineage为空。全部严格边逐字保留，完全不靠此Workflow字段做兼容判断。原Snapshot SHA仍指向原字节；新Epoch content_digest描述适配后的对象。

B5-C3物理切片证据单独锁SHA，核对同窗口、观察不变性、完整Conv/Temporal、节点Token和组件列表、槽及大小。B4复用既有冻结状态条件、物理一槽一项与非零引用拒绝。历史slot_generation仍UNKNOWN，不补造代际，不宣称任意现在时刻仍然有效。

生产Plan本来没有求解器标签。本轮Outcome在外层标记Exact/Greedy、optimality_certified及fallback_reason，不修改生产Plan字段。任何计划未来要执行，仍必须重新观察、绑定、Preflight，不能复用这里的历史Slot/Node。
