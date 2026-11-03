# 第一道门禁：原策略实现与573条件复现

**PASS：573/573条件、2292/2292策略结果。** 逐项比较集合语义、基数、每Pending的T/E/G及成本、总J，容差1e−9 ms；FlowState原始选择顺序573/573完全一致，PCB/LDR/WFC的返回顺序也与各自冻结结果一致。不是读取旧“同集”结论当复现。

## 实际选择入口

|策略|真实实现及规则|本轮复用方式|
|---|---|---|
|FlowState|原评估器`rq3_frozen_snapshot_evaluator.py`用`GlobalOptimizer.select`；当前该入口进入`_select_candidates`，动态最大边际，1e−9同分容差、ID字典序、非正边际停止|直接调用同一生产Greedy核心`_select_candidates`，成本闭包按冻结Formal矩阵和Phi计算。没有自行重写Greedy，也未调用plan_strict|
|PCB|`p1_strong_baseline_validation/pcb.py::select`：每Pending推荐最佳正独立收益项，按收益/ID/Pending排序，ID去重到K|导入已有冻结模块；不改变其推荐和停止规则|
|LDR|同目录`ldr.py::select`：每Pending最深合法项，按位置/ID/Pending排序去重|导入已有冻结模块；不读Phi作决策|
|WFC|同目录`wfc.py::select`：每步优先推进Pending数q，其次推进Token总数r，最后ID；q=0停止|导入已有冻结模块；完整保留动态更新及其诊断轨迹|

Greedy核心所需对象仅有checkpoint_id/memory_bytes/recurrent_resident；本轮使用只含这三个字段的不可变适配对象，完整候选/位置/兼容在独立成本闭包中保持原样。没有伪造Workflow或Lineage来绕过关系，也没有把Formal输入构造成已认证Strict Epoch。PCB/LDR/WFC接受其原common.Epoch对象。实际原公式与独立评分逐值核验，原始字节预算转K保持等大小合同。

**当前生产能力必须另述：** `GlobalOptimizer.plan_strict()`已经接入Exact–Greedy门禁：先Strict Epoch完整认证，n≤12、m≤4、K≤4、子集≤794时Exact，否则原Greedy；输入不合法不回退。它是生产源码能力，不是仅有独立实验Exact Planner。但本次历史Formal策略定义是Marginal Greedy，故不通过Strict混合入口替换它，Exact单列Oracle。

## 版本、同分与完整性

冻结PCB/LDR/WFC/common四文件SHA与implementation_lock全部一致；当前生产Optimizer、Phi、原始评估入口另记录完整SHA。当前Optimizer文件包含后续混合接线，不能说整个文件与早期版本相同；本轮复用Greedy循环并用2292份原结果复现建立行为对应。

没有执行旧脚本main，避免覆盖旧文件。原预算复现先完成并写reproduction_gate.json=PASS，随后才进入K扫描。原始FlowState逐步选择按原diagnostics核对，不从终集猜测。原列表顺序差异与跨策略集合一致性是不同概念。

全部读取文件前后内容摘要一致；输入去预算SHA三条件逐项相同，191唯一快照域完整。若任一复现、身份、公式或规模失败，run_budget.py保存BLOCKED及阶段/堆栈并停止，不静默补数据或扩大门禁。

依据：original_budget_reproduction.csv、reproduction_gate.json、validation.json、source_manifest.json及protocol.json。
