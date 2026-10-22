# 独立Exact纯规划合同

实现：本目录planner.py；唯一对外完整入口plan(epoch,evidence,allow_fallback=False,cancel=None)。无控制器、Engine或适配器实例，无物理执行能力。

数据流：完整StrictAllocationEpoch → prepare（严格对象和冻结物理证据校验）→ benefits（共同非负有限收益矩阵）→ 规模选择 → solve → finish → 原生产validate_plan → Outcome。任何校验异常直接抛出，不广泛catch后降级。

四个准入约束同时满足：n≤12，m≤4，K≤4，ΣC(n,i)≤794；K为非负整数，K0合法。前三界下第四项是冗余但显式检查的安全上界。K>n在原生产合同下并不非法；本轮如K>4仍拒绝Exact。测试覆盖负数、非整数、布尔K以及合法超限。

合法超限：默认LimitError；显式allow_fallback=True才调用现有GlobalOptimizer._select_candidates，以相同严格评分回调运行原Greedy循环。外层返回solver=Greedy、optimality_certified=False、明确原因。未使用Legacy Workflow谓词，也未改写Greedy。

不合法：UNKNOWN、缺失/重复配对、重复身份、Token/配置/Epoch摘要冲突、非等大小、组件缺失、非零引用、Node/Slot/组件别名、陈旧标记或错误证据范围均Fail-closed。超规模和UNKNOWN同时出现时仍拒绝，不进入Fallback。

Exact完整枚举大小≤K的子集，共同J评分；相对全局最小J误差≤1e-9的集合视为数值同分，按较少项、ID字典序取代表。这是B4-C合同，不是任意精度实数最优性声明。相同收益但不同集合不称性能胜负。无支配删除、无提前剪枝。

收益矩阵由冻结模型内部构建，不允许调用方指定新模型。solve防御性重新检查该矩阵一致性；Exact和Greedy都承担这项成本。两者共同score使用相同合法前沿及Phi，不通过矩阵缓存改变任何策略定义。Plan通过原validate_plan逐请求重算，拒绝非有限结果。

## 取消和deadline

cancel是可选协作谓词：求解入口和每个Exact子集前检查；为真抛Cancelled，不返回未穷尽计划、不自动Fallback。已有Greedy只有入口取消检查；矩阵构建、Epoch校验和Plan验证没有中途抢占。测试验证初始和第三次检查时取消均不返回计划。

因此没有硬deadline保证，也没有声称普通Python检查能强制中断。未来需要独立冻结端到端期限、预留Fallback预算、可取消工作单元及进程隔离方案；超时不得宣称Exact。不能在未验证安全输入时用Fallback“继续服务”。本轮没有新增生产时间上限或后台线程。
