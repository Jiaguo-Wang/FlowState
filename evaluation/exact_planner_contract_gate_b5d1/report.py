"""只汇总本轮纯规划合同、验证结果及统一计时。"""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
c=json.loads((P/'correctness_tests.json').read_text());n=json.loads((P/'negative_tests.json').read_text());v=json.loads((P/'independent_validation.json').read_text());rows=json.loads((P/'timing_summary.json').read_text())
def write(name,text):(P/name).write_text(text.strip()+'\n')
table='|快照|K|求解器|共同准备中位ms|共同矩阵中位ms|求解中位ms|Plan构建校验中位ms|全路径中位ms|全路径本地P95 ms|\n|---|---:|---|---:|---:|---:|---:|---:|---:|\n'
for r in rows:table+=f"|{r['group']}|{r['K']}|{r['solver']}|{r['prepare_ms_median']:.3f}|{r['benefits_ms_median']:.3f}|{r['solve_ms_median']:.3f}|{r['plan_validation_ms_median']:.3f}|{r['total_ms_median']:.3f}|{r['total_ms_p95']:.3f}|\n"
write('planner_interface_audit.md','''# 现有规划接口与绑定边界

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
''')
write('exact_planner_contract.md','''# 独立Exact纯规划合同

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
''')
write('runtime_integration_prerequisites.md','''# 后续Runtime集成前提

本轮READY只适用于冻结Epoch纯规划，不授权立即执行当前GPU状态操作。

1. 可信Strict Epoch必须在当前连续安全窗口建立，并重新核验Token、配置、完整候选域、候选外状态及当前引用。历史SHA不能认证当前GPU。
2. 必须重新生成PhysicalView，验证Node/Slot/Conv/Temporal切片及Keep/Evict无别名；原生产bind/preflight不能跳过。本轮无任何PhysicalBinding输出。
3. 生产接线需要显式求解器来源、Exact规模准入、错误与Fallback隔离；冻结快照的stale=False仅为证据使用范围声明，不是跨RPC活性租约。
4. 决策时间预算须覆盖输入认证、共同预处理、求解及计划校验；当前计时排除了初始读取和Epoch构造，更未包含RPC、Scheduler隔离、GPU摘要和Reconciliation。不能以求解器中位值估计完整控制成本。
5. 生产deadline/取消能力未实现，Greedy Fallback也没有强制抢占。任何超时方案先做CPU故障验证，不得把未穷尽结果标Exact。
6. 更换求解器后仍须独立真实协调与恢复正确性验证，及公平的控制含量性能测量。EPOCH_ADMISSIBLE不是RESUME_VALIDATED；本轮没有TTFT、GPU净收益或长期保留保护证据。

所有生产源码SHA在本轮前后相同。没有自动开始以上工作。
''')
write('final_report.md',f'''# Phase B5-D1 最终报告

**Gate：READY（独立CPU纯规划接口及冻结合同）**。

## 正确性与范围

五快照20条件：12个小规模条件（g001/g004/B5-C3，K1–4）走Exact，均与冻结Exact目标及代表集合一致，并经独立Decimal穷举Oracle复核；8个g002/g003条件只走显式Greedy Fallback，匹配冻结Greedy，未扩大Exact域。生产plan_strict的20条件Greedy回归也通过。

每份输出都是既有StrictAllocationPlan，经过生产validate_plan。外层Outcome独立标记求解器与最优性；物理信息仅用于冻结审计，没有任何执行绑定。

{len(n['rejections'])}项拒绝测试通过，另有5项同分/容差、5项规模谓词、4项非法Plan、零预算和中途协作取消检查。非法输入先于规模路由，UNKNOWN不能借Fallback通过。详细记录见negative_tests.json、independent_validation.json。故障输入仅为内存副本，没有修改冻结矩阵、候选或参数。

## 统一CPU计时

相同合法Epoch、同一共同prepare/benefits，每条件2次预热、20次正式重复，Exact/Greedy交替先后，12条件×2求解器×20=480条记录。共同阶段逐次复用并给两种规划路径相同计费；总时延逐样本为共同准备+共同矩阵+本求解器求解+本计划构建校验，不包含另一求解器的时间。它是分段实测和，不是两求解器连续执行的整段墙钟。

prepare包括原epoch.validate、原始证据SHA及JSON解析、物理身份检查；benefits构建矩阵。solve包括防御性矩阵复核和选择。finish包括Plan构建及原validate_plan，它会再次验证Epoch。初次文件读取、from_frozen构造和Manifest检查在计时外；比较起点已经持有合法Epoch。没有删除重复安全检查来追求速度。

全部均值、分位数、范围见solver_timing_comparison.csv，原始值见raw_timing.json。P95为20次重复第19项，只是本地观测，不是在线P95；不同阶段的中位数不必相加等于总时延中位数。

{table}

输入/Plan合同验证占主导，Exact并未表现为比Greedy更快；采用它的理由是小规模目标最优性及可接受性需要后续控制预算验证，而非声称CPU/GPU速度优势。B5-D0边界不同，不与该轮数字直接相减。

## 完整性与限制

本目录保存planner.py、run.py、verify.py、report.py及全部报告/数据。implementation_lock.json记录冻结来源和生产源码SHA；原始文件前后不变。preliminary/仅保留初轮实现的检查及计时历史，最终主结果只来自上级目录修订后实现的同一批测量，不混合两批。

B5-C3缺少B4式metadata，通过真实组件切片及已有逐请求来源记录做格式适配，未修改兼容语义。长期Slot generation仍未知，因此仅认证冻结边界；未来当前状态仍需重新绑定。无GPU、无驱逐、无迁移、无Restore、无Tree DP、无生产修改。

完成后停止，等待审查，不进入Runtime集成。
''')
manifest={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='evidence_manifest.json'}
(P/'evidence_manifest.json').write_text(json.dumps({'files':manifest},ensure_ascii=False,indent=2)+'\n')
print('READY：七项交付及独立验证证据已完成。')
