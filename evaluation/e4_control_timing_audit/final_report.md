# E4 控制成本与计时边界审计

**结论：现有记录足以定位主要时间区间，但不足以拆出纯生产控制成本或证明稳定的控制后净收益。** 约7.5秒主要包含请求执行，并非7.5秒的状态控制。下一轮应先补计时边界与客户端证据持久化隔离，再做同一w1/K2的最小对照，不需要新增工作负载。

本次仅只读检查E1、E4-1A、E4-1B源码与既有原始记录，没有GPU调用、重跑或生产代码修改。下面数值均为ms。E4-1B汇总单位为12个有效测量生命周期（A/C×Greedy/Exact×3），不含四个Smoke及两个Cold；它们是同一工作负载的重复，不是12个独立工作负载。

## 1. Control RPC：可拆到什么程度

来源：E4-1B各`r{1,2,3}_{A,C}_{Greedy,Exact}/reconciliation.json`的`timing`及`summary.json`。七个阶段由同一个`timed()`按顺序包围，互不嵌套；Native操作、组件检查、重复Observe等嵌套于这些阶段，不能另加一次。

|区间|均值|范围|证据与解释|
|---|---:|---:|---|
|完整客户端Control RPC（父区间）|814.493|703.828–879.971|客户端调用前到`_call`返回，包含持久化，不是纯网络/服务端时间|
|元数据观察|43.747|42.441–49.029|`metadata_observation`；FA前沿、候选及当前物理视图；不是独立轻量RPC|
|Strict Epoch认证|103.445|97.651–110.143|`strict_epoch_authentication`；含历史JSON读取、来源线索、编码/摘要及完整合同认证，不是单纯数学校验|
|收益矩阵构建|0.021|0.019–0.027|`benefit_matrix`|
|求解|0.129|0.105–0.167|`solver`；Greedy与Exact分别运行|
|Plan构建及验证|8.645|8.302–10.010|`plan_build_validate`|
|Apply-time观察、Bind及Preflight|37.334|36.714–39.665|`apply_time_freshness_preflight`；保留新鲜度与物理安全检查|
|Recurrent-only协调及后验|138.753|137.671–142.211|`strict_reconciliation_with_postchecks`；不等于纯Native释放耗时|
|上述七项之和|332.074|—|只加互不嵌套的阶段|
|RPC未分类余额|482.419|380.408–540.782|**端点差额**：RPC减七项之和；不是独立测得的写盘或网络耗时|

未分类余额中，源码能确认存在：严格视图等价检查、实验冻结来源/目标验证、计划序列化、末次Observe和FA映射检查、Admission基线构造、响应编码/传输、客户端解析和持久化。**各自耗时UNKNOWN**；不能将482.419全部标为fsync、通信、实验审计或可删除成本。

客户端`call_saved`在发送前保存请求，收到响应后保存原始响应及JSON，均有fsync，但没有这些步骤的独立起止时间。`operations[*].start_ns/end_ns`可以测量Adapter调用范围，仍包含Adapter安全检查，且已在协调阶段内；不能当作纯GPU释放时间并再次累计。轻量控制没有重型分阶段Journal，不代表客户端也没有证据写盘。

代码：`evaluation/lightweight_runtime_e1/control.py:33–43,66–119`（`certify/apply/timed`）；`evaluation/lightweight_runtime_e1/bridge.py:168–176`（返回前API与Admission基线）；`evaluation/native_resume_correctness_e4_phase1a/runner.py:27–29,41–46`；`evaluation/exact_greedy_paired_gpu_fix_b5d4a/persistence.py:6–8,24–38`。

## 2. 约7.5秒完整路径的实际组成

定义客户端时刻：a=前置重型参考RPC发起，b=该RPC返回，c=控制RPC发起，d=控制RPC返回，e=目标generate发起，f=首个非空输出，g=流结束。

- Control RPC = d−c；TTFT = f−e；Request Wall = g−e。
- Control-to-completion = g−c = **(d−c)+(e−d)+(g−e)**。
- Audit-inclusive Completion = g−a = **(b−a)+(c−b)+(g−c)**。
- TTFT包含在Request Wall内；Prefill CUDA、COW、外层run_batch等也位于请求内部且可能相互嵌套，不能加到Wall上。

|不重叠区间|均值|范围|认证等级|
|---|---:|---:|---|
|控制RPC d−c|814.493|703.828–879.971|直接记录|
|控制返回到请求发起 e−d|243.077|156.064–259.120|由相同端点代数差得到；具体子操作耗时UNKNOWN|
|Request Wall g−e|6411.764|6046.786–7080.660|直接记录|
|Control-to-completion g−c|7469.333|7110.491–8140.743|直接记录，包含前三行|
|前置重型Reference RPC b−a|1855.752|1762.787–2137.484|直接记录；含参考观察、认证和两种规划，不只是内容哈希|
|参考返回到控制发起 c−b|103.680|65.875–128.707|端点差额；源码包含reference_audit.json落盘，独立写盘计时UNKNOWN|
|Audit-inclusive g−a|9428.766|8965.804–10060.608|直接记录，包含前置参考、间隔和完整控制路径|

**可以进一步定位请求内的长区间。** 12个测量臂中，首个Decode的外层`run_batch` Wall均值6188.090，范围5781.630–6825.545；内层Eager CUDA事件均值1538.774，范围1333.169–1854.883。这些是嵌套区间，不能相加；内外差也不能直接命名为编译时间。例`r1_A_Greedy/post_request.json`的`trace.execution[1]`记录首Decode Wall约6058.53，随后六个Decode各约16.68–19.74；`stream_chunks.json`首、次输出间隔也约6.066秒。因而不能把整个6秒解释成客户端写盘或控制观察。

现有源码给出同流CUDA事件及同步、原生Host调用边界，但缺少编译器、模型Runner内部、CPU提交等待的细分证据。**首Decode长区间的具体原因UNKNOWN**。每个臂新Engine、独立Triton目录，历史请求只生成1 Token；首次Decode/编译影响是待检验解释，不能据此扣除“预热成本”。CUDA事件区间含提交空隙，不等于纯内核时间。

Engine加载、八条历史重放、请求后的`b5d4_state`完整审计、Cold生命周期和清理均不在g−c/g−a内。故“Audit-inclusive”也不是所有实验审计的总生命周期时间，更不是生产Serving延迟。E4-1A还在d与e之间执行完整Smoke后验；不能将其完整路径与1B直接作无混杂性能对照。

代码：上述`runner.py:41–61,77–83`；`evaluation/lightweight_runtime_e1/observation.py:35–57,75–89`。重型与轻量实际插桩均须按这些位置解释。

## 3. E1/E4能支持的比较与成本划分

E1同批四臂三次均值Control RPC：重型Greedy/Exact为4682.064/4926.071，轻量为788.566/806.993；支持在该受控实验中分离重型审计显著降低**所测客户端控制调用**成本。不能把跨工作负载E1与E4数值直接拼成同一对照。E4-1A八个正确性臂Control RPC均值4679.865，仅用于辨认其重型边界，不作为本轮性能样本。

E4-1B A的Exact与Greedy：Prefill CUDA均值42.618对73.026、TTFT74.258对102.601，三对均改善；Control RPC824.074对775.516，三对均更长。Control-to-completion均值7352.864对7527.624，但仅2/3对改善；Audit-inclusive9406.734对9422.254也仅2/3对改善。C的恢复前沿一致，Prefill均值37.460对37.062，不支持恢复位置收益。**真实恢复计算收益已观察到，稳定控制后净收益未证明。**

|比较对象|共同要求/成本|可能的增量与正确口径|
|---|---|---|
|使用同一严格观察和协调路径的Exact、Greedy、PCB/WFC等|相同输入认证、物理观察、RPC、Bind/Preflight、协调安全后验、等强度记录|求解器成本不同；选集不同还会改变物理操作、验证、序列化和恢复成本。相同阶段不等于数值必然相同，不能事后令其相消。收益矩阵是否共有取决于实际调用，不默认所有基线都必须构建它。未运行基线的开销仍UNKNOWN|
|Exact与Greedy当前小规模实例|两者共用矩阵及Plan合同，均驱逐5项|A求解均值0.145对0.110、C为0.152对0.109，不能用几十微秒解释RPC几十毫秒差异；后者归因尚未拆全|
|无额外FlowState控制的原生策略|共同模型、输入、硬件与合法初始条件；原生缓存维护仍可能有成本|应按共同“决策需求已知→请求完成”边界实测，有控制臂计入真实观察/认证/求解/协调，无控制臂计入自身Native管理。不能强加不使用的FlowState控制，也不能用Cold全量重算替代无控制基线；本批没有这种对照，因此净收益UNKNOWN|

内容审计、Oracle和持久化的“实验专用”标签不等于可以从已测值中扣除。要得到低审计路径数值必须真实测量该路径，不能相减估算生产时延。跨Workflow物理血缘仍为PROVENANCE_UNVERIFIED，逻辑K不代表HBM压力。

## 4. 最小缺口与唯一后续建议

当前可复核轻量阶段及含持久化的RPC时间；**不能从现有记录得到无客户端审计污染的端到端控制时间**。需要最少补齐：服务端控制入口/出口；发送/接收与编码/解析；每次证据写盘/fsync；控制返回到generate；首Decode外层和Eager之间的细分时间。服务器/client跨进程区间优先各自测时，不依赖未经校验的跨时钟相减。

**唯一建议：先完成CPU合同回归，再对同一w1/K2、A/C执行一次预注册的“计时补点＋延后证据持久化”的低审计配对复核。** 这是下一轮建议，本轮未实现或执行：

1. 保留连续Scheduler隔离、完整严格边、物理映射、Apply-time、逐项后验与Admission检查；重型参考仍在独立区间，不能未经验证移除其依赖。正常路径证据先保存在内存，完成测量后写盘；失败立即保存已有证据并Fail-stop。新增单调时钟只记录边界，不增加GPU同步或缓存操作。
2. CPU验证单次原生调用、异常不吞没、失败证据保全；先做独立GPU正确性Smoke，再进行固定A/C、Exact/Greedy的小规模独立交错配对。改变持久化时机可能改变Admission间隔，必须重验源状态新鲜度、COW、FA与保留组件、Cold输出，不能凭旧READY继承通过。
3. 继续明确标为新Engine首次使用测量，并定位首Decode长区间；若未来欲报告稳态，必须另行冻结对称预热和空状态重建协议，不能删掉本批长Decode或临时热身后替换结果。本次最小对照的价值是分清真实控制、持久化与请求执行，不承诺生产净收益。无控制基线若要做绝对净收益主张仍须另行授权和预注册，不在本建议中扩张范围。

## 5. 可复核来源

所有路径相对仓库根目录；报告数字由以下原始字段只读聚合，未重新运行求解或实验：

- `evaluation/native_resume_performance_e4_phase1b/experiment_protocol.json`、`execution_lock.json`、`validation.json`：12个正式测量臂、Smoke与18次清理。
- 该目录全部`r{1,2,3}_{A,C}_{Greedy,Exact}/summary.json`、`reconciliation.json`、`post_request.json`、`stream_chunks.json`：Wall端点、七项阶段、Native操作、Prefill/Decode与输出时间。`gpu_raw_timings.csv`、`paired_differences.csv`为交叉核对汇总。
- `evaluation/lightweight_runtime_e1/gpu_raw_timings.csv`及各臂`reconciliation.json/summary.json`；`final_report.md`：E1同批四臂范围。
- `evaluation/native_resume_correctness_e4_phase1a_run01/{A,B,C,D}_{Greedy,Exact}/summary.json`、`final_report.md`：重型正确性验证边界；不并入1B统计。
- `evaluation/native_resume_performance_e4_phase1b/launch.py`：独立Engine、固定顺序和每臂Triton目录；`evaluation/native_resume_correctness_e4_phase1a/runner.py`为实际复用执行入口。

本轮完成，停止于只读审计。没有修改E1/E4冻结文件，没有GPU操作或性能调优。
