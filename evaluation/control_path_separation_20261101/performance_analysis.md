# 控制路径分离：性能分析

**判定：SUCCESS_FULL_CYCLE。**

工作负载为冻结受控C/K3：16候选、4 Pending、K=3，Qwen3.5-9B、SGLang v0.5.17、单H100 PCIe TP=1。所有本轮测量在GPU1的新Engine生命周期中完成，不混入历史GPU0/失败批量路径结果。Smoke仅用于正确性，不进入性能分母。

## 新旧FlowState五轮配对（ms）

|轮次|Control差|Full差|Audit-inclusive差|
|---|---:|---:|---:|
|1|-291.723|-3046.818|-1824.373|
|2|-214.011|-2777.075|-1210.438|
|3|-390.479|-3106.370|-1608.902|
|4|-210.833|-3714.444|-2689.272|
|5|+47.932|-2982.624|-1689.140|

差值为serving−original，负值表示新路径更短。第五轮Control退化保留。

## 五轮原始分布（ms）

|阶段/模式/策略|指标|均值|中位数|最小|最大|
|---|---|---:|---:|---:|---:|
|paired/original/FlowState-Greedy|control_cycle_ms|8433.569|8446.388|8351.347|8496.257|
|paired/original/FlowState-Greedy|full_cycle_ms|21998.372|22021.692|21817.562|22148.197|
|paired/original/FlowState-Greedy|audit_inclusive_ms|23014.709|23002.321|22858.508|23203.282|
|paired/original/FlowState-Greedy|sum_ttft_ms|266.850|267.547|263.108|269.371|
|paired/original/FlowState-Greedy|sum_cuda_prefill_ms|242.673|243.537|238.832|245.150|
|paired/serving/FlowState-Greedy|control_cycle_ms|8221.747|8204.534|8082.573|8399.279|
|paired/serving/FlowState-Greedy|full_cycle_ms|18872.906|18958.628|18433.753|19286.083|
|paired/serving/FlowState-Greedy|audit_inclusive_ms|21210.285|21249.606|20514.011|21791.884|
|paired/serving/FlowState-Greedy|sum_ttft_ms|263.670|263.490|259.644|266.472|
|paired/serving/FlowState-Greedy|sum_cuda_prefill_ms|239.605|240.608|237.094|242.220|
|three_policy/serving/FlowState-Greedy|control_cycle_ms|8083.861|8031.346|7911.451|8349.304|
|three_policy/serving/FlowState-Greedy|full_cycle_ms|18853.904|19027.521|18158.986|19330.647|
|three_policy/serving/FlowState-Greedy|audit_inclusive_ms|21241.874|21366.763|20530.947|21529.159|
|three_policy/serving/FlowState-Greedy|sum_ttft_ms|263.889|263.364|259.341|270.775|
|three_policy/serving/FlowState-Greedy|sum_cuda_prefill_ms|238.389|237.702|234.651|242.882|
|three_policy/serving/Marconi-Evict|control_cycle_ms|7622.850|7588.912|7505.122|7754.989|
|three_policy/serving/Marconi-Evict|full_cycle_ms|18552.812|18520.140|18312.679|18795.712|
|three_policy/serving/Marconi-Evict|audit_inclusive_ms|20602.711|20529.374|20507.679|20808.132|
|three_policy/serving/Marconi-Evict|sum_ttft_ms|747.093|747.213|740.445|752.893|
|three_policy/serving/Marconi-Evict|sum_cuda_prefill_ms|713.674|713.286|710.194|718.428|
|three_policy/serving/KVFlow-inspired|control_cycle_ms|7961.648|7816.041|7722.467|8620.776|
|three_policy/serving/KVFlow-inspired|full_cycle_ms|18972.370|19226.674|18195.011|19615.482|
|three_policy/serving/KVFlow-inspired|audit_inclusive_ms|21053.161|21300.968|20158.042|21967.087|
|three_policy/serving/KVFlow-inspired|sum_ttft_ms|695.244|694.635|693.612|696.793|
|three_policy/serving/KVFlow-inspired|sum_cuda_prefill_ms|664.344|665.004|659.916|666.953|

## 同轮配对汇总（ms）

|比较|指标|差值中位数|胜/平/负|范围|
|---|---|---:|---:|---:|
|serving-original|control_cycle_ms|-214.011|4/0/1|[-390.479,+47.932]|
|serving-original|full_cycle_ms|-3046.818|5/0/0|[-3714.444,-2777.075]|
|serving-original|audit_inclusive_ms|-1689.140|5/0/0|[-2689.272,-1210.438]|
|serving-original|request_execution_ms|-27.863|4/0/1|[-38.053,+17.671]|
|serving-original|sum_ttft_ms|-2.899|4/0/1|[-7.903,+3.121]|
|serving-original|sum_cuda_prefill_ms|-4.543|4/0/1|[-6.443,+3.387]|
|serving-original|control_rpc_wall_ms|-1823.139|5/0/0|[-1975.577,-1655.092]|
|serving-original|rpc_tail_ms|-1648.193|5/0/0|[-1956.039,-1442.264]|
|serving-original|observation_ms|-8.122|3/0/2|[-87.055,+158.765]|
|serving-original|selection_call_ms|+0.846|2/0/3|[-1.153,+2.461]|
|serving-original|selection_stage_ms|-0.094|3/0/2|[-2.172,+4.408]|
|serving-original|freshness_ms|-4.566|3/0/2|[-44.486,+47.218]|
|serving-original|reconciliation_ms|-2.781|3/0/2|[-19.988,+11.173]|
|serving_FS-MC|control_cycle_ms|+406.329|0/0/5|[+279.001,+836.422]|
|serving_FS-MC|full_cycle_ms|+507.381|1/0/4|[-636.725,+683.150]|
|serving_FS-MC|audit_inclusive_ms|+775.819|0/0/5|[+1.573,+857.987]|
|serving_FS-MC|request_execution_ms|-479.261|5/0/0|[-536.318,-459.418]|
|serving_FS-MC|sum_ttft_ms|-483.345|5/0/0|[-490.582,-476.439]|
|serving_FS-MC|sum_cuda_prefill_ms|-475.543|5/0/0|[-480.726,-470.205]|
|serving_FS-MC|control_rpc_wall_ms|+923.201|0/0/5|[+667.561,+1267.272]|
|serving_FS-MC|rpc_tail_ms|+436.163|0/0/5|[+315.508,+646.679]|
|serving_FS-MC|observation_ms|-25.311|4/0/1|[-60.294,+114.762]|
|serving_FS-MC|selection_call_ms|+14.119|0/0/5|[+13.569,+15.294]|
|serving_FS-MC|selection_stage_ms|-2.284|5/0/0|[-3.935,-1.654]|
|serving_FS-MC|freshness_ms|-2.275|3/0/2|[-68.191,+57.199]|
|serving_FS-MC|reconciliation_ms|+11.515|2/0/3|[-12.903,+55.316]|
|serving_FS-KV|control_cycle_ms|+250.360|1/0/4|[-709.325,+481.332]|
|serving_FS-KV|full_cycle_ms|-36.025|3/0/2|[-583.110,+142.071]|
|serving_FS-KV|audit_inclusive_ms|+134.442|1/0/4|[-600.324,+972.754]|
|serving_FS-KV|request_execution_ms|-422.152|5/0/0|[-478.277,-391.191]|
|serving_FS-KV|sum_ttft_ms|-432.111|5/0/0|[-434.272,-425.982]|
|serving_FS-KV|sum_cuda_prefill_ms|-425.266|5/0/0|[-429.467,-423.668]|
|serving_FS-KV|control_rpc_wall_ms|+569.911|1/0/4|[-250.967,+1093.059]|
|serving_FS-KV|rpc_tail_ms|+355.159|0/0/5|[+308.814,+601.020]|
|serving_FS-KV|observation_ms|-46.751|4/0/1|[-880.691,+34.808]|
|serving_FS-KV|selection_call_ms|+11.823|0/0/5|[+11.702,+12.600]|
|serving_FS-KV|selection_stage_ms|-5.482|5/0/0|[-6.222,-3.116]|
|serving_FS-KV|freshness_ms|-11.614|4/0/1|[-66.323,+17.758]|
|serving_FS-KV|reconciliation_ms|-5.848|3/0/2|[-29.193,+51.326]|

## 测量边界与分离成本

Control为T3−T0，Full为T4−T0，均保留原始定义。T5是在全部原后验/Cold/Oracle检查、报告格式化及附加审计记录完成后、summary写盘和Engine清理前读取；Audit-inclusive为T5−T0。它不是包含模型启动和两轮历史重放的完整生命周期。

Control RPC包括控制和RPC尾段，不与Control重复相加；八个timing阶段与clock六段间隙不能混合求和。Selection Call嵌套在Selection阶段。ΣTTFT与ΣCUDA是四个串行受控请求等权总和，不能视为自然并发吞吐。所有Full/T5都来自同一臂直接时间戳，未拼接平均值或扣除估计审计成本。

Journal仍在原时刻捕获完整快照，仍在请求前fsync，只改变紧凑JSON格式；完整响应先持久化再解码。唯一延后的是已有完整持久化响应的重复缩进展示文件。其Wall/进程CPU、编码/压缩/解码开销、字节量及起止时间在separation_audit.json和performance_results.csv中；这些都是实际成本，不是免费优化。

新格式化在T4后同步执行，不与本臂四个目标请求竞争。编码、压缩和解码均在请求前，仍计入Full。未新增GPU调用或异步取证线程；CPU计时为process_time差，可能包含同期线程，不能声称独占函数CPU时间。原路径未单独记录的编码CPU成本保持UNKNOWN。

全量GPU状态SHA、29次协调观察、13次逐项驱逐及前后验全部保留。GPU同步、内容复制和RPC数量没有减少；收益是取证CPU/字节/重复展示路径的组合效果。本轮没有单独消融各项修改，不能按阶段中位数相减给每项分配因果收益。

三策略虽然共享取证代码与要求，保留集不同会改变FA保护区等审计工作量；不能把这类研究审计成本直接称作生产必须成本。形状预热仍计入Full，各自编译与预热成本实际保留。

## 正确性、适用域和负面结果

65项CPU测试、1个GPU Smoke及25个性能臂原合同均通过。26臂初始候选Token/内容、严格矩阵和FA映射签名一致，64条配对无UNKNOWN；每臂13次真实驱逐，3个保留状态及槽位账本正确。逐请求实际E/G/Prefill、Cold、COW与FA保护见pending_results.csv和verified_evidence_inventory.json。跨Workflow物理血缘仍为PROVENANCE_UNVERIFIED。

req_generation不是Mamba槽位Generation。未引入状态缓存或减少观测，安全限定原连续Scheduler/独占受控Engine合同；完整复原的ABA不能靠端点SHA证明不存在，不能据此升级为通用并发事务或长期Pin保证。codec SHA仅认证传输完整性，不是GPU状态认证。

GPU0初始已有其他SGLang任务，未启动模型或干预；预检失败及首次启动前GPU1绑定修订均保存。GPU1每臂检查进程/显存，另一张卡和CPU未声明独占。五轮只是该共享服务器上的描述性证据，不声称生产P95、显著性或普遍优势。

人类可读Baseline名称为Marconi-Evict Adaptation和KVFlow-inspired；均为受控策略级适配，不是原论文完整Serving System。正式FlowState入口实际为n=16下的Marginal Greedy分支，无新求解器。

## 证据索引

逐臂：performance_results.csv；逐Pending：pending_results.csv；逐轮差：paired_comparison.csv；均值/中位/范围与全部原值：statistics.json；完整物理与来源证据：verified_evidence_inventory.json、arms/*/run；代码/协议锁：execution_lock_gpu1.json、implementation.diff；原版本与历史未改：original_source_manifest.json。
