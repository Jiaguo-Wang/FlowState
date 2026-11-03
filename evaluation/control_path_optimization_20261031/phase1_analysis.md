# 控制路径归因（实现修改前冻结）

## 原始证据与范围

直接读取原始五轮十五臂的 summary.json 与 reconciliation.json；全部 PASS，哈希见 phase1_sources.json，逐臂结果见 phase1_raw_timings.csv。n=16、m=4、K=3，单卡 TP=1。所有新结果与历史隔离。

## 实际调用链

launch_round → device_entry → runner.main → 两轮16条历史重放（中间原生清空，仅初始化协议）→ 单次 Scheduler 同步 b5d4_apply → control.apply：完整观察 → 物理映射 → 严格 Epoch 认证 → 生产 plan_strict 的 Greedy 分支/两基线真实选择内核 → 新鲜度观察及 bind/preflight → FA 内容基线 → StrictReconciler（13次 Recurrent-only 驱逐，逐项前后观察，失败停止）→ FA 与选中组件内容核对 → T3 → 日志刷盘与 admission_baseline 全量审计 → RPC返回 → 四次形状预热 → 四个 Pending 串行恢复 → T4 → 请求后审计与 Cold/Oracle 对照。

## 实际阶段计时

下表为五轮中位数，单位 ms；各行取中位后不可相加恢复总中位数。

|阶段|FS|MC|KV|
|---|---:|---:|---:|
|final_heavy_observation|3157.225|2567.497|3123.221|
|cpu_physical_mapping|0.641|0.608|0.642|
|strict_epoch_authentication|153.581|149.185|151.537|
|policy_online_select|32.063|34.730|37.895|
|apply_time_freshness_preflight|1597.137|1251.442|1611.807|
|fa_content_audit_before|1256.561|1038.877|1237.818|
|strict_reconciliation_with_postchecks|908.522|906.281|900.128|
|fa_content_postcheck_and_resume_baseline|1872.684|1110.972|1398.085|
|unattributed_inside_control_ms|1120.212|1092.915|1106.389|
|control_cycle_ms|10138.858|8123.101|9596.033|
|rpc_tail_ms|4565.822|3836.390|4004.728|
|control_rpc_wall_ms|14775.452|11998.061|13511.437|
|cycle_ms_full_incl_warm|24107.215|21379.657|22840.290|

timing 数组的八阶段在 control.apply 中顺序执行，互不嵌套；剩余量仅表示未计时部分，不能推定全是写盘。clock.stages 只包含六段，FA 基线和协调在 clock 段间隙内，不可再把段间隙与八段一起求和。Control=T3−T0；Full=T4−T0 包含 RPC 尾段、形状预热、请求及间隙；客户端 Control RPC 与控制区间存在包含而非并列关系。request TTFT 仅为各自请求区间。

## 操作分类与安全边界

|操作|类别/位置|同步与重复|本轮处理|
|---|---|---|---|
|Scope、静止性、候选/组件/引用/槽位观察|A，T0–T3必要控制|CPU树遍历；协调29次观察各对应不同状态|全部保留，不复用跨驱逐快照|
|严格 Epoch、真实 FA Frontier、选择、绑定和新鲜度|A，T0–T3|同一 Scheduler 连续窗口；不可把结束窗口当租约|全部保留原函数及检查|
|逐项 Native 驱逐及 Postcheck|A，T0–T3|13次真实动作及验证|不批量绕过，不改 fail-stop|
|逐层 FA、逐槽 Conv/Temporal 内容哈希|C，当前也承担真实保护认证；T0–T3及RPC尾段|反复执行 GPU→CPU 同步，单次读中数十次小复制|只批量读取同一时刻张量，逐字节摘要不变，不删检查|
|重复构建小映射/认证|B潜在冗余|成本远小于内容读取；其新鲜度必要性未证明可删|不修改|
|DeferredJournal 序列化及T3后fsync、返回大对象|C，部分位于Full内|真实计入RPC尾段；原记录不足独立细分|保留原行为，不估计扣除|
|请求后审计、独立Oracle|C，T4后|不属于Full，生命周期仍有成本|原样保留|

## 最小优化与验证计划

仅在隔离实验包中切换内容读取实现：FA按层收集后批量设备到主机复制，Recurrent按组件一次收集驻留槽位，再按原始槽位切片的字节顺序计算SHA。保留每次实际观察、组件身份、内容摘要及全部拒绝条件；不缓存跨时刻内容，不引入 mutation_generation，不改变控制状态生命周期。CPU对照检验字节一致性、非连续布局、别名/过期/错误输入和原生严格控制负例；GPU Smoke同时执行旧/新摘要并要求完全相等。新增临时张量的显存和复制次数应记录，不能只凭CPU正确声称GPU成功。

原版与批量版使用相同T0/T3/T4和取证等级，五轮新鲜独立生命周期交错比较；有效后才进行同模式三策略五轮对照。保留所有负向结果，任何安全失败停止。

## 历史131.904 ms的口径

来源 evaluation/control_path_optimization_output/control_20260929/analysis/comparison.json 及 profiling/analysis/measurement_scope.md；24快照×3策略×3次旧协议，1输出Token、Legacy语义。66.225/24.985/0.240/0.929/39.525及总131.904为旧口径均值，不是当前Strict连续控制的同边界测量，不能直接用于速度比。

结论：主要可干预成本是内容读取及同步方式；尚未证明优化有效。没有证据允许删除必要安全检查或改变Full边界。
