# Phase 1：在线检查与取证消费者审查

## 原始版本与隔离

已独立复核原十五臂原始数据与18项生产/入口源码哈希，未被上一轮失败批量化覆盖；清单 original_source_manifest.json。新包从 evaluation/p0_control_cycle_gpu 原版复制，不导入上一轮 batched_payload，不改复制粒度、布局或Batch Size。Git提交8cba094d2c5466759b70e4a5ef8f4824b61aade5；仓库其他已有工作不重置。

## 真实调用链与计时

单卡TP=1、禁Overlap/HiCache/分离传输等配置下：launch → device_entry → runner两轮16条历史 → Scheduler.on_idle同步控制 → capture.observe_control → 当前全树PhysicalView/FA Frontier与严格矩阵 → census映射 → planner.from_frozen/prepare → plan_strict（n16的Greedy分支）→ 当前物理Preflight → FA/Selected内容基线 → StateController.reconcile_strict → 13次驱逐与逐项前后观察 → FA/Selected内容后验与恢复保护基线 → T3 → Journal写盘/fsync与admission_baseline → RPC响应编码/传输/客户端原始响应持久化及格式化报告 → 形状预热 → 4请求 → T4 → 完整审计/Cold/Oracle。

原始FS五轮中位数（ms；不同阶段中位不能直接相加）：

|阶段|中位ms|
|---|---:|
|final_heavy_observation|3157.225|
|strict_epoch_authentication|153.581|
|policy_online_select|32.063|
|apply_time_freshness_preflight|1597.137|
|fa_content_audit_before|1256.561|
|strict_reconciliation_with_postchecks|908.522|
|fa_content_postcheck_and_resume_baseline|1872.684|
|in_window_serialize_ms|244.943|
|after_window_serialize_ms|931.981|
|after_window_flush_ms|440.623|
|control_cycle_ms|10138.858|
|rpc_tail_ms|4565.822|
|cycle_ms_full_incl_warm|24107.215|

timing数组八段顺序不嵌套；clock六段的间隙包含FA基线和协调，不能再与八段重复相加。Selection Call是policy_online_select内部子段。Control=T3−T0，Full=T4−T0，RPC Wall与Control存在包含关系。逐臂见 phase1_timings.csv。

## 检查、实际消费者及A/B/C分类

|操作|消费者/源码|分类与处理|
|---|---|---|
|Scope、无活动请求/异步传输、真实FA Frontier|r2c0.bridge.scope/observe；b5c3.observe_control；strict_epoch准入及Apply-time check|A；全部保留，真实重复读取不缓存|
|Token、Node/Slot/组件切片、引用/别名/槽位记账|planner.prepare、strict_control.bind/preflight/postcheck、SGLangAdapter|A；全部保留|
|完整Recurrent SHA|capture.observe_control前后相等、control.check与初始相等、所选状态驱逐前后及请求后相等|当前合同的同步消费者明确；不删、不移、不伪造|
|完整FA SHA与保护区SHA|control.apply驱逐前后内容门禁、bridge.cache_state恢复基线及runner请求后核对|当前合同的同步消费者明确；不删、不移|
|29次协调物理观察|StrictReconciler每次驱逐前/后与最终postcheck|A；不同状态时刻，不属于可直接复用的重复|
|Journal缩进JSON、后续fsync|纯证据文件；不参与选择、bind或Native输入，但落盘失败原来会阻止请求|B/C；可用等价紧凑JSON代替，仍同阶段快照、T3后且请求前fsync；不改失败时机|
|apply完整响应重复JSON编码/大传输/重复格式化|probe.ControlServer → persistence.call_saved → runner；原始响应先落盘，再读取同一result字段|C；可采用可校验无损压缩封装，解码及持久化均在请求前；不复用任何Runtime观察|
|客户端74MiB缩进reconciliation.json|仅后续证据阅读；同一完整响应已被call_saved保存并fsync|B；仅此重复展示格式可移到T4后，原始完整可解码证据仍在请求前持久化；T5必须计入格式化开销|
|请求后审计和独立Oracle|最终正确性/测量认证|B但门禁；全部按原逻辑执行，未完成不能PASS|

## ABA、Generation与控制边界

冻结SGLang scheduler.py:1682 event_loop_normal在同一线程依次recv/process/run/on_idle；transport在on_idle的单次drain_one中调用控制，不能把返回后的控制窗口当租约。r2c0.scope排除ongoing insert、transfer、HiCache、Overlap等。Adapter.evict_mamba_only仅调用MAMBA组件释放，逐项Postcheck仍检查FA、节点、槽账本和非目标状态。

memory_pool.py中ReqToTokenPool.req_generation是请求池身份，不等于Recurrent槽位世代；现有PhysicalView组件ID为存储切片摘要，没有可跨窗口使用的Mamba allocation generation。单凭末态相等不能发现完整ABA；SHA也无法证明中间从未释放后原样恢复。因此没有证据允许通过缓存PhysicalView、Generation代用或删掉SHA来缩短必要验证。本轮不引入此类复用。

安全可行范围仅为原已验证的连续控制/独占受控Engine合同：新优化对每次实际观察、同步、状态读写、异常隔离及Native路径调用次数和顺序不作任何改变；日志编码和已持久化证据的展示格式不影响物理生命周期。控制返回后仍依赖既有Native Match/COW与源状态/Cold核验，不声称长期Pin或通用并发事务保证。

## 可实施的最小集合

1. Journal同一时刻用紧凑JSON编码：解析后与缩进JSON完全相同，保留原6个阶段文件及请求前fsync、失败处理；减少重复格式化，不缓存物理状态。
2. apply成功响应采用JSON→zlib→Base64封装，带长度/SHA；客户端仍用原call_saved先保存完整原始响应并fsync，再严格解码，任何不一致拒绝请求。失败响应保留原格式并Fail-stop。完整响应没有裁剪字段。
3. 只将已持久化完整响应的重复缩进reconciliation.json展示移至T4后；新增T5审计完成墙钟，记录格式化CPU/Wall、编码解码CPU/Wall、字节量及是否与请求重叠。T0/T3/T4定义不变，Audit-inclusive不得隐藏。

不减少GPU同步、全树/逐槽复制或安全观察次数，不减少RPC数量；拟减少的是取证CPU格式化、线上字节及重复请求前格式化。真正必要控制下限仍UNKNOWN，不能用总时间减去阶段中位数估算。

## 进入实现的判定

存在可证明等价的C类编码/展示优化，可进入最小实现；不存在已证明可安全删除的SHA/物理观察。本轮不宣称通用低审计Serving实现。先CPU逐字段/原始失败合同测试，再GPU Smoke与最小配对；若明显退化停止，禁止继续调参。
