# Phase R2-C1a：真实 Recurrent-only Reconciliation 正确性验证

## 最终门禁：READY

在一个新的独立Engine生命周期中，g001/K2严格计划通过正式 `StateController.reconcile_strict`、`StrictRuntimeBridge` 和 `SGLangAdapter` 完成 **10项真实Recurrent-only驱逐**。最终Managed Resident Set恰为 **RQ3_G001_B_TURN_001、RQ3_G001_B_TURN_003**。12个驻留槽变为2个，空闲槽从16变为26。所有生产Postcheck及归档复核通过，没有部分失败。

本轮只验证物理协调：没有提交4个Pending，没有请求恢复或COW正确性测试，没有性能A/B，没有TTFT或控制净收益结论。历史回放中的原生状态复制属于原有历史准备过程，不是本轮新增恢复实验。

## 1. 环境及设备选择

- Qwen3.5-9B，SGLang 0.5.17，Transformers 5.12.1，TP=1，28槽，原extra_buffer/256-token tracking及其余Engine配置逐项保持。
- GPU0存在其他用户进程，因此未使用或操作GPU0。本轮使用空闲的同型号 **GPU1 H100 PCIe**，UUID `GPU-051c9abc-864a-9f31-c4f7-a737aed997ff`，映射为容器内cuda:0。只改变宿主设备选择，不改变单卡模型执行配置；见gpu_before.json及command.json。
- 镜像ID `sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926`，与C0一致。模型、Tokenizer、输入及244项来源文件重新校验；容器内在Engine创建前再次核对Runtime源码、依赖版本与模板身份。
- 不复用C0的Engine、窗口或绑定。当前Engine身份 `r2c0-pid194`，当前窗口 `r2c0-194-466095643327703`。身份字段中的r2c0前缀来自复用的观察器命名，不表示复用旧生命周期；当前PID、单调时钟窗口与绑定均重新生成。
- `implementation_lock.json` 单独记录本轮脚本、生产控制器/Adapter及所复用C0代码摘要，旧锁未覆盖。

## 2. 当前严格状态与执行链

直接读取g001冻结的12个历史Token输入，按原分组、round3及组内顺序回放；4个Pending输入保持冻结且不提交。history_identity.json记录实际Runtime接收到的历史输入摘要与数量，全部一致；Planning Target、FA Frontier、候选位置及Token前缀与冻结合同一致。

在单次Scheduler队列回调内执行：

`当前只读Observe/严格48对条件核对/Bind/Preflight → GlobalOptimizer.plan_strict再次确认同一计划 → StrictRuntimeBridge.control_epoch → StateController.reconcile_strict → 生产StrictReconciler重新Bind/Preflight → 逐目标SGLangAdapter.evict_mamba_only与Postcheck → 最终Postcheck → Selected数值摘要核验`。

只读门禁复用C0函数但没有返回Scheduler主循环，再调用正式协调器；整个外层RPC为独立的 `r2c1a_reconcile`。不是把C0已经结束的租约拿来执行，也没有拆成多个Admission可能插入的RPC。当前48对关系为15条EPOCH_ADMISSIBLE、33条REJECTED、0条UNKNOWN，与冻结条件一致。

`bridge.py:control`建立本轮桥接；`RecordedAdapter`仅记录调用前后时间、目标和返回/异常，实际动作完整委托生产SGLangAdapter。没有重新调用Legacy选择器、修改Greedy、K或兼容关系，没有跳过任何生产检查。

## 3. 连续隔离与安全边界

延用已核实的普通Scheduler循环：接收/处理请求后，在无batch时同步调用on_idle，并在该调用返回后才可能再次Admission。当前回调同步完成只读门禁及协调；没有await、工作线程分派或中途释放调度权。scope逐次核对running/waiting为空，禁用overlap、HiCache、session cache、异步传输、HiSparse、额外ReplaySSM等写者；CUDA工作在进入观察前完成同步。

源证据：冻结Scheduler `event_loop_normal:1682–1713`；C0归档的runtime_sources.json；`tests/runtime/wp3b_end_to_end_transport.py:172–176`；本轮bridge.py的scheduler_process、control及held回调。

异常时生产协调器记录completed_evictions与失败对象、在窗口内invalidate；外层设置失败状态，on_idle停驻直到本轮父进程清理。该GPU运行未发生故障；真实故障停驻分支未注入破坏验证，CPU测试覆盖了失败前拒绝和中途停止合同。不能据此宣称跨RPC事务、回滚或长期Pin/Lease能力。

## 4. 实际驱逐账本

|次序|Checkpoint|当前Node|实际原Slot|Adapter返回及Postcheck|
|---:|---|---:|---:|---|
|1|RQ3_G001_A_TURN_001|18|3|PASS|
|2|RQ3_G001_A_TURN_002|24|12|PASS|
|3|RQ3_G001_A_TURN_003|28|20|PASS|
|4|RQ3_G001_B_TURN_002|25|14|PASS|
|5|RQ3_G001_C_TURN_001|21|8|PASS|
|6|RQ3_G001_C_TURN_002|26|16|PASS|
|7|RQ3_G001_C_TURN_003|30|24|PASS|
|8|RQ3_G001_D_TURN_001|23|10|PASS|
|9|RQ3_G001_D_TURN_002|27|18|PASS|
|10|RQ3_G001_D_TURN_003|31|26|PASS|

实际调用10项、成功返回10项、completed_evictions10项，恰为逻辑候选补集。failed_checkpoint_id=null，error=null，phase=complete。无重试、无额外目标、无部分成功后继续掩盖错误的情况。

保留B1对应当前Node20/Slot6；B3对应当前Node29/Slot22。Node/Slot恰与C0编号相同，但来自新Engine按完整Token匹配、当前组件读取与绑定，不是历史编号复用。

## 5. 实际物理后置条件

|项目|协调前|协调后|结论|
|---|---|---|---|
|Managed Recurrent Resident|12项|B1、B3两项|PASS|
|空闲Recurrent Slots|16|26|PASS|
|选中Conv/Temporal|完整驻留|相同切片身份、完整驻留|PASS|
|选中组件实际数值|本轮驱逐前SHA-256|两候选全部Conv/Temporal摘要一致|PASS|
|其余10项Recurrent|各占一个独立槽|无slots及可恢复组件映射|PASS|
|14个非根节点及所有Token前缀|当前完整树|不变|PASS|
|FA索引与树结构|当前全树摘要|不变|PASS|
|非目标FA、引用、recency等|当前合同摘要|不变|PASS|
|候选外Recurrent|0|0|PASS|
|重复释放、Slot/组件别名|无|无|PASS|

生产协调期间保存23次observe：初始绑定、Preflight、每步前后及最终检查；每次均重新验证当前物理身份和槽位记账。`verification_summary.json`从原始账本按实际操作完成时间再次核对这23个视图，未重新运行实验或分配算法。

额外读取了Selected实际Conv/Temporal内容并计算SHA-256，协调前后相同。完整FA payload没有逐字节哈希；FA结论限定为既定的索引、结构、分配器和非目标合同，加上Adapter的原对象/路径检查。原生Mamba LRU及evictable/free计数随目标驱逐合法改变，不要求这些目标元数据保持原值。

释放含义是槽位归还Allocator和撤销Recurrent映射，不要求擦除预分配张量中的残留字节，也不表示向操作系统释放相同数量HBM。物理池有空闲时仍落实逻辑K=2；这不等于GPU容量耗尽。

## 6. CPU门禁及失败记录

最终 `cpu_tests_locked.log`：57 passed、16 subtests passed。包括生产严格合同回归及新增入口成功、第二次操作失败立即停止、只读Preflight失败零操作测试。初版模拟测试未进入FakeRuntime自身控制上下文而失败，完整保留cpu_tests_initial.log；修正测试装配后通过，没有改动生产合同或降低安全条件。

实际GPU无部分失败。运行日志保留音视频依赖告警与Engine关闭时CUDA IPC告警；没有据此省略失败信息。实际返回码0、shutdown返回、资源清理验证通过。

## 7. 清理及完整性

Engine关闭、本轮容器已不存在，GPU1恢复14MiB、无计算进程；见engine_cleanup.json、process_result.json与gpu_cleanup.json。只对本轮命名容器清理，未终止GPU0其他用户任务。

执行前后源文件、模型和冻结输入摘要一致，C0完整证据Manifest及B4-B1/B4-C Manifest复核通过。未修改生产Optimizer、Controller、Adapter、Phi或冻结历史产物；新增代码与报告仅位于本目录。最终 evidence_manifest.json 记录本轮证据摘要。

## 8. 下一阶段条件

**具备进入限定g001的实际请求恢复验证准备条件。** 本轮已经证明严格选集通过生产协调路径落实为受控物理状态；尚未证明随后Pending实际命中哪个状态、恢复输出正确或性能改善。

进入后续恢复验证仍须另行授权，使用新生命周期并重新验证当前严格合同、协调结果及参考输出。不得复用已结束窗口的物理绑定，也不得把本轮Selected驻留解释为长期保护。本轮停止，未运行R2-C1b或R3。
