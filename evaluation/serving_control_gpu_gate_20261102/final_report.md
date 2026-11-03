# Serving精简接口最小GPU正确性验证

**最终裁决：GPU_CORRECTNESS_PASS。** 仅执行了A `shadow_validation` 与B `serving` 两个独立GPU生命周期。两臂均通过原完整Runtime合同和独立结果复核，共完成26次真实Recurrent-only驱逐、8个Pending的8 Token逐Token Cold一致性检查。未启动第三个GPU生命周期、三策略矩阵或参数搜索。

**本轮没有证明性能收益。** 两臂分别只有一次执行，A含双接口额外检查；B的实测Control与完整周期更长，全部保留。不能将体积减少解释为生产Serving加速。

## 1. 环境、输入与执行边界

- 冻结C/K3，16个候选、四Pending、逻辑K=3，28槽物理池；Qwen3.5-9B / SGLang v0.5.17 / H100 PCIe / TP=1。
- GPU1 UUID：`GPU-051c9abc-864a-9f31-c4f7-a737aed997ff`。镜像ID：`sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926`，RepoDigest为`lmsysorg/sglang@sha256:2fa998888b1b48e778e17f6360a6b60f3cc888b56fa35281669af1d007316337`。CPU只读容器确认Torch 2.11.0+cu129、Transformers 5.12.1。
- 策略固定FlowState-Greedy；实际入口仍是生产 `GlobalOptimizer.plan_strict`，n=16超Exact准入后走Marginal Greedy；没有加载冻结选集冒充在线求解。
- 原十份输入/映射/Cold文件逐字节复制且与旧协议SHA相等；模型配置、Tokenizer、模板核对相等。原16预热→flush→16重建→协调→四请求形状预热→四正式Pending协议不变。
- [执行协议](experiment_protocol.md)、[执行锁](execution_lock.json)、[启动日志](launcher.log)、[CPU门禁日志](cpu_tests.log)及两臂各自`command.json`完整保留。新增文件仅在本目录，生产与历史CPU/GPU包未修改。

## 2. GPU前发现的必要表示适配

上一轮79项CPU门禁主要覆盖JSON后的列表表示。此次增加真实Scheduler序列化前形态，发现`asdict`保留的Token/Slot元组与完整baseline列表直接比较会误拒绝。**GPU启动前**在本包将这两处比较改为双方tuple逐元素比较；没有改变Token、Slot、长度、引用或任何SHA条件。旧包、旧结果不修改。

新增六项CPU回归覆盖元组形态、旧新模式等价、完整生产者恰一次、异常传播、原安全函数一致及计时回传。合计**85项CPU测试全部通过**后才冻结和启动GPU。最小差异见[implementation.diff](implementation.diff)，表示问题复现/修复见`test_instrumentation.py::test_real_scheduler_tuple_representation`。除此之外，仅新增计时与新目录编排；`control.py`、`bridge.py`、`codec.py`、`protocol.py`、`observation.py`与CPU原型逐字节相同。

## 3. 两臂真实正确性结果

|检查|A：shadow_validation|B：serving|
|---|---|---|
|当前严格Epoch|16候选、4 Pending；24准入/40拒绝/0 UNKNOWN|相同逻辑合同；独立新Epoch|
|逻辑保留集|C_B_P05120、C_C_P06144、C_D_P07168|相同|
|模型化J|154.348955616 ms|相同；不是GPU延迟|
|当前物理选集|本臂观察到N57/N56/N55|本臂重新观察到N57/N56/N55；未复用A绑定|
|完整/精简Baseline|同次完整读取产生两接口并校验一致|仅返回精简Baseline；完整生产者仍执行|
|真实驱逐|13项均returned、账本完成13项|13项均returned、账本完成13项|
|驻留与空闲槽|Resident 16→3；Free 12→25|相同；各臂独立记账|
|FA与组件保护|协调FA断言、索引/结构、Selected内容及请求后保护内容通过|全部相同门禁通过|
|内容认证|两次实际cache_state完整审计；受保护FA每次480 MiB|没有减少读取或SHA，仍两次完整审计|
|Native / COW / Cold|四请求完成；B/C/D各一次已完成COW；全部8 Token Cold完全相同|相同|
|资源清理|Engine shutdown返回、宿主退出0、无本轮容器/目标GPU计算进程|相同|

同一逻辑Epoch的比较按CPU逻辑候选映射、Token摘要、位置、Pending T、所有严格边及其条件重新规范化，结果完全相同；没有要求两臂Epoch摘要或物理指针相等。逐臂绑定、组件、Native/COW源工作槽及完整输出见[correctness_evidence.json](correctness_evidence.json)，完整前后状态见各臂`run/reconciliation.json`和`run/post_request.json`。

协调FA全内容不变由原control在同一同步窗口内实际读取并断言；外部复核另检查记录中的Attention/非目标结构摘要、全部原FA索引和受保护FA内容。没有为本报告另开GPU读取，也不把未单独保存的每一个局部中间SHA声明为额外独立取证。

### 逐Pending实际恢复与计时

下表均为本轮单次真实值，时间单位ms。Prefill包含冻结32 Token动态尾，因此G=0仍有32 Token Prefill。

|臂|Pending|H|T|E|G|实际Prefill|CUDA Prefill|Client TTFT|Request Wall|已完成COW|Cold|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|A|C_A|4096|4096|0|4096|4128|155.184|162.918|284.334|0|逐Token一致|
|A|C_B|5120|5120|5120|0|32|26.322|31.807|150.885|1|逐Token一致|
|A|C_C|6144|6144|6144|0|32|27.534|33.021|154.311|1|逐Token一致|
|A|C_D|7168|7168|7168|0|32|26.185|32.132|151.740|1|逐Token一致|
|B|C_A|4096|4096|0|4096|4128|156.332|163.538|287.303|0|逐Token一致|
|B|C_B|5120|5120|5120|0|32|29.129|34.602|162.758|1|逐Token一致|
|B|C_C|6144|6144|6144|0|32|27.342|32.873|161.867|1|逐Token一致|
|B|C_D|7168|7168|7168|0|32|26.540|32.258|157.897|1|逐Token一致|

C_A 的E=0，合法退化为完整重算，没有Recurrent命中或COW；不能写成四请求均命中Recurrent。C_B/C_C/C_D实际E分别为5120/6144/7168，三个源槽与工作槽在本臂各自隔离，copy返回完成，历史Conv/Temporal身份与内容仍与协调前/后/请求后基线一致。全部8个输出数组及来源见[逐请求CSV](pending_timings.csv)和原`stream_chunks_*.json`。跨Workflow物理血缘继续为`PROVENANCE_UNVERIFIED`。

## 4. 实际响应体积

单位为实际字节；不是预计节约量，不用它推导延迟。

|对象|A|B|
|---|---:|---:|
|完整RPC编码前紧凑JSON|27,903,224|25,843,998|
|zlib后实际载荷|12,597,401|11,565,034|
|Base64载荷|16,796,536|15,420,048|
|实际线响应response.raw（含外层封装/换行）|16,796,970|15,420,482|
|response.json档案|16,797,027|15,420,539|
|返回的admission_baseline紧凑JSON|2,059,187|606|
|精简旁路admission_baseline_serving|606|0|

B的606字节仅指Baseline子对象，**整包线响应仍约15.4 MB**，因为完整Epoch、观测/协调账本等仍按原合同返回。完整生产与SHA也仍在关键路径；没有把响应裁剪冒充设备读取最小化。原codec的`full_evidence_preserved`表示对输入编码对象无损，不表示精简对象仍含被省略的完整Baseline字段。

## 5. 阶段计时与包含关系

下表单位ms，保留每臂单次实测，不提供收益百分比或胜负。完整数值见[stage_timings.csv](stage_timings.csv)。

|区间|A|B|
|---|---:|---:|
|完整Baseline生产（仍含原SHA/读取）|789.714|895.716|
|接口投影|12.144|14.161|
|服务端接口校验|11.636|11.115|
|客户端接口校验|13.119|10.210|
|post_apply_baseline父区间|813.511|921.945|
|完整响应编码父区间|492.748|464.732|
|　JSON编码【编码子项】|229.092|227.743|
|　压缩【编码子项】|229.903|209.859|
|客户端解码与校验|456.777|467.099|
|call_saved父区间：执行+传输+持久化|10148.800|12060.242|
|编码结束→客户端持久化结束混合尾段|479.670|437.724|
|Control T0–T3|7951.111|9813.778|
|Control RPC Wall|10621.001|12539.866|
|RPC尾段【完整周期内子区间】|2667.301|2721.746|
|Full Cycle T0–T4|18746.404|20785.824|
|Audit-inclusive T0–T5|21134.181|23387.206|
|T4–T5【T0–T5子区间】|2387.777|2601.382|

- 原T0/T3/T4/T5位置未改。T0–T3是Observe/在线Select/协调及必要后验；T0–T4包含原形状预热、RPC尾段、请求与间隙；T5还包含原请求后认证、Oracle和证据展示，不含Engine启动/历史重放/最终shutdown。
- Baseline生产→投影→服务端校验为相邻子区间，包含在post_apply_baseline和RPC中；不能再加到父区间。A的服务端校验和客户端校验还会重新验证精简投影，这是其额外验证职责。
- `response.raw`、`response.json`在原call_saved路径中实际flush/fsync后返回，再解码和验证接口，随后才发送Pending。每臂原control的6份Journal仍实际执行序列化及落盘；没有移动持久化或删除同步。
- **纯网络/socket传输时间为UNKNOWN。** 当前直接记录的call_saved父区间包含服务端执行、外层JSON发送、网络等待及原文件写盘/fsync；编码结束到持久化结束也包含多种开销，不能命名为纯传输。没有通过事后估计相减伪造该指标。
- B的Control及T0–T5数值均高于A。本轮样本量各1、A有双接口验证、GPU/Host运行存在波动；这些数值完整保留，既不能证明收益，也不能据此作统计退化归因。没有剔除重型审计成本或把J当成实测时间。

## 6. 异常、身份范围与清理

1. 初次沙箱nvidia-smi无法连接驱动；宿主只读检查确认是可见性限制，GPU1无计算进程、17MiB，随后原门禁再次检查后才启动。两臂前后资源均留在各自`preflight.json/gpu_cleanup.json`。最终目标GPU17MiB、无计算进程，控制端口空闲，原监控容器仍运行，GPU0未被本实验绑定或清理。磁盘最终仍大于20GiB。
2. GPU两臂没有Native/Cold/FA/组件或原测量门禁失败，没有重试。日志保留可选多模态TorchCodec/FFmpeg加载traceback及torchao SyntaxWarning，发生在Engine就绪前；退出时两臂均记录CudaIPC生产者退出警告。原始记录未删除；实际Engine shutdown返回、docker退出0、目标卡无计算进程/低显存及容器清理门禁均通过。不将该警告写成“完全没有任何warning”。
3. **两臂局部engine_id均为`r2c0-pid193`。** 容器PID命名空间可复用PID，不能把该字符串作为全局唯一Engine身份。事后分析最初错误地要求该字符串不同，触发的是分析断言而非GPU合同失败；原记录在[analysis_initial_failure.log](analysis_initial_failure.log)。随后按实际独立容器命令、非重叠宿主区间、各自初始空状态、不同Barrier/Epoch及中间GPU/容器清理证据复核。两臂没有复用任何绑定，分别核验本臂的engine/barrier/lease/plan/组件。
4. A Barrier=`b5c3-193-1981114532588498`，B Barrier=`b5c3-193-1981225681956947`；两个宿主进程均正常结束，A结束早于B开始。补充只读Docker事件保留了B的die/destroy与容器ID；A的完整容器ID未留存，标记UNKNOWN，不伪造。独立性不依赖A的未留存全ID或跨生命周期指针差异。
5. 未新增Slot Generation，不把元数据摘要当作GPU内容认证、跨RPC租约或ABA解决方案；仍限定原受控独占生命周期和既有同步窗口。异常拒绝/Fail-stop来自原安全路径及85项CPU回归，本轮没有故意破坏真实GPU做故障注入。

## 7. 证据索引与结论边界

- [correctness_evidence.json](correctness_evidence.json)：规范化逻辑比较、每臂当前绑定、Native源/工作槽、实际COW、Cold全部Token。
- [response_sizes.json](response_sizes.json)、[stage_timings.csv](stage_timings.csv)、[pending_timings.csv](pending_timings.csv)：实测字节与原始逐臂/逐请求时间，不混用历史性能数据。
- `arms/{A_shadow_validation,B_serving}/run/{initial,reconciliation,post_request,summary,engine_cleanup}.json`：当前物理状态与原门禁证据；`control_evidence/`及`rpc_*/response.raw/json`保留原日志和持久化响应。
- [gpu_cleanup.json](gpu_cleanup.json)、每臂`gpu_cleanup.json/host_exit.json`、[docker_lifecycle_events.json](docker_lifecycle_events.json)：资源、独立执行与异常边界。
- [evidence_manifest.json](evidence_manifest.json)和[execution_lock.json](execution_lock.json)：原始结果、输入与代码身份。报告生成时再次核对冻结源文件和两臂输入SHA不变。

本轮支持：精简Baseline返回接口在此C/K3受控配置下保持必要安全语义，能够执行真实协调、Native/COW及Cold正确性认证，可供另行授权的受控性能实验使用。

本轮不支持：生产部署、稳定端到端净收益、三策略性能排名、全部工作负载泛化、跨Workflow物理血缘认证，或删除任何尚未等价证明的SHA/状态读取/Postcheck。

**任务到此结束，未启动任何后续优化或正式性能矩阵。**
