# B5-D4 启动前冻结协议

唯一目标为 B5C2_C_TURN_003。使用原 B5-C2 四来源、十二份Token身份及八条历史 A1/B1/C1/D1/A2/B2/C2/D2。工作负载为 Trace-derived Controlled Composition，不是自然并发。历史每条生成1 Token；正式目标与独立Cold均 temperature=0、max_new_tokens=8、ignore_eos=True。正式请求采用流式API，stream_interval=1，根据真实TokenizerManager的incremental_streaming_output记录累计或增量模式；保存每条响应及抵达时间，按该模式恢复完整输出并核对completion_tokens。

每臂全新Engine，Qwen3.5-9B、冻结镜像SGLang v0.5.17、GPU0 H100 PCIe、TP1、28槽、原缓存参数与非重叠调度、禁用CUDA Graph配置。K固定2，不代表物理池压力。生产Greedy选择和D1 Exact使用同一目标、严格输入合同和原同分规则，不修改生产实现。

## 固定执行顺序与停止

最多5对，顺序为：1 Greedy→Exact，2 Exact→Greedy，3 Greedy→Exact，4 Exact→Greedy，5 Greedy→Exact。每臂独立生命周期、独立空Triton目录；不复用本轮其他臂编译缓存，不新增热身请求。其他系统/文件缓存无法强制完全冷却，记录并作为限制。由于奇数对数，首位策略3/2分布。

先执行第1对，再执行一次独立空缓存Cold Reference。逐Token、实际来源、COW、前后状态和配对语义均通过后，才执行余下4对；所有后续输出均与同一冻结Cold输出检查。最多11个生命周期。任意错误或不可比即停，无重试、补位、扩大样本或按胜负删除。失败与已完成运行均保留。

## 身份与控制

输入与依赖摘要在启动前冻结；当前Node/Slot来自新观察。单次Scheduler控制调用内依次观察、Strict Epoch认证、当前策略规划、独立Oracle、Apply-time校验、生产bind/preflight及reconcile_strict；无跨RPC旧绑定应用。Exact按当前Oracle核验，Greedy不要求最优。固定Pending C必须由当前所选、最深、已准入共享状态恢复；Exact来源须可追溯D历史，不能只凭输出一致通过。

配对按完整Token/位置、严格边与条件、配置、组件内容、FA索引/内容、Pending T/H、物理池占用比较。不同生命周期的Node/Slot地址不要求相同，也不跨生命周期复用。对各臂仍分别验证实时绑定、源状态保护及Admission新鲜度。

## 预注册指标与计时边界

1. 四Pending J与目标C单请求模型成本分别保存，不当作GPU时间。
2. 主要设备指标：COW完成后目标Prefill `EagerRunner.execute` 同一CUDA流start/end事件区间；另保留包含COW/审计的 `Scheduler.run_batch` 区间；非重叠调度，原调用恰好一次。事件前全设备同步，事件后等待，记录额外同步耗时。该设备时间线区间包含主机提交空隙及可能的首次JIT编译等待，不是内核耗时之和；仅作本审计化配置下的成对区间比较。
3. COW原调用同样记录CUDA事件及Host调用区间；Native Match单独记录Host原调用区间。COW/Match可与Prefill或准备阶段存在嵌套，不相加宣称独立总成本。
4. TTFT：调用流式Engine.generate前单调时钟至收到首个非空Token响应；Wall至流式迭代结束。含Admission检查、哈希/日志、调度、传输与编译。TPOT仅报告首末流式响应间隔/7的描述性均值，不作稳态推断。
5. 完整控制：客户端发起控制RPC至返回；完整实测路径从该RPC发起至目标流结束，包含控制后写盘/准备，不包含八条历史及Engine启动。后置完整性采集时间另列。子阶段包含观察、Epoch、收益矩阵、求解、Plan、Oracle、Apply检查、全树FA哈希、生产协调和原Adapter逐目标时间。子阶段与总量有包含关系，不重复求和。
6. 同步计时与完整审计会扰动原生异步执行；两臂一致。证据写盘、GPU内容哈希和Oracle保留，不从完整Wall中扣除形成虚构轻量生产成本。
7. 每对逐项差值及全部原始数据；最多5对只报告描述统计，无P95、显著性或四Pending吞吐/Goodput主张。READY不依赖Exact获胜。
