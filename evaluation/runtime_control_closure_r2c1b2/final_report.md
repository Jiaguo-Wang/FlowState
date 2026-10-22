# Phase R2-C1b-2：真实请求接口修复与跨Workflow恢复重验

## 最终门禁：READY

本轮只验证 `RQ3_G001_A_PENDING_TURN_004`，使用g001/K2严格计划B1/B3。没有运行其他Pending、策略性能比较或修改Greedy、Phi、严格兼容矩阵。R2-C1b-1的失败记录保持原样。

## 一、接口修复与CPU门禁

新建 `observation.py`，使用真实 `Req.get_fill_ids()`、`extend_range`、`prefix_indices` 及Batch的 `prefix_lens/extend_lens`。不再读取不存在的Req.fill_ids或Req.extend_input_len。包装原函数恰好一次，返回值与异常原样传递；可选字段缺失记UNKNOWN。没有额外Cache查找或决策修改。

冻结镜像中实际Req、ScheduleBatch与ForwardMode的CPU检查：43项测试及3个子测试通过。覆盖真实目标输入、Cold、分块、Decode、只读性、缺失诊断、必要字段冲突、调用次数与异常传播。没有用FakeReq代替真实类型。既有FakeRuntime测试仅用于协调合同。首次测试收集导入错误已修正，日志保留；不是GPU故障。源码、时机及CPU限度详见 `req_api_audit.md`、`req_api_source.json`、`cpu_tests.log`。

## 二、资源、身份与执行范围

首次宿主机预检查发现GPU0属于其他用户任务，未创建Engine；原始拒绝记录保留于 `preflight_gpu0/`。随后将宿主机GPU固定到空闲GPU1，同型号H100 PCIe，容器内cuda:0。没有操作其他用户进程。

Qwen3.5-9B、SGLang v0.5.17、TP=1、28槽及原Engine参数全部核对。12条历史请求使用冻结Token；Strict与Cold各自独立生命周期。Cold只提交相同完整目标输入，初始Recurrent Cache为空，不复用Strict状态。每臂均核对来源和实现SHA。没有使用旧Node/Slot作为当前映射。

当前源码身份：`implementation_lock.json`；来源身份：`source_identity.json`；实际命令及设备身份：`strict_command.json`、`cold_command.json`、各臂gpu_before文件。旧来源共244项再次校验，另核对C0、C1a及C1b-1完整清单。

## 三、严格协调与Admission

同一次Scheduler同步控制调用内完成当前Strict Epoch/计划/Bind/Preflight、10项真实Recurrent-only驱逐及全部Postcheck。Selected=B1/B3，12→2驻留，16→26空闲槽；FA侧及非目标状态合同通过。当前B1映射Node 20 / Slot [6]。

在真实 `process_input_requests` 分派目标之前再次读取缓存，完整节点/Slot、语义状态和Selected内容与协调结束基线相同，实际输入逐Token相同。原生Match前再次保存状态。没有把已结束的同步控制窗口解释为跨RPC租约。

## 四、实际恢复、COW与执行

|量|Strict实际值|Cold实际值|
|---|---:|---:|
|完整原始输入Token数|4394|4394|
|Attention前沿H|4224|0|
|原生Executable Prefix E|2432|0|
|H−E|1792|0|
|匹配节点|20|9|
|COW源槽|[6]|[]|
|工作槽|[27]|[]|

COW完成记录数：1。完成认证要求原生 `pool.copy_from` 返回，复制前源内容、复制后目标内容与复制后源内容SHA逐项相同；CPU拷回读取同步观察实际Conv/Temporal张量内容，不能只用cow_requested作为证明。工作槽与Selected槽必须互斥。

实际原生执行记录如下；返回只表明本次run_batch返回，最终输出另行精确核验。

|模式|Extend起点|Extend终点|原函数返回|
|---|---:|---:|---|
|1|2432|4394|True|
|2|不适用：非Prefill|不适用：非Prefill|True|
|2|不适用：非Prefill|不适用：非Prefill|True|
|2|不适用：非Prefill|不适用：非Prefill|True|
|2|不适用：非Prefill|不适用：非Prefill|True|
|2|不适用：非Prefill|不适用：非Prefill|True|
|2|不适用：非Prefill|不适用：非Prefill|True|
|2|不适用：非Prefill|不适用：非Prefill|True|

恢复输入区间与H−E分别解释：原始4394 Token，E=2432，首次Prefill为[2432,4394)，共1962 Token；其中1792对应H−E，另外170在H之后。Decode不以残留extend_range解释当前计算。没有把模型计算的E/G直接冒充实际执行记录。

## 五、输出正确性

冻结协议不变：temperature=0、max_new_tokens=8、ignore_eos=True；逐Token精确比较，停止行为和输出长度相同，禁止回撤。结果：PASS。

Strict输出：[550, 26679, 220, 17, 13, 94305, 25, 18661]。

Cold输出：[550, 26679, 220, 17, 13, 94305, 25, 18661]。

停止原因：{'type': 'length', 'length': 8}。

原协议没有Logits对照，本轮未增加或放宽任何数值容差。Cold只用于正确性，不据其Prefill时间宣称性能收益。

## 六、历史状态保护与正常副作用

执行后Selected组件内容与切片身份检查：PASS。B1/B3逐层Conv/Temporal内容使用SHA比较，源状态不得被COW或工作槽覆盖。

请求后新增节点：[]；移除节点：[]。请求前后完整语义视图相等：False。新增驻留的原有节点：[{'node_id': 28, 'position': 4224, 'slots': [28]}]。此处实际出现的是原有Attention节点重新获得请求执行新产生的Recurrent缓存，而不是Admission前意外恢复被驱逐的旧状态。原生 `mamba_component.py:517–549, 608–620` 的extra_buffer路径按本次请求tracking位置插入状态并回收工作槽；当前请求前后证据与该路径一致。没有逐条插入事件，因此不把源码解释提升为新增节点每条写入的独立跟踪认证。完整视图不要求相等，因为正常请求可插入新前缀节点、缓存状态并改变访问元数据；以Selected历史状态不变、槽隔离及实际新增状态清单区分正常副作用和损坏，详见 `state_integrity.json` 与 `strict/post_request.json`。

## 七、清理、边界及结论

清理：{'strict': 'PASS', 'cold': 'PASS'}。各臂独立Engine退出，本轮容器移除，实验GPU无计算进程且显存回到门限内；具体MiB见 `gpu_cleanup.json`。返回码不单独作为成功依据：要求每臂summary、匹配/COW/执行记录、输出比较及清理同时成立。

失败清单：[]。

此结果最多支持当前受控g001、当前计划、当前B1→A_PENDING边和8-Token确定性协议。不能推广到全部准入边、长期Pin/Lease、全部Pending、真实在线Agent并发、物理HBM容量压力或端到端性能净收益。

本轮到此停止，不运行其余三个Pending，不进入策略A/B或R3。
