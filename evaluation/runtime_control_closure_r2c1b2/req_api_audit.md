# 真实请求接口与观测时机审计

本轮使用与既有实验一致的本地 SGLang v0.5.17 镜像；不能仅凭版本字符串认定源码一致，实际文件摘要在 `source_identity.json`、`req_api_source.json` 和 `implementation_lock.json` 中核对。CPU 容器未挂载 GPU、未启动 Engine、禁用网络。

## 接口与坐标

原生文件：`/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py`。

- `Req.__init__`：771 行起定义真实类型，保存 `origin_input_ids`、`output_ids`、`full_untruncated_fill_ids`、`extend_range`。当前对象没有 `fill_ids` 和 `extend_input_len`，不能沿用这两个旧字段。
- `Req.set_extend_range`：1222 行，建立当前计算区间。
- `Req.get_fill_ids`：1225–1226 行，返回 `full_untruncated_fill_ids[:extend_range.end]`。这是截至本次 Extend 终点的序列，不是单独的未缓存后缀。
- `ScheduleBatch.prepare_for_extend`：2281 行起；2291 行通过 `r.get_fill_ids()[len(r.prefix_indices):]` 构建真实 Prefill 输入；2293–2296 行分别读取区间终点、前缀长度、区间长度；2321 行起设置 Batch 的 `prefix_lens` 与 `extend_lens`；2350 行附近检查 `seq_len - pre_len == req.extend_range.length`。
- `Scheduler.run_batch`：原生 `managers/scheduler.py`，准确起始行及全文见 `req_api_source.json`。无 overlap、无推测解码的冻结路径先 `resolve_forward_inputs`，再执行 `model_worker.forward_batch_generation`。包装函数位于这一入口外侧，记录前后事件，不修改原生路径。

本目标完整输入长度为 4394；严格计划的 T/Attention Frontier 为 4224；计划 E=2432、G=1792。实际 Extend 的预期区间是 `[2432,4394)`，长度1962，包含缓存 Attention 前沿后的170个输入 Token。因此不得把完整输入长度、H、E、H−E和Prefill输入长度混用。实际结果以运行记录为准。

Decode 不是 Prefill。新观测只在 `forward_mode.is_extend()` 为真时解释 `extend_range` 和 Batch 的 Extend 数组，不把 Decode 时的残留字段解释成当前恢复区间。

## 最小修改

新增独立 `observation.py`；本轮 `bridge.py` 使用 `wrap_run_batch`，旧 R2-C1b-1 文件保持原样。

- `observe_batch` 只读取原生 Req 和 Batch，不写字段、不调用 Cache Match、不改变引用或分配。
- `wrap_run_batch` 在观测后恰好调用一次保存的原函数，直接返回其原返回值。原生异常继续传播，不统一捕获并吞掉。
- 只有原函数返回才写入 `native_run_returned`；这一标志不单独充当完整输出正确性认证。
- 可选 `extend_batch_idx` 缺失记作 UNKNOWN。
- 必需字段和只读接口在 GPU 启动前由真实类型 CPU 测试覆盖，并在提交目标前再次调用 `real_api_preflight`。必要区间或 Batch 不一致时显式拒绝，不伪造证据。若实际执行中状态违反前置合同，仍然失败即停。
- 完整来源、匹配源、COW 内容摘要和执行后 Selected 状态检查保留。没有增加额外 Prefix Lookup，没有强制指定恢复节点。

## CPU 检查范围及限度

使用冻结镜像中实际导入的 `Req`、`ScheduleBatch`、`ForwardMode`、`SamplingParams`，以冻结目标 Token 输入构造真实 CPU 对象。检查严格恢复、Cold、分块区间、Decode、只读状态、可选字段缺失、必需字段冲突、原函数恰好调用一次及异常原样传播。

测试验证生产观测所使用的同一个包装函数；受测原函数使用调用计数桩，目的是不执行 GPU forward。原生 `prepare_for_extend` 的内存分配和完整 Scheduler 不能在此宣称已执行：CPU 测试核对其冻结源码及真实对象接口；实际执行闭环由后续 GPU 记录验证。

首次 pytest 收集因相对导入失败，记录在 `cpu_tests_collection_failure.log`；已改为绝对导入。最终结果以 `cpu_tests.log` 和 `cpu_gate.json` 为准。源码定位额外解开装饰器，避免把 `run_batch` 误归到装饰器文件。只有最终内容锁检查通过后才允许 GPU 启动。
