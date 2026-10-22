# 测量有效性

**没有有效Greedy/Exact配对，没有目标C的GPU计算、TTFT或输出对照测量。** paired_gpu_timings.csv保留失败臂并将未执行指标标为UNKNOWN；不能把空缺当作0，也不能引用历史B5-D3或模型J替代本轮测量。

control_cost_breakdown.csv只包含失败前真实完成的重型观察、Epoch认证、矩阵、Greedy求解、Plan验证和Oracle的Host区间。观察包含GPU同步和哈希；没有设备事件执行数据。当前Oracle返回J=48.228728759089606 modeled ms、29个子集，但这不是Greedy实际运行耗时，更不是目标C GPU性能。

实现中的COW后EagerRunner CUDA事件、run_batch外围事件、流式TTFT和完整控制到请求完成Wall均未进入正式目标执行。CPU替身测试通过不证明GPU计时有效。缺少首对状态可比性、恢复来源、8Token对照，故不继续重复或报告性能差值。

完整当前Epoch/Greedy选集未随失败响应持久化：ControlClient._call对ok=false提前抛异常，runner.py:37的写盘未执行。失败RPC对象从原failure.json的traceback用ast.literal_eval无执行地提取到failed_control_response.json，原日志保持不变。不能将历史预期选集冒充当前已保存计划。
