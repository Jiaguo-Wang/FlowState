# 首臂失败与状态控制边界

**未执行Reconciliation或Pending恢复。** 八条历史请求实际完成，提交与Runtime接收Token身份核对通过，保存于p01_Greedy/history_01.json至history_08.json。最终历史Census有7个驻留候选，位置3712/2432/3456/4160/4096/3904/3520，实际Slot为3/8/10/12/14/16/18。

控制函数完成观察、Strict Epoch认证、Greedy求解、Plan验证及当前Oracle；返回的失败对象见failed_control_response.json，其中operations=[]、observations=[]。失败位于control.py:97–99，早于control.py:101开始的Apply-time重验、bind/preflight与后续reconcile_strict。因此本轮实际策略驱逐0项、正式Pending 0项，不能声称协调、Native恢复或输出正确性通过。

现有CPU门禁验证Req/Batch/计时包装，没有覆盖“公共共享状态首次出现在目标同一来源Session的历史边界”这一目标选取情形。该缺口导致实验性来源门禁错误，不是已证实的生产Compatibility、Greedy或恢复计算错误。

在同一任务允许的范围内只做离线归因，不修改冻结执行代码、不重试Engine。清理见gpu_cleanup.json，唯一生命周期PASS。
