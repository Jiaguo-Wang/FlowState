# 失败证据保存测试

证据顺序：观察 → observed_epoch文件 → Strict Epoch文件 → 已验证Plan文件 → 来源预期文件 → Apply/协调。来源选择失败发生在Plan落盘之后，且在任何GPU状态操作之前。

集成CPU用例26通过实际修复版control.apply到来源选择失败路径，验证：

- 返回对象和失败阶段文件包含完整raw_epoch、Strict Epoch、Plan；
- phase/failed_stage为source_selection，原异常保留；
- Scheduler fail-stop标记已设置；
- GPU观察接口和StateController.reconcile_strict调用次数均为0；
- 将该完整失败响应送入新客户端线协议测试时，response.raw和response.json在异常抛出前已保存，Epoch/Plan保持一致。

用例22测试阶段日志。用例23直接使用旧failed_control_response.json测试原始失败响应字节保存。用例24、28分别验证损坏JSON和空响应仍保存原字节与异常；用例25验证拒绝覆盖已有目录。

这些都是CPU静态与故障注入测试，不能称为新Runtime成功记录。临时夹具由测试创建并清理，执行结果保存在cpu_regression_results.json及cpu_tests.log；可用原命令复现。

能力边界：进程被强制终止或磁盘失效时不承诺尚未采集的数据可恢复。只保存当时已经取得的证据，不补造缺失Epoch或Plan；文件保存失败不能继续驱逐。新增fsync属于实验审计成本，未来需要计入完整控制时间，不假称免费或轻量。
