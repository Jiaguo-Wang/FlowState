# 消费者最小化基线接口原型

本目录仅为隔离实验原型；生产入口和旧实验不变。当前不提供自动GPU启动器。

## CPU复核

在仓库根目录执行：

```bash
python3 -B -m unittest evaluation.serving_control_interface_gate_20261101.test_interface tests.test_strict_control tests.test_strict_epoch -v
python3 -B -m evaluation.serving_control_interface_gate_20261101.verify_frozen
```

第二条命令只读历史真实响应，核对已有派生产物；不同结果会拒绝，不覆盖旧证据。CPU测试模拟的物理故障不能当作本轮GPU故障注入或真实资源验证。

## 接口模式

- `FS_ADMISSION_BASELINE_MODE=validation`：默认；完整基线原样返回。
- `FS_ADMISSION_BASELINE_MODE=serving`：完整生产者照常执行一次，只返回 schema、当前响应身份摘要及 selected_components。
- `FS_ADMISSION_BASELINE_MODE=shadow_validation`：同一次完整读取生成完整基线及精简旁路响应，逐项核对。不是性能模式。

精简或双接口模式必须同时设置 `FS_SEPARATION_MODE=serving`，确保原异常隔离包装生效。模式名称分别作用于基线和既有 `FS_SEPARATION_MODE`（JSON/Journal优化模式）。当前 CPU 门禁没有执行任何 Engine 或 device_entry。后续获授权时必须复用原 GPU UUID/空闲门禁、冻结配置和独立输出目录，入口应指向本目录的 device_entry，不能覆盖源实验。

## 不变边界

本轮没有取消 baseline 的内容读取、SHA、同步、LRU结构验证或FA索引检查。`cache_state` 的完整生产者、请求后完整审计及两次审计计数不变；`control.py`、`codec.py`、`observation.py`、`protocol.py` 与源文件字节相同。精简摘要只是元数据响应身份，不能作为物理内容证明、Slot Generation 或跨RPC租约。
