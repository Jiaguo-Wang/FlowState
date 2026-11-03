# 安全等价证明与边界

## 等价关系

新模式保留原控制过程的全部状态读写及安全断言。对控制apply函数、桥接dispatch、Admission inputs和Native install_observation的AST逐项比较；除Journal编码和响应外围封装外，调用体保持一致。CPU检查 `test_original_safety_call_sequence` 可复核。

JSON缩进是展示属性。对同一阶段对象x，`loads(dumps(x,indent=2)) == loads(dumps(x,separators=(',',':')))`；编码仍在同一record时刻同步完成，既不持有可变对象等待后续读取，也不复用上次物理观察。flush/fsync仍在T3后、响应前；异常仍传播至原控制fail-stop。

响应封装满足 `unpack(pack(x)) == loads(原线协议dumps(x))`，包含全部字段，未传递可执行pickle。长度、完整zlib流、尾部、SHA和重复键均校验，损坏或缺失字段拒绝而不回退。客户端先执行原call_saved，完整原始封装response.raw和response.json均已fsync，然后解码及消费；任何持久化/解码失败都不会提交Pending。服务端封装失败设置原 `_r2c0_failed`，保持失败生命周期隔离。

## 分离的唯一执行位置

原客户端在请求前重复生成的缩进reconciliation.json延后到T4之后；同一完整响应在请求前已有可解码、带SHA的持久化副本，不删除或延迟第一份必要证据，也不延迟任何用于当前物理安全的检查。失败路径先保存原始响应，保留请求、异常；不会要求协调成功才能保留响应。

新增T5是全部原正确性与Oracle检查、重复报告格式化及附加取证记录完成时刻。两种模式均记录T5−T0和T5−T4，T0/T3/T4定义原样保留。格式化在全部请求执行完成后、同步进行，不与这些请求竞争CPU/GPU；记录process CPU及Wall，不能把T4后的工作当作不存在。

## ABA与Generation

没有增加PhysicalView/Slot或内容缓存，没有改变29次协调观察和13次驱逐的顺序，所有SHA仍在原时刻真实执行。不存在由取证快照复用而新引入的身份过期窗口。严格安全仍以TP=1、无Overlap、无异步传输/插入、单Scheduler同步控制和独占受控Engine为前提。

Recurrent allocation generation尚不存在于本PhysicalView，Req池req_generation不能代替它。完全复原的ABA不能由端点SHA或指针相等证明不存在；本轮不声称补齐这项通用并发能力。控制窗口内允许的变更仍只有Adapter指定的Recurrent-only释放；无需将旧视图用于新时刻。返回后不视为长期租约，仍走原Native Match/COW/源状态及Cold门禁。

因此此模式仅为原已验证运行域内的安全取证快速路径，不是支持任意外部并发写入的通用Serving事务。若配置打开异步功能或Scope未知，原门禁继续拒绝。

## 保持的消费者

Strict Epoch和准入矩阵的原始字节摘要不变；只在最终响应传输层重新编码，解码恢复相同计划、绑定、候选与T/E/G/J。控制SHA结果、FA保护基线和Recurrent组件摘要的消费者不变。原始源码/实验哈希受manifest保护，不导入失败批量方案。
