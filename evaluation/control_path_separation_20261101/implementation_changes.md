# 最小隔离修改

原包：evaluation/p0_control_cycle_gpu；本轮仅在evaluation/control_path_separation_20261101新增文件，原版可运行，不修改生产代码或冻结证据。基础提交与原始哈希见 original_source_manifest.json。

- control.py：DeferredJournal.record在serving模式生成紧凑JSON；原始模式保持原逻辑。阶段快照时刻、文件数、flush/fsync位置及异常处理不变。
- codec.py：完整apply响应无损JSON/zlib/Base64封装及有界解码，含长度/SHA/尾部检查，记录编码/压缩/解码CPU与Wall。仅CPU字节数据，不含设备操作。
- bridge.py：在既有dispatch外围按模式封装成功apply响应；内部控制/Native过程不变，封装异常进入已有隔离标记。Smoke额外校验与原线协议JSON等价。
- runner.py：先持久化完整原始响应再解码；仅重复缩进报告移至T4后，记录CPU/Wall及T5；原所有Native/Cold/FA/COW/测量断言保留。
- observation.py、protocol.py：原版逐字复制。device_entry.py仅改本轮包入口，单卡门禁保留。

不修改Batch Size、复制粒度、重排或Phi；不减少任何同步SHA/物理观察、驱逐检查，不新增Generation或状态缓存。

五轮原版/serving配对固定奇数原版先、偶数serving先；先serving正确性Smoke，再第一对。第一对必须控制更低且Full不高于原版，才能续接剩余四对；否则停止，保留负向结果，不调参。五对全部正确且控制配对差中位数<0、Full配对差中位数<=0，才执行同serving模式三策略五轮，原P0 ROUND_ORDER不变。

Smoke不进入性能。无损封装的完整性检查与全部原安全检查在所有serving臂一致；三策略如执行均走相同编码与取证时机，不与历史基线拼接。所有GPU臂独立生命周期、原形状预热、K=3与C输入不变。
