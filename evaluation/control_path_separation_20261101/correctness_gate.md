# 正确性门禁

## CPU/静态：PASS

冻结镜像、无GPU、无网络、源码只读挂载运行：65项全部通过（cpu_tests.log）。含11项新测试及原Strict Epoch/Control回归，覆盖完整真实74MiB报告的无损往返、边界/截断/摘要/重复键拒绝、跨Epoch/Slot变化无内容缓存、阶段日志即时快照、fsync失败传播、持久化先于解码、控制/Admission/Native钩子AST等价，以及冻结C/K3真实生产Greedy路由与完整计划一致。

静态等价不等于新增Generation保证；未删除或复用任何物理观察/SHA，原Scope及失败拒绝逻辑保留。未引入并发、异步取证或跨时刻状态缓存。完美恢复的ABA仍不是由现有端点摘要检测；本轮安全范围限定原连续Scheduler/独占Engine合同，详见 safety_equivalence.md。

GPU门禁须独立完成；CPU通过不视为Native恢复已认证。

## GPU1 Smoke：PASS

实际单卡绑定证据 arms/smoke_serving_fs/device_binding.json：宿主GPU1 UUID 051c9abc…，容器逻辑cuda:0，TP=1。GPU0已存在的SGLang进程未干预；选择设备发生于首次模型启动前，原GPU0预检失败和新设备锁均保留。

生产plan_strict实际执行Marginal Greedy，选中 C_B_P05120、C_C_P06144、C_D_P07168，模型化J=154.34895561599998ms；13项真实驱逐成功。E=[0,5120,6144,7168]，原Native/Prefill/COW/Cold/FA与组件保护及Oracle门禁全部通过。6阶段日志在原时刻额外检查解析后等价，完整响应与原线协议JSON逐字段等价；完整原始封装在解码前持久化。

传输体27,902,168B，压缩后12,608,071B，Smoke中编码约520.581ms、解码614.185ms，另外有954.471ms额外等价认证；这些真实成本保留且Smoke不作为性能样本。优化并非免费，也未减少GPU内容读取。后续正式样本只去掉Smoke额外双重对照，保留全部原安全检查。

## 五轮新旧路径：全部PASS

十个性能臂均通过原Native/Cold/FA/组件保护与Oracle检查。五轮逐字节物理证据复核 five_pair_physical_equivalence.json：全部候选Token/位置/Conv/Temporal摘要、严格状态及条件矩阵、Selected最终内容、FA映射与保护内容相同；每臂13项驱逐、Free Slots从12到25。每个serving臂请求前持久化的完整封装解码后，与T4后生成的完整reconciliation.json逐字段相同。

该复核在五轮全部结束、没有性能测量进行时执行，时间边界已入记录。首对额外复核也保留独立记录；其执行区间未用作任何策略输入。

五轮配对门禁 pair_gate_5.json：Control差中位数−214.011ms，Full差中位数−3046.818ms；第五轮Control+47.932ms不剔除。此门禁仅批准继续同模式三策略批次，不提前认定相对Baseline净收益。

## 最终完整批次：PASS

1个Smoke、10个新旧路径性能臂、15个三策略性能臂，共26个独立GPU1生命周期完成。所有原Native/Cold/COW/FA/组件/测量门禁通过。verified_evidence_inventory.json独立复核26臂初始候选Token/位置/内容、64条严格关系及FA映射签名相同；每臂13次真实驱逐，338次合计；驻留16→3，空闲12→25。每个完整持久化响应等于对应展示报告，6阶段Journal齐全，测量请求窗口无新Triton内核。

共104个目标请求均通过原执行及8 Token Cold输出合同；E>0时认证Native COW，E=0为合法完整重算。其他未实际采用的准入边仍仅EPOCH_ADMISSIBLE；血缘PROVENANCE_UNVERIFIED不升级。

最终清理gpu_cleanup_final.json：GPU1 17MiB、0%、无计算进程且无本轮容器残留；GPU0其他SGLang任务仍在运行，未干预。根盘约35.7GiB可用。
