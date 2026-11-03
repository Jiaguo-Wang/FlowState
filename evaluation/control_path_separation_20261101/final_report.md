# 最终报告

**判定：SUCCESS_FULL_CYCLE。**

## 实际实施与安全边界

1. 必要在线安全操作全部保留：Scope/静止边界、真实FA前沿、严格Epoch与兼容关系、Node/Slot/组件/引用/别名/容量、Bind/Preflight、13项驱逐和逐项Postcheck。
2. 内容SHA虽昂贵，但当前仍有安全与保护门禁消费者；本轮未证明可以删除或延迟，所有全量FA/Recurrent SHA保持原时刻真实执行。
3. 安全优化仅为Journal紧凑编码、完整apply响应的可校验无损封装，以及将已有持久化副本的重复缩进展示移至T4后。原始完整响应仍在请求前fsync，坏响应或请求前必需的持久化错误继续拒绝请求；T4后重复展示写入失败只能拒绝整臂认证，不能撤销已执行请求。
4. 没有减少GPU同步、RPC次数或全量状态复制；减少的是CPU格式化、线上字节和重复展示的关键路径成本，不涉及上一轮失败的批量读取。
5. 保持Strict Epoch、物理绑定与实际状态检查；不使用Slot Generation替代检查。现有跨窗口Generation缺口没有被修复，不能声称排除任意并发ABA，只在原已验证的连续控制范围内等价。

## 定量结果（五轮同轮配对中位差，ms）

|比较|Control|Full T0–T4|Audit-inclusive T0–T5|ΣTTFT|ΣCUDA Prefill|
|---|---:|---:|---:|---:|---:|
|serving-original|-214.011|-3046.818|-1689.140|-2.899|-4.543|
|serving_FS-MC|+406.329|+507.381|+775.819|-483.345|-475.543|
|serving_FS-KV|+250.360|-36.025|+134.442|-432.111|-425.266|

负值表示左侧更短。全部逐轮正负差、均值、中位数和范围见performance_analysis.md及CSV，未筛掉慢轮次。

6. 新旧FlowState控制时间：配对中位差-214.011ms，胜/平/负=4/0/1。第五轮退化保留。
7. Native恢复语义及8 Token Cold逐Token结果保持；相同选集下CUDA/TTFT仍有实测波动，不能因E/G一致宣称时延完全不变。每个Pending实测值见pending_results.csv。
8. Full的变化与Audit-inclusive的变化分别报告；T4后CPU/Wall未隐去，T5不含summary写盘/Engine退出，不等于完整服务生命周期。
9. 相对Baseline净收益：按预先保守条件（Full中位差<0、至少4/5获胜且Audit-inclusive中位差<0），满足的对象为无。不把仅Prefill/TTFT改善升级为完整周期净收益。
10. 仍需同步的大量内容审计是本轮未分离的主要剩余成本；必要Runtime成本下限仍UNKNOWN。尚无跨模型、自然并发或生产低审计路径的外推证据。

## 完成范围与证据

65项CPU测试通过；26个GPU1独立生命周期（1 Smoke、10新旧配对臂、15三策略臂），338次真实Recurrent-only驱逐，104个目标请求的原Native/输出合同通过。E=0请求为合法完整重算，只有E>0的请求认证COW；不将所有请求都称作Recurrent命中。Cold使用冻结独立参考，本轮未另外新跑Cold生命周期。

26臂初始状态物理签名相同；每臂Resident 16→3、Free Slots 12→25，全部响应完整、FA/组件保护及Pool账本通过。三策略同协议、同GPU1、同输入/候选/K/模型，未拼接历史基线。全部负向结果保留；GPU0最初资源预检失败且未启动模型，换卡发生于首次模型启动前。

精确实现身份以execution_lock_gpu1.json和最终manifest为准，基线Git提交8cba094d2c5466759b70e4a5ef8f4824b61aade5；当前代码仅新增隔离实验包。旧十五臂、生产源码及上一轮失败结果未修改。完整命令、环境、日志、物理记录和清理见execution文件、arms/*与最终清理记录。

Marconi-Evict Adaptation / KVFlow-inspired是适配策略，不是官方完整系统；PROVENANCE_UNVERIFIED保持不变。

本轮完成后停止，不修改论文，不继续扩大优化或实验。

## 最终资源与实现状态

GPU1已清理至17MiB、0%、无计算进程；无本轮实验容器残留。GPU0现有任务未处理，不能将此表述为整机无计算任务。根盘约35.7GiB可用，原始数据与缓存全部保留。

该改造只在独立实验包完成CPU/GPU验证，未修改或部署生产主路径。主报告与CSV包含全部有效的负向观测；Source/环境及命令见environment.json、commands.md和最终manifest。

计时和错误处理的精确定义补充见measurement_boundaries.md；跨臂离线复核、统计不在本臂T5中，且全部在GPU测量结束后执行。
