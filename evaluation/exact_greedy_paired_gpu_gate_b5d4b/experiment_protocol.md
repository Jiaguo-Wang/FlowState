# B5-D4b 冻结执行协议

原协议逐字保存在original_experiment_protocol.md；来源、Token、模型/Tokenizer/模板、缓存参数、K=2、Pending C、生成协议及五对交替顺序均保持原样。

唯一Runtime执行入口：evaluation.exact_greedy_paired_gpu_fix_b5d4a.runner。不得使用旧D4 runner/control；新目录只有独立launcher、配对核验和分析代码。全部D4a修复及生产依赖SHA纳入execution_lock.json/source_identity.json。

最多11个独立Engine：第1对Greedy→Exact，然后独立Cold；全部门禁通过才继续第2对Exact→Greedy，第3对Greedy→Exact，第4对Exact→Greedy，第5对Greedy→Exact。每个Engine八条固定历史后仅派发C，Cold无历史。每臂独立空Triton目录，不新增热身或补位。首对或后续任何关键失败均停止。失败和未完成运行保留，不自动重试。

## 来源与样本资格

沿用D4a拆分合同：Strict兼容与物理保护必须通过，Native实际来源必须匹配该生命周期Plan/Binding；仅有首次观察的来源保持PROVENANCE_UNVERIFIED。此标签不自动排除性能样本，也不自动证明跨Workflow恢复。Exact从D历史首次观察到的共享状态恢复只能报告该线索，不能称张量血缘认证。

本轮整体READY采用用户新门禁：至少一组完整可信配对；不要求Exact获胜或跨Workflow血缘认证。若后续出现关键错误仍立即停止并单列失败，不能隐藏为只报告成功样本。

## 计时

原D4a包装不改：COW后的EagerRunner.execute用同流CUDA事件；外围run_batch事件包含COW/审计；COW原调用另记设备事件与Host时间。事件前同步、后等待会影响执行；同流区间含Host提交与首次编译空隙，不是纯内核时间之和。

流式TTFT从Engine.generate前到首非空Token响应，Wall到流结束，8Token输出完整对照。控制RPC和从RPC发起到请求流结束的完整路径保留重型观察/哈希/Oracle/日志及同步成本。D4a新增逐阶段fsync是实验审计成本，保留在实测总量，未单独计时部分明确未知，不减掉宣称生产净收益。

完整路径不含启动和八条历史，也不含请求结束后的最终Census；这些有日志但不冒充已完整计时。主要报告各对全部差值、均值/中位/范围，最多5对不推断生产P95、显著性或四Pending吞吐。J与C单请求模型成本和实测GPU/Wall分别列出。

输出新目录；GPU0 H100空闲检查失败则不启动；只清理本轮专用容器。完成后停止。
