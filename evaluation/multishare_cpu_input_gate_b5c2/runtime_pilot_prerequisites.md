# 后续Runtime前提（仅记录，不执行）

本轮只认证目标CPU输入和内容结构。四个Pending分别取固定Session第三次请求，前两次历史输入都已保存。未来如获授权，可按A1/B1/C1/D1/A2/B2/C2/D2准备历史回放；四份Pending在Barrier前准备并记录controlled_input_ready，不称agent_native_ready。

预期内容边界：System完整前缀2428；四方共同LCP2438；A/B完整System＋User边界3723，最大LCP3765；C/D完整System＋User边界3502，最大LCP3566。它们只是观测到的Token位置，**不构成Checkpoint创建列表，也不要求Runtime保留这些精确位置**。原生track interval为256时，位置对齐、实际history输入路径、插入/分裂及保留行为都可能使实际节点不同。

未来必须另外验证：

1. 模型权重、Tokenizer、模板、镜像及配置身份；本轮未加载模型权重或验证GPU环境。配置锁中的B4 Engine参数仅作来源参照。
2. 历史请求实际提交/接收完整Token与本轮数组一致；不能直接把CPU重建写成原始Runtime Token。
3. 在真实静止边界观察原生生成的所有Checkpoint，不人工插入切点；完整Conv/Temporal、Node/Slot、所有权、引用、别名和候选外可恢复状态。
4. 同边界真实FA frontier及Planning Target。所有内容位置均不超过对应完整输入长度，但本轮T为UNKNOWN，不能假设T=N或全FA命中；须逐边检查L≤T。
5. 多个共享位置是否同时真实驻留且关联不同Pending支持集，是否还有更深专属状态。内容支持不等于EPOCH_ADMISSIBLE，更不等于RESUME_VALIDATED。
6. 真实持久保留配额与工作/COW空间、保护和native eviction；不能拿逻辑K或28槽配置宣称真实容量竞争。本轮没有K、K_min或收益矩阵。
7. 不能用后续Pending执行生成的节点回填此前Barrier。若实际仍单共享或候选不完整，原样报告，不更换Session、预算或输入。

下一步是否启动Runtime由人工另行决定；本轮到此停止。
