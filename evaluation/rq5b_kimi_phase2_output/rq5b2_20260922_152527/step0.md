Step 0：NEED_MINIMAL_ADAPTER

直接复用现有兼容性、H/E/G、GlobalOptimizer、RecoveryCostModel、SGLangAdapter.evict_mamba_only 以及 RQ6-E 的统一证明函数。
新增 TP=2 批量分发、首次物理句柄发现、替换元数据只读校验以及 absent 保持 absent 的协调分支。每个 epoch 一次聚合观测与一次聚合协调，每次内部各分发两个 rank RPC；统一聚合后验验证不额外发起观测。
使用 2K/4K 检查点，并核对完整祖先路径，防止隐藏循环状态影响真实 E。逻辑 K=2 只约束候选集合，非候选分叉末端状态必须保持不变。恢复模型仅作冻结接口功能验证，不代表 Kimi 校准或性能结论。
SGLang 镜像版本与提交匹配；两卡均空闲。没有核心修改需求。
