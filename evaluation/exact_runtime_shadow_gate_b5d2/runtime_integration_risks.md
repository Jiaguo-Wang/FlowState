# Runtime集成边界

1. 两次只读观察窗口是独立Scheduler同步调用，CPU规划在窗口外，不享有持续Admission隔离或租约。本轮独占Engine没有其他请求，实际前后状态一致；未来应用计划必须重新观察、Bind和Preflight。
2. D1只接受frozen_snapshot证据范围。本轮新Runtime观察在当前边界即时序列化成不可变证据，再交给原D1，不伪造current_runtime认证，不修改Strict语义。候选来源采用本生命周期首次Census与真实历史Session，不据此推断独占物理Owner。
3. 本次只能认证Epoch内准入及规划，没有Pending实际恢复、逐边RESUME_VALIDATED或策略性能证据。历史回放正常COW和缓存变化不属于Shadow无副作用期间。
4. Exact遵守n≤12、m≤4、K≤4、子集≤794；本轮只K2。缺失或UNKNOWN输入Fail-closed，超限明确拒绝，不调整候选或扩大范围。无硬deadline承诺。
5. 当前源文件与模型摘要检查结果：True；GPU清理：PASS。实际CUDA IPC退出警告：['[W1017 18:56:18.952881435 CudaIPCTypes.cpp:16] Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]']。保留原始日志，不能以警告或进程返回码单独替代资源清理证据。

本轮不修改生产Optimizer、Scheduler或StateController；复用C3已验证的专用观察进程入口，没有新增缓存动作。完成后停止。
