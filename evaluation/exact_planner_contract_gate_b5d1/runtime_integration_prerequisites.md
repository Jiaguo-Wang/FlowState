# 后续Runtime集成前提

本轮READY只适用于冻结Epoch纯规划，不授权立即执行当前GPU状态操作。

1. 可信Strict Epoch必须在当前连续安全窗口建立，并重新核验Token、配置、完整候选域、候选外状态及当前引用。历史SHA不能认证当前GPU。
2. 必须重新生成PhysicalView，验证Node/Slot/Conv/Temporal切片及Keep/Evict无别名；原生产bind/preflight不能跳过。本轮无任何PhysicalBinding输出。
3. 生产接线需要显式求解器来源、Exact规模准入、错误与Fallback隔离；冻结快照的stale=False仅为证据使用范围声明，不是跨RPC活性租约。
4. 决策时间预算须覆盖输入认证、共同预处理、求解及计划校验；当前计时排除了初始读取和Epoch构造，更未包含RPC、Scheduler隔离、GPU摘要和Reconciliation。不能以求解器中位值估计完整控制成本。
5. 生产deadline/取消能力未实现，Greedy Fallback也没有强制抢占。任何超时方案先做CPU故障验证，不得把未穷尽结果标Exact。
6. 更换求解器后仍须独立真实协调与恢复正确性验证，及公平的控制含量性能测量。EPOCH_ADMISSIBLE不是RESUME_VALIDATED；本轮没有TTFT、GPU净收益或长期保留保护证据。

所有生产源码SHA在本轮前后相同。没有自动开始以上工作。
