# 测量与有效性边界

本轮仅验证真实历史回放物化、同一静止边界的严格准入及CPU模型化分配，不测量GPU策略性能。每组独立Engine，固定八条历史请求，每条temperature=0、max_new_tokens=1、ignore_eos=True；四个第三次请求只作为已准备的受控输入，不提交Prefill/Decode。历史输出来自原始轨迹，目标模型新生成的Token仅作为回放请求的正常缓存副作用，不回填Pending。

预算在任何新GPU观察之前冻结为K=1..min(4,n)，全部候选保留。每个候选必须独占一个等大小Recurrent槽，Conv/Temporal切片字节成本一致；别名、UNKNOWN或完整性冲突拒绝进入完整比较。在线Exact适配域为n≤12、m≤4、K≤4、枚举数≤794；域外只允许预注册n≤24、m≤4、单预算子集数≤20000的离线Exact。超过域标记NOT_TRACTABLE，不伪装最优。枚举包含空集及所有大小≤K的集合；目标容差1e-9 modeled ms，同分先较少项，再排序ID字典序。

PCB、LDR、WFC直接调用与B4-C摘要一致的历史模块；StandaloneTopK按静态独立收益和ID排序；Greedy使用D1适配器调用生产选择核心，严格矩阵直接评分，不重新调用Workflow/Lineage谓词。Exact由独立组合枚举及共同E/G评价核验，并与D1准入域内结果对照。所有输出计划经过既有validate_plan，只做CPU校验，不绑定或执行物理计划。

物化观察复用冻结C3入口：同一次Scheduler同步静止窗口内只读遍历FA树、Recurrent组件与切片，前后完整状态及内容摘要一致。历史实际提交输入与Native接收的摘要核对。对四个Pending，只读FA遍历与Token前缀检查不等同于Native请求实际命中；EPOCH_ADMISSIBLE不升级为RESUME_VALIDATED。首次观察到某节点不是其张量血缘证明，始终保留PROVENANCE_UNVERIFIED。

四个受控组合是四个来源工作负载；每组四预算是重复条件，不是16个独立工作负载。结果不能证明自然Agent并发、GPU容量压力、TTFT、吞吐、Goodput或控制开销后的净性能收益。K_min仅表示达到全候选可实现前沿的最小逻辑保留数，不保证Gap为零。
