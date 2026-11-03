# T2：机制与共同可行域

依据及版本见 [原始来源审计](paper_source_audit.md)。下表中的“未提供”仅指已检查论文/源码，不作不存在性证明。

## 1. 外部方法

|维度|Marconi|KVFlow|TOPAS|FlowState 当前合同|
|---|---|---|---|---|
|主要目标|Hybrid 缓存复用、计算/内存效率、TTFT|重复 Agent 调用的前缀复用及工作流延迟|任务 JCT 的联合状态调度|固定 Pending 的总模型化恢复成本 J|
|对象|radix 节点的 KV 与 SSM|固定/动态 KV 节点及 CPU 副本|Agent 前缀与已准入请求状态|已物化完整 Recurrent Checkpoint；FA 保持|
|价值|recency + α×FLOPs/字节|ASG→STE→固定前缀祖先优先级|条件任务进展与转移成本|当前保留集合决定每请求最大可执行前沿|
|Workflow 信息|不需要显式 DAG；历史请求树|显式/可推知的执行关系与进度|DAG、阶段进度和任务年龄|当前已知 Pending 与严格兼容边；不等于完整 DAG|
|Hybrid 特性|明确支持，已讨论 SSM 精确匹配|论文未认证 Recurrent 控制|论文未认证 Recurrent 控制|Qwen3.5 / Kimi 配置下有受控正确性证据|
|共享|共享分叉状态被准入/保护|共享前缀用关联 STE 最小值|同 Agent 请求共享前缀成本|一个候选可提升多个 Pending 的 E|
|集合决策|逐项效用驱逐，随树更新；非当前 J 的联合求解|树形优先级驱逐；非当前 J 的联合求解|明确联合搜索前缀集合与请求分配|Exact 或动态边际 Greedy|
|额外机制|准入、历史调参、树合并|预取、CPU 备份、状态感知调度|Admission、Preemption、Swap、Aging|本轮研究合同为无调度重排的保留/驱逐|
|预算|混合状态实际字节|KV 字节/设备容量|前缀+请求 KV 容量及调度限制|等大小 Checkpoint 个数 K；非 HBM 压力|

原始定位：Marconi §3、§4.1–4.3 和作者 `evict_v2`；KVFlow 正式版 §3.1–3.4、算法 1/2；TOPAS III、IV-B–E。来源链接和代码行号集中在 [paper_source_audit.md](paper_source_audit.md)，避免把三个系统对不同目标的优化混为一个排行榜。

## 2. 内部对照：不是外部 SOTA

|方法|实际决策规则与输入|集合更新|实现依据|
|---|---|---|---|
|StandaloneTopK|按每项对全部 Pending 的独立收益总和排序；只取正收益，ID 同分|不扣重复覆盖|[E3 audit.py:58–64](../../greedy_swap_structure_e3_step1/audit.py)|
|WFC|先最大化本步能推进前沿的 Pending 数 q，再最大化 Token 推进总量 r，再 ID|更新各 E；q=0 停止|[wfc.py:4–24](../../p1_strong_baseline_validation/wfc.py)|
|PCB|每 Pending 推荐收益最大项；按单请求收益排序去重|静态推荐；不是全候选独立总收益 Top-K|[pcb.py:5–24](../../p1_strong_baseline_validation/pcb.py)|
|LDR|每 Pending 推荐最深合法项；按位置排序去重|静态；不是简单全局按 L 取 K 项|[ldr.py:4–20](../../p1_strong_baseline_validation/ldr.py)|
|Marginal Greedy|选择使共同 J 降低最多的项；零边际停止|动态边际|[optimizer.py:_select_candidates](../../../flowstate/optimizer.py)|
|1-Swap|Greedy 起点，按移出/移入 ID 固定顺序接受第一个严格改善交换，重复到局部停机|更新 J；不接受等值交换|[E3 audit.py:42–56](../../greedy_swap_structure_e3_step1/audit.py)|
|Exact|枚举全部合法大小≤K的集合；同分偏少项再 ID|全子集最优；受规模限制|[optimizer.py:145](../../../flowstate/optimizer.py)、独立 [E3 Oracle](../../greedy_swap_structure_e3_step1/audit.py)|

现生产 `plan_strict()` 已为 Hybrid：n≤12、m≤4、K≤4、枚举数≤794 全部成立时 Exact，否则 Greedy；UNKNOWN/非法 Epoch 不得通过 Fallback。最新 [GPU 双分支门禁](../../paper_hybrid_solver_gpu_gate.md)已记录真实调用及正确性，而不是仅 CPU 集成。其 Epoch 仍复用冻结准入证据，不能据此称为完全在线构建；C 正式 60 臂历史性能矩阵仍加载冻结选集，不能追溯升级为在线 Hybrid 性能。

## 3. 同一个候选域下哪些机制会消失

- 固定全候选域、冻结 FA、统一 K 后，可以比较优先级给出的 Recurrent 保留集合，但 **Marconi 准入和混合字节管理**不在被测问题内。
- KVFlow 原 KV 树存在祖先依赖；只移除 Recurrent 且 FA 路径保留时可删 Recurrent 祖先而留后代。这是资源对象适配，不是原 KV 树驱逐的逐操作复现。
- TOPAS 的请求驻留/执行容量与任务进展不能由当前 E/G/J 替代；移除调度部分会改变问题。
- 因此“统一 J 更低”只证明给定当前需求和资源合同的状态选择质量，不证明原系统工作流延迟、Goodput 或全局缓存效率更差。
