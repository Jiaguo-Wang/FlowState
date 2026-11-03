# T2：原始论文与作者源码审查

审查日期：2026-10-26。只读文献、源代码和既有结果；未执行作者代码、策略评估或 GPU 实验。论文机制、公开代码行为和 FlowState 适配三者分别记录。用户消息末尾问题被截断，不据此补写额外任务。

## 1. 版本与来源

|工作|论文版本与日期|作者公开仓库及本次核验版本|可复现范围|
|---|---|---|---|
|Marconi|[MLSys 2025 正式论文](https://proceedings.mlsys.org/paper_files/paper/2025/file/7c180af017258d239bac6248d1eb26ac-Paper-Conference.pdf)；[arXiv v3](https://arxiv.org/html/2411.19379v3)，2025-04-10；v1 为 2024-11-28|[ruipeterpan/marconi](https://github.com/ruipeterpan/marconi)，本地作者代码 HEAD `08016617b1524e6bf6ac29b680641cc945bda7f0`|公开 artifact 的主要执行入口是 trace 驱动模拟；不等于已有可接入 Qwen3.5/SGLang v0.5.17 的 GPU 服务实现|
|KVFlow|[NeurIPS 2025 正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7971d31a7d5eb0f1eed2f8f6f368195-Paper-Conference.pdf)，[正式收录页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b7971d31a7d5eb0f1eed2f8f6f368195-Abstract-Conference.html)；[arXiv v1](https://arxiv.org/html/2507.07400v1)，2025-07-10|[PanZaifeng/KVFlow](https://github.com/PanZaifeng/KVFlow)，本地 HEAD `7ef897eaedc16626a281ccfe0d453fae80251e8f`|该提交晚于论文，并混载 ScaleSim；不能把当前混合源码逐行归属到论文实验版本|
|TOPAS|[arXiv:2608.25523v1](https://arxiv.org/html/2608.25523v1)，2026-08-26；本次未核实正式会议版本|论文/摘要页未找到作者项目链接；按完整标题及作者检索 GitHub 未定位可核验仓库|代码地址、commit、实现模块均为 **UNKNOWN**，不是已证明作者未公开代码|

本地作者源码位于 [B5-A raw](../../sota_baseline_audit_b5a/raw/)。本次实际执行 `git rev-parse HEAD` 核对上述两个提交，并重读关键实现；没有将其称为截至今日最新远端 HEAD。初次版本记录见 [source_lock.json](../../sota_baseline_audit_b5a/source_lock.json)。本次关键文件摘要见 [source_identity.json](source_identity.json)。

TOPAS 通过网页工具核实正文第 III、IV 节；shell 直接下载因沙箱网络限制、随后 TLS EOF 失败，未保存完整本地副本。结论基于已读取官方正文，不依据二手摘要推导代码能力。

## 2. Marconi：必须保留的真实机制

论文定位：§3 说明 Recurrent State 不能任意回退，KV 与 SSM 必须支持相同前缀；§4.1 准入；§4.2 驱逐；§4.3 实现；附录 A 成本。它已经认识到 Hybrid 的精确状态匹配约束，不能在论文中写成“不区分物理命中与可执行恢复”。FlowState 可比较的差异是固定 FA 驻留条件下、针对当前 Pending 的 Recurrent 集合恢复目标，而不是首次发现 Hybrid 的匹配困难。

|机制|作者代码位置（相对 `raw/marconi_repo/`）|核实结论|
|---|---|---|
|准入|`radix_cache_hybrid.py:147–169,220–263,322–343`；`policy_exploration.py:53–156`|请求历史 radix tree、分叉处投机状态及请求结束状态；不是对任意冻结候选做一次 Top-K|
|驱逐域|`_collect_leaf_and_single_child_nodes:383–394`；`evict_v2:503–559`|叶子和单孩子节点；每轮重新收集；分叉祖先并非任意可删|
|价值|`evict_v2:518–545`|相对父节点计算 FLOPs；效率除以模型状态内存；与 recency 归一化后组合|
|时序|`evict_v2:535`；`match_prefix:256–261`|recency 为 `1/(current_ts-last_access_time)`；V2 命中更新被访问节点，不是所有祖先|
|释放|`_delete_leaf:409–415`；`_evict_intermediate_node:417–434`；`:553–559`|叶子释放 KV+SSM；单孩子节点释放 SSM、KV 合并到孩子。不是始终只释放 Recurrent|
|参数|`config_tuner.py:58–118`；`radix_cache_hybrid.py:465–482`|历史 bootstrap 调权重；并非统一固定 α=1|
|公开执行形态|`artifact_evaluation.md:50–58`；`policy_exploration.py`|预先 token 化请求轨迹、模拟缓存策略；不能把 artifact 的运行直接当作真实 GPU 服务复现|

细节冲突必须留档：作者 `evict_v2:526` 的 MLP 差分调用了 `get_attn_flops(parent)`；调参 `config_tuner.py:110` 用总 FLOPs 节省，而论文调参描述为命中率。当前任务不修正作者源码，也不选择更利于 FlowState 的版本。后续若复现须明确采用原代码还是论文语义，二者不能混称一致。[既有逐行审计](../../sota_baseline_audit_b5a/marconi_audit.md)可辅助定位，但其中“无树”“适配等价叶子驱逐”的概括不适用于当前 FlowState：当前 Runtime 有真实 radix tree，Recurrent-only 控制保留 FA，和 Marconi 叶子释放不同。

作者公式含 Mamba 状态与层计算；Qwen3.5 的 GatedDeltaNet 不能无核验套用 Mamba1 FLOP 式。统计访问频率不是其单次效用式的必需字段，但真实节点访问时刻、父子结构、层配置、字节成本和历史调参轨迹是忠实复现所需信息。

## 3. KVFlow：正式版本补充了重要合同

正式论文 §3.1、算法 1/2：ASG 的 AND/OR 聚合得到 STE；固定前缀末端向祖先传播最小 STE，动态后缀最先驱逐；**实际驱逐从叶子开始，释放后才把成为叶子的父节点加入堆**。§3.2 包含 CPU 备份、预取及加载状态感知调度。§3.3 明示基于 **SGLang v0.4.4**，提供显式固定末端标记或按持续命中历史识别固定部分两种方式。§3.4 对不可预测工作流采用等优先级退化。以上不是本项目只保留 Recurrent 槽位的完整行为。[正式 PDF，pp.3–6](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7971d31a7d5eb0f1eed2f8f6f368195-Paper-Conference.pdf)。

因此需修正两种过强说法：① 原方法并非必须由用户显式声明 Fixed Prefix，也允许历史启发式；② 无 ASG 时原系统不一定拒绝服务。本项目为证据完整性保守拒绝 UNKNOWN，是自身适配合同。当前历史数据既没有显式声明，也没有足以复原这一启发式的逐 Agent 持续命中、分段及失效过程，不能用一次 LCP 替代。

作者本地源码核验位置：

- `python/sglang/srt/managers/agent_manager.py:21–72`：接收执行步数、同节点多个 Agent 取最小值、优先级版本缓存。
- `python/sglang/srt/mem_cache/lora_hiradix_cache.py:269–305,933–947`：阈值保护、叶子遍历、写回/加载路径与步数更新。
- `python/sglang/srt/mem_cache/lora_radix_cache.py:119–141` 以及 `radix_cache.py:140`：比较器先返回 LRU，其后优先级比较不可达；这是该提交事实，**不能据此断言论文算法就是 LRU**。
- `SScheduler/Scheduler.py:129–233`、`SScheduler/policy/base_policy.py:37–72`：外部步数合并与传输；混合版本没有可核实为论文原实验的完整 ASG 生成路径。README 同时列出 KVFlow 与 ScaleSim。

官方同分语义未闭合。本项目固定 ID 同分是明确改写；同 STE 的多个近邻祖先/末端可能占用重复名额，结果必须同时报告同分组，不能全部归因于工作流感知失效。[本地源码审计](../../sota_baseline_audit_b5a/kvflow_audit.md)。

## 4. TOPAS：相关性很强，但不是可直接剥离的缓存评分

论文第 III、IV 节联合选择驻留前缀和在运行/获准请求，以任务剩余最长服务路径为目标；式(13)组合条件任务进展、搬运、重算、撤销及一步复用项。还需要 DAG 阶段服务分布、已执行服务量、任务年龄、请求专属 KV、交换带宽。第 IV-E 节含小规模枚举和较大规模增删/交换搜索，不能把它描述为“只做简单排序”。[原文](https://arxiv.org/html/2608.25523v1#S4)。

该联合决策与 FlowState 固定 Pending、只选 Recurrent、计数预算 K 的可行域不同。冻结请求调度或仅抽取复用项会丢失主要机制，只能命名新适配，不能称为 TOPAS。未找到 Hybrid Recurrent 可执行性验证；这表示所读论文未提供对应证据，不是断言绝不支持。未定位代码，函数级复现为 UNKNOWN。

## 5. 对创新性措辞的约束

Marconi 已处理 Hybrid 匹配及异构价值；KVFlow 已处理多 Agent 共享前缀和执行距离；TOPAS 已显式联合考虑前缀集合与请求分配。不能以“它们不懂共享/不会联合优化”概括竞争工作。当前可保留的区别须限定为：**同一严格 Epoch、固定 FA 与已知 Pending 下，按可执行 Recurrent 前沿及恢复成本进行集合分配，并经独立的组件控制合同落实**。性能胜负必须另由相同输入域的实验支持。
