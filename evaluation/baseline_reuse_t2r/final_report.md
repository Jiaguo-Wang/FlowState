# T2-R：复用既有外部 Baseline 的最小接线结果

日期：2026-10-27。**总门禁 PARTIAL：CPU共同接口与冻结回归通过；三策略正式GPU对照未就绪，本次未启动GPU。** 没有新建Marconi/KVFlow策略类，没有改生产FlowState、原策略评分、Phi、冻结输入或历史结果。

## 1. 此前已经实现了什么

| 能力 | 实际入口与已有证据 | 本轮判断 |
|---|---|---|
| Marconi Adaptation | `evaluation/sota_policies.py:96–140,216–258`：`MarconiStylePolicy.select`、归一化效用、字典序兜底 | 可直接复用；不是完整官方Marconi |
| 旧访问/增量代理 | `evaluation/sota_metadata.py:78–133`：LRU顺序转rank，按旧Workflow/Lineage父关系计算增量Token跨度 | 旧实验定义保持；不得直接套到新跨Workflow严格域 |
| 新KVFlow-inspired | `evaluation/kvflow_inspired_retention_cpu/policy.py:18–92`：Ready=0，AND=max+1，OR=min+1，固定末端向Token祖先传播，共享取最小STE，动态后缀优先淘汰 | 核心保留机制已实现，无需重写 |
| 旧KVFlow-style | `evaluation/sota_policies.py:26–94`：外部STE+旧兼容谓词 | 不用于此次共同接口，避免重新引入旧Workflow过滤 |
| 三结构CPU对照 | `kvflow_multiworkflow_cpu_main/`：A/B/C × K3/5 × 五策略，30行 | 直接复用，当前三策略对应18行 |
| 10-Agent初期CPU | `kvflow_synthetic_stage1/` | 早期末端域未包含后来观察到的祖先，不能直接替代新完整预算域 |
| 旧Marconi GPU | T2既有核验：24快照、216生命周期；对Marconi适配的24.264401%为逐快照TTFT改善平均 | 历史适配证据；本轮未重复审计/运行，不升级Strict或完整系统结论 |
| C/K3真实物理域 | `c_shared_gpu_pilot_k3/census/run/census.json`及`summary.json` | 16候选、16槽、12空闲，全部Token/位置可一对一映射；原Pilot只跑FS/ST/WFC |
| 正式C性能矩阵 | `c_formal_perf_gpu/protocol.json` | 60主臂是FS/ST/WFC，不是Marconi/KVFlow正式臂 |

原文献、作者代码与版本沿用 [T2审计](../paper_audits/t2_sota_baselines/paper_source_audit.md)，不另起大规模文献调查。用户列出的根目录`sota_policies/sota_policies.py`等路径没有用作第二套实现；实际复用的是上述evaluation入口。

## 2. 这次改了什么，为什么

全部新增文件仅在本目录：

- [interface.py](interface.py)：薄的纯CPU接线。共享候选、K、当前Ready域和评价入口；分别调用既有Greedy内核、MarconiStylePolicy和KVFlow的`select`。返回选集、各Pending E/G、共同J、原有评分/优先级和单次调用诊断时间。
- 同一入口先核验调用图/声明域、Epoch/配置、Token关系、Ready身份、声明访问顺序、祖先Token关系、等大小逻辑预算及UNKNOWN。没有用Phi替代KVFlow优先级。
- [test_interface.py](test_interface.py)：18个冻结条件回归与非法输入拒绝；显式禁止把CPU声明计划直接传给物理控制器。此拒绝入口不是新的Runtime安全框架，也没有实现物理协调。
- [collect_evidence.py](collect_evidence.py)：复制原有CPU结果并保留原行号，提取历史真实物理映射及独立组件字节数。不是新策略评估器。

**策略/生产代码修改为零。** 没有为“凑齐正式Marconi”猜测GatedDeltaNet FLOPs，也没有把旧合成访问改写为原生访问。

## 3. 核心机制是否满足要求

### KVFlow：CPU保留机制可用；物理执行仍由已有严格控制器负责

ASG→STE→固定前缀/动态后缀→确定性保留已闭合。未知前驱、未解循环、缺Fixed Prefix或缺映射均拒绝。UNKNOWN前驱存在时不擅自用OR其他分支补出最小值，是已冻结的保守合同。共享取关联STE最小值，但未知关联仍拒绝，不推测未来执行。

同分ID升序是明示适配，并非官方同分认证；没有实现原系统预取、调度和KV树物理驱逐。候选是完整Recurrent Checkpoint，FA保持不变。严格物理释放、引用与COW由共同控制器承担；不能说`policy.select()`本身已经完成Native Eviction。

C的全部16个候选已在历史GPU中物化，且其原始Token与声明候选逐项相等，[historical_physical_mapping.json](historical_physical_mapping.json)保存验证。**这些是历史绑定，不能用于新Engine。** `PROVENANCE_UNVERIFIED`不变。

### Marconi：适配可复用，忠实FLOP效率仍未闭合

| 项目 | 当前事实 | 处理 |
|---|---|---|
| Recency、归一化、排序 | 已实现rank/min-max/效用降序，α=1 | 保留冻结定义 |
| FLOPs分子 | 旧轨迹用增量Token跨度；A/B/C用累计位置L | 继续明确标为Token代理，不改名为真实FLOPs |
| 真实状态大小 | 本次从Census `component_slices.byte_ranges`确认每候选51,511,296 bytes | 已补充独立原始证据；不把整池storage_bytes当单槽 |
| 等成本分母 | 既有CPU memory_bytes=1代表逻辑单位 | 用实测相同字节分母复核K3/5选集不变；min-max归一化消去统一比例，不必改旧结果 |
| Qwen3.5 GDN计算成本 | 所核对入口没有已验证的精确分子；作者utils.py提供Mamba-1/Attention/MLP公式 | 不将Mamba-1公式直接移植到GatedDeltaNet |
| 原论文驱逐行为 | 作者`radix_cache_hybrid.py:515–545`使用当前叶/单孩子集合、父相对收益、逆时间间隔recency并随驱逐更新 | 当前静态Top-K没有复现全部行为；不能宣称官方核心驱逐规则完整一致 |
| α依据 | 1.0为既有预注册等权适配，未按本轮结果调参 | 可用于固定参数适配，不能称作者bootstrap后的最优α |

因此，“现有Marconi Adaptation可复算”为READY；“已补齐忠实Marconi保留/驱逐机制”为PARTIAL。仅把分母改成真实字节不会解决分子、父相对估值和动态更新的差异。本轮没有重建模拟器，也没有在信息缺失时重写策略。

## 4. 是否有三策略共同输入，CPU结果能否直接使用

**有：A/B/C的合成声明CPU域。没有：已经具备三者全部原生决策元数据、可立即用于忠实SOTA对照的新Strict Epoch。**

| 结构 / K | FlowState-Greedy J | Marconi代理 J | KVFlow-inspired J |
|---|---:|---:|---:|
| A / 3 | 0.000000 | 154.348956 | 0.000000 |
| A / 5 | 0.000000 | 154.348956 | 0.000000 |
| B / 3 | 0.000000 | 230.764345 | 236.250843 |
| B / 5 | 0.000000 | 230.764345 | 2.493390 |
| C / 3 | 154.348956 | 348.234011 | 627.614977 |
| C / 5 | 0.000000 | 348.234011 | 510.368355 |

单位均为modeled ms，分母6个条件、3种预声明结构。结果直接复制自冻结CSV，不重新生成工作负载。FS对Marconi为6胜；对KVFlow为4胜2平。**不省略未来评价：** 冻结后续调用静态成本中，FS对Marconi为5负1平，对KVFlow为2负4平。它不模拟在线状态更新，也不是未来真实延迟。

完整30行含LRU/WFC和未来静态目标均保存在 [cpu_comparison_reused.csv](cpu_comparison_reused.csv)；候选STE、priority、Marconi评分及所有选集见 [candidate_priorities_reused.csv](candidate_priorities_reused.csv) 和 [adapter_regression.json](adapter_regression.json)。

适用限制：访问事件是`DECLARED_SYNTHETIC_ACCESS`，不是原生Cache访问；CPU候选是PROPOSED，不能以后来C物化记录反向升级当年的A/B/C数据。短目标位置的非零Phi预测位于旧校准范围之外。A/K3、K5中FS和KVFlow成本相同但选集不同，不能算收益差异。C的KVFlow多项STE=0，ID同分及祖先/末端粒度会影响选集，不能概括为STE机制普遍无效。

## 5. CPU验证和版本状态

- `unittest` **21/21方法通过**：包括原KVFlow的14个测试及本次7个接口测试；子用例包含18个冻结条件、12类输入破坏、4类预算错误、Marconi无效元数据、实际等大小字节分母与逻辑单位的选集一致性。
- 18/18共同接口结果与冻结三策略结果的选集、E/G、J一致；不是新正式性能重复。
- 冻结输出清单15文件摘要全部通过。旧输入锁仅`flowstate/optimizer.py`与当前版本不同；本轮没有修改它，也没有刷新旧锁。新接口明确调用Greedy `_select_candidates`，不是当前`plan_strict`自动Exact分支。当前版本回归只证明这六个条件一致，不证明所有输入下与旧版本等价。
- 旧pytest套件尝试执行但uv312及系统Python均没有pytest，故**没有宣称这两套旧测试本轮重新通过**。原始错误日志保留；已用现有unittest完成本轮独立接口及核心机制检查，未安装依赖。
- [adapter_call_diagnostics.csv](adapter_call_diagnostics.csv)仅记录单次接口回归的准备/验证与策略调用边界；旧正式CPU结果未记录独立策略耗时，标UNKNOWN。不能把该单次诊断当稳定延迟或与旧GPU比较。

## 6. 是否具备正式GPU对照条件

**目前不能启动。** GPU0只读查询发现PID3492611，`sglang::scheduler_TP0_EP0`占65,736 MiB；未启动Engine、未等待重试、未干预其他任务。详见 [gpu_gate.json](gpu_gate.json)。

资源空闲之外还需区分两条路线：

1. **现有适配的集合效果对照：** 可以提交单独协议，明示冻结合成访问、累计Token代理、α=1、ID同分。候选优先沿用C/K3的16项域。三者仅输出选集，然后原样复用`c_shared_gpu_pilot_k3/control.py:physical_cpu_census→apply`的当前Token映射、Strict计划验证、Bind/Preflight、`reconcile_strict`与Native Resume/COW/Cold合同。每臂独立生命周期；此路径尚未加入Marconi/KVFlow正式臂，也没有新通过记录。不得以Marconi与Standalone同集借用其策略选择/控制耗时。
2. **声称忠实Marconi核心机制的正式对照：** 还缺验证过的GDN计算节省输入、符合该评分语义的原生访问时间，以及父子收益/逐次淘汰更新合同。当前CPU代理结果不能填补这些缺口。本轮不凭任意常数或Phi补齐它们。

不把恢复正确性、Token前缀或历史物化提升为跨Workflow张量血缘认证。不把逻辑K或实测独立槽成本称为真实容量不足。正式多轮工作负载与协议须另行提交审核，本轮未自行安排或执行。

**本次实际完成：复用实现、共同CPU接口、冻结结果回归、真实字节/映射证据补齐；未完成：忠实Marconi分子及动态驱逐适配、新三策略GPU臂。** 至此停止。
