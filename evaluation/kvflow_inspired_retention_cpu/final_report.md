# KVFlow-inspired Recurrent Retention 最小实现与CPU结果

结论：**实现及14项基本单测通过；真实快照比较受输入缺口阻断。** 已立即运行 E2 w1–w4 和 B5-C4，共5个严格冻结快照、20个预算条件。不能忠实计算的结果标记UNKNOWN，没有生成代理打分或虚构选集。

## 已交付实现

- `policy.py`：独立纯CPU策略，无Phi、收益矩阵、Workflow相等判断、调度、预取或Runtime调用。
- `compute_ste`：有当前Epoch声明证据的Ready调用为0；AND为前驱最大值加1；OR为前驱最小值加1；未知前驱、缺声明、未解循环均保持UNKNOWN。未知OR前驱也保守保持UNKNOWN，不推测其步数。
- `priorities`：根据实际Token数组验证候选是固定前缀末端的祖先，向上传播关联调用STE的最小值；只有声明并验证的动态后缀取无穷大保留优先级。固定与动态角色重叠时，已知固定用途优先。无映射或关联STE不完整时保持UNKNOWN。
- `select`：按优先级升序、Checkpoint ID升序确定性保留K项。任何候选优先级UNKNOWN时拒绝生成选集，不把未知当作动态后缀，不用字典序掩盖缺证据。
- `test_policy.py`：14项测试（含参数化子项），验证Ready、AND/OR、缺失前驱、循环、未来信息、共享传播、动态后缀、同分、身份、位置、预算、无副作用和拒绝行为。测试Token仅为单测夹具，不属于真实工作负载。
- `run_cpu.py`：读取冻结来源和现有完整性清单，提取元数据并尝试执行策略；独立重算既有Greedy/WFC选集的共同J和逐Pending E/G。选择模块不访问这些评价分数。

本模块是逻辑保留适配，不是官方完整KVFlow；调用方仍须先验证Strict Epoch物理准入、等大小预算和冻结身份。它不执行绑定或驱逐，不取代现有Strict合同。

## 论文与官方实现的边界

采用[论文§3.1](https://arxiv.org/html/2507.07400#S3.SS1)的步骤聚合与固定前缀优先级传播，动态后缀最先淘汰。本实现不包含§3.2预取或调度。

核对[官方仓库](https://github.com/PanZaifeng/KVFlow)及已有本地副本，锁定提交 `7ef897eaedc16626a281ccfe0d453fae80251e8f`。当前 `lora_radix_cache.py:119` 的比较器直接返回LRU；`lora_hiradix_cache.py:269–295` 使用阈值保护及叶子遍历，而非本文要求的统一K项排序。因此不能从该混合版本确认论文优先级机制的官方同分规则。本实现明确使用ID升序，不冒称继承官方同分语义，也不额外引入LRU或深度作为代理。

已有旧 `sota_metadata.py` 固定steps的代理和 `rq3_kvflow_controlled_comparison.py` 手工工作流图没有真实固定Token末端映射，未用于生成本轮“真实CPU结果”。

## 实际输入适配结果

|快照|候选|Pending|严格配对|预算|KVFlow结果|
|---|---:|---:|---:|---|---|
|w1|7|4|28|1–4|UNKNOWN|
|w2|7|4|28|1–4|UNKNOWN|
|w3|7|4|28|1–4|UNKNOWN|
|w4|7|4|28|1–4|UNKNOWN|
|B5-C4（使用B5-C3物理快照）|7|4|28|1–4|UNKNOWN|

这5个快照的Token和严格恢复关系存在，但不能提供KVFlow特有元数据：

1. Snapshot只保存Session/Workflow和请求身份，没有已声明Agent invocation到依赖图节点的映射、前驱或AND/OR语义。
2. `controlled_input_ready=true`说明受控输入已准备；`agent_native_ready`仍为未认证。即使允许将当前请求视为立即可执行，也不能据此识别所有缓存Agent及其固定前缀末端。
3. 保存完整Token输入和LCP，不等于保存固定提示声明。不能把LCP自动指定成fixed prompt，也不能把完整历史自动视作固定提示。
4. 缺少已确认动态后缀分类；未知候选不能直接设为最高淘汰级别。

`asg_extraction.json`记录原字段和空的声明结果；空数组表示**未观测到ASG声明**，不是已证明没有依赖的图。`pending_ste.csv`的20个调用映射和`candidate_priorities.csv`的35个优先级均明确为UNKNOWN。

## CPU对照CSV

`cpu_comparison.csv`包含100行：20个条件×5种策略。

- KVFlow-inspired：20行UNKNOWN，选集、J和E/G为空，不当作空选集或零收益。
- FlowState-Greedy与WFC：40行既有冻结选集及共同目标重新计算通过，含真实J及逐Pending E/G；不是新运行求解器的结果。
- LRU与Marconi Adaptation：40行UNAVAILABLE。所读取五组严格结果没有二者可复核的选集与对应完整访问输入；没有借用旧24快照或旧逻辑域的元数据临时补分。
- 与KVFlow的同集/异集、胜平负全部UNKNOWN。**有效KVFlow对比为0，不是0胜20负或20平。**

本轮没有忠实KVFlow选集，因此不能给出其实际模型化J、前沿差异或性能优劣。合法Strict恢复矩阵不自动满足ASG和固定提示的输入合同。

## 接口所需的最小现有数据形式

`graph`逐调用需要：`invocation_id`、`epoch_id`、`known_at_epoch`、`evidence_path`、`ready`、`predecessors`、`aggregation`。

`declarations`逐调用需要：同一身份、配置摘要、当时可见性证据、`fixed_token_ids`、`input_token_ids`及是否已确认动态后缀。候选直接使用冻结`checkpoint_id/token_ids/token_position_L/barrier_id/configuration_digest`。

这些是实现接口，不是要求修改冻结快照；本轮没有创建声明、读取未来轨迹或推断不存在的Agent关系。要产生忠实真实对比，需要来源自身提供上述声明证据，不能靠更换打分方式补齐。

## 复现与停止

```bash
/home/wjg/uv312/bin/python -B -m unittest evaluation.kvflow_inspired_retention_cpu.test_policy -v
/home/wjg/uv312/bin/python -B -m evaluation.kvflow_inspired_retention_cpu.run_cpu
```

所有输出仅写本目录；`source_integrity.json`记录已读冻结来源的SHA及前后不变检查。未修改生产FlowState、SGLang、Phi、Strict合同或冻结数据，未运行GPU。已完成最小实现、测试及真实输入适配尝试，停止，不自动采集新元数据或启动其他实验。
