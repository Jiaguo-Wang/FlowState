# KVFlow合成顺序工作流第一阶段

**总体：PARTIAL。CPU生成、优先级与比较完成；真实固定前缀物化成功；Reconciliation/Resume未完成。** 一次独立GPU尝试在任何驱逐前按冻结门禁停止，已清理，未自动重试。

## 实现及预注册输入

参考[KVFlow §4.1](https://arxiv.org/html/2507.07400#S4.SS1)的十Agent、随机固定前缀和变化后缀机制独立重建，标记为`reconstructed_synthetic_sequential_workflow`。本地[官方仓库](https://github.com/PanZaifeng/KVFlow)提交为7ef897eaedc16626a281ccfe0d453fae80251e8f；未找到可直接复现该论文配置的生成入口。没有把通用HiCache benchmark冒充论文生成器。

`protocol.json`在计算结果之前固定：种子20261021，10个Agent，固定4096 Token、动态32 Token，固定提示预热两轮，完整工作流输入生成两轮，预算K=3与5。Runtime仅K3、目标Agent00，temperature=0、max_new_tokens=8、ignore_eos=True；历史预热生成1 Token。没有改变来源、长度、种子、预算或Phi追求优势。

- `generate.py`及`workload.json`保存完整Token数组、调用身份、Ready、前驱、AND语义、Fixed/Input Token、模型及Tokenizer摘要。
- 每轮首调用Ready，之后依赖前一调用，STE为0至9。所有动态输入在实验开始前独立生成，不依赖前一次实际输出。
- 两轮是同一合成工作流的重复输入，不是两个独立工作负载。CPU规划将未来一轮十次已知调用等权纳入目标，**不是十个同时Ready Agent**。
- 通过Qwen3.5-9B冻结词表直接采样非特殊Token IDs，不做文本编码或Chat Template渲染。词表、模板、配置身份仍被保存。
- uv312 Transformers因已有huggingface-hub版本冲突无法导入；使用原词表JSON生成解决CPU输入问题，未安装、修改依赖。Runtime使用原已验证镜像。

## CPU结果

复用`evaluation/kvflow_inspired_retention_cpu/policy.py`，未修改其公式或同分规则。新增16项CPU检查通过。候选域为每Agent一个固定末端，共10个等逻辑成本的**PROPOSED**状态；T=4096是CPU情景FA前沿假设，不冒充GPU观察。

|预算|策略|选中Agent|共同J（模型化ms）|
|---|---|---|---:|
|3|KVFlow-inspired|00、01、02|1080.442689|
|3|FlowState-Greedy|00、01、02|1080.442689|
|3|WFC|00、01、02|1080.442689|
|3|LRU|09、08、07|1080.442689|
|3|Marconi Adaptation|09、08、07|1080.442689|
|5|KVFlow-inspired|00、01、02、03、04|771.744778|
|5|FlowState-Greedy|00、01、02、03、04|771.744778|
|5|WFC|00、01、02、03、04|771.744778|
|5|LRU|09、08、07、06、05|771.744778|
|5|Marconi Adaptation|09、08、07、06、05|771.744778|

第二轮同样结果，全部20条策略记录保留在`cpu_comparison.csv`。候选STE/Priority为0至9，详见`candidate_priorities.csv`。每条CSV包含逐调用E/G：选中Agent为E=4096、G=0；未选为E=0、G=4096。对应下一请求还有32个动态Token，本恢复目标不把这32个新Token当作历史恢复缺口。

KVFlow与Greedy/WFC同集；与LRU/Marconi异集且至少部分调用前沿不同；**共同J全部持平，未发现模型化优势**。原因是长度相等、支持互不相交、每保留一个末端提供相同收益。静态十调用等权J不表达近期调用顺序，也不是完整循环缓存管理性能。

LRU元数据来自CPU模型中已声明的两轮固定提示访问次序，Marconi复用既有recency加增量Token跨度/逻辑大小的适配，α=1。此处是完整生成输入下的CPU事件模型，**不是实际Native Node最近访问计数**，也不是官方Marconi系统。随后GPU出现更多节点，不能把该十候选表当作二十个真实节点的策略结果。

## 最小真实执行结果

本轮新Engine：Qwen3.5-9B、SGLang v0.5.17、H100 PCIe、TP=1、28 Recurrent Slots。预启动GPU0无计算进程，显存14 MiB。输入、运行源码和镜像身份检查通过。

20次固定提示预热均完成，实际接收Token摘要与提交一致。`strict/history_01.json`到`history_20.json`保存逐次输入、输出、Native匹配和完整物理观察。**尚未执行两轮完整的带动态后缀工作流**；GPU只完成固定提示预热，随后门禁停止。

最终真实观察：

- 10个4096位置的固定末端，Node18–27，各自Token数组与一个Agent完整Fixed Prefix完全一致。
- 另有10个4032位置的Recurrent祖先，Node29、31、33、35、37、39、41、43、45、47；逐Token核对分别为Agent00–09固定前缀的真实祖先。
- 共20个驻留Recurrent状态，8个空闲Slot，无记录到的Native Eviction。
- `runtime_materialization.csv`保存位置、Slot、对应Agent、Conv/Temporal完整性及引用。Node/Slot仅对该已结束生命周期有效。
- 第二轮固定提示重放记录出现H=4095、E=0以及mamba_branching_seqlen=4032；不能据“同一提示重复”推断已经从4096末端复用。实际行为见history日志，本轮未放宽匹配要求或强制Native来源。

## 停止原因：实验适配候选域过窄

`control.py:15`要求每个实际候选都等于一个完整固定提示末端，后续还要求候选数恰为10。真实出现的4032祖先不满足该**实验层附加假设**，故在Bind、Reconciliation和Pending提交之前拒绝。

这不是SGLang无法物化Fixed Prefix，也不是KVFlow优先级无法表达祖先；既有策略本身支持祖先传播。本轮入口没有把实际二十候选域接入，而是过早要求与CPU十候选域相同。因此不能报告Runtime闭环通过，也不能称为已证实恢复计算错误。

按照已冻结的失败停止规则保留该尝试，不改门禁后重试、不删除额外候选，也不把CPU选集冒充已落实物理集合。该入口不能直接宣称为可用的完整Runtime适配。

实际Recurrent-only驱逐 **0次**，正式Pending **0次**，Cold生命周期未启动；没有Native Resume、COW输出正确性认证。失败响应原文与traceback在`strict/failure.json`及`strict.runtime.log`，其中operations为空。ControlClient异常抛出使`reconciliation.json`未生成；本报告不宣称持有未保存的最终Strict Epoch内容哈希。

## 清理与证据边界

Engine正常进入清理。专用容器已由`--rm`移除，后续删除检查返回“No such container”符合该事实。连续三次检查均GPU空闲、14 MiB，无本轮容器残留；`gpu_cleanup.json`为PASS。没有操作其他用户进程。

- 已证明：可复现合成Workflow→声明ASG→Fixed Token→CPU KVFlow选集与共同模型化对照；原生GPU能物化固定末端及额外祖先。
- 未证明：KVFlow物理保留/驱逐效果、选择后的Native前沿、COW和Cold输出一致性、真实缓存复用收益、性能或完整KVFlow系统优势。
- 不将逻辑K称为物理内存压力，不将PROPOSED或观察到的Token关系自动升级为完整EPOCH_ADMISSIBLE/RESUME_VALIDATED。

## 产物与复现

`generate.py`、`cpu.py`、`test_cpu.py`、`protocol.json`、`workload.json`、`input_lock.json`为CPU实现与身份；`cpu_comparison.csv`、`candidate_priorities.csv`为CPU结果；`control.py/bridge.py/runner.py/launch.py`为本轮独立Runtime实验入口；`runtime_materialization.csv`、`strict/`、`runtime.log`、`gpu_cleanup.json`为真实证据。

CPU入口：`/home/wjg/uv312/bin/python -B -m evaluation.kvflow_synthetic_stage1.cpu`。本目录已运行Runtime，启动器会拒绝覆盖；**不要直接重新执行生成器改变已锁定文件**。

本轮未修改生产算法、Phi、现有KVFlow Policy、SGLang安装源码或历史冻结数据。`result_summary.json`记录已锁源文件不变检查。第一阶段结束，停止，等待下一阶段指令。
