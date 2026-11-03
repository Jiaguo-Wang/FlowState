# Qwen 27B 接入 FlowState：模型与 TP=2 只读可行性审计

日期：2026-10-30。**最终结论：PARTIAL。** 本地模型具备与冻结 SGLang 实现匹配的 Hybrid 架构及 TP 分片路径，双 H100 的容量估算没有显示必然不可部署；但未执行27B加载。**当前 P0 正式链路明确限定 TP=1，尚无覆盖两个 Rank 的 Strict 观察、绑定、连续隔离和一致协调合同，不能直接启动27B三策略正式实验。**

本轮只新增本报告；未修改 Python、Shell、配置、冻结输入或历史结果；未启动模型、Engine、CUDA计算或GPU实验，未安装/下载/构建。检查镜像时使用 `--read-only --network none --runtime runc` 的无GPU临时容器，仅读取版本与源码；未传 `--gpus` 或挂载模型加载权重。没有调用现有实验启动器。

## 1. 模型真实身份

|项目|实际读取结果|
|---|---|
|本地模型名称|README标题 **Qwen3.8-27B**；不是根据目录推断Qwen3.5-27B|
|入口路径|`/data1/models/Qwen3.8-27B`|
|真实路径|`/data1/models/.cache/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master`；入口为符号链接，未来容器必须使其目标同样可见|
|architecture|`Qwen3_5ForConditionalGeneration`|
|根model_type / text model_type|`qwen3_5` / `qwen3_5_text`|
|模型类型|Dense Hybrid、带Vision Encoder的条件生成模型；`language_model_only=false`，不是MoE。文字实验无需图像，但不能假定加载器不会创建Vision模块|
|语言层数及组成|64层：48层`linear_attention`、16层`full_attention`，每4层一层Full Attention|
|隐藏/FFN维度|5120 / 17408|
|Full Attention|24个Query heads，4个KV heads，head_dim=256，带输出门|
|Linear/Recurrent|Gated DeltaNet实现；16个Key heads、48个Value heads，Key/Value head_dim均128，Conv kernel=4|
|状态dtype配置|`mamba_ssm_dtype=float32`；它与权重BF16不同。冻结源码默认Conv为BF16，环境变量可覆盖，后续须锁定|
|最大上下文|`text_config.max_position_embeddings=262144`，即256 Ki tokens；这是配置上限，不是本轮验证过的服务长度或显存保证。README提到的托管服务1M不等于本地配置|
|其他|词表248320，`tie_word_embeddings=false`；另声明1层MTP。config记录Transformers `5.8.0.dev0`，不意味着必须使用该旧版本|

来源：[模型配置](/data1/models/Qwen3.8-27B/config.json)、[本地README](/data1/models/Qwen3.8-27B/README.md)、[权重索引](/data1/models/Qwen3.8-27B/model.safetensors.index.json)。本轮未联网验证模型发布身份或README外部主张，报告仅认证上述本地内容。

### 1.1 权重元数据：独立逐分片核对

仅读取18份safetensors的8字节头长及JSON头，没有把Tensor载入内存或GPU。逐Tensor检查shape、dtype、data_offsets，并与索引的weight_map精确比对。

|项目|结果|
|---|---:|
|分片|18/18，`model-00001-of-00018.safetensors`至`model-00018-of-00018.safetensors`|
|Tensor数|1199；所有Tensor头的dtype均为 **BF16**；未发现量化Tensor或FP16/FP32存储Tensor|
|形状乘积总元素数|27,781,427,952，约27.7814B；不是只凭“27B”标签估计|
|语言主干及lm_head|26,895,998,464元素|
|Vision|460,730,096元素|
|MTP|424,699,392元素|
|纯Tensor payload|55,562,855,904 bytes，等于全部元素×2，与索引`metadata.total_size`一致|
|全部分片文件大小|55,563,006,776 bytes =55.563 GB = **51.7471 GiB**，包括头部|
|索引一致性|实际Tensor名→分片映射与weight_map相同，没有重复Tensor键|

该检查证明头部、清单和磁盘尺寸一致，**不是55GB权重内容逐字节校验，也不是模型加载成功证明**；未与外部发布方权重摘要比对。小文件SHA-256：

- `config.json`：`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`
- `model.safetensors.index.json`：`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`
- `tokenizer_config.json`：`b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`

## 2. 冻结镜像与 SGLang 实际支持

本轮读取当前P0启动器指定的**同一个镜像ID**，不是最新tag或宿主Python环境：

|身份|实际结果|
|---|---|
|镜像ID|`sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926`|
|RepoDigest|`lmsysorg/sglang@sha256:2fa998888b1b48e778e17f6360a6b60f3cc888b56fa35281669af1d007316337`；它和镜像ID是不同对象|
|SGLang / Transformers|**0.5.17 / 5.12.1**，由容器内distribution metadata读取|
|PyTorch / Tokenizers|2.11.0+cu129 / 0.22.2|
|真实源码根|`/sgl-workspace/sglang/python/sglang/`；不是普通site-packages下猜测的源码路径|

以下路径均相对此源码根，行号对应本轮冻结镜像实际文件：

|检查|实际源码依据|结论|
|---|---|---|
|architecture注册|`srt/models/qwen3_5.py:2319` 的EntryClass包含`Qwen3_5ForConditionalGeneration`；`srt/models/registry.py:111–118`扫描EntryClass|存在对应注册声明和原生实现。不是仅Transformers回退；本轮未导入完整模型注册表、实例化模型或验证权重加载|
|配置识别|`srt/configs/qwen3_5.py:15–34`继承Qwen3NextConfig，规范化rope_parameters；实际模型配置model_type吻合|源码接口匹配；不能由模型市场名“3.8”判为未知architecture|
|Hybrid组成|`qwen3_5.py:216`实现Qwen3_5GatedDeltaNet；`:830`实现Full Attention层；`qwen3_next.py:271–312`分离线性/Full层并生成状态参数|不是把线性层当普通Attention KV缓存|
|Attention TP|`qwen3_5.py:848–858`按attn_tp_size切分Q heads/KV heads；两者整除检查|24/2=12 Query heads、4/2=2 KV heads每Rank，可满足整除条件|
|GDN TP|`qwen3_5.py:216起`使用attn_tp_rank/size、Column/RowParallel投影；`:320–328`分片dt_bias/A_log及其loader|16/2=8 Key heads、48/2=24 Value heads每Rank；不存在本配置的头数不可整除问题|
|Recurrent状态切分|`srt/configs/qwen3_next.py:298–312`→`mamba_utils.py:189–236`，Conv各Q/K/V子块按头分片；Temporal形状`(num_heads/TP, head_dim, state_size)`|每Rank持有不同状态切片；Rank0内容不代表Rank1内容，不能比较两RankTensor哈希相等来认证|
|状态分配|`srt/mem_cache/memory_pool.py:329起`的MambaPool；缓存参数含Conv和Temporal及dtype，标准池含额外哨兵槽|每个Scheduler/设备有本地Pool；需逐Rank完整性与账本|
|TP进程创建|`srt/entrypoints/engine.py:841–892`按tp_rank创建Scheduler子进程并指定gpu_id；普通请求在`managers/scheduler_components/request_receiver.py`经TP组广播|原生Engine存在TP=2执行机制；不等于外部FlowState控制socket自动广播|
|多模态/MTP|`qwen3_5.py:1771起`ConditionalGeneration继承VL包装；普通权重loader可跳过MTP，Vision由VL路径处理|不能把磁盘全权重简单除2当加载显存实测；本轮不启MTP、Vision推理或推测解码|

**SGLang子结论：源码层支持路径存在，未发现本配置明确不兼容的架构/TP头数条件；27B BF16 TP=2实际加载、内核形状、32K恢复和Cold正确性仍未验证。** 没有运行推理，因此“模型可以运行”只能作为可部署候选，不能写成已实测通过。未检查外部issue全集，不声称不存在其他已知缺陷。

主要镜像源码SHA：`qwen3_5.py`=`be3def820e9448d808abcfddeb01338d7376586d67203a6d22aed7da423f0897`；`registry.py`=`3c5b4e87c5943147c28cb5b68e187cdf8f8f9d880c42f1f6941fce99fe757e41`；`memory_pool.py`=`29ef7b37d2e12b1a4b674b97c33b3ceb05cd8421a3742eb33cb6a75f52896ddc`；`engine.py`=`63f285e2d1866f4e9cc34867b330809d71ddac7124e2f442f308124f7fa9923f`。

## 3. 双 H100 显存可行性：理论分项，不是部署测量

只读nvidia-smi返回两卡均为H100 PCIe，每卡81559 MiB≈**79.6475 GiB**；GPU0/GPU1占用14/17 MiB，计算进程查询为空。UUID分别为`GPU-bfff4d31-0b94-7de3-9946-f593437ae566`和`GPU-051c9abc-864a-9f31-c4f7-a737aed997ff`。这是读取瞬间状态，不是实验资源预留。`nvidia-smi topo -m`报告两卡互联 **NV12**；不能仅因名称为PCIe就断言无NVLink，也未测NCCL带宽/延迟或P2P正确性。

### 3.1 权重

全部BF16磁盘payload平均平分约 **25.874 GiB/Rank**，仅为粗略基线。实际Dense文字权重按TP切分，加上复制的小参数、可能复制的Vision Encoder，且普通路径不加载MTP，分配并非严格一半。

用“语言主干完全平分 + Vision完整复制每Rank”的简化情形估计：语言约25.0489 GiB/Rank、Vision约0.8582 GiB/Rank，共 **25.9070 GiB/Rank**。具体Vision分片方式、norm复制、allocator及加载瞬时副本未测；可用约26 GiB作为数量级，不将其作为峰值上限。CPU权重加载和临时Host内存也未测。

### 3.2 Attention KV

BF16 KV、TP=2、16个Full Attention层、每Rank2个KV heads、head_dim256：

`每个缓存Token每Rank =16×2(K/V)×2(KV heads)×256×2 bytes =32768 bytes =32 KiB`。

|驻留的不同Token总数|每Rank纯Attention KV|
|---|---:|
|7200|0.2197 GiB|
|22528|0.6875 GiB|
|32768（32K）|**1 GiB**|
|四条互不共享的32K路径，总131072|**4 GiB**|
|131200|4.0039 GiB|
|262144|8 GiB|

这里是**全树去重后的驻留Token数**，不是只有当前单请求长度。共享前缀只计算一次；历史、输出、预热随机分支和工作Token同样占用缓存，页对齐与Pool预分配另计。当前C/K3四Pending输入为4128/5152/6176/7200 Tokens，16条预热为两轮固定长度；22528仅是四个T之和，不是实测全树占用，不能据它认证容量足够。

### 3.3 Recurrent State

冻结源码默认Conv BF16、Temporal FP32，无ReplaySSM额外缓冲时：

- 每Rank每线性层Conv：`((2×16×128+48×128)/2, 4−1)=(5120,3)`，30720 bytes。
- 每Rank每线性层Temporal：`(48/2,128,128)=(24,128,128)`，1572864 bytes。
- 48层合计每逻辑状态每Rank：Conv **1.40625 MiB** + Temporal **72 MiB** = **73.40625 MiB**；两Rank合计146.8125 MiB。
- 28个可分配槽约 **2.0072 GiB/Rank**；若按MambaPool `size+1`包括哨兵槽计，约 **2.0789 GiB/Rank**。配置中的28不是“全系统只有28份单卡张量”，而是两Rank共同表示28个逻辑状态；不能将每Rank槽数相加当56个可独立Checkpoint。

这是标准Conv/Temporal基础池估算。工作/COW占可用槽，额外tracking/intermediate状态、预填充激活和临时buffer并未包含。两RankSlot编号可以不同，正确性要求映射对应，不能要求物理地址或内容哈希相等。若环境改变状态dtype、ReplaySSM、布局或跟踪策略，上式需重新核对，不能为省显存擅自从FP32改BF16。

### 3.4 激活、图和工作区及当前0.4预算风险

|项目|当前可得判断|
|---|---|
|CUDA Graph|正式9B配置`disable_cuda_graph=true`，保持该设置时无图捕获池；仍有CUDA上下文、CUDA事件、NCCL和内核工作区|
|激活与Prefill工作区|27B更深、更宽，受实际chunk长度、临时投影、Attention/GDN backend影响；**未测，UNKNOWN**。当前chunked_prefill_size=45056不能凭9B通过就认证27B峰值|
|NCCL/通信及allocator|需要TP通信buffer、可能复制参数和碎片；**未测，UNKNOWN**，不能填任意固定GiB冒充实测|
|单32K的基础合计|权重约25.91 + KV1 + Recurrent2.079 ≈ **28.99 GiB/Rank**，未含上述未知项|
|四独立32K路径的基础合计|约 **31.99 GiB/Rank**，同样不含未知项|
|物理容量层面|相对79.65 GiB仍有较大未分配容量；没有从权重+基础缓存发现必然OOM。但这不是实际余量认证|
|原配置`mem_fraction_static=0.4`|静态总预算数量级仅 **31.86 GiB/Rank**；四独立32K的基础合计已接近/略超，不能原样担保。单32K的基础剩余也仅约2.87 GiB静态预算|

冻结源码 `srt/mem_cache/kv_cache_configurator.py:1614起`用当前空闲显存减去 `pre_model_load_memory×(1−mem_fraction_static)` 预留；权重已经占用会计入预算，非“加载26 GiB权重后另有总卡40%可作KV”。最终还按分布式可用内存决定池大小，任何Rank不足都影响整体。

**显存结论：硬件总容量具有合理可行性，当前9B的0.4静态预算、131200服务上下文、45056 chunk、28槽组合未被27B验证。** 需要独立部署记录实际每Rank加载峰值、Pool大小及工作区，才能锁定27B配置；本轮不修改任何容量参数，也不推荐未经测量直接增至某个比例。

## 4. 当前 Attempt 5 正式调用链与配置

本轮读取真实 [Attempt 5报告](p0_control_cycle_gpu/attempt5_audit_report.md)、[summary](p0_control_cycle_gpu/arms/r1_ck3_fs/run/summary.json)及当前源码：FlowState-Greedy状态PASS、实际生产`GlobalOptimizer.plan_strict`→Marginal Greedy、16个物理候选完整映射，选集B/C/D三个末端。它证明9B单卡单臂，不证明27B或两基线已完成同轮。

调用链：

`launch.py::docker_cmd` → `device_entry.main` → `runner.main/ObservedEngine` → `bridge.scheduler_process` → `wp3b_end_to_end_transport._wrapped_run_scheduler_process` → 每个Scheduler本地`on_idle/drain_one` → `control.apply` → 当前物理观察/Strict Epoch → 在线策略Select → `StateController.reconcile_strict` → `StrictReconciler` → `StrictRuntimeBridge` → **本地**`SGLangAdapter.evict_mamba_only` → Native Match/COW/推理 → Cold与后验。

|配置/入口|当前实际值及限制|
|---|---|
|launch.py MODEL|`/home/wjg/models/qwen3.5-9b`，只挂载此路径；27B符号链接及真实目录当前未挂载|
|Docker设备|`--gpus device=<单个UUID>`；`CUDA_VISIBLE_DEVICES=<单UUID>`、`EXPECTED_GPU_UUID=<单UUID>`。当前链未用一个`GPU_VISIBLE_DEVICES`字段管理双卡；NVIDIA runtime的设备可见性来自实际Docker参数|
|device_entry.main|明确`torch.cuda.device_count()==1`；只校验设备0、一个UUID及H100 PCIe型号，写入单卡device_binding|
|Engine配置|`tp_size=1`；`max_mamba_cache_size=28`；`mamba_track_interval=256`；extra_buffer；`mem_fraction_static=0.4`；context_length131200；chunked_prefill45056；禁CUDA Graph与overlap|
|模型/Tokenizer/模板身份|prepared_inputs配置及source_identity锁定9B路径、config、Tokenizer、Chat Template和引擎配置摘要。27B的配置/Tokenizer摘要不同，不能直接替换路径后沿用旧执行锁|
|控制port|单个`FLOWSTATE_STEP5D_PORT=50087`，一个ControlClient；没有rank→port表|
|分布式初始化|当前TP1只有一个Scheduler Rank；SGLang TP2能创建两个子进程并初始化通信，但P0没有与之配套的双Rank控制协议|

## 5. TP=2 状态观察与协调：八项明确答复

|问题|现有能力及实际阻断|结论|
|---|---|---|
|1. 是否硬编码单GPU？|launch选一UUID，device_entry断言仅一GPU；冻结输入tp_size1|**是**，直接挡住原样启动|
|2. 能否设置TP2？|SGLang Engine有该参数；但本链`targeted_probe._validate_runtime_scope:293起`遍历tp/pp/dp，任一不为1即抛错；R2-C0 `scope`及观察复用此函数|原生参数存在，**当前FlowState安全路径拒绝TP2**|
|3. 两Rank能否都观察？|可在各自Scheduler调用本地探针；但当前transport在每个Scheduler init安装相同端口的control server，没有rank偏移或汇聚|**没有当前可用的双Rank观察入口**；放开device检查可能出现端口bind冲突或只有一个Rank可达|
|4. 逻辑Checkpoint如何映射？|当前census按完整Token前缀+L映射到一份PhysicalView；`PhysicalNode`只有node_id/slots/component_ids，Binding只有单engine/window/view|**缺少逻辑ID→每Rank本地Node/Slot/组件集合的认证与汇聚**。两个Rank的Node/Slot不应硬编码相同|
|5. 身份/驻留/完整性怎么验证？|现有slice_identity记录本地device、storage地址及范围，能区分本地切片；PhysicalView engine_id为pid，不含world/rank共识|可复用本地检查，但需全Rank完整报告及共同epoch/config/Token/L/选集一致性；一个Rank PASS不能认证整个状态|
|6. 驱逐能否跨Rank一致执行？|Adapter只访问传入本地tree_cache，StrictReconciler逐项处理本地视图；没有广播、双Rank预检汇聚、统一提交或跨Rank失败传播|**当前不能认证**。Rank0驱逐不代表Rank1驱逐；必须任何Rank失败均停止所有Rank Admission，不能恢复半状态|
|7. 恢复能否保证同一Checkpoint？|Native TP正常请求会广播输入并协同计算；但FlowState本地控制socket不经过原生request广播，可能先使各Rank缓存不同|**尚未建立**。需逐RankNative Match同逻辑源/E/G、COW工作槽隔离及源保护，不能只看Rank0输出|
|8. Cold验证适用于TP2？|逐Token对照方法可复用；当前Cold文件是9B/TP1，不能作为27B参照|需独立27B/TP2同参数Cold，再认证各Rank路径和最终输出；不能先把输出差异解释为浮点误差|

### 5.1 为什么不是“删掉单卡断言”即可

1. [targeted_probe.py:280–306](../motivation/artifacts/wp3b_gate_20260820/targeted_probe.py:280)明确拒绝TP2，属于当前受控正确性前提，不是无意义限制。
2. [wp3b_end_to_end_transport.py:147起](../tests/runtime/wp3b_end_to_end_transport.py:147)为每个Scheduler安装控制服务；[install_control_server:525–564](../motivation/artifacts/wp3b_gate_20260820/targeted_probe.py:525)执行同地址socket.bind并用本进程队列。它不会把控制动作广播到TP组；不能将普通推理请求广播误当控制广播。
3. [R2-C0 observe](runtime_control_closure_r2c0/bridge.py:60)及[PhysicalView/Binding](../flowstate/strict_control.py:29)只表示一份本地池、单控制窗口与本地FA摘要。把两Rank局部Tensor拼成一串组件ID并不能证明同时静止、所有Rank都通过Preflight或协调全局成功。
4. 同一窗口下必须做到**所有Rank均进入且保持控制隔离→全Rank只读预检通过→统一选集→各Rank动作→全Rank后检汇聚→共同允许请求**。每Rank分别发一次RPC但其中一个已返回Scheduler主循环，不等价于这项合同；还须避免其他Rank停在TP collective而导致控制死锁。
5. 现有本地fail-stop不提供跨Rank回滚。若一Rank操作失败、另一Rank已经释放，必须将整个Engine标失效并清理，不允许继续Native Resume。无需发明回滚算法，但必须实现跨Rank失败传播与停止。

生产Optimizer的数学选择主要依赖逻辑Strict Epoch，不依赖GPU数量，**不需要更换Greedy或收益目标**。缺口位于分布式证据与应用合同：一个逻辑状态由两份Rank分片共同构成；只能在全部Rank有效时标EPOCH_ADMISSIBLE。逻辑K应统一选一次，不允许两个Rank自行选出不同集合。

## 6. Kimi-Linear TP=2：可借鉴但不能冒认已经接通

现存 `tests/runtime/rq5b_kimi_phase1_transport.py` 专门定义TP2 runtime_scope和 `base_port+tp_rank`；`evaluation/rq5b_kimi_phase2.py::rank_facts/BatchedClient` 对两个Rank并发RPC、对逻辑事实及每Rank映射汇聚；phase3 transport记录逐Rank恢复。历史产物含 `per_rank_state/rep*_rank{0,1}.json`。

**当前P0 bridge未导入这些Kimi transport或BatchedClient**，仍导入WP3B单卡链和R2-C0门禁。Kimi是KDA而此模型是GDN，状态形状、字节数、模型加载和完整Cold参考都不同。可以借鉴“rank端口/映射/结果汇聚”的工程方式，不能据此声称P0已具备两Rank连续控制、Strict Binding或27B正确性。

## 7. 最小修改范围（仅建议，本轮零实施）

优先在**新的独立27B/TP2实验包**接线，不覆盖Attempt5或9B配置；保留核心优化目标及Native原语。下表描述需修改/适配的接口边界，不批准放宽门禁。

|模块/函数|最小必要工作|不能省略的验证|
|---|---|---|
|新launch，参照`p0_control_cycle_gpu/launch.py::pick_gpu/docker_cmd`|双UUID共同独占门禁；只读挂载27B真实模型目录；Docker两设备及CUDA可见顺序；独立端口/输出/执行锁；TP2参数|设备0/1↔UUID↔Rank映射、双卡子进程及清理记录，不影响9B或他人任务|
|新device_entry，参照`main`|要求准确两设备，逐设备UUID/型号核验；保留spawn主入口保护|不能仅把`==1`改`>=1`；记录两个Rank的实际设备身份|
|新prepare/配置锁，参照`prepare.py`|建立27B模型/模板/Tokenizer/完整Token及引擎配置身份；工作负载仍用既定规则，重建CPU预检|不能复制9B的Cold、模型摘要、预期物理ID及状态大小当27B证据。直接Token replay可保留相同数组，但须验证其27B输入语义/词表/配置，不冒称重新渲染相同Prompt|
|TP2控制transport/连续窗口，参照WP3B与Kimi分Rank入口|两个Rank均可达；共同控制序号、全部Rank静止/Admission关闭、预检汇聚、动作分发、失败统一停止|不是简单双端口即安全；需证明控制期间不会进入不匹配的TP collective或让一个Rank提前接收新请求|
|观察/映射协调层，参照R2-C0 `observe`、`PhysicalView/Binding`、`StrictRuntimeBridge`|维护每个Rank的本地视图/绑定，逻辑一致性与分片差异分开；同一Plan在各Rank落实，汇聚全部Postcheck|可复用现有本地Strict检查，新增分布式协调包装或经审查的聚合合同；不得把缺Rank当UNKNOWN后仍执行|
|Native观测/COW/状态保护，参照P0 `bridge/observation`|逐Rank记录Native源、L/E/G、源/工作槽、组件、FA保护，结果按rid/rank归档|Tensor哈希只作同Rank前后比较，不能要求两Rank不同分片内容相等|
|Cold及CPU负面合同|全新27BTP2 Cold；缺Rank/错映射/引用冲突/单Rank失败/超时等先CPU测试，再独立GPU正确性门禁|CPU不能替代模型加载、双RankGDN恢复及输出实测|

若采用聚合包装保留现有核心本地Strict类型，至少也要有独立的**全Rank绑定与协调证据对象**，不能只复用同一个单RankPASS。若必须改核心物理类型，应单独审查，不能在此任务自动进行；优化器和三种策略公式无需因此重写。

## 8. 三策略27B正式对照之前，最少还需要什么

**需要的不是重新实现三种策略，而是建立27B TP2的统一安全执行合同：**

1. **独立部署配置与内存门禁**：两卡绑定、模型实际路径、TP2、标准状态dtype、Graph/overlap设置，实测每Rank加载/32K及既定工作负载峰值；确定静态比例、chunk与缓存容量。当前0.4配置不能直接认证。
2. **双Rank Observe–Bind–Preflight–Reconcile–Postcheck**：统一逻辑Plan、每Rank物理映射、共同连续窗口、全Rank后检和失败停止。完成前不得尝试策略驱逐。原生SGLang支持TP2不自动满足此项。
3. **27B输入及成本身份**：同样的候选/预算/已知ASG/访问声明可作为共同协议基础，但当前9B Phi不能称为27B已校准预测。若用它选择27B的FS保留集合，必须明示迁移模型；若要主张27B位置感知估价有效，需独立27B校准及验证，不调整Greedy目标来追求结果。Marconi的FLOPs/byte及状态大小改用27B真实结构、两Rank合计字节，KV的固定前缀映射按当前Token/物化状态认证。
4. **三策略最小正确性门禁**：先同一合法当前Epoch完成CPU一致性；再各策略从等价独立初始边界完成真实逐Rank协调、Native Match/COW/FA保护与全新27BTP2 Cold逐Token对照。FS当前n16仍应按既有规模合同走Greedy；MC/KV复用已有真实选择内核，不能把冻结加载计时冒充在线选择。
5. **再冻结共同性能矩阵和时间窗口**：所有Rank正确性通过后，才审查三策略配对重复、TP同步下CUDA事件与TTFT/完整周期边界，保留通信、必要控制和全部负向结果。32K容量验证与当前短请求性能是两个问题，不混成一个结果。

这五项分别需要后续授权与实际验证；本轮没有执行部署或为其预留GPU，也不自动开始下一阶段。**目前可以进入的是TP2适配与合同设计审查，不是原样运行三策略正式GPU矩阵。**

## 9. 审计身份与停止声明

- 当前Attempt5运行版runner SHA与实际文件相同：`b6cee2d3ffa04e8f85e9658bf166a37cfb063ad21983c9488b37fb8b4eaf9f84`；本轮直接读取summary为PASS，不能沿用前一日“P0尚无GPU产物”的旧状态。
- 当前 `device_entry.py`：`2d7d20881078e22c02bef80e5bdcbc276f52a6da5997ae4dab2803dbac47c4a3`；`control.py`：`b1c3a64a61720e69fe75497a09933a5b1f383b47658c3f9679e677aaab1694f9`；`flowstate/strict_control.py`：`5ffeeaeef14606f3f1d280f41fa26a575b2f10c66ea5db28832939481a3ebab3`。
- nvidia-smi及Docker socket在初始沙箱内受限；仅对上述只读命令使用了批准的权限执行。无其他用户GPU进程操作，无任何模型CUDA加载。容器为无网络、无GPU、只读文件系统、退出自动删除的源码检查容器，不是模型服务。
- 未对目录名或“27B”标签作架构判断；未用新版网络文档替代冻结镜像源码；未因Kimi历史TP2实验存在而推定P0已支持。

**最终结论：PARTIAL。模型/冻结SGLang在源码及资源数量级上具备TP2接入基础；当前FlowState正式P0路径不支持直接安全应用TP2计划，必须补齐逐Rank状态认证与协调，并另做27B部署及Cold门禁。到此停止，等待人工确认。**
