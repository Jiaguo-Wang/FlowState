# Serving 控制接口消费者最小化与CPU门禁

**裁决：READY_FOR_GPU_VALIDATION。** 范围严格限定为 `admission_baseline` 的响应投影：保留完整生产者、全部内容SHA与同步、现有逐项控制及请求后认证，只省略没有当前在线消费者的嵌套返回副本。CPU门禁通过，可以进入另行授权的最小GPU验证；这不是生产Serving认证，也不是性能收益结论。

**关键限制：本轮没有减少GPU内容读取、FA保护范围或完整基线的构建工作。** 没有足够证据证明这些生成过程可被元数据替代，因此选择先验证消费者接口，而没有以安全标准换取速度。新增响应身份校验也有CPU成本，实际收益未知。

## 1. 范围、来源和证据身份

读取最新 `control_path_separation_20261101` 的 SUCCESS_FULL_CYCLE 及 `control_path_remaining_attribution_20261101` 的 VALIDATE_SERVING_PATH，使用前者 `arms/t{1..5}_{fs,mc,kv}/run/` 的15个真实响应与请求后状态。没有读取失败批量优化的性能数据来推导收益，没有混入Legacy时间。

- Frozen来源：Qwen3.5-9B / SGLang v0.5.17，C/K3，n=16、m=4，三策略各五次历史运行。
- 本轮仅CPU：读JSON、源码和既有CPU合同；不启动Docker/Engine，不查询或占用GPU，不安装依赖。
- 输入及源码摘要见 [evidence_manifest.json](evidence_manifest.json)；原桥接文件初始摘要见 [source_manifest.json](source_manifest.json)。历史Node/Slot仅用于离线接口复核，不作为新Engine绑定。
- 逐响应身份、选集、E/G/J及投影核对见 [frozen_response_verification.json](frozen_response_verification.json)。本轮没有重算或修改三策略公式。

## 2. 完整消费者矩阵

范围：`control_path_separation_20261101` 最新 SUCCESS_FULL_CYCLE 的实际执行链。全仓库字段引用文件清单见 `baseline_reference_files.txt`。下列行号指原始版本；新版本最小差异见 `implementation.diff`。旧 E1、B5-D3 的同名接口不是本次精简接口的通用消费者，不能替换其实现。

|字段/操作|生产者|直接、嵌套或异常消费者|安全职责与隐式依赖|Serving 必须性|本轮处理|
|---|---|---|---|---|---|
|`admission_baseline.nodes[].node_id/token_ids/slots/full_present/references`|bridge.py:51–72，全树 `_tree_items`、向父节点拼完整 Token、读取 FULL/MAMBA、三种引用|最新 runner 无嵌套读取；`_cache_state` 内将节点转成 selected_payload 的 view；完整结果写盘/编码|循环拒绝 bigram；类型转换可报错；节点与槽位读取为组件内容检查提供位置|生成过程保留；此嵌套响应不是当前在线决策输入|原生产过程不变；精简响应省略；投影前追加 Node/Token/Slot/引用与 `after` 对照，保留根哨兵特例|
|`semantic.tree`|bridge.py:70–72 → `probe._global_maps`|semantic构建和证据序列化；最新 Admission 不读取|全树结构观察；原消费者协议曾用于跨检查对比|不能把生成操作整体视为冗余|照常构建，仅省略返回副本|
|`semantic.accounting`|同上 → `probe._accounting_snapshot`|同上；Strict协调另有独立 PhysicalView 账本|池/组件记账观察，不替代 Strict 的完整 resident/free partition 校验|现有生成保留|仅省略嵌套响应|
|`semantic.recency_rows`|`barrier_fa_frontier_control.py:152–213`|档案；旧 E1 Admission 对整份 semantic 比较|读访问时间、创建时间、hit_count；不更新 LRU|原生成保留|仅省略嵌套响应|
|`semantic.mamba_lru_order_mru_to_lru`|同上 → `lru_node_order`|档案；旧协议整体 semantic 消费|遍历有环/哨兵/节点索引不一致会抛错|隐式异常检查保留|不删遍历，仅省略返回|
|`semantic.reference_rows`|同上|档案；旧协议整体 semantic 消费|读取 FULL 与 MAMBA 的 lock/host/session ref；转换错误仍拒绝|现有生成保留|仅省略嵌套响应|
|`semantic.full_evictable_leaf_ids`|同上|档案；旧协议整体 semantic 消费|反映 Attention 叶节点集合|现有生成保留|仅省略嵌套响应|
|`ns`|bridge.py:73，monotonic_ns|档案；不是租约，也不是 T0/T3/T4/T5 的边界|采样时点，不能证明跨 RPC 新鲜度|不是本链在线输入|保留采样，精简响应不返回；不改原计时区间|
|`selected_payload[cid][].sha256/bytes`|bridge.py:79 → r2c1a.selected_payload|此嵌套字段仅档案；runner.py:369–371 实际比较顶层 `selected_payload_after`|实际 Conv/Temporal 内容读取及 `.cpu()` 同步；缺失映射/非法切片可能报错|没有证明可跳过此读取和同步|完整计算保留；投影前额外与顶层内容基线相等检查；仅省略嵌套副本|
|`selected_components[cid][]`|bridge.py:80–87 → r2c0.slice_identity|runner.py:370 与 post.state.selected_components 比较；当前唯一直接读取的 baseline 嵌套值|身份含 storage base/byte ranges/slot/shape/stride/dtype/device；连续布局和越界检查|必要|原样保留并复制；投影前与当前 `after` 的组件身份一致性检查|
|`fa_map`|bridge.py:90–92 → E1.control.fa_map → B5-D3.control.fa_map|此副本仅档案；runner.py:373–375 实际读顶层 fa_map_before|索引长度、完整 Token 前缀到索引唯一性检查；读索引张量可同步|尚未证明可去掉生产者|完整构建保留；投影前与顶层索引比较；仅省略嵌套返回|
|`fa_protected_payload[].layer/component/sha256/bytes`|bridge.py:93 → fa_payload(protected_indices)|此副本仅档案；runner.py:372 实际比较顶层 fa_protected_payload|HybridLinearKVPool 类型、禁止量化、两层传输计数为空、index_select 实际可访问性、内容读取及同步|这些隐式拒绝条件及读取不能直接删|完整计算保留；投影前与顶层保护内容比较；仅省略返回副本|
|`TRACE.audit[].kind/ns`|bridge.py:95–100|runner.py:377–380，恰好两次完整 hash 审计；post_request 档案|证明本实验没有将全量内容读取插入正式请求测量窗口|实验有效性门禁仍保留|仍两次完整生产者调用，不伪造 metadata 计数，不修改门槛|
|`post_apply_baseline_ns`|bridge.py:223–225|报告/剩余开销分析|覆盖完整 baseline 及新增投影，不是纯 SHA|真实计时记录|起止位置不动，不减估算审计时间|
|完整响应编码、SHA、压缩、Base64|codec.py:17–76|runner.AuditedControlClient → call_saved → unpack|包长度/摘要/重复JSON键/尾流/格式损坏拒绝|传输完整性保留|codec 原样；`full_evidence_preserved` 只表示对**传入 codec 对象**无损，不代表精简对象含旧完整 baseline|
|raw/json 写盘、flush/fsync|B5-D4a.persistence.call_saved；runner.py:134|成功前持久化，异常写 exception；runner 顶层失败与 finally shutdown|不得丢失失败响应，不得持久化失败后继续 Pending|本实验合同保留|原样；CPU 注入 fsync 失败及损坏响应验证拒绝|
|格式化 reconciliation 写盘与 oracle|runner T4 后|审计报告与最终认证|不是在线选择依据，但完整实验仍需|取证职责|全保留；Serving 档案诚实保存精简响应；shadow_validation 同时保存完整基线|

## 非直接消费者与适用范围

- 原始 `BASELINE` 指向 apply 的整份 result。`inputs()` 确认 scope 和单次 Admission，实际调用当前包 `protocol.admission_gate(BASELINE, None, received, expected)`；该函数只比较完整 Token。未读取 baseline 的 semantic/nodes。此事实**不等于**建立跨 RPC 连续租约。
- `b5d4_state` 使用相同 `cache_state` 再次返回全部内容；本轮未精简它。runner 的保护内容、FA Map、COW/Native/Cold 门禁均保留。
- 旧 `lightweight_runtime_e1/protocol.py:7–8`、bridge.py 的 Admission 会消费 `nodes/semantic/selected_components`；旧 B5-D3 也有完整 semantic/nodes 消费。这些入口不得导入精简响应，本原型仅在独立副本中显式启用。
- 原 apply 内 `prior.scope`、Strict Epoch认证、真实 FA frontier、29次协调观测、Bind/Preflight/逐驱逐Postcheck、FA前后SHA、选中组件前后SHA完全没有移动或精简。它们不能因 baseline 字段未被消费就省略。
- 异常路径：任一原生产者异常仍向上传播；`FS_SEPARATION_MODE=serving` 的原 wire_dispatch 设置 `_r2c0_failed`，维持失败隔离。客户端先持久化 wire response，再解码和接口验证，验证失败不会发送 Pending；原 finally 清理仍执行。失败后不回退 Validation 或重新协调。

## 操作分类

A：真实 scope、FA frontier、Epoch/兼容性、映射/组件/引用/别名/容量、绑定/逐项控制、Native/COW 及必要内容保护。

B：目前承载隐式断言或同步的 baseline 内容读取、树遍历、SHA。本轮均保留；没有建立替代检查前不能删。

C：重复的嵌套完整响应、完整展示和档案。只将 baseline 的六类无直接在线消费者字段从 Serving **返回值**省略；完整 Validation 保留，其他日志/fsync不动。

## 3. 隔离实现与文件变更

|文件|改动与职责|
|---|---|
|`interface.py`|新增纯CPU接口合同：模式、当前响应元数据身份、原Strict PhysicalView校验、保留组件映射、完整基线投影及客户端校验|
|`bridge.py`|独立副本；仅在原位置以 `interface.build` 包装同一个 `cache_state`；精简模式必须使用现有异常隔离包装|
|`runner.py`|独立副本；持久化及解码后、发送任何Pending前验证接口；报告写明基线模式。Native/COW/Cold/FA和两次审计断言不变|
|`device_entry.py`|独立副本；只改变 runner 模块目标，GPU单卡/UUID/H100门禁原样保留；本轮未执行|
|`control.py`、`observation.py`、`protocol.py`、`codec.py`|隔离副本与原文件逐字节相同；没有新求解器或Runtime实现|
|`test_interface.py`、`verify_frozen.py`|新增CPU合同、真实冻结响应核对、证据摘要与复核入口|

全部新增文件位于本目录。完整修改见 [implementation.diff](implementation.diff)，操作说明见 [README.md](README.md)。生产 Optimizer、Controller、Strict Epoch、Strict Control、Adapter、SGLang及旧实验均未修改。已有scope、FA前沿、计划构造、13项逐驱逐及29次视图检查仍由原代码执行。

## 4. 新旧调用链与数据流

原链：

`Scheduler同步dispatch → 原control.apply（Observe/Select/Bind/Preflight/Reconcile/Postcheck/SHA） → cache_state完整基线 → 完整响应 → 原codec → call_saved落盘 → 解码 → 四Pending → 请求后全量认证/Cold/Oracle`。

新链：

`同一dispatch及原apply → interface.build → 同一个cache_state恰一次 → 模式分派 → 原codec → 原call_saved落盘 → 解码 → validate_response → 原四Pending与全部原后验门禁`。

|模式|返回对象|证据与行为|
|---|---|---|
|默认 `validation`|原7字段 `admission_baseline`|原完整对象保留，不移动任何验证或内容读取|
|显式 `serving`|`schema`、`context_sha256`、`selected_components`|完整生成后才投影；投影前复核当前组件、Node/Slot/引用、FA索引与内容摘要；客户端按当前响应验证身份|
|显式 `shadow_validation`|原完整基线及 `admission_baseline_serving` 旁路|同一次完整生成产生两份接口，检查等价，完整认证证据仍可独立保存；不能作为低审计性能样本|

`FS_ADMISSION_BASELINE_MODE` 与原 `FS_SEPARATION_MODE` 各管一层；精简和双接口模式要求后者为 `serving`，否则在Engine前拒绝。禁止未知模式或损坏响应回退。

`context_sha256` 覆盖当前响应已有的计划与物理元数据，**只防止混用响应**；不是GPU内容摘要，不是新鲜度租约，不是Slot Generation。必要内容保护仍由原实际SHA承担。`validate_view` 使用原生产物理合同检查已返回的元数据；它不冒充新的实时Observe。

完整JSON编码口径下，15个历史基线从 **2,059,170–2,059,187字节**投影为 **606字节**。这只是该嵌套对象的离线序列化体积；外层仍保留原完整Epoch、账本与其他审计，不能据此推断整包大小、RPC节约或GPU性能。

## 5. CPU正确性门禁

最终命令与完整输出见 [cpu_tests.log](cpu_tests.log)、[cpu_results.json](cpu_results.json)，环境见 [cpu_environment.json](cpu_environment.json)。

- **79项 unittest 全部通过**：新增接口测试25项；复用 Strict Control/Strict Epoch 测试54项。子测试故障组合不冒充独立实验。
- **15/15真实冻结响应通过**：三策略×五轮。精简前后Epoch、候选、兼容矩阵、预算、选集、Pending E/G/J、物理后验及其他顶层结果完全不改；保留组件身份与冻结请求后完整状态逐项相同。
- 默认Validation完整对象与原对象一致；shadow同时保留全部7字段并与投影核对。所有模式恰一次调用原完整生产者，异常不吞、不自动回退。
- 拒绝Epoch/Plan摘要、K或配置不一致、生命周期响应混用、Admission未关闭、传输中、缺失Conv/Temporal、错误Node/Slot、活动引用、主机副本、保护冲突、Node/Slot/组件别名、容量与空闲账本不闭合。
- 拒绝新的FA索引或保护内容变化、选中Recurrent内容变化、必要组件字段缺失、未知schema、混合旧字段、损坏传输及缺失完整基线字段。
- 复用生产Strict合同负例核验UNKNOWN、缺边/重复身份、过期映射、未托管状态、FA变化、部分驱逐失败、原语静默无操作及破坏Selected。模拟故障保持Fail-stop，不继续后续驱逐。
- 真实持久化函数的CPU注入确认：fsync失败不继续发送请求；损坏原始响应先落盘并保存异常后拒绝。包装源码核对确认持久化先于解码/接口验证。
- AST/字节核对确认 `_cache_state`、`cache_state`、Admission输入钩子、Native Match/COW安装、idle失败隔离、完整control/codec/observation/protocol没有改动。设备入口仅变更目标包名。

这些是离线合同与模拟Runtime证明，不是本轮GPU物化、Native Resume或Cold的新增实测。旧响应的真实GPU成功只作为冻结输入来源，不能转移成新原型的GPU PASS。

## 6. 仍不能消除的安全依赖

1. 基线内容生成的 `.cpu()` 同步、FA池类型/量化/传输检查、索引可访问性、组件切片范围和LRU链一致性尚未被等价元数据机制替代，全部保留。
2. 完整FA后验、Selected内容前后保护、请求后FA保护及两次完整hash审计均保留。消费者未读取某个嵌套摘要，不意味着摘要的生成过程无职责。
3. 原连续控制语义仅来自同一Scheduler串行同步回调；模式/schema/摘要不能建立隔离。apply结束到后续RPC的原受控协议没有被升级成跨窗口租约，完美ABA仍未验证。
4. 原型仅适用于最新四Pending测量链。旧E1/B5-D3的Admission消费者需要完整nodes/semantic，不能替换为本精简接口。
5. 原本请求前执行的校验没有延后；Journal、持久化、完整请求后Validation及Oracle没有挪动。T0–T5和原区间起止不改，新增投影如实计入RPC尾段及完整周期。
6. `PROVENANCE_UNVERIFIED` 不变。组件身份、Token兼容或Native匹配不构成跨Workflow张量血缘认证。
7. 不存在性能结论。当前仍是完整内容认证加精简响应；可能减少传输/重复序列化，也可能被新增元数据核验抵消，须实测。

## 7. 下一步最小GPU验证计划（未执行）

需要另行授权。先核对本目录代码摘要、原模型/镜像/输入和CPU门禁，再按原独占GPU UUID门禁检查；不满足即停止，不干预其他任务。

最小两次独立生命周期，沿用冻结C/K3和原四Pending顺序：

1. `shadow_validation` / FlowState：新的实际Node/Slot重新绑定；同一回调中生成原完整基线和精简旁路，核对组件、Strict Epoch/Plan、E/G/J、13次真实驱逐、FA保护、Native Match/COW、8 Token Cold及资源清理。任何失败停止。
2. `serving` / FlowState：在等价新生命周期重复同样的真实控制与四Pending门禁，保留完整请求后Validation。与第一臂只比较接口差异，不能把shadow额外检查差值解释为生产收益。

仍使用现有计时边界完整记录 baseline、Control RPC、TTFT、T0–T4及T4–T5，不从旧数值减审计成本。两次只用于正确性和计时可采集性，不能形成稳定性能主张；本轮不授权三策略正式矩阵。进一步删除SHA或构建步骤须另有隐式检查/同步的等价证明。

**本任务到CPU接口门禁结束，未启动GPU，未自动进入下一阶段。**
