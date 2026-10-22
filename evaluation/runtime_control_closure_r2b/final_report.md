# Phase R2-B：严格计划物理绑定与安全协调合同

## 门禁：READY（仅 CPU 级安全接线）

StrictAllocationPlan 已可通过正式 `StateController.reconcile_strict` 进入独立严格协调路径。当前实现覆盖完整物理绑定、统一 Preflight、逐次单组件驱逐、每步及最终 Postcheck、失败账本和 fail-stop。CPU 测试通过，不表示 GPU 实际安全性已验证，也不授权正式 Runtime A/B。

本轮没有启动 GPU、SGLang、Docker、模型服务，没有实际驱逐 Recurrent State。使用 g001/K2 的冻结逻辑计划作为主要模拟输入，包含共享祖先 B1 与专属深层 B3；物理 Node 从 10000 开始、Slot 独立构造，绝非历史 Snapshot 中 Node/Slot 的复用认证。

## 1. 修改文件与正式接口

- `flowstate/strict_control.py`：新增不可变 PhysicalNode、PhysicalView、BoundCandidate、PhysicalBinding；以及 validate_plan、validate_view、bind、preflight、postcheck、StrictReconciler、StrictRuntimeBridge。
- `flowstate/controller.py:115`：新增静态显式入口 `StateController.reconcile_strict(epoch, plan, runtime)`，不改动 Legacy reconcile 的选择或驱逐逻辑。
- `tests/test_strict_control.py`：28 项纯 Fake Runtime 安全测试，包括正常流程、驱逐前拒绝、故障原语和部分失败。
- 本目录 `record_evidence.py`：记录实际 CPU 模拟执行和当前源码/输入 SHA-256。

R2-A 的 strict_epoch.py、optimizer.py 及 tests/test_strict_epoch.py 保持不变，并核对其 R2-A 身份摘要。Phi、Legacy 兼容谓词、Baseline 和冻结数据没有改变。

## 2. 控制链与隔离边界

```text
R2-A StrictAllocationEpoch + StrictAllocationPlan
 → StateController.reconcile_strict
 → StrictReconciler
   → runtime.control_epoch：一次性、连续隔离、禁止 Admission
   → observe 当前完整物理视图
   → bind：校验计划并绑定全部当前候选
   → observe + preflight：重新核验完整视图与绑定
   → 对逻辑补集按原计划顺序逐项执行：
       observe + 验证前面已完成操作之外的状态不变
       RuntimeCheckpointHandle（当前 Node、完整 Token、预期摘要）
       StrictRuntimeBridge.evict_mamba_only
         → 注入的现有 Adapter.evict_mamba_only
       observe + postcheck
   → 最终 postcheck
 → 返回关联 epoch/plan/binding 的账本
```

**没有 GlobalOptimizer.select、plan_strict 或旧 Workflow-ID/Lineage 谓词的重新调用。** validate_plan 仅按严格关系重算给定选集的 T/E/G 与成本，检查计划一致性，不执行选择。CPU 测试在进入协调后将 Legacy 选择和谓词设置为调用即失败，正常流程仍通过。

`StrictRuntimeBridge` 复用现有 Adapter 的 `evict_mamba_only(handle)` 接口，不重写 SGLang 原生 detach/free。测试注入 FakeAdapter；真实运行可以注入 SGLangAdapter，但还必须接入真实的 observe、control_epoch 和 invalidate 回调。本轮没有安装新的 scheduler 传输入口，也没有声称仅凭桥接类存在即可安全启动真实 Engine。

现有 `tests/runtime/rq6_batched_control_transport.py` 是参考：统一当前 view 和后验检查。没有直接复用其只接受 selected_ids 的 RPC，因为它未绑定严格计划、K、完整别名检查及本轮连续隔离合同；本轮在独立严格路径复用单组件动作，保留 Legacy 批量实现不变。

## 3. 当前物理绑定合同

PhysicalBinding 保存：epoch ID/digest、完整 plan digest、当前 Engine/Barrier/lease、绑定视图摘要、K、selected/evicted IDs，以及每个候选完整 Token、当前 node、slots、组件切片身份、驻留和保护状态。

绑定匹配使用严格候选完整 Token 前缀与当前观测节点的精确数组相等；必须唯一命中。历史节点号和槽位不参与绑定。组件身份必须是每个槽位对应的存储切片身份，不能使用整个共享池对象的 ID，否则无法区分切片所有权。当前采用严格一对一策略，任何候选别名或组件别名均拒绝，不实现通用引用计数。

PhysicalView 的 nodes 包含全部非哨兵缓存节点；根节点、完整 FA 树和其他非目标状态由 attention_digest/non_recurrent_digest 覆盖。存在任意候选域外 resident recurrent node 即拒绝，采取比“只拒绝影响当前请求者”更保守的准入。当前合同为单池、正整数可分配 Slot 1..P、完整 Conv/Temporal 状态、无工作占用的静止边界；TP 多 rank、host backup、非连续控制或允许共享组件的部署不在范围内。

**没有 Slot Generation 时的边界：** lease 不是代际计数，更不是通用 ABA 认证。运行时提供者必须保证从 bind 开始至 postcheck 结束连续禁止 Admission、native 状态变化和其他控制操作。observe 的 idle/admission 标记只能作为证据字段，不能代替实际调度隔离。若真实 Runtime 无法提供此能力，R2-C 必须停止，不能填 true 强行通过。

## 4. 驱逐前统一 Preflight

以下均在第一项破坏性操作之前检查：

|检查|实现与拒绝方式|
|---|---|
|计划与 epoch/source/K 一致|validate_plan 对比完整身份、来源摘要、K 和目标成本；不重新选择|
|选集合法、数量不超预算|唯一选中 ID、域内、selected_count 一致且不超过 epoch.K|
|驱逐补集恰好正确|与候选全集减选集的排序元组一致，逻辑 Keep/Evict 不重叠|
|全部当前映射存在|每个候选包含选中项均必须唯一映射到完整驻留节点|
|物理互斥|所有 node、slot 和组件切片标识唯一，并显式检查 Keep/Evict 交集为空|
|组件完整与保护|Conv/Temporal 完整、Attention 驻留、refs=0、无 protection/host copy|
|当前视图没有失效|重新 bind 当前观测，必须与原绑定完全相等；Engine/Barrier/lease/config 均匹配|
|状态可安全观察|idle、admission_closed、无 pending transfer；视图为不可变结构|
|完整候选域|全树 resident recurrent 节点集合恰等于候选绑定集合|
|Slot 记账|resident/free 不重叠、不重复、不越界，二者并集恰为池容量|

因此，输入已经显示 alias、stale view、保护或缺组件时，操作列表必须为空。物理池有空闲槽不会跳过逻辑驱逐：成功模拟以 12 个候选、16 个空闲槽开始，K=2，仍执行 10 项逻辑补集驱逐，最终只有 B1/B3 保留。

## 5. 每步与最终后置条件

每个驱逐后以及下一次驱逐前，重新检查当前 scope、Engine/Barrier/lease、配置、全树节点集合及记账。相对于绑定前视图：

1. 已完成驱逐的目标 node 必须无 slots、无 Conv/Temporal 可恢复组件，但保留 Token/Attention。
2. 选中及尚未驱逐的节点、非目标节点逐项不变。
3. attention_digest 与 non_recurrent_digest 不变；池容量不变。
4. 空闲 slots 恰为原空闲集合加已完成释放的 slots；不允许额外释放、丢失或重复。
5. Managed resident 集合恰为全集减已完成项；最终恰为 Selected，且数量不超过 K。
6. 账本包含 plan digest、完整 binding、最后视图、完成列表、阶段及失败对象。

这些是 **CPU 合同测试通过**，不是物理 GPU 后置条件通过。实际 FA 摘要的覆盖范围、组件切片身份可靠性、原生释放行为和真实输出都仍需 R2-C 验证。

### 是否存在先驱逐、后发现 Selected 被破坏的路径？

**要区分原因。** 由输入中已知的 Node/Slot/组件别名或失效引起的错误映射，在统一 Preflight 就拒绝，不会先驱逐再检查。

但如果底层原语违反合同、实际释放了非目标 Selected，或隔离提供者允许未授权并发变化，任何纯前置检查都不能保证阻止该外部错误。测试专门注入“原语破坏 Selected”及“FA 意外变化”，立即 Postcheck 检出、停止后续操作并使生命周期失效。不能将该检测称为撤销损坏或通用物理安全证明。

## 6. Fail-stop 与失败账本

Preflight 失败、操作失败、部分成功、期间视图变化、Postcheck 失败均抛 StrictReconciliationError，不返回成功。

在已经进入的连续控制上下文内先采集失败视图、调用 invalidate，再退出上下文；避免先解除 Admission 才标记失效。桥接自身一次性消费、失效后禁止执行；运行时 invalidate 回调还须对整个 Engine 生效，防止换一个桥接对象重用旧状态。异常上下文不得吞异常，失败退出必须保持隔离直到上层清理。

账本区分：completed_evictions 是已经返回成功的操作；failed_checkpoint_id 可能在抛异常前已经部分改变状态，不能因其不在 completed 中就认为未发生变更。故记录 failure_view、阶段、错误原因、cleanup_required，并且不自动重试、不回滚、不继续提交请求。失败视图读取或 invalidate 自身异常同样保留错误字段，交给上层终止；不能视为成功清理。

`integration_trace.json` 含正常流程和第 2 项驱逐失败的实际模拟记录：第 1 项完成后立即停止，没有第 3 项调用。另有“驱逐已变更后才异常”的测试，证明失败账本不把操作异常解释成无变更。

## 7. CPU 测试与回归

最终结果：**118 passed，16 subtests passed**，见 tests_final.log；其中 28 项新增物理合同安全测试。safety_tests.json 列出测试名称和结果。其余覆盖 R2-A、Legacy Optimizer/Controller、E/G、成本模型与多 Workflow 分配回归。

新增覆盖：正常共享祖先/专属分支计划；Slot/Node/组件别名；选中项缺映射；未选中项保护；引用/host copy；epoch 摘要、K、选集域与补集；缺组件/空 Slot；View、Engine 身份变化；活动请求、Admission、传输；域外驻留；FA 破坏；部分失败；期间外部变化；原语静默 no-op；原语破坏 Selected；失败操作已部分变更；超预算；记账错误。

R2-A 16 个正式条件继续按冻结 B4-C 核对选中集合、顺序、逐步边际、T/E/G/成本；12 条跨 Workflow 边仍保留。没有重新运行 B4-C 正式比较或 Exact，没有改变原预算，也没有生成新的性能结果。

执行环境为 uv312，使用已存在本地缓存 pytest，禁用插件自动加载；无依赖安装。可复核命令：

```bash
PYTHONPATH=/home/wjg/.cache/uv/archive-v0/Wt-t6EKpnGZnOzKw/lib/python3.12/site-packages \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/wjg/uv312/bin/python -B -m pytest \
-q -p no:cacheprovider tests/test_strict_control.py tests/test_strict_epoch.py \
tests/test_optimizer.py tests/test_controller.py tests/test_executable_state.py \
tests/test_recovery_model.py tests/test_recovery_model_freeze.py tests/test_multiworkflow_allocation.py
```

## 8. 新版本身份与产物

- `source_identity.json`：当前核心、Adapter、测试、参考 Transport、冻结四快照、B4-C 结果和旧锁文件的实际 SHA-256；独立的新 CPU 合同版本身份。
- `safety_tests.json`：CPU 门禁及测试日志摘要。
- `integration_trace.json`：模拟当前物理视图、绑定、逻辑计划、操作和故障账本。
- `record_evidence.py`：只执行 CPU Fake 的归档入口。

旧锁和旧结果均未覆盖。B4-B1/B4-C Manifest 内容核验通过，R2-A 记录的生产 optimizer、strict_epoch 和测试摘要不变。本轮 controller.py 的新增入口由新 source_identity 记录。此身份文件不代替 R2-C 的真实模型/镜像/Runtime 版本锁。

## 9. R2-C 前仍需补齐的真实证据

1. 将 StrictRuntimeBridge 的控制上下文接到实际 scheduler 安全点，证明持续禁止 Admission/native 状态变更，失败保持隔离；不能仅传入标志。
2. 当前 census 转换必须提供全部非哨兵节点、slot 与组件切片身份、配置、全 FA/非目标摘要及可核验记账；不得复制历史绑定。
3. 将桥接的单组件驱逐明确连到当前 SGLangAdapter，并在真实生命周期验证本轮全部 Preflight/Postcheck。
4. 核验选集物理驻留、FA 保留、实际 match/COW、请求恢复前沿和 cold reference 输出；保持 EPOCH_ADMISSIBLE 与实际恢复等级分离。
5. 新建真实执行版本锁、异常清理与 GPU cleanup 门禁。真实条件不满足时 fail-stop，不放宽合同，不直接开展策略性能 A/B。

**结论：READY 仅表示纯 CPU 安全接线可进入下一阶段的真实正确性验证准备。真实桥接提供者、隔离能力及 GPU 后置条件尚未认证；本轮停止，不自动进入 R2-C。**
