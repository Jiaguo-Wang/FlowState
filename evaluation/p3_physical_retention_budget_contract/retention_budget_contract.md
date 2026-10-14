# 物理 Retention 资源合同（设计）

本轮只读审查与 CPU 整数容量推导，未启动 GPU、Engine、Agent、Pilot、replay 或策略比较。日期：2026-10-09。

源码约定：`S/` 为 `/tmp/audit_sglang/sglang-v0517-tag/python/sglang/`，使用 P3-G0 与停止的正式实验容器逐字节核实的版本；本轮重新核验其 16 个文件摘要。其余相对路径以 `/home/wjg/code/FlowState/` 为根。新扩展仅为设计，没有实施；UNKNOWN 不当作零。

**RESOURCE_CONTRACT_NEEDS_EXTENSION**。可在现有池与 recurrent-only 原语之上增加有限的 admission/配额/事件接口；不是已经部署的资源隔离机制。

## 1. 五种量

- Physical capacity：N 个可分配 recurrent slots；普通 payload N×b，另有不可借用 padding b。实际总 HBM 单独记录。
- Persistent retention：跨受控 epoch 被授权保留的至多 K 个唯一完整 checkpoint，`B_ret=K b`。指池内可重用状态的驻留权，不声称驱逐后立即归还 GPU 驱动显存。
- Working/transient：即将批准的最多 Q 个运行请求的 writable 与 tracking 状态；普通 no-overlap extra_buffer 基本角色各Q槽。实际动态峰值还含 donation 和新缓存生成。
- Fork/COW headroom：为分支 execution/copy/donation 的峰值和 admission free 检查预留空间。纯 Agent fork 不立即制造 Q 份 GPU 状态；COW destination 与 Working 同槽，不重复计数。
- Native runtime reserve：通过既有 native admission 所需的空闲量及本轮未覆盖瞬态的保守余量；它属于同一个执行 reservation 包络，不能在包络之外重复相加。当前没有原生独立 reserve ownership。

为避免虚构逐项峰值，先定义总 reservation `R_exec`，并用事件分类核验 W/T/H/native 子项的并集是否被它覆盖。固定同池其他不可管理占用上界 O 后，`K = max(0,N−R_exec−O)`。若 N<R_exec+O，应拒绝配置，而非靠把 K 置零仍启动。若 O 含 session/未知 owner，与限定模式冲突，则直接失败。等大小 b 必须现场确认，不能把不同 TP/压缩模式混用。

## 2. 为什么候选可以多于 K

空闲阶段 W/T 尚未使用全部执行 reservation，允许真实已计算的 checkpoint 临时借用其中空闲槽，记 Borrowed lease。lease 只改变审计/配额归属，不复制状态、不填充未来节点、不凭空加内存。

边界前必须有 `C_unique + O + U_now ≤ N`，且 borrowed 不超过已获批的空闲 reservation；U_now 是实时 working/transient 并集。所有 C 真正驻留。关闭 admission 并确认 idle 后，FlowState 决策可以看到 `|C|>K`。在下一批服务承诺兑现前，reconciliation 归还临时借用：全部长期 retained 的唯一槽总数≤K，所有借用 lease 清零，free≥R_exec，FA 完全不变。没有做到这些就不放行请求。

释放之后 `|S|≤K`，并不会维持所有 C 原地驻留；未选中的 checkpoint 已失效并留有 detach/generation 日志。这正是竞争的物理含义：同一批有限槽不能既长期保留全部状态，又满足已经事前约定的执行空间。候选多于 K 不是越过 N。

**严格适用边界：**若 K 被定义为此刻所有驻留 checkpoint 的硬物理上限，则全 C 驻留推出 K_min≤|C|≤K，不能 K<K_min。上述设计采用跨 epoch 的 retention quota + 可撤销的空闲执行区借用，必须以这一本名报告，不能称为物理池硬上限实验。

## 3. 可执行的状态机与准入

1. 配置冻结：N、Q、R_exec、O 上界、每请求最大输入/输出/中间插入数量的容量界、lease 上限/有效期、barrier 触发规则，在看到 candidate/K_min 或策略结果之前确定。
2. BUILD/执行阶段：为新请求事前检查整个已批准阶段的峰值界，匹配、COW、tracking、插入都纳入。新 checkpoint 从本阶段 execution reservation 获得有期限的 lease；一个 generation 只归一个 owner。已批准的一小段执行中 lease 增长有界，不允许无限积累。
3. QUIESCE/CAPTURE：预注册阶段结束或下次 admission 前触发，Agent 所有 Ready 请求留在外部 gate；等待 GPU 和所有先前请求完成，scheduler fully idle，无 transfer/insert；一次捕获 C/R/全部槽账目。不能先让请求进入 native match/alloc 再做决定。
4. RECONCILE：只在此处确定/实现 retained set；使用原有 adapter，先验证所有 targets 的引用、身份、generation、映射与 selected 物理并集，后验证释放量/FA/驻留/账本；失败不开放 gate。Pilot 本身不执行策略比较，本节描述后续合同执行入口。
5. COMMIT/ADMIT：只有 free≥R_exec、总 persistent≤K、lease=0、无未知 allocated slot 时批准下一阶段。重新生成的状态不能偷偷永久保留在 K 之外；它们是下一阶段的有期限 lease，下一 admission 前再次结算。

这是一项**长期/边界保留上限**，不是保证任何执行瞬间 cache 节点数≤K。所有批次之间都必须重新兑现合同；不能只在第一次做一次驱逐，然后任由后续缓存绕过 quota。外部不相关 checkpoint 也必须计入 O 或全局 K，不能只统计对当前 P 有收益的候选而忽略其物理占用。

当前 transport 仅在 fully idle 时处理控制；若申请的生成长度/分块数无法给出整个活动阶段的有限峰值界，必须在发请求前拒绝/延后该阶段，不能指望中途 idle 执行 reconciliation。不可用事后修改 K 或隐式 native eviction 弥补。支持任意长度的持续生产负载可能需要更深 scheduler/insert admission hook，不属于首个 Pilot 的可证明范围。

## 4. Native eviction 的顺序与非干扰

当前存在三条会先改变 C 的路径：allocation.py:283 的 preallocation evict；MambaComponent:193/:477 的 COW/slot 分配失败后 evict；MambaComponent:248 的 path cap 自动清理。FA 分配不足也可能走原生树驱逐。新合同保持 path cap=-1、池分离且保证 FA headroom；只有在完整的事前容量检查通过时才允许进入这些执行路径。

不修改原生 eviction 算法、不把 selected lock_ref 人为设大、不静默忽略保护。已有 native eviction 事件必须记录来源和代际；若在捕获前发生，只能按真实剩余 C 描述，不能恢复旧候选列表冒充驻留。对目标无污染 Pilot，则此类事件直接使容量资格失败；发生在 capture→reconcile→admit 的保护边界更应失败并停止。记录失败不是防止驱逐的实现：未来必须补齐真正的 admission 检查和 lease 权限，不能只靠 post-hoc 日志宣称硬隔离。

## 5. K_min 与共享/专属取舍

精确 K_min_frontier 仍按 P2 的全候选最大前沿覆盖定义。合法出现 K<K_min 的必要条件：真实 lineage/pending；全部 C 在 N 内已物化；原生状态有效；K 来自独立的资源服务合同；临时借用可安全撤销；所有 native/working/COW 引用与 generation 正确；释放不与 selected 物理并集相交。

单纯 K<K_min 不保证是“共享祖先 vs 专属”取舍，还需同一个共同祖先和至少两个更深专属项具有重叠收益覆盖。两 pending 各有不同的最深 checkpoint 时 K_min=2；若共同最深 checkpoint 可覆盖两者则 K_min=1。一般总有 K_min≤有效 pending 数，因为每个 pending 至多选一个最深 checkpoint 即可覆盖。

因此两个 pending 要严格容量不足，非零 retention K 必须为1。不能为此先设 K=1，再倒算一个虚构 reserve 或任意很小的 N。当前没有独立业务/硬件合同支持本两分支任务的 K=1；是否能事先绑定这种配置为 UNKNOWN。即使以后有合法 K=1 仍不保证 FlowState 胜过 PCB/LDR/WFC。

## 6. 最小缺失接口

- 在任意 native match/COW/alloc 之前的全 ingress admission 权限及 quiescence ACK。
- 全 pool 的 canonical ownership/generation 与 Retained/Borrowed/Working/Tracking 账本；现有 request_generation 不等于 recurrent generation。
- 明确 N、实际 b、free/峰值、native evict 来源的联合 capture；全槽账目不得只列候选。
- `commit_retention_contract`：验证全局 persistent≤K、lease 清零、free reservation，并以 barrier/配置摘要约束下一次 admission。
- 有限活动阶段的峰值可验证上界与超限拒绝。不能在活动状态上调用现有 idle-only eviction。

这些可先以独立 Engine/control/Agent adapter 扩展，复用原 allocator 和 eviction 操作；本轮未改实现。缺少这些接口时只能说逻辑预算实验，不能说资源合同已经强制执行。
