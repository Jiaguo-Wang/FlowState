# admission_baseline 字段、操作与消费者矩阵

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
