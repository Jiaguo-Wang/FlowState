# Phase R2-A：严格 Compatibility 生产分配接线

## 门禁：READY

生产 GlobalOptimizer 已通过显式 StrictAllocationEpoch 生成纯逻辑 Allocation Plan。g001–g004、K=1–4 共 16/16 冻结条件全部一致；12/12 跨 Workflow 准入边保留。最终 CPU 测试为 **90 passed，16 subtests passed**。没有启动 GPU、SGLang、Docker 或模型服务，没有调用物理驱逐，也没有更改历史矩阵、Phi、Baseline 或冻结结果。

READY 仅对应 R2-A 的生产规划接口，不代表 R1 整体 Runtime 闭环已解除阻断，更不代表 Reconciliation 或实际恢复成功。

## 1. 修改范围与接口

|文件|函数/类型|作用|
|---|---|---|
|flowstate/strict_epoch.py|StrictAllocationEpoch、StrictCandidate、StrictPending、StrictEdge|新增不可变 epoch、身份、输入与完整三态矩阵合同|
|同上:101|validate|唯一 ID、完整笛卡尔配对、合法域、位置、Token/配置/边界/内容摘要和状态一致性；UNKNOWN 拒绝|
|同上:149|from_snapshot_bytes|消费 B4 冻结格式，要求调用方传入来源清单中的可信 SHA-256，不重建 Token|
|同上:175|frontiers|仅依赖 EPOCH_ADMISSIBLE 边计算 E，不调用 Workflow/Lineage 谓词|
|同上:62–84|PendingPlan、StrictAllocationPlan|输出身份、K、选中顺序/数量、每请求 T/E/G/成本、总 J、收益及逻辑驱逐补集|
|flowstate/optimizer.py:36|select|保留 Legacy 路径及原 `_recovery_cost` 扩展接口，拒绝 Strict 包装对象混入|
|同上:52|plan_strict|新增真实生产严格入口；不使用 B4-C MatrixOptimizer 子类|
|同上:78|_select_candidates|抽出原 Greedy 循环，两路径共用；排序、同分容差、停止和预算换算不变|
|tests/test_strict_epoch.py|StrictTests|26 项测试方法，包含 16 个冻结条件子测试及负例|
|本目录 regression.py|main|输出独立回归明细、计划及完整性记录，不执行物理协调|

生产修改只有新增 strict_epoch.py 和 optimizer.py 的入口接线与共享循环提取；未修改 Controller、Adapter、state_catalog.is_compatible、Phi 或其他策略。Legacy 仍可通过旧评分路径及已有子类工作。新的 Strict 路径不依赖覆写 `_recovery_cost`。

## 2. 实际严格数据流

```text
冻结 snapshot 原始 bytes ＋ 可信清单 SHA-256 ＋ K
 → StrictAllocationEpoch.from_snapshot_bytes
 → 不可变候选/Pending/完整配对及配置/边界/内容摘要校验
 → GlobalOptimizer.plan_strict(epoch)
 → 再次 validate
 → _select_candidates，共用原 Greedy
     → epoch.frontiers，只读取 EPOCH_ADMISSIBLE 边
     → G=T−E → 原 RecoveryCostModel.estimate → Σ成本
 → StrictAllocationPlan
 → 停止：无 Controller、RuntimeHandle 或 Adapter 调用
```

严格边为 REJECTED 时不参与前沿，其单候选恢复收益为零。输入保留原 Workflow ID 与 Lineage，没有归并或伪造身份。严格入口测试将旧 `executable_state.is_compatible` 与 Legacy `_recovery_cost` 替换为调用即抛异常的哨兵，16 条生产回归仍通过；另有 Legacy 单独拒绝该跨 Workflow 候选、Strict 保留其前沿的对照测试。

原始 Snapshot 摘要覆盖整个输入，包括候选物理证据；epoch 内容摘要覆盖规范化输入和 K。各候选、Pending 绑定相同 barrier/config；每条边限定在当前域，重复、冲突、缺失、UNKNOWN 都失败。位置及精确 Token 条件与实际数组重新核对。导入缺字段会显式异常，不会回退 Legacy。对象使用冻结 dataclass 和 tuple；修改 K 或其他内容而不重新构造一致合同会导致摘要失败。

**信任边界：** SHA-256 是内容一致性检查，不是签名或真实性认证。调用方必须从已验证来源清单获取 expected_sha256，不能把任意输入的即时哈希当作 Runtime 认证。导入器消费已有完整准入证据，不再次采集 GPU 张量，也不能证明历史 Node/Slot 在新生命周期仍有效。直接构造 StrictAllocationEpoch 的调用方同样承担证据来源责任；validate 检查合同一致性，不认证物理事实。

## 3. 全部正式回归结果

单位 modeled ms。表中短名按本组完整 ID 缩写，如 g001 的 B1 表示 RQ3_G001_B_TURN_001。完整 ID、补集在 regression.csv，全部 T/E/G、逐请求成本及 epoch 摘要在 plans.json。

|组|K|生产选择顺序|生产 J|冻结 J|结果|
|---|---:|---|---:|---:|---|
|g001|1|B1|334.563447662|334.563447662|PASS|
|g001|2|B1 → B3|224.359648669|224.359648669|PASS|
|g001|3|B1 → B3 → C3|136.436789948|136.436789948|PASS|
|g001|4|B1 → B3 → C3 → A3|68.218394974|68.218394974|PASS|
|g002|1|B1|545.999749991|545.999749991|PASS|
|g002|2|B1 → C4|363.184899992|363.184899992|PASS|
|g002|3|B1 → C4 → A4|223.086373602|223.086373602|PASS|
|g002|4|B1 → C4 → A4 → D4|82.987847211|82.987847211|PASS|
|g003|1|A5|959.964932377|959.964932377|PASS|
|g003|2|A5 → B5|552.288450383|552.288450383|PASS|
|g003|3|A5 → B5 → C5|193.885055650|193.885055650|PASS|
|g003|4|A5 → B5 → C5 → D5|0.000000000|0.000000000|PASS|
|g004|1|B1|253.455257084|253.455257084|PASS|
|g004|2|B1 → C2|153.168924167|153.168924167|PASS|
|g004|3|B1 → C2 → A2|97.217634680|97.217634680|PASS|
|g004|4|B1 → C2 → A2 → B2|48.608817340|48.608817340|PASS|

`test_frozen_16_conditions` 逐条件核对：完整返回顺序、冻结 Greedy trace 中每步 chosen、每步全部剩余候选边际、每个 Pending 的 ID/T/E/G/成本、总 J、选中数及逻辑补集。选择顺序包含同分行为；冻结 trace 的所有边际与生产严格前沿评分一致。对 12 条跨 Workflow 准入边逐条检查单候选前沿，均未被旧谓词过滤。

没有重新运行 B4-C 正式脚本或其他 Baseline，也没有重新枚举 Exact。已有四个 Greedy 非最优条件原样保留，未因接线改变其成本或选择。

## 4. 测试与失败即拒绝

最终测试日志：`tests_final.log`。90 项测试含新增 26 项及原 Optimizer、Controller、Executable State、Recovery Model、模型冻结和多 Workflow 分配回归；其中正式快照另列 16 个子测试。

|负面或隔离条件|结果|
|---|---|
|UNKNOWN、缺失配对|拒绝|
|重复配对、同一配对冲突|拒绝|
|Candidate/Pending ID 重复、关系域外|拒绝|
|候选位置与 Token 长度不符、T 超出输入|拒绝|
|不同 Epoch、不同配置|拒绝|
|Token 摘要、原始 Snapshot SHA、epoch 内容摘要不符|拒绝|
|准入条件缺失、准入状态与条件矛盾|拒绝|
|候选声明不驻留、可变输入集合|拒绝|
|Legacy 参数混入 Strict 对象，Strict 入口接受非 Epoch|拒绝|
|严格跨 Workflow 边经过旧谓词|哨兵测试确认没有调用|
|计划触发物理驱逐|Adapter 哨兵测试确认没有调用|
|REJECTED 单边贡献|前沿为零|
|零预算、饱和后停止、同分 ID 顺序|通过|

K=0 与 K=100 仅为边界单元测试，后者用于检查饱和后零边际停止；不是新增正式预算条件，不计入 16 条分配结果，不形成新的性能结论。

### 测试执行记录

uv312 默认未安装 pytest；系统 pytest 启动器也缺少对应可导入包。未安装依赖，使用已有本地缓存中的 Python 3.12 pytest，并禁用插件自动加载，避免额外运行设施。早期 unittest 的新测试通过，但旧 pytest 模块导入失败；补充逐步检查时出现过测试代码 NameError，已修复，相关早期日志保留，最终以 tests_final.log 为准。

最终命令：

```bash
PYTHONPATH=/home/wjg/.cache/uv/archive-v0/Wt-t6EKpnGZnOzKw/lib/python3.12/site-packages PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/wjg/uv312/bin/python -B -m pytest -q -p no:cacheprovider tests/test_strict_epoch.py tests/test_optimizer.py tests/test_controller.py tests/test_executable_state.py tests/test_recovery_model.py tests/test_recovery_model_freeze.py tests/test_multiworkflow_allocation.py
```

独立回归产物入口为本目录 regression.py，仅调用本次生产纯规划方法；不调用旧 evaluate.main，不写入旧结果目录。

## 5. 完整性与版本边界

`integrity.json` 保存 B4-B1 根 Manifest 与 B4-C 产物 Manifest 的内容核对、当前核心文件和测试 SHA-256。历史 Snapshot、结果、旧执行锁、T1/T2 未被改写。旧执行锁中 optimizer.py 的 SHA 仍指旧生产版本，本次授权修改后自然不同；不能声称当前源码仍通过旧 Runtime 执行锁，也没有为了通过而改写旧锁。后续新阶段需要独立的新版本身份锁。

## 6. 尚未解决的物理缺口与 R2-B 前提

本轮已修复 R1 的严格 Compatibility 进入生产规划问题，但以下未解决：

1. StrictAllocationPlan 的逻辑 ID 尚未绑定新 Engine 的实时 node/slot/generation，历史驻留不是当前有效性证明。
2. 未建立 keep/free 物理组件互斥、全树候选外状态检查、选中状态保护和驱逐前完整安全合同。
3. 未将计划发送给 SGLangAdapter；logical_eviction_ids 只是候选补集，不能解释为已驱逐集合。
4. 未验证物理协调后的驻留恰等于计划、FA 不变、请求间状态变化、实际恢复前沿或输出正确性。
5. 未验证多策略等价初态，没有 Runtime A/B 或控制开销后性能证据。

进入 R2-B 前应明确：同一当前 epoch 的严格证书与物理映射绑定、静止/admission 边界、变更前安全拒绝规则、预算与后置条件、失败即停账本，以及新版本内容锁。不得将本轮 READY 自动提升为可直接运行正式 GPU 比较。

**最终判定 READY，限纯规划生产接线。完成后停止，不进入 R2-B。**
