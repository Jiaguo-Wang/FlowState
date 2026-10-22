# FlowState E4-Phase 1A：w1/K2 Native Resume正确性门禁

## Gate：BLOCKED

阻断来自执行前GPU资源门禁，不是已证实的恢复计算错误。GPU0存在不属于本轮的SGLang计算进程PID 3059291，进程显存79554 MiB；设备总占用79578/81559 MiB。本轮没有启动Engine、创建实验容器、驱逐状态或提交任何请求。遵循失败即停规则，未等待后自动重试、未干预该进程。

## 完成的只读与CPU准备

- 核验E2/E3冻结产物、w1完整输入与E1/B5-D4b实现摘要；准备12个独立生命周期的固定协议（每个A–D Pending分别Cold、Greedy、Exact）。
- 新实验适配只扩展固定目标和冻结语义核验：完整Token前缀双射对齐当前与旧候选，禁止旧Node/Slot作为当前绑定；重放状态不等价则拒绝。
- 新增CPU合同10/10通过，其中一个测试覆盖两策略×四Pending全部8个目标；既有生产Strict Epoch/Control测试54/54通过。uv312没有pytest，首次pytest启动失败日志保留，随后使用这些测试原本的unittest入口完成54项；未安装依赖或改变环境。
- CPU证据仅验证代码接口与拒绝合同，不代表真实物理Bind、Reconciliation或Resume成功。

## 冻结选集核对

| 策略 | 冻结逻辑候选 | Model J (ms) | 实际物理应用 |
|---|---|---:|---|
| Greedy | B5C3_NODE_20, B5C3_NODE_21 | 126.590493308 | 未执行 |
| Exact | B5C3_NODE_18, B5C3_NODE_21 | 53.770758869 | 未执行 |

上述Node式ID仅是E2冻结逻辑命名，不代表新生命周期物理映射。E3的Greedy+1-Swap在w1/K2与Exact同集，无需单独Arm。

## 四个Pending完整执行账本

| Pending | 策略 | 模型T/E/G | Runtime、COW、Cold对照 |
|---|---|---|---|
| IMW1_A_TURN_003 | Greedy | 3648/2432/1216 | 全部未执行 |
| IMW1_A_TURN_003 | Exact | 3648/3392/256 | 全部未执行 |
| IMW1_B_TURN_003 | Greedy | 3712/2432/1280 | 全部未执行 |
| IMW1_B_TURN_003 | Exact | 3712/3392/320 | 全部未执行 |
| IMW1_C_TURN_003 | Greedy | 4096/3712/384 | 全部未执行 |
| IMW1_C_TURN_003 | Exact | 4096/3712/384 | 全部未执行 |
| IMW1_D_TURN_003 | Greedy | 4160/3712/448 | 全部未执行 |
| IMW1_D_TURN_003 | Exact | 4160/3712/448 | 全部未执行 |

完成范围：0/8个策略—Pending恢复；0/4个Cold；0项物理驱逐。所有真实前沿、COW与输出比较均UNKNOWN/NOT_EXECUTED，不用模型预测填充Runtime结果。

## 实现与状态安全边界

独立runner沿用E1重型路径；独立bridge只设置当前目标并接入冻结映射校验，原Scheduler隔离、Bind/Preflight、生产StateController.reconcile_strict、Native Recurrent-only控制、Admission前物理/内容核对、COW及FA保护逻辑均复用。重型参考和每个严格臂协调后内容审计保留；本轮不做性能调优。原生来源校验继续区分Token/严格准入、实际Node/Slot匹配及血缘。

FA不变性分两个阶段：协调阶段要求全部原FA索引及内容不变；正常请求阶段允许目标恢复区间重算和新状态生成，要求保留源Recurrent内容及非重算FA内容不变。不能要求请求后全树机械不变，也不能把重算副作用掩盖为状态损坏或反之。以上是待执行合同，本轮未获得新GPU证据。

## 证据等级

- EPOCH_ADMISSIBLE：沿用E2已有冻结证据；未采集新Epoch。
- RECONCILIATION_VALIDATED：本轮未取得。
- RESUME_VALIDATED：本轮未取得。
- PROVENANCE_UNVERIFIED：保持原级别，无新增来源血缘认证。

## 关键文件

`source_identity.json`、`execution_lock.json`、`experiment_protocol.json`、`frozen_snapshot.json`、`frozen_plans.json`记录来源与预定条件；`cpu_tests.log`和`production_contract_unittest.log`记录CPU结果；`launch_failure.json`保存启动失败堆栈，attempted为空；`resource_preflight.json`记录实际占用及无本轮容器证据；`per_pending_status.csv`列出全部8个未执行条件；`gpu_cleanup_evidence.json`明确无本轮资源需要清理。

## 能证明和不能证明

能证明冻结w1计划与目标适配通过CPU核验，且启动前资源安全检查拒绝了当前不满足条件的环境。不能证明两个选集在本轮真实安全落地、Pending实际恢复、输出正确或FA/组件保持。没有GPU性能或端到端收益结论。

唯一下一步：在GPU0恢复空闲并获得继续执行安排后，以新独立执行目录重新进行相同w1/K2门禁；保留本次失败目录，不覆盖或复用execution_started标记。本轮已停止，不自动重试、不进入Phase 1B，也不扩展w2–w4。
