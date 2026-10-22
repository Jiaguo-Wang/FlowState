# 输入、版本与预算合同

**PASS**。只读取B5-C3和B4-C冻结证据，前后SHA全部一致。完整路径与摘要见input_integrity.json；B5-C3及B4-C各自evidence_manifest逐文件检查通过。

来源：`evaluation/multishare_runtime_gate_b5c3/strict_epoch_snapshot.json`、`physical_state_census.json`、`sharing_support_matrix.json`。七个唯一候选、四个唯一Pending，28个完整无重复配对，13准入、15拒绝、0未知。逐边核对Token前缀、L≤T、Epoch及配置身份。正收益与B5-C3单边记录一致；拒绝边收益为零。四个T按A/B/C/D为4160/4096/3904/3566，不使用后续执行状态。

## 等大小证据

每个候选有一个独占Slot及完整Conv/Temporal组件。以冻结physical_state_census.json → audit_before.component_slices → byte_ranges求和，每项为 **51511296字节**：Conv为24×49152=1179648字节，Temporal为24×2097152=50331648字节。七个Slot互异。逻辑成本均为1个完整Checkpoint；K=1、2、3、4，无加权预算或候选删除。以上是Recurrent切片大小，不包含Attention、不声称归还Slot释放预分配HBM。Node/Slot仅用于历史证据，不能充当未来生命周期映射。

## 锁定策略

PCB/LDR/WFC/common.py直接复用`evaluation/p1_strong_baseline_validation/`，当前SHA与B4-C/input_integrity.json一致。

生产optimizer.py在R2阶段已变化；本轮从Git提交`42e494026b0d39398ea0d2629d324690ae65b383`读取历史文件，核对SHA `eabf84c7c5d50db1d5e9e60821b0803f760f2c81750a846d960f4cb638f87b98` 后隔离保存为frozen_optimizer.py。执行原Greedy循环，仅按B4-C的MatrixOptimizer方式替换成本接口，使严格矩阵直接决定前沿。未调用后来改变的生产选择入口，未修改任何生产文件；占位Workflow字段不参与评分。

StandaloneTopK完全沿用B4-C/evaluate.py定义：静态总收益降序，ID升序，收益大于1e-9，截取K；最终输出ID排序。results中的selection_order对该策略表示静态排名次序，而非动态决策。Exact没有决策顺序，展示字典序代表集合。PCB/LDR保存原推荐记录；WFC保存原动态排序。

Phi源码摘要与B4-C一致，元数据逐项一致：a=37.828150，b=0.345974143，c=-0.156201917，G和T以1024归一化。未拟合。比较容差为B4-C既有1e-9 ms；Greedy仍沿用原源码的非正收益停止、容差及ID同分规则。

## 独立核验与复现

`validate.py`从原始边和Token位置出发，以Decimal独立实现冻结公式；128个子集、24个策略结果、448个加一候选的边际变化、4个Exact预算全部通过。详见validation.json。没有导入或调用Runtime启动接口。

在仓库根目录依次运行：
```
/home/wjg/uv312/bin/python -B -m evaluation.multishare_allocation_gate_b5c4.evaluate
/home/wjg/uv312/bin/python -B -m evaluation.multishare_allocation_gate_b5c4.validate
/home/wjg/uv312/bin/python -B -m evaluation.multishare_allocation_gate_b5c4.report
```
这些入口只写本轮新目录，不修改冻结源。
