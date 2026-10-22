# 工作负载结构与机制归因

全部四组均按原顺序保留。以下成本单位为modeled ms，非实测GPU耗时。

## w1

K_min=4；正收益支持层次化=True。所有7个候选均保留，未进行支配剪枝。

| 候选 | L | 正收益支持 |
|---|---:|---|
| B5C3_NODE_18 | 3392 | A,B |
| B5C3_NODE_20 | 2432 | A,B,C,D |
| B5C3_NODE_21 | 3712 | C,D |
| B5C3_NODE_22 | 3648 | A |
| B5C3_NODE_24 | 3712 | B |
| B5C3_NODE_25 | 4096 | C |
| B5C3_NODE_27 | 4160 | D |

| K | Exact | Greedy | StandaloneTopK | PCB | LDR | WFC |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 223.842223 | 223.842223 | 223.842223 | 431.140986 | 431.140986 | 223.842223 |
| 2 | 53.770759 | 126.590493 | 126.590493 | 276.792030 | 276.792030 | 126.590493 |
| 3 | 21.953375 | 53.770759 | 53.770759 | 137.171261 | 137.171261 | 53.770759 |
| 4 | 0.000000 | 36.635926 | 36.635926 | 0.000000 | 0.000000 | 36.635926 |

逐预算组合及冗余：

- K=1：FlowState-Greedy=B5C3_NODE_20（J=223.842223）；FlowState-Exact=B5C3_NODE_20（J=223.842223）；PCB=B5C3_NODE_27（J=431.140986）；LDR=B5C3_NODE_27（J=431.140986）；WFC=B5C3_NODE_20（J=223.842223）；StandaloneTopK=B5C3_NODE_20（J=223.842223）
- K=2：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_21（J=126.590493）；FlowState-Exact=B5C3_NODE_18,B5C3_NODE_21（J=53.770759）；PCB=B5C3_NODE_27,B5C3_NODE_25（J=276.792030）；LDR=B5C3_NODE_27,B5C3_NODE_25（J=276.792030）；WFC=B5C3_NODE_20,B5C3_NODE_21（J=126.590493）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_21（J=126.590493）
- K=3：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18（J=53.770759）；FlowState-Exact=B5C3_NODE_18,B5C3_NODE_25,B5C3_NODE_27（J=21.953375）；PCB=B5C3_NODE_27,B5C3_NODE_25,B5C3_NODE_24（J=137.171261）；LDR=B5C3_NODE_27,B5C3_NODE_25,B5C3_NODE_24（J=137.171261）；WFC=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18（J=53.770759）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18（J=53.770759）
- K=4：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18,B5C3_NODE_27（J=36.635926）；FlowState-Exact=B5C3_NODE_22,B5C3_NODE_24,B5C3_NODE_25,B5C3_NODE_27（J=0.000000）；PCB=B5C3_NODE_27,B5C3_NODE_25,B5C3_NODE_24,B5C3_NODE_22（J=0.000000）；LDR=B5C3_NODE_27,B5C3_NODE_25,B5C3_NODE_24,B5C3_NODE_22（J=0.000000）；WFC=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18,B5C3_NODE_27（J=36.635926）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18,B5C3_NODE_27（J=36.635926）

代表Exact集合不嵌套的相邻预算：[[1, 2], [2, 3], [3, 4]]；枚举全部同分最优集合后，仍不存在嵌套最优延伸的预算：[[1, 2], [2, 3], [3, 4]]。后者说明不可撤回的Greedy选择存在结构性限制，不能用不同集合本身宣称性能差异。

当K<K_min时不能保留达到全部候选最佳前沿的完整集合，存在有效逻辑取舍；K≥K_min时Exact可达到该前沿，但Greedy先前选择可能继续占用预算。共享祖先和专属/子集深层候选的收益重叠，导致初始覆盖优势与最终组合收益不一致。具体选集、边际值及各Pending E/G见allocation_results.json和allocation_comparison.csv。
## w2

K_min=4；正收益支持层次化=True。所有7个候选均保留，未进行支配剪枝。

| 候选 | L | 正收益支持 |
|---|---:|---|
| B5C3_NODE_18 | 3456 | A,B |
| B5C3_NODE_20 | 2432 | A,B,C,D |
| B5C3_NODE_21 | 3520 | C,D |
| B5C3_NODE_22 | 3776 | A |
| B5C3_NODE_24 | 3712 | B |
| B5C3_NODE_25 | 4032 | C |
| B5C3_NODE_27 | 3840 | D |

| K | Exact | Greedy | StandaloneTopK | PCB | LDR | WFC |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 214.019643 | 214.019643 | 214.019643 | 426.216764 | 426.216764 | 214.019643 |
| 2 | 53.733180 | 131.431285 | 131.431285 | 281.692530 | 281.692530 | 131.431285 |
| 3 | 21.965538 | 53.733180 | 53.733180 | 139.620769 | 139.620769 | 53.733180 |
| 4 | 0.000000 | 34.177019 | 34.177019 | 0.000000 | 0.000000 | 34.177019 |

逐预算组合及冗余：

- K=1：FlowState-Greedy=B5C3_NODE_20（J=214.019643）；FlowState-Exact=B5C3_NODE_20（J=214.019643）；PCB=B5C3_NODE_25（J=426.216764）；LDR=B5C3_NODE_25（J=426.216764）；WFC=B5C3_NODE_20（J=214.019643）；StandaloneTopK=B5C3_NODE_20（J=214.019643）
- K=2：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_21（J=131.431285）；FlowState-Exact=B5C3_NODE_18,B5C3_NODE_21（J=53.733180）；PCB=B5C3_NODE_25,B5C3_NODE_27（J=281.692530）；LDR=B5C3_NODE_25,B5C3_NODE_27（J=281.692530）；WFC=B5C3_NODE_20,B5C3_NODE_21（J=131.431285）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_21（J=131.431285）
- K=3：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18（J=53.733180）；FlowState-Exact=B5C3_NODE_18,B5C3_NODE_25,B5C3_NODE_27（J=21.965538）；PCB=B5C3_NODE_25,B5C3_NODE_27,B5C3_NODE_22（J=139.620769）；LDR=B5C3_NODE_25,B5C3_NODE_27,B5C3_NODE_22（J=139.620769）；WFC=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18（J=53.733180）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18（J=53.733180）
- K=4：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18,B5C3_NODE_25（J=34.177019）；FlowState-Exact=B5C3_NODE_22,B5C3_NODE_24,B5C3_NODE_25,B5C3_NODE_27（J=0.000000）；PCB=B5C3_NODE_25,B5C3_NODE_27,B5C3_NODE_22,B5C3_NODE_24（J=0.000000）；LDR=B5C3_NODE_25,B5C3_NODE_27,B5C3_NODE_22,B5C3_NODE_24（J=0.000000）；WFC=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18,B5C3_NODE_25（J=34.177019）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_21,B5C3_NODE_18,B5C3_NODE_25（J=34.177019）

代表Exact集合不嵌套的相邻预算：[[1, 2], [2, 3], [3, 4]]；枚举全部同分最优集合后，仍不存在嵌套最优延伸的预算：[[1, 2], [2, 3], [3, 4]]。后者说明不可撤回的Greedy选择存在结构性限制，不能用不同集合本身宣称性能差异。

当K<K_min时不能保留达到全部候选最佳前沿的完整集合，存在有效逻辑取舍；K≥K_min时Exact可达到该前沿，但Greedy先前选择可能继续占用预算。共享祖先和专属/子集深层候选的收益重叠，导致初始覆盖优势与最终组合收益不一致。具体选集、边际值及各Pending E/G见allocation_results.json和allocation_comparison.csv。
## w3

K_min=4；正收益支持层次化=True。所有7个候选均保留，未进行支配剪枝。

| 候选 | L | 正收益支持 |
|---|---:|---|
| B5C3_NODE_18 | 3456 | A,B |
| B5C3_NODE_20 | 2432 | A,B,C,D |
| B5C3_NODE_21 | 3392 | C,D |
| B5C3_NODE_22 | 3776 | A |
| B5C3_NODE_24 | 3520 | A,B |
| B5C3_NODE_25 | 3840 | C |
| B5C3_NODE_27 | 3456 | C,D |

| K | Exact | Greedy | StandaloneTopK | PCB | LDR | WFC |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 185.820309 | 185.820309 | 185.820309 | 405.326286 | 405.326286 | 185.820309 |
| 2 | 25.559190 | 103.251492 | 103.251492 | 263.254526 | 263.254526 | 103.251492 |
| 3 | 10.909073 | 25.559190 | 103.251492 | 130.976144 | 130.976144 | 25.559190 |
| 4 | 1.142854 | 10.909073 | 25.559190 | 1.142854 | 1.142854 | 10.909073 |

逐预算组合及冗余：

- K=1：FlowState-Greedy=B5C3_NODE_20（J=185.820309）；FlowState-Exact=B5C3_NODE_20（J=185.820309）；PCB=B5C3_NODE_25（J=405.326286）；LDR=B5C3_NODE_25（J=405.326286）；WFC=B5C3_NODE_20（J=185.820309）；StandaloneTopK=B5C3_NODE_20（J=185.820309）
- K=2：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_24（J=103.251492）；FlowState-Exact=B5C3_NODE_24,B5C3_NODE_27（J=25.559190）；PCB=B5C3_NODE_25,B5C3_NODE_22（J=263.254526）；LDR=B5C3_NODE_25,B5C3_NODE_22（J=263.254526）；WFC=B5C3_NODE_20,B5C3_NODE_24（J=103.251492）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_24（J=103.251492）
- K=3：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_27（J=25.559190）；FlowState-Exact=B5C3_NODE_24,B5C3_NODE_25,B5C3_NODE_27（J=10.909073）；PCB=B5C3_NODE_25,B5C3_NODE_22,B5C3_NODE_24（J=130.976144）；LDR=B5C3_NODE_25,B5C3_NODE_22,B5C3_NODE_24（J=130.976144）；WFC=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_27（J=25.559190）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_18（J=103.251492）
  StandaloneTopK按其选择顺序出现零边际候选：B5C3_NODE_18。静态独立收益不扣除已有覆盖；最终目标仍按每个Pending的最大收益评价。
- K=4：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_27,B5C3_NODE_25（J=10.909073）；FlowState-Exact=B5C3_NODE_22,B5C3_NODE_24,B5C3_NODE_25,B5C3_NODE_27（J=1.142854）；PCB=B5C3_NODE_25,B5C3_NODE_22,B5C3_NODE_24,B5C3_NODE_27（J=1.142854）；LDR=B5C3_NODE_25,B5C3_NODE_22,B5C3_NODE_24,B5C3_NODE_27（J=1.142854）；WFC=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_27,B5C3_NODE_25（J=10.909073）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_18,B5C3_NODE_27（J=25.559190）
  StandaloneTopK按其选择顺序出现零边际候选：B5C3_NODE_18。静态独立收益不扣除已有覆盖；最终目标仍按每个Pending的最大收益评价。

代表Exact集合不嵌套的相邻预算：[[1, 2]]；枚举全部同分最优集合后，仍不存在嵌套最优延伸的预算：[[1, 2]]。后者说明不可撤回的Greedy选择存在结构性限制，不能用不同集合本身宣称性能差异。

当K<K_min时不能保留达到全部候选最佳前沿的完整集合，存在有效逻辑取舍；K≥K_min时Exact可达到该前沿，但Greedy先前选择可能继续占用预算。共享祖先和专属/子集深层候选的收益重叠，导致初始覆盖优势与最终组合收益不一致。具体选集、边际值及各Pending E/G见allocation_results.json和allocation_comparison.csv。
## w4

K_min=4；正收益支持层次化=True。所有7个候选均保留，未进行支配剪枝。

| 候选 | L | 正收益支持 |
|---|---:|---|
| B5C3_NODE_18 | 3456 | A,B |
| B5C3_NODE_20 | 2432 | A,B,C,D |
| B5C3_NODE_21 | 3456 | C,D |
| B5C3_NODE_22 | 3776 | A |
| B5C3_NODE_24 | 3520 | A,B |
| B5C3_NODE_25 | 3840 | C |
| B5C3_NODE_27 | 3968 | D |

| K | Exact | Greedy | StandaloneTopK | PCB | LDR | WFC |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 205.344724 | 205.344724 | 205.344724 | 419.981093 | 419.981093 | 205.344724 |
| 2 | 45.066824 | 122.775420 | 122.775420 | 275.456858 | 275.456858 | 122.775420 |
| 3 | 10.871358 | 45.066824 | 45.066824 | 133.385098 | 133.385098 | 45.066824 |
| 4 | 1.105138 | 25.521474 | 45.066824 | 1.105138 | 1.105138 | 25.521474 |

逐预算组合及冗余：

- K=1：FlowState-Greedy=B5C3_NODE_20（J=205.344724）；FlowState-Exact=B5C3_NODE_20（J=205.344724）；PCB=B5C3_NODE_27（J=419.981093）；LDR=B5C3_NODE_27（J=419.981093）；WFC=B5C3_NODE_20（J=205.344724）；StandaloneTopK=B5C3_NODE_20（J=205.344724）
- K=2：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_24（J=122.775420）；FlowState-Exact=B5C3_NODE_21,B5C3_NODE_24（J=45.066824）；PCB=B5C3_NODE_27,B5C3_NODE_25（J=275.456858）；LDR=B5C3_NODE_27,B5C3_NODE_25（J=275.456858）；WFC=B5C3_NODE_20,B5C3_NODE_24（J=122.775420）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_24（J=122.775420）
- K=3：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_21（J=45.066824）；FlowState-Exact=B5C3_NODE_24,B5C3_NODE_25,B5C3_NODE_27（J=10.871358）；PCB=B5C3_NODE_27,B5C3_NODE_25,B5C3_NODE_22（J=133.385098）；LDR=B5C3_NODE_27,B5C3_NODE_25,B5C3_NODE_22（J=133.385098）；WFC=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_21（J=45.066824）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_21（J=45.066824）
- K=4：FlowState-Greedy=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_21,B5C3_NODE_27（J=25.521474）；FlowState-Exact=B5C3_NODE_22,B5C3_NODE_24,B5C3_NODE_25,B5C3_NODE_27（J=1.105138）；PCB=B5C3_NODE_27,B5C3_NODE_25,B5C3_NODE_22,B5C3_NODE_24（J=1.105138）；LDR=B5C3_NODE_27,B5C3_NODE_25,B5C3_NODE_22,B5C3_NODE_24（J=1.105138）；WFC=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_21,B5C3_NODE_27（J=25.521474）；StandaloneTopK=B5C3_NODE_20,B5C3_NODE_24,B5C3_NODE_21,B5C3_NODE_18（J=45.066824）
  StandaloneTopK按其选择顺序出现零边际候选：B5C3_NODE_18。静态独立收益不扣除已有覆盖；最终目标仍按每个Pending的最大收益评价。

代表Exact集合不嵌套的相邻预算：[[1, 2], [2, 3]]；枚举全部同分最优集合后，仍不存在嵌套最优延伸的预算：[[1, 2], [2, 3]]。后者说明不可撤回的Greedy选择存在结构性限制，不能用不同集合本身宣称性能差异。

当K<K_min时不能保留达到全部候选最佳前沿的完整集合，存在有效逻辑取舍；K≥K_min时Exact可达到该前沿，但Greedy先前选择可能继续占用预算。共享祖先和专属/子集深层候选的收益重叠，导致初始覆盖优势与最终组合收益不一致。具体选集、边际值及各Pending E/G见allocation_results.json和allocation_comparison.csv。

## 适用范围

层次化支持与路径上最大恢复位置符合已有条件性树DP讨论，但本轮没有实现或验证新的DP求解器。相同目标值表示在本固定输入与预算下简单策略足够；不代表一般算法数学等价。位置感知Phi始终固定，没有Gap-only消融，不能单独归因于位置项。

## 具体机制证据补充

- w1：K2 Greedy的J=126.590493，Exact=53.770759，差额72.819734 modeled ms。Exact同时保留AB/CD两个子域状态；Greedy保留公共祖先及一个子域状态。K3 Greedy选集中删除公共祖先B5C3_NODE_20的目标增量为0.000000000，说明该项此时已被更深覆盖替代，但不可回退的Greedy仍为其占用预算。
- w2：K2 Greedy的J=131.431285，Exact=53.733180，差额77.698105 modeled ms。Exact同时保留AB/CD两个子域状态；Greedy保留公共祖先及一个子域状态。K3 Greedy选集中删除公共祖先B5C3_NODE_20的目标增量为-0.000000000，说明该项此时已被更深覆盖替代，但不可回退的Greedy仍为其占用预算。
- w3：K2 Greedy的J=103.251492，Exact=25.559190，差额77.692302 modeled ms。Exact同时保留AB/CD两个子域状态；Greedy保留公共祖先及一个子域状态。K3 Greedy选集中删除公共祖先B5C3_NODE_20的目标增量为0.000000000，说明该项此时已被更深覆盖替代，但不可回退的Greedy仍为其占用预算。
  相同支持集合下的严格支配（支配者→被支配者）：[['B5C3_NODE_24', 'B5C3_NODE_18'], ['B5C3_NODE_27', 'B5C3_NODE_21']]；主比较仍保留全部候选。
- w4：K2 Greedy的J=122.775420，Exact=45.066824，差额77.708596 modeled ms。Exact同时保留AB/CD两个子域状态；Greedy保留公共祖先及一个子域状态。K3 Greedy选集中删除公共祖先B5C3_NODE_20的目标增量为-0.000000000，说明该项此时已被更深覆盖替代，但不可回退的Greedy仍为其占用预算。
  相同支持集合下的严格支配（支配者→被支配者）：[['B5C3_NODE_24', 'B5C3_NODE_18']]；主比较仍保留全部候选。

w3的StandaloneTopK在K3选择已被更深AB状态替代的Node18，边际为0，J=103.251492；Greedy转选CD状态，J=25.559190。K4两者J分别25.559190与10.909073。w4在K4同样出现Node18静态冗余，StandaloneTopK J=45.066824，Greedy J=25.521474。w1/w2未出现改变最终结果的静态排序冗余，Greedy与StandaloneTopK全部预算持平。

WFC与Greedy在16/16条件均得到相同选集和E/G。该观察不能证明动态恢复收益评分相对WFC有独立优势，也不证明两规则一般等价。四组K1均以公共共享状态达到Exact；K4均由PCB/LDR的最深需求覆盖达到Exact。K2/K3共8个条件，Exact严格优于本轮全部五种其他规则。

四组K_min=4：K1–K3共12个条件存在逻辑预算截断，K4无需复杂联合求解也可由PCB/LDR达成全候选最佳前沿。w1/w2此时四个G均为0；w3/w4仍有未被任何候选覆盖的Gap（分别对应B/D的15/15 Token、B的29 Token），因而最优J仍为1.142854、1.105138。不能把K_min等同于零Gap。

每组物理池均为28槽，7槽驻留、21槽空闲，无本轮逻辑预算引起的物理驱逐。真实共享与逻辑紧预算已经建立；真实HBM不足或在线容量压力没有建立。
