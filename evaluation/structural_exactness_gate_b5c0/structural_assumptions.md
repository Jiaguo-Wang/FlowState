# 结构合同重新确认

结论：四组全部满足单共享候选条件。只以B4-B1原始snapshot及B4-C冻结results.json确认，不以B5-B0结论替代检查。

## 权威来源和检查方式

仓库根 `/home/wjg/code/FlowState`。快照根为 `evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/`；对应g001–g004的snapshot.json、summary.json及manifest.json。结果根 `evaluation/strict_compatibility_b4c/`，使用results.json的benefit_matrix_ms、input、rows和traces，核验evidence_manifest.json。原始清单全部通过，路径及SHA见verification.json。

不导入生产Optimizer，不重新执行B4-C或Baseline。当前optimizer与B4-C历史版本的变化不进入本次数值核验。没有使用g000、g005/g006或旧573条件。

|条件|实际核验|
|---|---|
|唯一共享|按严格正收益b>0统计支持，不用Workflow ID推断；每组仅B1支持4请求|
|其余候选|均支持恰好1个Pending，没有第二个共享候选|
|非负固定收益|全部条目有限且≥0；拒绝边严格为0|
|等大小|每组所有候选memory_bytes=51,511,296；K是逻辑项数而非HBM压力|
|逐请求最大收益|对两个分支实际选集直接按Σmax计算F，与分支公式一致；逐轮日志边际也独立复核|
|同Epoch与Phi|候选/请求barrier_id与configuration_digest一致；位置/T与冻结输入一致；全部收益用冻结参数Decimal算术核对，误差≤1e−9|

Phi只用于核验既有值，没有拟合或改变矩阵。参数为a=37.82815、b=0.345974143、c=−0.156201917，g=G/1024、t=T/1024，Phi=a g+b g t+c g²；冻结模型名position_aware_quadratic_v1。

## 逐快照合同与q/r

### g001

12候选、4Pending；共享候选RQ3_G001_B_TURN_001。快照SHA：`3bd4a6124bfa97391db4088d97356b9e45a9ed71bafb291f6d80fd101ff49ad7`；Barrier：`b4b1-g001-365275016404246`。

|Pending|专属最佳代表|q（ms）|r（ms）|max(0,q−r)（ms）|
|---|---|---:|---:|---:|
|A_PENDING_TURN_004|A_TURN_003|159.270211783031|91.051816809062|68.218394973969|
|B_PENDING_TURN_004|B_TURN_003|201.340328378570|91.136529385680|110.203798992891|
|C_PENDING_TURN_004|C_TURN_003|179.014540271781|91.091681551000|87.922858720781|
|D_PENDING_TURN_004|D_TURN_003|159.270211783031|91.051816809062|68.218394973969|

### g002

16候选、4Pending；共享候选RQ3_G002_B_TURN_001。快照SHA：`9095b94e303d3de04db243ea0809a66de767da716e564f3c8c72434c272cef8c`；Barrier：`b4b1-g002-365353730960912`。

|Pending|专属最佳代表|q（ms）|r（ms）|max(0,q−r)（ms）|
|---|---|---:|---:|---:|
|A_PENDING_TURN_005|A_TURN_004|231.294852889258|91.196326498586|140.098526390672|
|B_PENDING_TURN_005|B_TURN_004|174.069562576500|91.081715365516|82.987847210984|
|C_PENDING_TURN_005|C_TURN_004|274.095889074000|91.281039075203|182.814849998797|
|D_PENDING_TURN_005|D_TURN_004|231.294852889258|91.196326498586|140.098526390672|

### g003

20候选、4Pending；共享候选RQ3_G003_B_TURN_001。快照SHA：`8033ccb947bfd0a2834272612b4cfacd278576d517b67e3efd8ee10311b0b0a4`；Barrier：`b4b1-g003-365435852774246`。

|Pending|专属最佳代表|q（ms）|r（ms）|max(0,q−r)（ms）|
|---|---|---:|---:|---:|
|A_PENDING_TURN_006|A_TURN_005|420.732039635883|91.565075361508|329.166964274375|
|B_PENDING_TURN_006|B_TURN_005|407.676481994125|91.540159897797|316.136322096328|
|C_PENDING_TURN_006|C_TURN_005|358.403394733008|91.445481135695|266.957913597312|
|D_PENDING_TURN_006|D_TURN_005|193.885055650000|91.121580107453|102.763475542547|

### g004

8候选、4Pending；共享候选RQ3_G004_B_TURN_001。快照SHA：`3c3dfae9c577d07ab36192d966f051158111ff375ace8a7248ae74735de66622`；Barrier：`b4b1-g004-365513360700058`。

|Pending|专属最佳代表|q（ms）|r（ms）|max(0,q−r)（ms）|
|---|---|---:|---:|---:|
|A_PENDING_TURN_003|A_TURN_002|146.978190831820|91.026901345352|55.951289486469|
|B_PENDING_TURN_003|B_TURN_002|139.620769407281|91.011952067125|48.608817340156|
|C_PENDING_TURN_003|C_TURN_002|191.402929931508|91.116597014711|100.286332916797|
|D_PENDING_TURN_003|D_TURN_002|139.620769407281|91.011952067125|48.608817340156|

四组K_min=4：各请求最大前沿对应不同的唯一专属最深项，共享B1不能达到这些最深前沿；四项既必要又充分。本批J(C)=0，不推广到其他负载。

准入边是EPOCH_ADMISSIBLE；本轮不追加逐边恢复认证。没有改变冻结收益、候选身份或严格关系。
