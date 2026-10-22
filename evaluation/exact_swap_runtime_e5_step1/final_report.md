# E5-Step 1：Exact与固定1-Swap真实恢复取舍

**Gate：READY**。Gate 0为59项CPU测试通过；已认证Smoke 8/8、正式测量 24/24、生命周期清理 36/36。

三组差值方向不一致，当前重复中的波动使方向结论不稳；未证明Exact具有实际联合恢复收益，也不将混合符号当作统计等价证明。

## 冻结与正确性边界

固定w1/K3。Exact {N18,N25,N27}，模型J=21.953374716；实验冻结首改善1-Swap {N21,N22,N24}，J=31.817384152。历史ID只对齐完整Token前缀，当前Node/Slot由每个Engine重新Observe/Bind/Preflight。1-Swap不称生产求解器，其计时项为冻结选择读取/映射，不是假定在线重算Swap耗时。

全部有效臂要求当前严格矩阵/Plan不变、连续Scheduler控制、4项真实Recurrent-only驱逐、恰好3项保留、Native Match来源符合Plan、COW工作槽与保留状态隔离、8Token与独立Cold精确一致、FA及保留内容合同通过。Smoke额外检查控制后全内容及COW复制内容；正式轻量臂延续E4合同及请求后独立内容验证。PROVENANCE_UNVERIFIED不升级。

|Pending|策略|实际H/E/G|实际Prefill Tokens|
|---|---|---|---|
|A|Exact|[(3648, 3392, 256)]|[858]|
|A|Swap|[(3648, 3648, 0)]|[602]|
|B|Exact|[(3712, 3392, 320)]|[916]|
|B|Swap|[(3712, 3712, 0)]|[596]|
|C|Exact|[(4096, 4096, 0)]|[401]|
|C|Swap|[(4096, 3712, 384)]|[785]|
|D|Exact|[(4160, 4160, 0)]|[422]|
|D|Swap|[(4160, 3712, 448)]|[870]|

## 主要指标：四次独立恢复的等权和

|配对组|Exact CUDA Prefill总和ms|1-Swap总和ms|Exact−1-Swap ms|
|---|---:|---:|---:|
|1|164.689|178.087|-13.399|
|2|166.245|164.307|1.938|
|3|167.824|170.926|-3.102|

三组总和均值为Exact 166.253 ms、1-Swap 171.107 ms，均值差-4.854 ms。均值更低不能替代三组方向不一致的事实。

额外观察：A在Exact下实际Prefill Tokens更多，但CUDA均值更低；C在Exact下Tokens更少，但三次CUDA均更高。当前记录不支持把计算耗时简单视为G或实际Prefill Tokens的单调函数；具体内核/提交/波动归因尚未验证，本轮不调参、不删样本。

## 单请求三次均值与控制路径

|Pending|指标ms|Exact|1-Swap|
|---|---|---:|---:|
|A|prefill_cuda_ms|43.126|47.832|
|A|ttft_ms|75.253|80.796|
|A|control_rpc_ms|821.498|822.224|
|A|request_wall_ms|6598.401|6677.339|
|A|control_to_completion_ms|7644.727|7721.896|
|A|audit_inclusive_completion_ms|9985.886|10147.597|
|B|prefill_cuda_ms|45.400|44.209|
|B|ttft_ms|77.447|76.838|
|B|control_rpc_ms|853.461|929.788|
|B|request_wall_ms|7077.691|6671.016|
|B|control_to_completion_ms|8170.318|7856.474|
|B|audit_inclusive_completion_ms|10476.459|10353.663|
|C|prefill_cuda_ms|38.994|36.372|
|C|ttft_ms|71.247|66.863|
|C|control_rpc_ms|779.888|823.565|
|C|request_wall_ms|6373.387|6807.847|
|C|control_to_completion_ms|7408.191|7877.361|
|C|audit_inclusive_completion_ms|9424.299|10012.680|
|D|prefill_cuda_ms|38.732|42.694|
|D|ttft_ms|69.974|74.503|
|D|control_rpc_ms|846.090|825.546|
|D|request_wall_ms|6545.856|6793.048|
|D|control_to_completion_ms|7643.683|7876.637|
|D|audit_inclusive_completion_ms|9721.965|10064.536|

完整逐次、逐Pending差值、各组等权和、均值/中位数/范围见对应CSV。控制及完整路径和也是四次独立生命周期测量的描述性加总，不是一个自然四请求服务周期，更不是吞吐。不能只报告C/D收益而略去A/B代价。

## 测量合同与不能证明的结论

沿用E4-1B客户端写盘、fsync及原生同流CUDA事件位置，不重构计时，不删首次Decode、不单侧预热。Control RPC包含客户端证据持久化；观察/认证/求解/协调为其子区间。Control-to-completion从控制RPC发起到目标流结束，Audit-inclusive另包含前置重型参考；请求后全内容检查和清理位于结束端点之外。CUDA事件含提交空隙，不称纯内核和。主要Prefill测量位于Native COW之后的原Eager路径，Smoke不纳入性能样本。

三组是固定受控组合的重复；无统计显著性或生产P95结论。模型J差距不能证明CUDA优劣，G=0也不代表完整输入无需Prefill。逻辑K3不等于GPU容量压力。未扩展来源、未改变生产求解器/Phi/Strict合同。若失败，详情见launch_failure.json和validation.json，不自动重试。

本轮完成即停止，不扩展到其他工作负载或性能优化。
