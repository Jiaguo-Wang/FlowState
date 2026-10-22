# Phase B5-D4 最终报告

**Gate：BLOCKED。有效配对0/5。** 阻断是本轮新增实验性来源门禁错误，不是已证实的FlowState恢复错误或Exact/Greedy性能结论。

## 已完成与停止位置

冻结五对交替顺序、固定Pending C、K2、原输入/配置与8Token生成协议；真实Req/Batch及计时包装CPU门禁通过。仅启动首对的Greedy独立Engine，八条冻结历史全部成功。当前观察与Strict Epoch认证、Greedy规划/验证、独立Oracle完成后，在Apply-time验证之前停止。

实际Recurrent-only策略驱逐 **0项**；正式Pending **0项**；Exact生命周期 **未启动**；Cold Reference **未启动**；后续四对 **未启动**。没有可信成对GPU耗时，不能比较性能。

## 已确认的错误原因

`control.py:97`将 `h['workflow_id'] != p['workflow_id']` 施加于两种策略所有目标来源，然后在第99行要求唯一跨Workflow来源。这是从上一阶段“选择任意跨Workflow目标”流程迁移时引入的过强条件。

本任务要求Greedy从公共共享Checkpoint恢复，同时特别要求Exact的CD共享Checkpoint来源可追溯D；并未要求Greedy的公共共享源也必须来自与C不同的Session。

本轮实际Census：Node20/2432/Slot8首次在 `B5C2_C_TURN_001` 完成后可见（history_03.json），按当前归因规则其Workflow恰与目标C相同。其Token数组确为目标C前缀，却被上述新增来源条件排除。Node27/3520/Slot18首次在D2之后可见（history_08.json），但本轮未进入Exact执行。首次观察边界本身不是精确insert事件，也不单独认证源张量由哪个请求生成；本报告不把它升级为实际恢复来源证明。

证据见failure_source_audit.json、failed_control_response.json、p01_Greedy/failure.json及p01_Greedy.runtime.log。原始失败保留；未修改门禁后重跑，也未更换请求、候选或预算。

## 证据范围与记录缺口

失败前控制Host区间已保存到control_cost_breakdown.csv。当前Oracle枚举29个子集并返回Exact候选N18/N27与J=48.228728759 modeled ms，但该结果不能替代当前Greedy最终选集或GPU结果。

失败RPC未返回完整Epoch和Plan，且ControlClient抛异常使runner原计划的reconciliation.json写盘未发生。原异常完整含失败对象，本轮只将其解析归档，没有补造Snapshot、重算分配或推断实际恢复E/G。

CPU门禁缺少真实历史来源同目标Session的目标选择负面/正面覆盖；因此“CPU通过”没有保证该实验门禁符合任务范围。CUDA计时包装尚未在正式目标执行中验证。

## 清理与完整性

唯一Engine已关闭、专用容器不存在；连续三次GPU0观测均无计算进程、显存14MiB，清理PASS。退出CUDA IPC警告原样保留，不从警告推导状态损坏，也不只凭返回码认定清理成功。

冻结来源和启动时执行锁摘要复核不变。所有新产物仅在本目录。没有修改生产Greedy、Exact目标、Phi、Strict Compatibility或历史冻结证据。

## 唯一后续建议

先修正独立实验入口的目标来源条件：Greedy允许其合法公共共享源的首次观察历史来自同一Session；Exact仍必须证明D历史来源，双方都保留实际Native Match、COW、输出和状态保护门禁。同时用本轮真实Census增加CPU来源归因回归，并在失败时先保存完整Epoch/Plan/响应。之后另行冻结并授权新运行，不能覆盖此次失败。

本轮遵守失败即停；不自动修复、不重试、不扩展工作负载。
