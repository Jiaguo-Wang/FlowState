# 来源合同与独立资格

本合同只用于独立修复版实验入口，不修改生产Strict Compatibility或原B5-D4协议、执行锁及BLOCKED结果。没有新GPU运行。

|判断|证据要求|本轮行为|
|---|---|---|
|严格可执行兼容|合法Strict Epoch与Plan、完整已准入关系、原始Token前缀、L≤T；随后真实当前Bind/Preflight|不使用Workflow不同作为条件；UNKNOWN、缺失或冲突关系拒绝|
|Native实际恢复来源|实际请求Token身份、实际Match的Node/源Slot/E/H与本生命周期绑定及计划一致|继续强制检查；不能用输出相同替代|
|恢复计算正确性|上述条件加Native COW完成、源与工作槽隔离、完整执行、Cold输出一致、历史状态保护|沿用原检查，来源标签既不能授权，也不能替代这些检查|
|跨Workflow血缘|可信原生物化/继承/复制事件关联创建请求、完整Token、组件与物理生命周期；再关联目标实际恢复|当前首次Census不足，统一PROVENANCE_UNVERIFIED，cross_workflow_resume_certified=false|

## 同Workflow与不同Workflow线索

真实失败记录中Node20首次出现在C1后，来源线索与目标C同Workflow；严格前缀共享不因此失效。Node27首次出现在D2后，来源线索与C不同，但不证明其张量由D创建。两者都可通过合法来源选择，不因Workflow标签被拒绝，也都不因此获得血缘认证。

保留Exact“希望认证D历史来源”的目标，输出requested_origin= D历史及原始first_observed_after；即使该字段写作D2，也保持PROVENANCE_UNVERIFIED。当前没有实现可验证完整原生血缘的认证器，输入中自称certified的标记不会被采信。不能把字段名称source_history当作认证结论。

## 预先确定的独立资格

1. 来源选择资格：生产validate_plan通过，原始与Strict关系一致，固定C有唯一最深、已准入、正位置共享选中源；无需Workflow不同。
2. 请求恢复正确性资格：真实Apply、Admission、Match/COW、输出和状态后验全部通过。来源血缘未知不能被记录成安全成功的替代条件。
3. 成对性能数据资格：双方恢复正确性通过、起始状态可比、相同冻结输入/配置/预算、真实GPU计时边界可信。若满足这些独立条件，来源血缘未知须单列，不得计入“已证明跨Workflow恢复”。本轮没有实际数据满足这些资格。
4. 跨Workflow声明资格：必须额外通过完整血缘链。本修复不会输出该认证成功。
5. B5-D4原整体READY还含跨Workflow目标；本轮不追溯修改该门禁。未来若仅正确性/性能资格成立而血缘仍未知，应拆分报告并按原完整要求保留PARTIAL，不能用本次CPU READY替代。

物理安全仍由原生产bind/preflight、连续Scheduler窗口、Reconciliation及后验合同保障。本轮测试中的历史Slot仅为静态CPU拒绝夹具，绝非新Engine当前绑定。未来必须重新观察并绑定。
