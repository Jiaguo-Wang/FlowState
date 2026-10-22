# CPU正确性与负面门禁

最终66/66通过，见cpu_regression_results.json及cpu_tests.log。

- 新增8项：冻结真实Census驱动轻量构建，候选/三态边/T与重型记录完全相同；同Epoch两求解器选集、每Pending E/G/成本及J一致；组件未知保留UNKNOWN；FA前沿/槽/引用变化拒绝；不采用未验证generation；生产协调入口未替换。
- 原生产严格控制28项：Node/Slot/组件别名、缺候选、保护与引用、Host Copy、Epoch及预算/补集错误、陈旧View、活动请求、Admission、Transfer、域外状态、FA破坏、部分失败、中途变化、静默无操作、所选组件破坏、状态变更后抛错及记账等。
- D4a原30项：严格准入与来源血缘分开、UNKNOWN/不兼容拒绝、Native来源/Slot/输入/前沿不匹配拒绝、原始失败响应和异常证据保存。

使用既有D4b真实冻结记录作为CPU输入；Scope和CUDA同步仅在CPU测试中模拟，不能宣称GPU证据。第一次测试因本机uv312没有Torch而在测试补丁装载时报错，保留cpu_tests_attempt01.log；改为显式模拟同步接口后重跑全部66项通过，没有放宽生产检查。每个真实Engine创建前仍运行原真实Req/Batch接口预检查。

Runtime Smoke另完成轻重真实视图/计划一致、五项实际驱逐、协调前后全FA内容及Selected保护、Native Match/COW内容、8Token Cold对照。这些动态证据与CPU测试分别保存。
