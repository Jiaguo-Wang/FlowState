# 来源与认证等级

输入来自`evaluation/independent_multishare_workloads/`，不重新分词、不重新选择Session、不更改历史截断。四组按w1、w2、w3、w4执行，各自四Session、两个任务，每任务两个Session。12份/组冻结完整Token数组按原始文件SHA验证；八条历史按A1、B1、C1、D1、A2、B2、C2、D2回放。第三次输入来自已截断历史，不能把离线已准备输入称为自然在线Ready。

| 组 | 任务一 | 任务二 |
|---|---|---|
| w1 | 12rambau__sepal_ui-747 | 3YOURMIND__django-migration-linter-258 |
| w2 | 4degrees__clique-26 | ARMmbed__mbed-tools-293 |
| w3 | ASFHyP3__hyp3-sdk-53 | ASPP__pelita-696 |
| w4 | ASPP__pelita-798 | Aarhus-Psychiatry-Research__timeseriesflattener-186 |

精确Session ID、原始行号、Task ID、切断点和来源SHA保留在源`source_selection_lock.json`及每组`prepared_inputs.json`。原始parquet为`/home/wjg/data/agentic_coding_trajectories/sessions.parquet`，SHA-256为`3dd8ec3546cf771ce4ab2ac6c51ccefdd621197fa997a2cefb430b50df808fb6`。本轮`source_identity.json`锁定432项来源、模型、代码和配置摘要；执行前重新核对。

唯一旧清单差异是`cpu_run.log`：清单生成后追加了最终中文READY行。当前文件去掉固定末行后的SHA精确匹配原清单；`prepare.py`仅接受这一逐字节可证明的差异。`source_manifest_differences.json`保留原始不一致记录，新来源锁记录完整当前文件SHA。没有修改旧日志、清单、Token或冻结实验；该差异不影响数据身份。

所有新节点号和槽号来自本轮各自Engine，历史物理绑定不跨生命周期复用。候选沿用观察器`B5C3_NODE_*`命名，跨组必须连同group/epoch解释，不能仅凭相同数字认定物理对象相同。CPU适配器用首次出现记录提供Workflow标签，但这只是观察线索；评分仅使用Strict Epoch边，不用该标签创造兼容关系。

认证等级严格分开：

- 原始历史请求：实际提交Token与Native接收Token摘要核对。
- 四个Controlled Pending：完整Token、配置、同边界驻留、组件及引用满足条件时为EPOCH_ADMISSIBLE。
- Pending的Native物理匹配、实际恢复及输出正确性：本轮未执行，NOT_TESTED。
- 跨Workflow张量来源血缘：PROVENANCE_UNVERIFIED，不因Prefix共享或首次观察时间升级。

因此本轮可证明同一边界的严格共享恢复资格，不能证明四个Pending逐边实际恢复，更不能称为原生Agent Fork/Join或自然并发。独立性指来源任务/Session与此前B5-C2及其他组不重叠，不是统计总体独立性保证。
