# Phase B5-D4a 最终报告

**Gate：READY。仅指实验来源门禁修复及CPU合同回归完成，不是GPU恢复、性能配对或跨Workflow血缘认证READY。**

## 修复结果

原B5-D4错误地将不同Workflow当作通用合法恢复条件。修复后：

- Greedy使用Node20的同Workflow首次观察线索，不因该标签拒绝；仍要求严格准入、合法位置、唯一计划支持源及原物理安全门禁。
- Exact使用Node27的不同Workflow线索，同样按Strict合同处理；首次D2后可见仅为线索，保持PROVENANCE_UNVERIFIED。
- Native实际匹配继续核对当前绑定的Node/Slot、E/H和输入Token，原COW、Cold输出、组件和FA保护检查不变。
- 兼容性、实际匹配、来源血缘分开记录；所有本轮来源回归均不产出已认证跨Workflow恢复声明。
- 当前观察、Strict Epoch、Plan及来源预期逐阶段落盘；控制响应不论ok与否先保存原始字节和解析对象，再返回或抛异常。

## CPU结果

**30/30通过。** 原始结果见cpu_regression_results.json、cpu_tests.log。

|范围|用例|
|---|---|
|真实失败来源回归|01：Node20同Workflow允许；02：Node27不同Workflow线索允许|
|不升级血缘|03、27：首次观察和自称认证标记不成为血缘证据|
|严格兼容拒绝|04–09：不兼容、UNKNOWN、关系变化、缺失配对、边界变化、组件条件冲突|
|原生产物理合同|10：静态记录正例；11–18：已驱逐、错Slot、旧Engine/窗口、Conv/Temporal缺失、保护/引用|
|实际来源核验|19–21、29：错误Node/Slot/前沿/Token拒绝；30：旧D3真实Match记录仅用于CPU核验，不升级血缘|
|失败保存|22–26、28：阶段日志、旧失败RPC、坏JSON、空响应、禁止覆盖、实际修复control.apply失败分支到客户端落盘|

用例26使用明确标注的静态观察夹具，在来源处注入失败；真实GPU观察与Reconciliation调用数为0。既有D3严格Epoch只作为CPU计划合同夹具，D4真实Census用于来源和物理负面回归；不冒充恢复了D4未持久化的内部Plan。

两次CPU夹具/断言开发错误日志保留：漏填Prepared.ids及JSON tuple/list比较。修正后全部通过，没有放宽实际安全条件。

## 修改和完整性

所有修复仅在evaluation/exact_greedy_paired_gpu_fix_b5d4a/。与旧版相比，control.py和runner.py有修改，bridge.py、observation.py、protocol.py原样复制以保持独立模块路径。新source_contract.py和persistence.py分别负责来源合同与证据保存。

每个副本的修改前后SHA和Diff见source_integrity.json及*.py.diff；546项旧证据、冻结输入、实现锁与依赖摘要全部匹配。原B5-D4的BLOCKED报告、日志、代码和执行锁未覆盖，生产Greedy、Exact、Phi、Compatibility、StateController未修改。

## 尚未认证与下一轮边界

当前没有能把Node27张量创建/继承事件绑定到D请求的完整原生血缘链。不同Workflow首次Census线索、Native命中相同Node以及输出一致，均不能单独补齐这一缺口。provenance_contract.md预先分开定义来源选择、恢复正确性、性能数据资格及跨Workflow声明资格；未追溯放宽B5-D4整体READY。

修复版控制函数原有status=READY只表示该次控制合同完成，不代表完整成对实验或跨Workflow血缘门禁通过。本轮没有新GPU调用，没有Eviction或Pending执行，没有计时比较。

未提供自动启动脚本、未切换旧冻结入口、未重跑B5-D4。未来执行须另建新输出目录和执行锁，明确使用本修复副本并另行授权；不能复用旧Engine的Node/Slot绑定。

## CPU复现

在仓库根目录执行：

```bash
/home/wjg/uv312/bin/python -B -m evaluation.exact_greedy_paired_gpu_fix_b5d4a.test_regression
/home/wjg/uv312/bin/python -B -m evaluation.exact_greedy_paired_gpu_fix_b5d4a.finalize
```

本轮完成后停止，等待审查。
