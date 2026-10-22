# 根因与最小修复

## 已确认根因

原B5-D4 control.py:97将 h.workflow_id != p.workflow_id 纳入所有策略通用的来源合法性条件，第99行再要求唯一跨Workflow源。这混淆了严格兼容性与来源血缘。p01_Greedy/history_03.json记录Node20/2432/Slot8首次在C1后出现，首次观察归因与C相同，因此被错误过滤。原失败没有实际驱逐或正式Pending执行。

另一个缺口是原control.py直到协调成功才把raw_epoch/epoch/plan加入返回对象；原runner.py:36的ControlClient._call对ok=false提前抛异常，导致第37行写盘不执行。

## 独立副本修改

原B5-D4目录完全只读。修复版control.py、runner.py以及原样复用的bridge.py、observation.py、protocol.py全部位于本目录；每个副本都有原文件SHA、新文件SHA和统一Diff。没有修改生产或历史执行入口，也没有生成新GPU启动锁。

- `source_contract.py:choose_source`：先调用生产validate_plan；检查原始与严格边域、状态及条件完全一致；校验目标Token/位置；从实际选集选固定C的唯一最深合法共享源。不使用Workflow不同作为一般合法性条件。
- `source_contract.py:validate_native_source`：复用原Node/Slot/E/H与Token校验要求，独立返回实际匹配与PROVENANCE_UNVERIFIED，明确不认证跨Workflow血缘。原COW、输出、组件与FA保护检查仍由runner执行。
- `control.py:apply`：在观察后、Strict Epoch构造后、Plan验证后和来源选择后分阶段保存已有证据。失败先设置生命周期fail-stop，再保存失败阶段、当前已有Epoch/Plan/原始观察和异常。没有绕过bind/preflight、生产reconcile_strict或改动策略计算。
- `persistence.py:Journal`：阶段文件独立、排他创建、flush/fsync；不可覆盖旧结果。磁盘写失败不授权继续控制。
- `persistence.py:call_saved`及runner内`AuditedControlClient`：保持原请求/响应线协议，在解释ok之前保存请求、原始响应字节和解析JSON；失败保留原始异常。空响应、损坏JSON也先保留原始字节。没有修改公共ControlClient。
- `runner.py`：保存来源校验结果，并在摘要中显式标记provenance_status及cross_workflow_resume_certified=false。

## CPU夹具边界

来源线索和物理拒绝夹具来自本次D4八条真实历史Census。由于D4失败未保存完整Epoch/Plan，测试使用D3既有严格Epoch作为CPU合同夹具，逐Token核对D4记录的候选存在性，再构造声明为测试用途的保留计划；不是恢复D4未保存的内部Plan，也不构建新的Runtime认证矩阵。

失败集成测试调用修复版control.apply，静态观察被替换为明确标注的CPU夹具；只执行CPU规划至注入的来源失败。实际GPU观察和Reconciliation被设为一旦调用即报错，并断言调用次数为0。没有Fake Runtime成功实验。

首次CPU运行有夹具漏填Prepared.ids，第二次有tuple/list持久化比较断言错误。两次失败日志保留；修正的是测试夹具与JSON断言，没有放宽生产或来源门禁。
