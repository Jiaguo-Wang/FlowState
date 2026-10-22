# 冻结证据索引

仓库：`/home/wjg/code/FlowState`。本轮仅CPU文件解析，未运行策略、GPU、Docker或Replay。

## 权威来源

|来源|作用/身份|
|---|---|
|evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/manifest.json、batch_summary.json、execution_lock.json|根清单及正式入选g001–g004；g005/006不纳入比较|
|同根g001–g004/snapshot.json、summary.json、manifest.json|实际Token、准入矩阵、物理驻留、来源组及同边界；各snapshot摘要与summary一致|
|evaluation/strict_compatibility_b4c/results.json|权威已冻结收益、选择集合、J、Exact所有最优集合、各步trace；本轮不导入evaluate模块|
|同目录evidence_manifest.json、input_integrity.json、evaluate.py|冻结结果完整性；EPS=1e−9；历史生产输入版本|
|evaluation/sota_baseline_audit_b5a/source_lock.json、final_report.md、adaptation_contract.md|仅作为已有源码审计；输入充分性主张本轮独立核对|
|B5-A raw/marconi_repo|HEAD=08016617b1524e6bf6ac29b680641cc945bda7f0，git工作树无差异|
|B5-A raw/kvflow_repo|HEAD=7ef897eaedc16626a281ccfe0d453fae80251e8f，git工作树无差异；论文版对应UNVERIFIED|
|evaluation/strict_compatibility_b4b0/capture.py:52–88；evaluation/rq3_openhands_neutral_collector.py:480–510|实际采集与受控recency来源|
|evaluation/sota_metadata.py:64–77、107起|步骤固定1和同Workflow增量长度代理，不是原生SOTA输入|
|/home/wjg/models/qwen3.5-9b/config.json|真实模型层数/维度；SHA需与各snapshot配置一致|
|evaluation/runtime_control_closure_r2a/final_report.md|当前optimizer与B4-C历史版本变化的来源：严格生产接口接线，已有16/16回归|
|evaluation/runtime_control_closure_r2c1b2/final_report.md|只支持g001一条跨Workflow恢复，不推广到全部68边|

## 已完成摘要核验
- {"manifest": "/home/wjg/code/FlowState/evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/manifest.json", "files": 339, "status": "PASS"}
- {"manifest": "/home/wjg/code/FlowState/evaluation/strict_compatibility_b4c/evidence_manifest.json", "files": 11, "status": "PASS"}
- {"manifest": "/home/wjg/code/FlowState/evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g001/manifest.json", "files": 14, "status": "PASS"}
- {"manifest": "/home/wjg/code/FlowState/evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g002/manifest.json", "files": 14, "status": "PASS"}
- {"manifest": "/home/wjg/code/FlowState/evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g003/manifest.json", "files": 14, "status": "PASS"}
- {"manifest": "/home/wjg/code/FlowState/evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g004/manifest.json", "files": 14, "status": "PASS"}
- {"repository": "/home/wjg/code/FlowState/evaluation/sota_baseline_audit_b5a/raw/marconi_repo", "commit": "08016617b1524e6bf6ac29b680641cc945bda7f0", "status": "PASS"}
- {"repository": "/home/wjg/code/FlowState/evaluation/sota_baseline_audit_b5a/raw/kvflow_repo", "commit": "7ef897eaedc16626a281ccfe0d453fae80251e8f", "status": "PASS"}

B4-C历史输入摘要存在一项差异，必须显式保留：[{"path": "/home/wjg/code/FlowState/flowstate/optimizer.py", "frozen_sha256": "eabf84c7c5d50db1d5e9e60821b0803f760f2c81750a846d960f4cb638f87b98", "current_sha256": "b0060e1cc9fc5d14352a998e8ef95b9b9a8ed3a28a562303595bea749e6f360c"}]

这是当前optimizer已在R2-A接入Strict Epoch的版本变化；冻结B4-C结果本身和四张snapshot摘要未变。故本轮不使用当前optimizer重跑或替代原始结果。所有结果与其历史执行合同关联，而非宣称当前源码与历史完全相同。

B5-A source_lock记录commit但不是全面文件SHA清单；本轮核验本地commit和clean状态，关键输入即时SHA见source_hashes.json。SHA保证内容一致性，不是独立真实性认证。不下载或刷新仓库/论文。

根执行锁作为历史环境身份保留；本轮不重新认证当前GPU、镜像或扫描全部模型权重。只核验本次用到的冻结数据和配置，不能声称完成动态环境复验。

已有同名输出目录包含早先结构脚本与metrics。本轮未覆盖它们；新增audit.py只读冻结收益、补充完整支持统计/Native事件映射。两个版本主要结构计数一致，旧脚本未在本轮执行。

复核入口：`/home/wjg/uv312/bin/python -B -m evaluation.sota_baseline_gate_b5b0.audit`。仅写本目录派生统计，不实现策略；K_min覆盖穷举由任务明确允许。
