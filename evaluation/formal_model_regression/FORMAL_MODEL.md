# 正式恢复模型集成回归

本目录记录 `position_aware_quadratic_v1` 集成后的纯离线回归。输入 workload、候选、
compatibility、baseline 信号和历史 GPU 制品均未修改，也未执行 GPU 或 TraceLab policy
comparison。

正式公式、参数、单位和适用域见 `evaluation/FORMAL_RECOVERY_MODEL.md`。本目录分别保存：

- `controlled_regression.json`：全部受控 workload 的旧新选择和正式模型指标；
- `selection_diff.csv`：逐 workload、预算、策略的选择差异；
- `oracle_regression.csv`：正式模型下的 exact Oracle 与 FlowState regret；
- `h100_selection_audit.json`：四个历史 H100 代表点的选择与可复用性；
- `h100_artifact_reusability.md`：H100 制品复用结论；
- `trace_model_compatibility.json`：冻结 TraceLab workload 的非策略兼容性抽检。

Global-LRU、Equal-Share、Workflow-Only、KVFlow-style 与 Marconi-style 的 checkpoint
selection 在旧新模型间保持不变。Recovery-Only、FlowState 与 Oracle 可以合法地因正式
objective 改变而产生新选择。
