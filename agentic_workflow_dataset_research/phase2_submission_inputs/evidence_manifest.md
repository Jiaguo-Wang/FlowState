# 证据、版本与复现

本目录是独立离线交接产物。输入来源仅读取；没有修改旧正式数据/矩阵/模型/Runtime，没有 GPU、Engine、Replay、策略比较或预算调整。

## 复现入口

在 `/home/wjg/code/FlowState` 执行（仅写本交接目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/wjg/uv312/bin/python agentic_workflow_dataset_research/phase2_submission_inputs/audit_weka.py
PYTHONDONTWRITEBYTECODE=1 /home/wjg/uv312/bin/python agentic_workflow_dataset_research/phase2_submission_inputs/prepare_samples.py
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' /home/wjg/uvgood/bin/python agentic_workflow_dataset_research/phase2_submission_inputs/tokenize_samples.py
PYTHONDONTWRITEBYTECODE=1 /home/wjg/uv312/bin/python agentic_workflow_dataset_research/phase2_submission_inputs/build_reports.py
```

audit_weka 使用已归档的官方文件树；prepare_samples 只使用五个本地分片。若 /tmp 原始分片已移走，应以 selection_lock 的摘要找到相同内容，不得换版本。已保存 16 组完整消息，因而 tokenizer 验证与交接无需再次扫描分片。脚本没有网络请求、模型加载、策略执行。

## 文件身份

`evidence_hashes.json` 包含原始五分片、Weka 全文件、本轮脚本、报告、原始消息及生成 token 文件、所引用源码和官方代码的 SHA-256；清单排除自身及本索引以避免循环摘要。tokenizer 文件摘要另见 `tokenizer_identity.json`，对应本地 `/home/wjg/models/qwen3.5-9b/`。

`official/revision.txt` 保存 AIPerf commit；官方文件版权与许可证头保留原样，不是新增英文注释。`official/weka_tree.json` 保存公开源文件的 LFS 身份。本轮源码摘取供审计，不代表已运行该软件版本。

## UNKNOWN 和限制

- 原始 LMCache 顶层工具定义、原生 fork / ready / queued 事件、任务 ID 的原生调度含义未知。
- 本地 tokenizer Hub revision 未保存，文件 SHA 固定；旧 GPU 软件环境与当前 5.13.0 完全等价未知。
- target engine 的 context limit、tracking checkpoint 生成、owner/generation、FA frontier、容量和恢复正确性均未动态认证。
- Weka hash 不能还原原文本，检测链不是直接 fork 事件；旧 Formal 70 条边没有因本次数据准备得到修复。
- source_type“实验回放”描述本次交接材料的使用方式及已有同源会话结构，不声称已证明每条原始会话的采集来源。
- 输入内容仅本地保存，不在报告打印全文；不得将本目录发布行为推定为已经授权。

`representative_retokenization_check.json` 记录三个代表案例共六个输入重新模板化后逐 token 一致。

`validation_summary.json` 为本次 16 组输入的结构门禁结果，PASS 不包含任何 GPU 或算法胜负结论。
