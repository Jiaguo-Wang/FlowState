# 实证核查工件说明

本目录存放 Phase 1 调研中实际执行的核查脚本，均可复现 `evidence_manifest.md` 中的 [实测] 结论。

- `inspect_tracelab.py`：解析 TraceLab v0.0.2 全量 trace（syfi_coding_trace.jsonl，1.2GB），
  统计顶层字段出现次数、session/rounds 分布、timing_events 事件类型分布。
  用法：`python3 inspect_tracelab.py <抽样行数>`
- `analyze_lmcache.py`：解析 lmcache-agentic-traces 全量 5 个 parquet 分片（2.37GB），
  输出 任务×模型 矩阵、每模型 session 数、来源分布，并对多模型任务抽查 input 消息流的第一个分歧点。
  依赖：duckdb（数据文件路径硬编码为 /tmp/lmcache_check/，复现时可修改）。
- cc-traces-weka 的核查为一次性 curl + python 内联解析（见 evidence_manifest.md 第 1 节命令），
  样本留存于 /tmp/weka_head.jsonl（4MB 头部）。
