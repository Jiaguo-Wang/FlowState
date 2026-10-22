# 独立多 Workflow 工作负载扩展——最终报告

**Gate：READY**（预注册来源、CPU 输入、因果截断与 Token 结构均可复核，成功与失败样本完整保留；本轮失败数为 0。READY 不表示 GPU Checkpoint 已物化，也不表示 FlowState 具有性能优势。）

## 1. 完成范围

基于 B5-C1 全量共享结构审计（1216 个同任务多 Session 组）与 B5-C2 已验证的 Tokenizer/模板/截断路径，新准备 **4 组独立多 Workflow 工作负载**（每组 2 个真实 instance_id × 各 2 条真实 Session = 4 个 Controlled Pending，共 16 Pending）。本轮只完成来源冻结、CPU 输入重建、Token 共享结构与因果截断核验；未启动 GPU、未构造 Strict Runtime Snapshot、未运行任何分配算法。

## 2. 来源与预注册

- 选择规则先于一切 Token 核验固定（`sampling_protocol.md`）：B5-C1 instance_id 字典序中剔除 B5-C2 已用 2 任务后依序取前 8 任务，连续成组 w1–w4；每任务取 session_id 字典序前 2 Session。
- 排除核验通过：8 任务、16 Session 与 B5-C2 零重叠，组间零重叠。
- `source_selection_lock.json` 在分词前写出（含 parquet 行号、source_id、assistant 索引、cut、截断历史 SHA256）；16/16 来源均存在合法第三次请求（cut=6），**无替换、无失败样本**。
- 未依据恢复收益、Exact 胜负或预期 Checkpoint 位置选择任何样本。

## 3. CPU 输入与结构结论

正式 Token 一律在冻结镜像 CPU 容器生成（runc、无 GPU、断网、只读源码/模型、仅本目录可写；Transformers 5.12.1/Tokenizers 0.22.2/SGLang 0.5.17，与 B4/B5-C2 身份链一致；主机 uv312 的 Transformers 4.57.6 未用于正式 Token）。48 份数组（16 标签 × 3 轮）全部落盘；规范化/模板/截断语义与 B5-C2 完全一致（冻结函数 AST 原样加载，B4 执行锁逐文件 SHA 核对通过）。

| 组 | 任务一 / 任务二 | Pending Token A/B/C/D | 四方LCP | AB | CD | 跨任务 | 分类 |
|---|---|---|---:|---:|---:|---:|---|
| w1 | 12rambau__sepal_ui-747 / 3YOURMIND__django-migration-linter-258 | 4250/4308/4497/4582 | 2438 | 3451 | 3771 | 2438 | nested_two_task |
| w2 | 4degrees__clique-26 / ARMmbed__mbed-tools-293 | 4209/4122/4191/4667 | 2438 | 3511 | 3570 | 2438 | nested_two_task |
| w3 | ASFHyP3__hyp3-sdk-53 / ASPP__pelita-696 | 4215/4184/4548/4543 | 2438 | 3535 | 3471 | 2438 | nested_two_task |
| w4 | ASPP__pelita-798 / Aarhus-Psychiatry-Research__timeseriesflattener-186 | 4866/4847/4513/6971 | 2437 | 3549 | 3488 | 2437 | nested_two_task |

- 每组均有共同内容前缀（2438/2437 = System 2428 ＋ 用户头部 10/9 Token，支持 {A,B,C,D}）与两个互不重叠的任务内更深子集（支持恰为 {A,B}、{C,D}）；消息边界逐条精确对齐（exact_token_prefix 全真）。
- 逐 Session 因果截断一致：只消费索引 0–5，第 3 条 Assistant 仅作定位标记；64 条工具结果（8/组）全部可回溯先前 tool_call_id；采集器五项未来信息检查全为 false；墙钟 UNKNOWN。
- 结构同质（四组皆嵌套、与 B5-C2 同型）是字典序确定性规则的产物，非筛选结果；共同前缀主体是数据集统一 System 提示，跨任务分叉在 User 工作区路径——**不是原生 Fork**。
- **文本匹配、Token LCP 与历史 Session 重复都不是可执行 Recurrent State 共享**：全部 `runtime_materialization=UNKNOWN`、`checkpoint_created=False`；FA frontier 与 Planning Target 未观测，L≤T 逐边确认留给 Runtime 阶段。

## 4. 独立复核与完整性

`finalize.py`（主机、只做哈希与重算，不分词）复核：parquet 与 B5-C1/B5-C2 产物摘要、B5-C2 排除、48 份数组长度/双摘要、全部 6×4 配对 LCP 独立重算、边界精确性、工具因果顺序、容器清理（`docker ps -a` 按本轮名称过滤为空）。结果 `verification.json=READY`；`artifact_manifest.json` 覆盖本轮全部文件。

执行异常记录：一次协议文档章节编号笔误（"7→4"）在任何脚本运行前修正，不影响选择与 Token；除此之外 prepare/容器/finalize 三阶段无失败重试，日志见 `prepare_run.log`、`cpu_run.log`。

## 5. 交付物

`sampling_protocol.md`、`prepare_sources.py`、`source_selection_lock.json`、`tokenize_cpu.py`+`run_cpu.sh`、`input_configuration_lock.json`、`causal_checks.json`、`requests.json`、`token_prefix_matrix.json`、`token_sharing_summary.csv`、`cpu_input_audit.json`、`verification.json`、`artifact_manifest.json`、`workload_inventory.md`、`runtime_collection_plan.md`、本报告；复核输入在 `w1/`–`w4/`（各组截断历史、48 份 Token、渲染文本、逐 Token offset）。

**本轮完成后停止，不自动进入 Runtime 采集，等待审查。**
