# T2-G1 Smoke 输出一致性只读审计

日期：2026-10-27。**判定：PASS**（三策略输出一致性结论成立；所报三字段"冲突"为检查脚本的文案缺陷与字面布尔，非数据不一致。按只读约束未修改任何代码与结果。）

## 1. 疑点与三字段的生成逻辑追踪

smoke_results.json 中同时出现：`check.pass=true`、`check.detail="输出与独立Cold参照不一致"`、`arms[].output_match_cold=true`。生成逻辑在 [verify_smoke.py](../verify_smoke.py)：

| 字段 | 代码位置 | 生成方式 | 审计结论 |
|---|---|---|---|
| `pass` | verify_smoke.py:48 + 31-34 | `s['output_ids'] == cold['output_ids']` 真实比较；`check()` 失败即抛 AssertionError（文件根本不会写出） | **真实比较结果，无默认值/硬编码布尔转换** |
| `detail` | verify_smoke.py:48 | 常量字符串 `'输出与独立Cold参照不一致'` **无条件附带**（不区分 pass/fail 的文案分支缺失） | **缺陷：文案错误**——pass=true 时 detail 仍写失败式文案 |
| `output_match_cold` | verify_smoke.py:60 | 行内**字面 True**，但位于 check() 之后，仅当真实比较通过才可达 | 语义上有比较结果背书，属代码质量问题（字面常量而非计算值） |
| `verdict` | verify_smoke.py:63 | 字面 'PASS'，但聚合方式为异常即停（24 项任一失败则不落盘） | 聚合正确（存在性⇒全过）；24=8 检查×3 臂，计数核对一致 |

其余 23 项检查的 detail 逐一扫描：均为动态值（如 `E实际=%s 预期=%s`、`驱逐数=%d`）或空串，**无同类常量文案问题**；唯一缺陷即第 48 行。

## 2. 原始数据独立逐 Token 比较（不依赖 smoke_results.json）

从四份原始 `run/output.json`（流式落盘的权威输出）直接重比，并交叉核对 summary 派生字段：

| 来源 | output_ids | 与 Cold 逐 Token |
|---|---|---|
| pilot cold_C_B（独立空缓存生命周期） | [220, 197, 1, 198, 197, 1, 198, 197] | 基准（summary 派生一致 ✓） |
| smoke FlowState-Greedy | [220, 197, 1, 198, 197, 1, 198, 197] | **一致**（summary 派生一致 ✓） |
| smoke Marconi-Evict | [220, 197, 1, 198, 197, 1, 198, 197] | **一致**（summary 派生一致 ✓） |
| smoke KVFlow-inspired | [220, 197, 1, 198, 197, 1, 198, 197] | **一致**（summary 派生一致 ✓） |

**不存在第一个不同 Token：三策略与 Cold 全部 8/8 Token 相同，无受影响策略。**

## 3. 比较前提一致性核验（请求/配置/参数）

| 前提 | 结果 | 证据 |
|---|---|---|
| 同一请求输入 | 三臂 request.json 的 pending token_ids 与 cold 逐 Token 相同，token_prefix_sha256 相同 | 各 `run/request.json` |
| 同一生成参数 | 四者均为 `{'temperature': 0, 'max_new_tokens': 8, 'ignore_eos': True}` | 各 `run/request.json` |
| 同一模型与引擎配置 | prepared_inputs.json 与 pilot **逐字节一致**（sha256 复核；qwen3.5-9b / sglang 0.5.17 / transformers 5.12.1 / chat_template a4aee8af…） | smoke case 副本 vs pilot 原件 |
| 同一镜像 | image_id sha256:1dfce903… 相同；smoke_lock.json 已锁共享文件与 pilot 模块摘要 | 各 `source_identity.json`、`smoke_lock.json` |

Cold 与 strict 的差异仅在缓存状态（Cold=空缓存全量 Prefill 5152；strict=按各自 E 恢复），这正是被验证的语义：恢复路径不影响生成结果。

## 4. 结论与建议

- **Smoke 正确性结论成立**：`pass=true` 与 `output_match_cold=true` 均由真实比较背书，三策略输出与独立 Cold 参照逐 Token 一致；"不一致"字样纯属 verify_smoke.py:48 的常量文案缺陷。
- 建议修复（**本轮未执行**，遵守只读约束）：将 detail 改为按结果分支（如 `detail=''` 于通过时、或统一写 `'输出与独立Cold参照逐Token一致=%s' % ok`），并可将行级 `output_match_cold` 改为计算值 `out == cold_out`。修复后重跑 verify_smoke.py 会重写 smoke_results.json，须在下一任务中与结果刷新一并处理。

**证据路径**：`evaluation/t2g1_three_policy_gpu/smoke_results.json`、`evaluation/t2g1_three_policy_gpu/smoke/strict_{FlowState-Greedy,Marconi-Evict,KVFlow-inspired}_C_B/run/{output.json,summary.json,request.json}`、`evaluation/c_shared_gpu_pilot_k3/cold_C_B/run/{output.json,summary.json,request.json}`、`evaluation/t2g1_three_policy_gpu/verify_smoke.py`。
