# Step 13G-B1 · AgentX Runtime-Compatible Shared-Coverage Population Census 最终报告

**状态：** `AGENTX_RUNTIME_POPULATION_CENSUS_READY`
**报告生成时间：** 2026-09-06 21:57:30
**Canonical Census Root：** `/tmp/pytest-of-wjg/pytest-130/test_run_census_integration0/census`

## 1. Frozen Input

- SHA256 match：`True`
- traces：`393`
- B0.1 audit root：`/home/wjg/data/agentx/audits/agentx_online_compatibility_20260904_185622`
- runtime context limit：`131200` tokens

## 2. Context Census

- all online-safe shared traces：`84`
- shared traces with ≥1 context-compatible epoch：`23`
- shared context-compatible epochs：`1206`
- shared context-ineligible epochs：`13773`
- chain-only context-compatible epochs：`17105`

## 3. Context-Length Distribution

### 3.1 全 corpus requests
- count：`98827`
- min=128, mean=218921.77, median=142016.0, p75=310464.0, p90=549504.0, p95=682880.0, p99=863488.0, max=989824
- requests ≤ 131200：`46442`
- requests > 131200：`52385`

### 3.2 online-safe shared-coverage traces
- count：`55547`
- min=128, mean=244106.15, median=160960.0, p75=355968.0, p90=603520.0, p95=734528.0, p99=894080.0, max=989824
- requests ≤ 131200：`23347`
- requests > 131200：`32200`

### 3.3 SHARED_CONTEXT_ELIGIBLE epochs
- count：`82762`
- min=128, mean=51684.86, median=48064.0, p75=72320.0, p90=95168.0, p95=104384.0, p99=118208.0, max=130624
- contexts ≤ 131200：`82762`
- contexts > 131200：`0`

## 4. Replay Semantics

- corpus stores：`BLOCK_SHAPE`
- AIPerf replay method：`AIPerf HashIdsPromptSynthesisMixin + ConversationReconstructor：通过 decode_block_tokens(hash_ids) 把每个 hash block 映射为确定性 Qwen token 序列，剩余 tail 用 sha256-keyed 采样补齐，保证 sum(tokens) == in_tokens。`
- exact length preservable：`True`
- prefix topology preservable：`True`
- FORK semantics preservable：`True`
- tokenizer compatibility：`CONDITIONAL`
- **REPLAY_CONSTRUCTABILITY：`READY`**

> REPLAY_CONSTRUCTABILITY = READY：AIPerf 的 deterministic block-token synthesis 可以保持物理 prefix-sharing 拓扑、精确 input length 与 conversation/FORK 语义；唯一前提是目标 tokenizer alias 可解析。

## 5. Formal Population Protocol

- selection：`EARLIEST_ELIGIBLE_SHARED_COVERAGE_EPOCH`
- one snapshot per trace：`True`
- policy blind：`True`
- future blind：`True`
- no candidate truncation：`True`

## 6. Policy Blindness Gate

- POLICY_BLIND：`PASS`
- FUTURE_BLIND：`PASS`

## 7. Candidate Population

- formal snapshots：`23`
- pending：min=2, mean=3.65, median=3.0, p75=4.0, p90=5.0, p95=7.0, p99=8.0, max=8
- candidates：min=6, mean=24.7, median=12.0, p75=21.0, p90=38.0, p95=109.0, p99=149.0, max=149
- shared candidates：min=1, mean=1.35, median=1.0, p75=2.0, p90=2.0, p95=2.0, p99=3.0, max=3
- max d_t(c)：`2`

## 8. Budget Feasibility

### 8.1 r=0.25
- mean K：`5.83`
- median K：`3.0`
- p95 K：`27.0`
- max K：`37`
- Exact tractable snapshots：`18`
- Exact intractable snapshots：`5`

### 8.2 r=0.5
- mean K：`12.09`
- median K：`6.0`
- p95 K：`54.0`
- max K：`74`
- Exact tractable snapshots：`16`
- Exact intractable snapshots：`7`

### 8.3 r=0.75
- mean K：`18.04`
- median K：`9.0`
- p95 K：`81.0`
- max K：`111`
- Exact tractable snapshots：`14`
- Exact intractable snapshots：`9`

- Exact tractable estimated cases：`18`

## 9. OpenHands Comparison

- OpenHands max d：`1`
- AgentX runtime-compatible max d：`2`
- OpenHands branching：`False`
- AgentX runtime-compatible branching：`True`
- Fills shared-coverage gap after 131200 filtering：`True`

## 10. Controlled Fork-DAG Role

`MICROBENCHMARK_ONLY`

## 11. Scientific Conclusion

1. 84 个 shared-coverage traces 中，在 131200 context limit 下仍至少有一个 eligible epoch 的 trace 数：`23`。
2. survive 后 runtime-compatible 的 max d_t(c) = `2`。
3. shared coverage 在 131200 过滤后仍然丰富：`YES`。
4. 确定性 replay 构造能力：`READY`（CONDITIONAL tokenizer）。
5. AgentX 可以成为第二个正式 RQ3 workload：`YES`，前提是 context-length blocker 在 collection 阶段被采样/过滤策略处理。
6. population selection 完全 policy blind：`YES`。
7. 不需要 candidate truncation：`YES`（formal population 不因 Exact tractability 过滤）。
8. Exact OPT 在 r=0.25 下预计可覆盖 case 数：`18`。
9. Controlled Fork-DAG 角色：`MICROBENCHMARK_ONLY`。
10. 下一步值得进入 neutral AgentX runtime collection：`YES`（在 tokenizer alias 验证后）。

---

*报告生成于 FlowState RQ3 Step 13G-B1 · CPU-only census*