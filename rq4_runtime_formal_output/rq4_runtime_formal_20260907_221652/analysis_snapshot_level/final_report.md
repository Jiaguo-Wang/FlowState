# RQ4-D Snapshot-Level 正式统计报告

- 状态：RQ4_STATISTICS_READY
- 正式统计单位：snapshot
- population：24/24
- runs consumed：216/216
- requests consumed：864/864
- 聚合层次：四个 pending requests 的 run 均值，再取三次 repetition 的均值
- bootstrap：round-stratified，10000 次，seed=20260907
- outliers removed：NO
- formal artifact modified：NO

## Snapshot-level TTFT

- LRU：mean=164.705002 ms，median=158.995561 ms，n=24
- Marconi：mean=129.097170 ms，median=121.177012 ms，n=24
- FlowState：mean=94.024580 ms，median=97.397916 ms，n=24

### FlowState vs LRU

- mean paired absolute reduction：70.680422 ms
- 95% CI：[63.806243, 78.300269] ms
- median paired absolute reduction：55.779853 ms
- mean paired relative reduction：43.057902%
- 95% CI：[39.940314%, 46.066374%]
- median paired relative reduction：40.346645%
- win/tie/loss：24/0/0

### FlowState vs Marconi

- mean paired absolute reduction：35.072590 ms
- 95% CI：[27.670502, 43.174819] ms
- median paired absolute reduction：44.023248 ms
- mean paired relative reduction：24.264401%
- 95% CI：[17.740919%, 31.321582%]
- median paired relative reduction：28.855116%
- win/tie/loss：22/0/2

## Executable gap

- LRU：mean G=2789.322917 tokens，median G=2824.000000 tokens
- Marconi：mean G=1856.656250 tokens，median G=2176.000000 tokens
- FlowState：mean G=767.989583 tokens，median G=544.000000 tokens
- FlowState vs LRU mean paired G reduction：2021.333333 tokens；W/T/L=24/0/0
- 与 LRU 比较时 DeltaG>0 且 DeltaTTFT>0：24/24；G 更小但 TTFT 更差：0/24；Spearman=0.969341
- FlowState vs Marconi mean paired G reduction：1088.666667 tokens；W/T/L=21/3/0
- 与 Marconi 比较时 DeltaG>0 且 DeltaTTFT>0：19/24；G 更小但 TTFT 更差：2/24；Spearman=0.940358

## 稳定性与诊断

- repetition median/P95/max CV：0.008588 / 0.041852 / 0.298240
- CV>0.20 的 snapshot-policy：2/72；仅报告，未删除。
- runtime drift diagnostic：WARNING
- request-level preliminary aggregate 未作为独立样本进入正式推断。
- RQ3 的 C(S) 未混入 RQ4 primary metric。

## 结论

在 24 个完整 formal snapshots 上，FlowState 相对 LRU 与 Marconi 的 snapshot-level TTFT 配对改善及 executable-gap 变化由冻结 artifact 直接重建；bootstrap 区间以 round-stratified snapshot 重采样获得。这些结果支持 allocation→executable gap→TTFT 的一致性链条，但相关性仅作为支持性证据。
