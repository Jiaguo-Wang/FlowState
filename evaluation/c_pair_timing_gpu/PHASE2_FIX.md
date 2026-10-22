# phase2 修正记录（attempt1 → attempt2）

- 失败：w1_CB_FS 在 b5d4_apply 溯源检查 BLOCKED（"缺少本生命周期来源线索"）。
- 根因：node_id 为进程生命周期计数器，跨 flush 重建不稳定——同内容节点在 warmup1 观察为 18、
  warmup2 重建后为 51（w1_CB_FS.attempt1/run/history_01.json vs history2_01.json）。
  control.py 逐字节复用 pilot，按 node_id 在 history_XX.json 中找来源线索，首轮预热的历史无法匹配
  flush 后重建的候选。
- 修正：仅改 runner 接线——strict_warm 模式下首轮预热写 history_pre_XX.json（留证），
  二次预热（真正物化当前候选的一轮）写 history_XX.json（控制溯源输入）。
  control.py / bridge.py 与所有正确性检查零改动。溯源语义更正确：v1.2 时间线合同对二次预热同样成立
  （同一 16 条冻结请求、round0/round1 物化轮次不变）。
- 证据处置：w1_CB_FS* 与 execution_started_p2/implementation_lock_p2/launch_p2_failure
  均改名 *.attempt1 保留。
