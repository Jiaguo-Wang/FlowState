# 实验来源路径索引

编号前两位对应手册实验。全部普通文件路径、大小及核对范围见[artifact_inventory.csv](artifact_inventory.csv)；分类存在性与缺项见[evidence_categories.csv](evidence_categories.csv)。不存在原始命令记录时，仅给脚本入口并标记RECONSTRUCTED_FROM_CODE，不伪造shell历史。

| 证据编号 | 完整artifact root |
|---|---|
| 01_semantics | /home/wjg/code/FlowState/motivation/artifacts/runtime_validation_gap_replay_20260819 |
| 01_direct | /home/wjg/code/FlowState/motivation/artifacts/wp3a_direct_eviction_20260819 |
| 01_gate | /home/wjg/code/FlowState/evaluation/runtime_artifacts/openhands_recurrent_only_causal_gate_20260831_052756_001767 |
| 01_sota | /home/wjg/code/FlowState/evaluation/runtime_artifacts/sota_correctness_20260825_104513_833972 |
| 02_qwen | /home/wjg/code/FlowState/motivation/artifacts/replay_cost_20260819 |
| 02_long | /home/wjg/code/FlowState/evaluation/runtime_artifacts/recovery_profiler_128k_20260826_133712_596813 |
| 02_failed | /home/wjg/code/FlowState/evaluation/runtime_artifacts/recovery_profiler_128k_20260826_133234_448476 |
| 02_position | /home/wjg/code/FlowState/evaluation/runtime_artifacts/recovery_position_audit_20260826_144654_852303 |
| 02_kimi | /home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555 |
| 03_model | /home/wjg/code/FlowState/evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020 |
| 03_audit | /home/wjg/code/FlowState/evaluation/rq2_model_audit_output/20260909_055110 |
| 04_collection | /home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017 |
| 04_oh | /home/wjg/code/FlowState/evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011 |
| 04_ax | /home/wjg/code/FlowState/rq3_agentx_policy_eval_20260906_214956 |
| 04_g1 | /home/wjg/code/FlowState/evaluation/p4_evaluation_restructuring |
| 05_p0 | /home/wjg/code/FlowState/evaluation/p0_allocation_mechanism_audit |
| 05_p1b | /home/wjg/code/FlowState/evaluation/p1_strong_baseline_validation |
| 05_ablation | /home/wjg/code/FlowState/evaluation/runtime_artifacts/rq5_frozen_snapshot_ablation_20260908_110726 |
| 06_ttft | /home/wjg/code/FlowState/rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652 |
| 07_fine | /home/wjg/code/FlowState/evaluation/rq6_overhead_output/rq6_overhead_20260909_061500 |
| 07_diagnostic | /home/wjg/code/FlowState/evaluation/rq6_runtime_bottleneck_output/rq6d_20260910_01 |
| 07_batch | /home/wjg/code/FlowState/evaluation/rq6_batched_control_output/rq6e_20260911_085820 |
| 07_profile | /home/wjg/code/FlowState/evaluation/control_path_optimization_output/control_20260929/profiling |
| 08_initial | /home/wjg/code/FlowState/evaluation/resume_epoch_e2e_output/resume_epoch_final_20260928 |
| 08_optimized | /home/wjg/code/FlowState/evaluation/control_path_optimization_output/control_20260929/optimized_formal |
| 08_serving | /home/wjg/code/FlowState/evaluation/serving_path_e2e_output/serving_20260929 |
| 09_capacity | /home/wjg/code/FlowState/evaluation/runtime_artifacts/openhands_4workflow_occupancy_calibration_20260831_050831_071744 |
| 09_size | /home/wjg/code/FlowState/motivation/artifacts/checkpoint_size_20260819 |
| 09_multiround | /home/wjg/code/FlowState/evaluation/runtime_artifacts/openhands_final_multi_round_20260903_025836_200373 |
| 09_kimi1 | /home/wjg/code/FlowState/evaluation/rq5b_kimi_phase1_output/rq5b_20260922_064043 |
| 09_kimi2 | /home/wjg/code/FlowState/evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527 |
| 09_kimifreeze | /home/wjg/code/FlowState/evaluation/rq5b_kimi_final_output/rq5b_frozen_20260923_090154 |
| 09_p2 | /home/wjg/code/FlowState/evaluation/p2_shared_cohort_feasibility_audit |
| 09_p3g0 | /home/wjg/code/FlowState/evaluation/p3_native_instrumentation_feasibility |
| 09_p3g1 | /home/wjg/code/FlowState/evaluation/p3_physical_retention_budget_contract |
| 09_kvflow | /home/wjg/code/FlowState/evaluation/rq3_kvflow_controlled_20260907_123928 |
| 09_boundary | /home/wjg/code/FlowState/motivation/artifacts/workflow_boundary_value_20260819 |
| 09_legacy | /home/wjg/code/FlowState/evaluation/runtime_artifacts/sota_latency_20260825_113526_839592 |
| 09_rerun | /home/wjg/code/FlowState/evaluation/runtime_artifacts/formal_model_latency_rerun_20260827_015154_313413 |
| 09_v2 | /home/wjg/code/FlowState/evaluation/recovery_profiler_v2 |
| 09_tracelab | /home/wjg/code/FlowState/evaluation/public_agent_trace/formal_policy_results_20260827_075548_356403 |
| 09_regression | /home/wjg/code/FlowState/evaluation/formal_model_regression |

历史命令见[historical_commands.md](historical_commands.md)。源码函数及行号见[source_code_index.md](source_code_index.md)，指标口径和各批次缺口见手册；图表来源见[figures/figure_sources.csv](figures/figure_sources.csv)。

`/workspace`、`/repo`与`/model`是历史容器内路径。对应仓库文件存在时以本索引宿主机路径为准；模型运行环境仅按冻结配置陈述。本轮未启动容器验证。
