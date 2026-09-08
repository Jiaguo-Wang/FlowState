"""验证 RQ3 KVFlow 工作流距离敏感受控比较。"""

from __future__ import annotations

import json

from evaluation.rq3_kvflow_controlled_comparison import (
    BUDGET_K,
    build_workload,
    run_comparison,
    write_artifacts,
)
from flowstate.state_catalog import is_compatible


def test_workload_has_three_explicit_workflows_and_distinct_ste() -> None:
    """三个 STE 必须由显式工作流图产生。"""
    workload = build_workload()

    assert len(workload.specs) == 3
    assert len(workload.candidates) == 3
    assert len(workload.continuations) == 3
    assert set(workload.steps_by_continuation.values()) == {1, 3, 5}
    assert workload.budget_k == BUDGET_K == 1


def test_compatibility_uses_only_explicit_workflow_lineage() -> None:
    """每个候选只能覆盖显式声明的本工作流待续请求。"""
    workload = build_workload()

    for candidate in workload.candidates:
        compatible = [
            continuation.continuation_id
            for continuation in workload.continuations
            if is_compatible(candidate, continuation)
        ]
        assert compatible == [
            next(
                spec.continuation_id
                for spec in workload.specs
                if spec.workflow_id == candidate.workflow_id
            )
        ]


def test_policy_rankings_expose_proximity_recovery_tradeoff() -> None:
    """更近的候选必须具有更小 STE，但恢复收益低于远端候选。"""
    result = run_comparison()
    rows = {
        row["candidate_id"]: row
        for row in result["candidate_signals"]["candidates"]
    }

    assert rows["CP_NEAR"]["steps_to_execution"] == 1
    assert rows["CP_FAR"]["steps_to_execution"] == 5
    assert (
        rows["CP_NEAR"]["flowstate_benefit_from_empty_ms"]
        < rows["CP_FAR"]["flowstate_benefit_from_empty_ms"]
    )
    rankings = result["candidate_signals"]["rankings"]
    assert rankings["KVFlow Adaptation"] == ["CP_NEAR", "CP_MID", "CP_FAR"]
    assert rankings["FlowState marginal order"] == ["CP_FAR", "CP_MID", "CP_NEAR"]


def test_selections_and_exact_common_objective() -> None:
    """冻结策略必须共享输入，Exact 必须独立枚举并匹配最优成本。"""
    result = run_comparison()
    selections = result["policy_selections"]["selections"]
    objective = result["common_objective_results"]

    assert selections == {
        "KVFlow Adaptation": ["CP_NEAR"],
        "Marconi Adaptation": ["CP_FAR"],
        "FlowState": ["CP_FAR"],
        "Exact": ["CP_FAR"],
    }
    assert objective["exact_subset_evaluations"] == 4
    assert (
        objective["results"]["KVFlow Adaptation"]["total_cost_ms"]
        > objective["results"]["FlowState"]["total_cost_ms"]
    )
    assert (
        objective["results"]["FlowState"]["total_cost_ms"]
        == objective["results"]["Exact"]["total_cost_ms"]
    )


def test_all_pass_gates_and_determinism() -> None:
    """全部预注册门禁和重复选择必须通过。"""
    result = run_comparison()

    assert result["status"] == "RQ3_KVFLOW_CONTROLLED_READY"
    assert all(result["policy_selections"]["pass_gates"].values())
    assert result["policy_selections"]["determinism"] == {
        "repetitions": 20,
        "pass": True,
    }
    provenance = result["workload_definition"]["provenance"]
    assert provenance == {
        "hidden_future_trajectory_used": False,
        "radix_tree_used_for_ancestry": False,
        "token_lcp_used_for_ancestry": False,
        "external_population_snapshot_used": False,
        "policy_output_used_to_construct_workload": False,
        "workload_parameters_tuned_for_flowstate_win": False,
        "openhands_168_snapshots_run": False,
        "agentx_23_snapshots_run": False,
        "gpu_used": False,
    }


def test_artifact_writer_creates_required_files(tmp_path) -> None:
    """正式写入器必须生成五个要求文件且内容可解析。"""
    result = run_comparison()
    root = tmp_path / "artifact"

    write_artifacts(result, root)

    expected = {
        "workload_definition.json",
        "candidate_signals.json",
        "policy_selections.json",
        "common_objective_results.json",
        "final_report.md",
    }
    assert {path.name for path in root.iterdir()} == expected
    for name in expected - {"final_report.md"}:
        assert json.loads((root / name).read_text(encoding="utf-8"))
    report = (root / "final_report.md").read_text(encoding="utf-8")
    assert "更近但恢复价值更低" in report
    assert "RQ3_KVFLOW_CONTROLLED_READY" in report
