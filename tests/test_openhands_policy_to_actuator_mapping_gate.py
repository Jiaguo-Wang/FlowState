from __future__ import annotations

import inspect

import pytest

from evaluation.openhands_common_barrier_snapshot_gate import (
    ENGINE_CONFIGURATION_COMMON_BARRIER,
    SCHEDULE,
)
from evaluation.openhands_policy_to_actuator_mapping_gate import (
    ENGINE_CONFIGURATION_ACTUATOR_MAPPING,
    EXPECTED_EVICTIONS,
    SELECTED_SETS,
    FrozenSelectedSetOptimizer,
    RecordingRuntimeAdapter,
    build_controller_report,
    build_summary,
    evaluate_mapping_invariants,
)
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import ReconcileExecutionError, StateController
from flowstate.state_catalog import CheckpointCandidate


def candidate(checkpoint_id: str, order: int) -> CheckpointCandidate:
    """构造控制器单测使用的等大小驻留候选。"""
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id=checkpoint_id[0],
        lineage_path=("openhands", checkpoint_id[0]),
        token_pos=order * 64,
        memory_bytes=10,
        recurrent_resident=True,
        fa_resident=True,
    )


def handle(checkpoint_id: str, order: int) -> RuntimeCheckpointHandle:
    """构造带节点和摘要约束的运行时句柄。"""
    return RuntimeCheckpointHandle(
        checkpoint_id=checkpoint_id,
        token_ids=(order, order + 1),
        expected_node_id=order,
        expected_prefix_digest=f"摘要-{checkpoint_id}",
    )


def formal_response() -> dict[str, object]:
    """构造通过正式 actuator proof 检查的最小响应。"""
    return {
        "formal_primitive": (
            "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
        ),
        "before": {
            "path": {
                "target_mamba_present": True,
                "target_full_present": True,
                "path_full_all_present": True,
            }
        },
        "after": {
            "path": {
                "target_mamba_present": False,
                "target_full_present": True,
                "path_full_all_present": True,
            }
        },
        "proof": {
            "same_node": True,
            "fa_unchanged": True,
            "path_unchanged": True,
            "tree_unchanged": True,
            "only_target_mamba_changed": True,
            "sanity_check": True,
            "cascade_called": False,
            "fa_identity_unchanged": True,
        },
    }


class FakeDelegate:
    """模拟成功的 scheduler 适配器并保存正式响应。"""

    def __init__(self) -> None:
        self.eviction_responses: list[dict[str, object]] = []

    def evict_mamba_only(self, runtime_handle) -> None:
        del runtime_handle
        self.eviction_responses.append(formal_response())


class FailingDelegate:
    """模拟首个 actuator 调用失败。"""

    eviction_responses: list[dict[str, object]] = []

    def evict_mamba_only(self, runtime_handle) -> None:
        raise RuntimeError(f"拒绝驱逐 {runtime_handle.checkpoint_id}")


def frozen_candidates_and_handles():
    """构造四候选及其一一对应句柄。"""
    candidates = tuple(
        candidate(f"{label}1", order)
        for order, label in enumerate("ABCD", start=1)
    )
    handles = {
        item.checkpoint_id: handle(item.checkpoint_id, order)
        for order, item in enumerate(candidates, start=1)
    }
    return candidates, handles


def state(checkpoint_id: str, resident: bool) -> dict[str, object]:
    """构造 invariant 检查所需的候选状态。"""
    order = "ABCD".index(checkpoint_id[0]) + 1
    return {
        "node_id": order,
        "token_pos": order * 64,
        "prefix_digest": f"摘要-{checkpoint_id}",
        "path_node_ids": [order],
        "path_full_digest": f"FA-{checkpoint_id}",
        "fa_resident": True,
        "recurrent_resident": resident,
        "tree_structure_digest": "同一树",
        "full_tree_digest": "同一FA树",
        "full_allocator": {"available": 100},
    }


def census(*, before: bool) -> dict[str, object]:
    """构造 LM 条件前后精确释放两个 slot 的 census。"""
    resident = (
        [
            {"node_id": order, "slots": [order]}
            for order in range(1, 5)
        ]
        if before
        else [
            {"node_id": order, "slots": [order]}
            for order in (3, 4)
        ]
    )
    return {
        "resident_mamba_nodes": resident,
        "mamba_available_slots": 24 if before else 26,
        "device_resident_mamba_slots": 4 if before else 2,
        "mamba_node_count": 4 if before else 2,
        "removed_mamba_node_ids": [] if before else [1, 2],
        "changed_existing_mamba_node_ids": [],
        "full_device_node_ids": [1, 2, 3, 4],
        "structure_node_ids": [0, 1, 2, 3, 4],
        "removed_full_device_node_ids": [],
        "removed_structure_node_ids": [],
    }


def test_frozen_selected_sets_match_step12h4_result() -> None:
    assert SELECTED_SETS == {
        "LM": (
            "OPENHANDS_BARRIER_C_TURN_001",
            "OPENHANDS_BARRIER_D_TURN_001",
        ),
        "F": (
            "OPENHANDS_BARRIER_A_TURN_001",
            "OPENHANDS_BARRIER_C_TURN_001",
        ),
    }
    assert EXPECTED_EVICTIONS["LM"] == (
        "OPENHANDS_BARRIER_A_TURN_001",
        "OPENHANDS_BARRIER_B_TURN_001",
    )
    assert EXPECTED_EVICTIONS["F"] == (
        "OPENHANDS_BARRIER_B_TURN_001",
        "OPENHANDS_BARRIER_D_TURN_001",
    )


def test_frozen_optimizer_rejects_non_candidate_and_over_budget() -> None:
    candidates, _ = frozen_candidates_and_handles()
    with pytest.raises(ValueError, match="非候选"):
        FrozenSelectedSetOptimizer(("Z1",)).select((), candidates, 20)
    with pytest.raises(ValueError, match="超出预算"):
        FrozenSelectedSetOptimizer(("A1", "B1", "C1")).select(
            (), candidates, 20
        )


def test_controller_maps_selected_set_to_sorted_unselected_evictions() -> None:
    candidates, handles = frozen_candidates_and_handles()
    adapter = RecordingRuntimeAdapter(FakeDelegate())
    controller = StateController(
        FrozenSelectedSetOptimizer(("C1", "D1")),
        adapter,
    )
    allocation = controller.reconcile((), candidates, handles, 20)
    report = build_controller_report(
        allocation=allocation,
        adapter=adapter,
    )
    assert report["selected_checkpoint_ids"] == ["C1", "D1"]
    assert report["requested_eviction_ids"] == ["A1", "B1"]
    assert report["successful_eviction_ids"] == ["A1", "B1"]
    assert report["failed_evictions"] == []
    assert len(report["formal_responses"]) == 2


def test_recording_adapter_preserves_controller_partial_failure_report() -> None:
    candidates, handles = frozen_candidates_and_handles()
    adapter = RecordingRuntimeAdapter(FailingDelegate())
    controller = StateController(
        FrozenSelectedSetOptimizer(("C1", "D1")),
        adapter,
    )
    with pytest.raises(ReconcileExecutionError):
        controller.reconcile((), candidates, handles, 20)
    report = build_controller_report(allocation=None, adapter=adapter)
    assert report["requested_eviction_ids"] == ["A1"]
    assert report["successful_eviction_ids"] == []
    assert report["failed_evictions"][0]["checkpoint_id"] == "A1"


def test_mapping_invariants_accept_only_exact_recurrent_changes() -> None:
    candidates, handles = frozen_candidates_and_handles()
    candidate_ids = [item.checkpoint_id for item in candidates]
    before_states = {
        checkpoint_id: state(checkpoint_id, True)
        for checkpoint_id in candidate_ids
    }
    after_states = {
        checkpoint_id: state(checkpoint_id, checkpoint_id in {"C1", "D1"})
        for checkpoint_id in candidate_ids
    }
    report = {
        "selected_checkpoint_ids": ["C1", "D1"],
        "requested_eviction_ids": ["A1", "B1"],
        "successful_eviction_ids": ["A1", "B1"],
        "failed_evictions": [],
        "formal_responses": [formal_response(), formal_response()],
    }
    result = evaluate_mapping_invariants(
        candidate_ids=candidate_ids,
        selected_ids=("C1", "D1"),
        expected_evicted_ids=("A1", "B1"),
        handles=handles,
        before_states=before_states,
        after_states=after_states,
        before_census=census(before=True),
        after_census=census(before=False),
        controller_report=report,
    )
    assert result["status"] == "PASS"
    assert result["selected_residency_exact"] is True
    assert result["recurrent_change_exact"] is True
    assert result["mamba_available_slot_delta"] == 2
    assert result["native_mamba_capacity_eviction"] is False
    assert result["fa_kv_cascade"] is False


def test_mapping_invariants_reject_unexpected_mamba_change() -> None:
    candidates, handles = frozen_candidates_and_handles()
    candidate_ids = [item.checkpoint_id for item in candidates]
    before_states = {
        checkpoint_id: state(checkpoint_id, True)
        for checkpoint_id in candidate_ids
    }
    after_states = {
        checkpoint_id: state(checkpoint_id, checkpoint_id in {"C1", "D1"})
        for checkpoint_id in candidate_ids
    }
    after_census = census(before=False)
    after_census["changed_existing_mamba_node_ids"] = [3]
    report = {
        "selected_checkpoint_ids": ["C1", "D1"],
        "requested_eviction_ids": ["A1", "B1"],
        "successful_eviction_ids": ["A1", "B1"],
        "failed_evictions": [],
        "formal_responses": [formal_response(), formal_response()],
    }
    result = evaluate_mapping_invariants(
        candidate_ids=candidate_ids,
        selected_ids=("C1", "D1"),
        expected_evicted_ids=("A1", "B1"),
        handles=handles,
        before_states=before_states,
        after_states=after_states,
        before_census=census(before=True),
        after_census=after_census,
        controller_report=report,
    )
    assert result["status"] == "FAIL"
    assert result["native_mamba_capacity_eviction"] is True


def test_summary_requires_two_independent_matching_runs(tmp_path) -> None:
    def run(condition, retained):
        return {
            "condition": condition,
            "status": "PASS",
            "engine_lifecycle": "independent_fresh",
            "invariants": {"actual_retained_ids": retained},
            "controller_report": {
                "formal_responses": [formal_response(), formal_response()]
            },
        }

    boundary = [
        {
            "r_plus_2_message_consumed": False,
            "r_plus_2_request_materialized": False,
            "pending_assistant_output_read": False,
        }
    ]
    summary = build_summary(
        artifact=tmp_path,
        runs=(
            run("LM", ["C1", "D1"]),
            run("F", ["A1", "C1"]),
        ),
        boundary_audit=boundary,
        environment=None,
    )
    assert summary["status"] == "PASS"
    assert summary["same_actuator_path"] is True
    assert summary[
        "different_selected_sets_produced_corresponding_residency"
    ] is True
    assert summary["pending_requests_executed"] is False
    assert summary["future_leakage"] is False


def test_summary_reports_partial_run_as_failure_without_secondary_error(
    tmp_path,
) -> None:
    boundary = [
        {
            "r_plus_2_message_consumed": False,
            "r_plus_2_request_materialized": False,
            "pending_assistant_output_read": False,
        }
    ]
    summary = build_summary(
        artifact=tmp_path,
        runs=(
            {
                "condition": "LM",
                "status": "FAIL",
                "engine_lifecycle": "independent_fresh",
                "invariants": None,
                "controller_report": None,
            },
        ),
        boundary_audit=boundary,
        environment=None,
    )
    assert summary["status"] == "FAIL"
    assert summary["verdict"] == "PARTIAL"
    assert summary["same_actuator_path"] is False


def test_gate_uses_two_fresh_lifecycles_and_only_executes_round_one() -> None:
    assert ENGINE_CONFIGURATION_ACTUATOR_MAPPING == (
        ENGINE_CONFIGURATION_COMMON_BARRIER
    )
    assert ENGINE_CONFIGURATION_ACTUATOR_MAPPING is not (
        ENGINE_CONFIGURATION_COMMON_BARRIER
    )
    assert tuple(turn for _, turn in SCHEDULE) == (1, 1, 1, 1)
    module = __import__(
        "evaluation.openhands_policy_to_actuator_mapping_gate",
        fromlist=["unused"],
    )
    source = inspect.getsource(module)
    assert "StateController(" in source
    assert "controller.reconcile(" in source
    assert "SchedulerRuntimeAdapter(client)" in source
    assert "engine.shutdown()" in source
