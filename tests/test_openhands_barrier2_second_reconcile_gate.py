from __future__ import annotations

import inspect

import pytest

import evaluation.openhands_barrier2_second_reconcile_gate as gate
from evaluation.openhands_barrier2_second_reconcile_gate import (
    ENGINE_CONFIGURATION_SECOND_RECONCILE,
    EXPECTED_SECOND_ELIGIBLE,
    EXPECTED_SECOND_SELECTION,
    build_second_reconcile_plan,
    build_summary,
)
from evaluation.openhands_policy_to_actuator_mapping_gate import (
    FrozenSelectedSetOptimizer,
    RecordingRuntimeAdapter,
    build_controller_report,
    evaluate_mapping_invariants,
)
from evaluation.openhands_round2_dynamic_registry_selection_gate import (
    ENGINE_CONFIGURATION_DYNAMIC_REGISTRY,
    POLICY_ORDER,
)
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import StateController
from flowstate.state_catalog import CheckpointCandidate


def candidate(checkpoint_id: str, order: int) -> CheckpointCandidate:
    """构造 Barrier 2 单测使用的驻留候选。"""
    label = checkpoint_id.removeprefix("OPENHANDS_BARRIER_")[0]
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id=f"workflow-{label}",
        lineage_path=("openhands", f"workflow-{label}"),
        token_pos=order * 64,
        memory_bytes=10,
        recurrent_resident=True,
        fa_resident=True,
    )


def candidates_for(policy: str) -> tuple[CheckpointCandidate, ...]:
    """按指定 policy 的真实 path-dependent universe 构造候选。"""
    return tuple(
        candidate(checkpoint_id, order)
        for order, checkpoint_id in enumerate(
            EXPECTED_SECOND_ELIGIBLE[policy],
            start=1,
        )
    )


def selection_for(policy: str) -> dict[str, object]:
    """构造已冻结的 Barrier 2 selector 输出。"""
    return {
        "selection_valid": True,
        "selected_checkpoint_ids": list(
            EXPECTED_SECOND_SELECTION[policy]
        ),
    }


def handle(checkpoint_id: str, order: int) -> RuntimeCheckpointHandle:
    """构造带稳定节点 identity 的运行时句柄。"""
    return RuntimeCheckpointHandle(
        checkpoint_id=checkpoint_id,
        token_ids=(order, order + 1),
        expected_node_id=order,
        expected_prefix_digest=f"摘要-{checkpoint_id}",
    )


def state(checkpoint_id: str, order: int, resident: bool) -> dict[str, object]:
    """构造 recurrent-only 不变量需要的候选状态。"""
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
    """模拟成功 actuator 并记录每次正式响应。"""

    def __init__(self) -> None:
        self.eviction_responses: list[dict[str, object]] = []

    def evict_mamba_only(self, runtime_handle) -> None:
        del runtime_handle
        self.eviction_responses.append(formal_response())


def clean_boundary() -> list[dict[str, object]]:
    """构造只读取当前 Round 3 pending 的在线边界。"""
    return [
        {
            "round_3_assistant_output_read": False,
            "round_4_message_consumed": False,
            "round_4_request_materialized": False,
            "future_timing_read": False,
            "future_checkpoint_read": False,
        }
    ]


def test_configuration_is_a_frozen_copy() -> None:
    assert ENGINE_CONFIGURATION_SECOND_RECONCILE == (
        ENGINE_CONFIGURATION_DYNAMIC_REGISTRY
    )
    assert ENGINE_CONFIGURATION_SECOND_RECONCILE is not (
        ENGINE_CONFIGURATION_DYNAMIC_REGISTRY
    )
    assert ENGINE_CONFIGURATION_SECOND_RECONCILE[
        "max_mamba_cache_size"
    ] == 28


def test_expected_path_dependent_universes_are_frozen() -> None:
    assert EXPECTED_SECOND_ELIGIBLE["LRU"] == (
        EXPECTED_SECOND_ELIGIBLE["Marconi"]
    )
    assert EXPECTED_SECOND_ELIGIBLE["FlowState"] != (
        EXPECTED_SECOND_ELIGIBLE["LRU"]
    )
    assert set(EXPECTED_SECOND_SELECTION) == set(POLICY_ORDER)


@pytest.mark.parametrize("policy", POLICY_ORDER)
def test_second_reconcile_plan_matches_each_policy(policy: str) -> None:
    plan = build_second_reconcile_plan(
        policy,
        candidates_for(policy),
        selection_for(policy),
    )
    assert set(plan["eligible_candidate_ids"]) == set(
        EXPECTED_SECOND_ELIGIBLE[policy]
    )
    assert set(plan["selected_checkpoint_ids"]) == set(
        EXPECTED_SECOND_SELECTION[policy]
    )
    assert set(plan["expected_evicted_ids"]) == (
        set(EXPECTED_SECOND_ELIGIBLE[policy])
        - set(EXPECTED_SECOND_SELECTION[policy])
    )
    assert plan["expected_eviction_count"] == 4


def test_plan_uses_actual_candidate_universe() -> None:
    candidates = candidates_for("LRU")[:-1]
    selection = selection_for("LRU")
    selection["selected_checkpoint_ids"] = [
        candidate.checkpoint_id for candidate in candidates[-2:]
    ]
    plan = build_second_reconcile_plan("LRU", candidates, selection)
    assert set(plan["eligible_candidate_ids"]) == {
        candidate.checkpoint_id for candidate in candidates
    }
    assert plan["matches_step12h8_candidate_universe"] is False


def test_plan_uses_actual_valid_selected_set() -> None:
    selection = selection_for("LRU")
    selection["selected_checkpoint_ids"] = list(
        EXPECTED_SECOND_SELECTION["Marconi"]
    )
    plan = build_second_reconcile_plan(
        "LRU",
        candidates_for("LRU"),
        selection,
    )
    assert set(plan["selected_checkpoint_ids"]) == set(
        EXPECTED_SECOND_SELECTION["Marconi"]
    )
    assert plan["matches_step12h8_selection"] is False


def test_plan_rejects_invalid_selector_result() -> None:
    selection = selection_for("FlowState")
    selection["selection_valid"] = False
    with pytest.raises(RuntimeError, match="不是合法结果"):
        build_second_reconcile_plan(
            "FlowState",
            candidates_for("FlowState"),
            selection,
        )


@pytest.mark.parametrize("policy", POLICY_ORDER)
def test_controller_evicts_exact_second_complement(policy: str) -> None:
    candidates = candidates_for(policy)
    handles = {
        item.checkpoint_id: handle(item.checkpoint_id, order)
        for order, item in enumerate(candidates, start=1)
    }
    plan = build_second_reconcile_plan(
        policy,
        candidates,
        selection_for(policy),
    )
    adapter = RecordingRuntimeAdapter(FakeDelegate())
    controller = StateController(
        FrozenSelectedSetOptimizer(plan["selected_checkpoint_ids"]),
        adapter,
    )
    allocation = controller.reconcile((), candidates, handles, 20)
    report = build_controller_report(
        allocation=allocation,
        adapter=adapter,
    )
    assert set(report["selected_checkpoint_ids"]) == set(
        EXPECTED_SECOND_SELECTION[policy]
    )
    assert report["requested_eviction_ids"] == sorted(
        plan["expected_evicted_ids"]
    )
    assert report["successful_eviction_ids"] == sorted(
        plan["expected_evicted_ids"]
    )
    assert len(report["formal_responses"]) == 4


def test_four_eviction_slot_delta_and_all_invariants_pass() -> None:
    policy = "FlowState"
    candidates = candidates_for(policy)
    selected = set(EXPECTED_SECOND_SELECTION[policy])
    candidate_ids = [item.checkpoint_id for item in candidates]
    handles = {
        item.checkpoint_id: handle(item.checkpoint_id, order)
        for order, item in enumerate(candidates, start=1)
    }
    before_states = {
        item.checkpoint_id: state(item.checkpoint_id, order, True)
        for order, item in enumerate(candidates, start=1)
    }
    after_states = {
        item.checkpoint_id: state(
            item.checkpoint_id,
            order,
            item.checkpoint_id in selected,
        )
        for order, item in enumerate(candidates, start=1)
    }
    expected_evicted = sorted(set(candidate_ids) - selected)
    removed_node_ids = sorted(
        handles[checkpoint_id].expected_node_id
        for checkpoint_id in expected_evicted
    )
    before_census = {
        "resident_mamba_nodes": [
            {"node_id": order, "slots": [order]}
            for order in range(1, 7)
        ],
        "mamba_available_slots": 20,
        "device_resident_mamba_slots": 6,
        "mamba_node_count": 6,
        "removed_mamba_node_ids": [],
        "changed_existing_mamba_node_ids": [],
        "full_device_node_ids": list(range(1, 7)),
        "structure_node_ids": list(range(0, 7)),
        "removed_full_device_node_ids": [],
        "removed_structure_node_ids": [],
    }
    after_census = {
        **before_census,
        "resident_mamba_nodes": [
            {"node_id": order, "slots": [order]}
            for order, item in enumerate(candidates, start=1)
            if item.checkpoint_id in selected
        ],
        "mamba_available_slots": 24,
        "device_resident_mamba_slots": 2,
        "mamba_node_count": 2,
        "removed_mamba_node_ids": removed_node_ids,
    }
    report = {
        "selected_checkpoint_ids": list(selected),
        "requested_eviction_ids": expected_evicted,
        "successful_eviction_ids": expected_evicted,
        "failed_evictions": [],
        "formal_responses": [formal_response() for _ in expected_evicted],
    }
    result = evaluate_mapping_invariants(
        candidate_ids=candidate_ids,
        selected_ids=list(selected),
        expected_evicted_ids=expected_evicted,
        handles=handles,
        before_states=before_states,
        after_states=after_states,
        before_census=before_census,
        after_census=after_census,
        controller_report=report,
    )
    assert result["status"] == "PASS"
    assert result["actual_retained_ids"] == sorted(selected)
    assert result["expected_released_slots"] == 4
    assert result["mamba_available_slot_delta"] == 4
    assert result["resident_mamba_slot_delta"] == 4
    assert result["mamba_node_count_delta"] == 4


def test_summary_accepts_three_exact_path_dependent_runs(tmp_path) -> None:
    runs = []
    for policy in POLICY_ORDER:
        runs.append(
            {
                "policy": policy,
                "status": "PASS",
                "second_reconcile_plan": {
                    "eligible_candidate_ids": list(
                        EXPECTED_SECOND_ELIGIBLE[policy]
                    )
                },
                "second_mapping_invariants": {"status": "PASS"},
                "barrier_2_fa_query_state_unchanged": True,
                "native_mamba_capacity_eviction": False,
                "fa_kv_cascade": False,
                "future_leakage": False,
            }
        )
    summary = build_summary(
        artifact=tmp_path,
        runs=runs,
        boundary_audit=clean_boundary(),
        environment=None,
    )
    assert summary["status"] == "PASS"
    assert summary["second_reconcile_mappings_exact"] is True
    assert summary["candidate_universe_path_dependent"] is True
    assert summary["candidate_universe_equality_required"] is False
    assert summary["round_3_requests_executed"] is False


def test_summary_rejects_any_mapping_failure(tmp_path) -> None:
    runs = [
        {
            "policy": policy,
            "status": "FAIL" if policy == "Marconi" else "PASS",
            "second_reconcile_plan": {
                "eligible_candidate_ids": list(
                    EXPECTED_SECOND_ELIGIBLE[policy]
                )
            },
            "second_mapping_invariants": {
                "status": "FAIL" if policy == "Marconi" else "PASS"
            },
            "barrier_2_fa_query_state_unchanged": True,
            "native_mamba_capacity_eviction": False,
            "fa_kv_cascade": False,
            "future_leakage": False,
        }
        for policy in POLICY_ORDER
    ]
    summary = build_summary(
        artifact=tmp_path,
        runs=runs,
        boundary_audit=clean_boundary(),
        environment=None,
    )
    assert summary["status"] == "FAIL"
    assert summary["second_reconcile_mappings_exact"] is False


def test_summary_rejects_future_leakage(tmp_path) -> None:
    boundary = clean_boundary()
    boundary[0]["round_4_request_materialized"] = True
    summary = build_summary(
        artifact=tmp_path,
        runs=(),
        boundary_audit=boundary,
        environment=None,
    )
    assert summary["status"] == "FAIL"
    assert summary["future_leakage"] is True


def test_gate_stops_after_second_reconcile() -> None:
    source = inspect.getsource(gate)
    lifecycle = inspect.getsource(gate._run_policy_lifecycle)
    assert lifecycle.count(".reconcile(") == 2
    assert lifecycle.count("execute_request(") == 2
    assert '"round_3_requests_executed": False' in source
    assert "build_round_three_pending(" in lifecycle
    assert "ROUND_THREE_SCHEDULE" not in source
