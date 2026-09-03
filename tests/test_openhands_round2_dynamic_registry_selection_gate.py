from __future__ import annotations

import inspect

import evaluation.openhands_round2_dynamic_registry_selection_gate as gate
from evaluation.openhands_common_barrier_snapshot_gate import (
    CHECKPOINT_SIZE_BYTES,
)
from evaluation.openhands_round2_dynamic_registry_selection_gate import (
    ENGINE_CONFIGURATION_DYNAMIC_REGISTRY,
    EXPECTED_BARRIER_ONE_SELECTION,
    POLICY_ORDER,
    DynamicRegistryEntry,
    _boundary_has_future_leakage,
    apply_materialization_observation,
    build_dynamic_metadata,
    build_round_three_pending,
    build_summary,
    materialize_round2_visible_requests,
    registry_candidates,
    run_policy_selector,
)
from evaluation.openhands_policy_runtime_heg_outcome_gate import (
    ENGINE_CONFIGURATION_HEG_OUTCOME,
)
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


class FakeTokenizer:
    """返回与历史消息数量稳定对应的测试 token。"""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        return list(range(1, len(messages) + 2))


class FakeBarrierClient:
    """提供无副作用 FA frontier 查询结果。"""

    def __init__(self, frontiers):
        self.frontiers = iter(frontiers)
        self.calls = []

    def inspect_fa_frontier(
        self,
        input_ids,
        *,
        extra_key,
        limit,
        nonce,
    ):
        self.calls.append((list(input_ids), extra_key, limit, nonce))
        return {
            "state_equal": True,
            "scope_before": {"idle": True},
            "scope_after": {"idle": True},
            "resident_fa_frontier": next(self.frontiers),
            "changed_fields": [],
            "traversed_node_ids": [1, 2],
        }


def make_entry(
    checkpoint_id: str,
    label: str,
    *,
    turn: int,
    token_pos: int,
    order: int,
    resident: bool = True,
) -> DynamicRegistryEntry:
    """构造一个具有稳定 runtime identity 的注册表条目。"""
    digest = f"摘要-{label}-{token_pos}"
    return DynamicRegistryEntry(
        checkpoint_id=checkpoint_id,
        workflow_label=label,
        workflow_id=f"workflow-{label}",
        turn=turn,
        lineage_path=("openhands", f"workflow-{label}"),
        token_pos=token_pos,
        memory_bytes=CHECKPOINT_SIZE_BYTES,
        recurrent_resident=resident,
        fa_resident=True,
        handle=RuntimeCheckpointHandle(
            checkpoint_id=checkpoint_id,
            token_ids=tuple(range(1, token_pos + 1)),
            expected_node_id=order,
            expected_prefix_digest=digest,
        ),
        creation_order=order,
        last_access_order=order,
        node_id=order,
        slots=(order,),
        materialization_events=[
            {"turn": turn, "event_order": order, "kind": "测试创建"}
        ],
    )


def make_registry() -> dict[str, DynamicRegistryEntry]:
    """构造四个 Round 1 checkpoint 的注册表。"""
    return {
        f"{label}1": make_entry(
            f"{label}1",
            label,
            turn=1,
            token_pos=100 * order,
            order=order,
        )
        for order, label in enumerate("ABCD", start=1)
    }


def make_pending():
    """构造四个仅与各自 workflow 兼容的 pending。"""
    return tuple(
        PendingContinuation(
            continuation_id=f"{label}3",
            workflow_id=f"workflow-{label}",
            lineage_path=("openhands", f"workflow-{label}"),
            anchor_pos=1_000,
            resident_fa_frontier=1_000,
        )
        for label in "ABCD"
    )


def clean_boundary():
    """构造没有 Round 4 或其他未来信息的审计行。"""
    return [
        {
            "round_3_assistant_output_read": False,
            "round_4_message_consumed": False,
            "round_4_request_materialized": False,
            "future_timing_read": False,
            "future_checkpoint_read": False,
        }
    ]


def test_three_policy_lifecycles_and_frozen_engine_configuration() -> None:
    assert POLICY_ORDER == ("LRU", "Marconi", "FlowState")
    assert set(EXPECTED_BARRIER_ONE_SELECTION) == set(POLICY_ORDER)
    assert ENGINE_CONFIGURATION_DYNAMIC_REGISTRY == (
        ENGINE_CONFIGURATION_HEG_OUTCOME
    )
    assert ENGINE_CONFIGURATION_DYNAMIC_REGISTRY is not (
        ENGINE_CONFIGURATION_HEG_OUTCOME
    )
    assert ENGINE_CONFIGURATION_DYNAMIC_REGISTRY[
        "max_mamba_cache_size"
    ] == 28


def test_visible_request_builder_stops_before_turn_three_output() -> None:
    raw = [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "请求"},
        {"role": "assistant", "content": "回答一"},
        {"role": "tool", "content": "结果一"},
        {"role": "assistant", "content": "回答二"},
        {"role": "tool", "content": "结果二"},
        {"role": "assistant", "content": "回答三"},
        {"role": "tool", "content": "未来结果"},
        {"role": "assistant", "content": "未来回答"},
    ]
    requests, audit = materialize_round2_visible_requests(
        FakeTokenizer(),
        raw,
        workflow_label="A",
        workflow_id="workflow-A",
    )
    assert tuple(requests) == (1, 2, 3)
    assert audit["maximum_assistant_turn_consumed"] == 3
    assert audit["round_3_assistant_output_read"] is False
    assert audit["round_4_message_consumed"] is False
    assert audit["round_4_request_materialized"] is False


def test_evicted_checkpoint_rematerialization_updates_current_generation() -> None:
    registry = make_registry()
    entry = registry["A1"]
    entry.recurrent_resident = False
    probe = RuntimeCheckpointHandle(
        checkpoint_id="探针",
        token_ids=entry.handle.token_ids,
        expected_node_id=9,
        expected_prefix_digest=entry.handle.expected_prefix_digest,
    )
    observed, kind = apply_materialization_observation(
        registry,
        request={
            "workflow_label": "A",
            "workflow_id": "workflow-A",
            "turn": 2,
        },
        event_order=5,
        executable_frontier=0,
        node_id=9,
        token_pos=100,
        slots=[13],
        handle=probe,
        fa_resident=True,
        recurrent_resident=True,
        previously_resident_ids=set(),
    )
    assert kind == "REMATERIALIZED"
    assert observed.checkpoint_id == "A1"
    assert observed.turn == 1
    assert observed.creation_order == 5
    assert observed.last_access_order == 5
    assert observed.recurrent_resident is True
    assert len(registry) == 4


def test_deeper_round_two_checkpoint_enters_registry() -> None:
    registry = make_registry()
    probe = RuntimeCheckpointHandle(
        checkpoint_id="探针",
        token_ids=tuple(range(1, 151)),
        expected_node_id=9,
        expected_prefix_digest="更深摘要",
    )
    observed, kind = apply_materialization_observation(
        registry,
        request={
            "workflow_label": "A",
            "workflow_id": "workflow-A",
            "turn": 2,
        },
        event_order=5,
        executable_frontier=100,
        node_id=9,
        token_pos=150,
        slots=[12],
        handle=probe,
        fa_resident=True,
        recurrent_resident=True,
        previously_resident_ids={"A1", "B1", "C1", "D1"},
    )
    assert kind == "CREATED"
    assert observed.checkpoint_id == "OPENHANDS_BARRIER_A_TURN_002"
    assert observed.turn == 2
    assert registry["A1"].last_access_order == 5
    assert len(registry) == 5


def test_dynamic_metadata_uses_round_two_access_and_adjacent_span() -> None:
    registry = make_registry()
    deeper = make_entry(
        "A2",
        "A",
        turn=2,
        token_pos=160,
        order=5,
    )
    registry["A1"].last_access_order = 5
    registry["A2"] = deeper
    candidates = registry_candidates(registry)
    rows, recency = build_dynamic_metadata(registry, candidates)
    by_id = {row["checkpoint_id"]: row for row in rows}
    assert by_id["A1"]["last_access_order"] == 5
    assert by_id["A2"]["creation_order"] == 5
    assert by_id["A2"]["marconi_parent_position"] == 100
    assert by_id["A2"]["marconi_incremental_span"] == 60.0
    assert max(item.last_access_order for item in recency) == 5


def test_round_three_pending_uses_read_only_barrier_frontier() -> None:
    requests = {
        (label, 3): {
            "workflow_label": label,
            "workflow_id": f"workflow-{label}",
            "turn": 3,
            "input_ids": list(range(10 + order)),
        }
        for order, label in enumerate("ABCD")
    }
    client = FakeBarrierClient([5, 8, 20, 4])
    pending, rows = build_round_three_pending(
        client,
        requests,
        policy="LRU",
    )
    assert [item.resident_fa_frontier for item in pending] == [5, 8, 20, 4]
    assert [item.planning_target for item in pending] == [5, 8, 12, 4]
    assert all(row["query_state_equal"] for row in rows)
    assert len(client.calls) == 4


def test_nonresident_registry_refresh_accepts_null_slots(monkeypatch) -> None:
    entry = make_entry(
        "A1",
        "A",
        turn=1,
        token_pos=100,
        order=1,
        resident=False,
    )
    monkeypatch.setattr(
        gate,
        "inspect_checkpoint",
        lambda client, checkpoint_id, token_ids: {
            "after": {
                "path": {
                    "prefix_tokens": 100,
                    "prefix_sha256": "摘要-A-100",
                    "node_id": 1,
                    "target_full_present": True,
                    "path_full_all_present": True,
                    "target_mamba_present": False,
                    "target_mamba_slots": None,
                }
            }
        },
    )
    rows = gate.refresh_registry(
        object(),
        {"A1": entry},
        phase="测试",
    )
    assert entry.row()["slots"] == []
    assert entry.candidate().recurrent_resident is False
    assert rows[0]["slots"] == []


def test_all_three_existing_selectors_accept_dynamic_registry() -> None:
    registry = make_registry()
    registry["A2"] = make_entry(
        "A2",
        "A",
        turn=2,
        token_pos=150,
        order=5,
    )
    candidates = registry_candidates(registry)
    metadata, _ = build_dynamic_metadata(registry, candidates)
    pending = make_pending()
    for policy in POLICY_ORDER:
        result = run_policy_selector(
            policy,
            candidates,
            pending,
            metadata,
        )
        assert result["selection_valid"] is True
        assert result["selected_count"] <= 2
        assert set(result["selected_checkpoint_ids"]).issubset(
            result["eligible_candidate_ids"]
        )
    flowstate = run_policy_selector(
        "FlowState",
        candidates,
        pending,
        metadata,
    )
    assert "recovery_cost_before_ms" in flowstate
    assert "recovery_cost_after_ms" in flowstate
    assert "total_benefit_ms" in flowstate
    assert "used_bytes" in flowstate


def test_selector_never_selects_nonresident_history() -> None:
    registry = make_registry()
    registry["D1"].recurrent_resident = False
    candidates = registry_candidates(registry)
    metadata, _ = build_dynamic_metadata(registry, candidates)
    for policy in POLICY_ORDER:
        result = run_policy_selector(
            policy,
            candidates,
            make_pending(),
            metadata,
        )
        assert "D1" not in result["selected_checkpoint_ids"]


def test_path_dependent_candidate_universes_are_allowed(tmp_path) -> None:
    runs = []
    for policy, ids in (
        ("LRU", ("A1", "B1")),
        ("Marconi", ("A1", "B1", "C2")),
        ("FlowState", ("A1", "A2", "C1")),
    ):
        runs.append(
            {
                "policy": policy,
                "status": "PASS",
                "barrier_2_registry": [
                    {
                        "checkpoint_id": checkpoint_id,
                        "recurrent_resident": True,
                    }
                    for checkpoint_id in ids
                ],
                "round_2_selection": {
                    "selection_valid": True,
                    "selected_count": 2,
                },
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
    assert summary["candidate_universe_equality_required_at_barrier_2"] is False
    assert summary["candidate_universe_path_dependent"] is True
    assert len(
        {
            tuple(value)
            for value in summary["eligible_registry_by_policy"].values()
        }
    ) == 3


def test_future_boundary_rejects_round_four_materialization() -> None:
    assert _boundary_has_future_leakage(clean_boundary()) is False
    leaked = clean_boundary()
    leaked[0]["round_4_request_materialized"] = True
    assert _boundary_has_future_leakage(leaked) is True


def test_gate_source_stops_after_second_selection() -> None:
    module = __import__(
        "evaluation.openhands_round2_dynamic_registry_selection_gate",
        fromlist=["unused"],
    )
    source = inspect.getsource(module)
    assert "for ordinal, policy in enumerate(POLICY_ORDER" in source
    assert "FormalEndToEndGateEngine(" in source
    assert "for offset, (label, turn) in enumerate(\n            ROUND_TWO_SCHEDULE" in source
    assert source.count("controller.reconcile(") == 1
    assert '"barrier_2_selection_applied": False' in source
    assert '"round_3_requests_executed": False' in source
