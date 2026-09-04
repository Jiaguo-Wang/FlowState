"""验证 RQ3 OpenHands population 采样与 neutral collector 的 CPU 行为。"""

from __future__ import annotations

import json

import pytest

from evaluation.openhands_common_barrier_snapshot_gate import token_digest
from evaluation.openhands_single_workflow_smoke import (
    _template_input_ids,
    normalize_message,
)
from evaluation.rq3_frozen_snapshot_evaluator import select_exact_opt, select_lfu
from evaluation.rq3_openhands_neutral_collector import (
    CheckpointObservation,
    GroupReplayTrace,
    PendingObservation,
    assemble_group_snapshot,
    collect_group,
    load_allocation_snapshot,
    materialize_group_requests,
    prepare_artifact_directories,
    replay_group_to_barrier,
    run_designated_groups,
    run_group_via_worker,
    summarize_population,
    write_snapshot_artifact,
)
from evaluation.rq3_openhands_population import (
    MAIN_GROUP_COUNT,
    SENSITIVITY_GROUP_COUNT,
    WorkflowGroup,
    allocation_round_for_ordinal,
    build_workflow_groups,
    designated_main_groups,
    evaluate_session_eligibility,
    k_sweep_for_candidate_count,
    order_sessions_by_digest,
    reference_budget_for_candidate_count,
    reserved_sensitivity_groups,
    session_order_digest,
)


def _session_ids(count: int) -> list[str]:
    """构造确定性的合成 session 标识。"""

    return [
        f"nebius-swe-rebench-openhands::chatcmpl-{index:032x}"
        for index in range(count)
    ]


def _group(ordinal: int, round_id: int) -> WorkflowGroup:
    """构造一个使用合成 session 的 group。"""

    base = ordinal * 4
    return WorkflowGroup(
        group_ordinal=ordinal,
        population_segment="main",
        allocation_round=round_id,
        session_ids=tuple(_session_ids(10000)[base : base + 4]),
    )


def _trace(
    ordinal: int,
    round_id: int,
    *,
    resident: bool = True,
    fa_clean: bool = True,
) -> GroupReplayTrace:
    """构造与真实 replay 结构一致的合成观测 trace。"""

    group = _group(ordinal, round_id)
    checkpoints = []
    ordinal_counter = 0
    for turn in range(1, round_id + 1):
        for label in ("A", "B", "C", "D"):
            ordinal_counter += 1
            session_id = group.session_by_label[label]
            token_pos = 512 * turn
            rid = f"rq3e-g{ordinal:03d}-{label.lower()}-turn-{turn:03d}"
            contributing = tuple(
                f"rq3e-g{ordinal:03d}-{label.lower()}-turn-{past:03d}"
                for past in range(turn, round_id + 1)
            )
            checkpoints.append(
                CheckpointObservation(
                    checkpoint_id=f"RQ3_G{ordinal:03d}_{label}_TURN_{turn:03d}",
                    workflow_label=label,
                    workflow_id=session_id,
                    turn=turn,
                    token_pos=token_pos,
                    node_id=ordinal_counter,
                    slots=(ordinal_counter,),
                    prefix_digest=token_digest([turn, ordinal]),
                    creation_order=ordinal_counter,
                    last_access_order=ordinal_counter,
                    fa_resident=resident,
                    recurrent_resident=resident,
                    contributing_request_ids=(rid,) + contributing[1:]
                    if turn == round_id
                    else contributing,
                )
            )
    pendings = tuple(
        PendingObservation(
            continuation_id=(
                f"RQ3_G{ordinal:03d}_{label}_PENDING_TURN_{round_id + 1:03d}"
            ),
            workflow_label=label,
            workflow_id=group.session_by_label[label],
            anchor_pos=512 * (round_id + 1),
            resident_fa_frontier=512 * (round_id + 1),
            input_token_digest=token_digest([round_id + 1, ordinal]),
            query_state_equal=fa_clean,
            scope_stable=fa_clean,
            traversed_node_ids=(1,),
        )
        for label in ("A", "B", "C", "D")
    )
    return GroupReplayTrace(
        group=group,
        executed_round=round_id,
        checkpoints=tuple(checkpoints),
        pendings=pendings,
        residency_snapshot_digest="c" * 64,
        request_rows=(),
        census_rows=(),
        boundary_audit=(),
        native_mamba_eviction=False,
        fa_cascade=False,
        oom=False,
        truncation=False,
        fa_query_side_effect_free=fa_clean,
    )


def _fake_messages(turns: int) -> list[dict[str, object]]:
    """构造交替 user/assistant 的合成消息序列。"""

    messages: list[dict[str, object]] = []
    for turn in range(1, turns + 1):
        messages.append({"role": "user", "content": f"用户消息 {turn}"})
        messages.append({"role": "assistant", "content": f"助手输出 {turn}"})
    return messages


class _FakeTokenizer:
    """产生与 history 长度成正比的确定性 token 序列。"""

    def __init__(self) -> None:
        self.calls = 0

    def apply_chat_template(
        self,
        history: list[dict[str, object]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        self.calls += 1
        tokens: list[int] = []
        for message in history:
            content = str(message.get("content", ""))
            if "SENTINEL" in content:
                tokens.append(999)
            else:
                tokens.append(7)
        tokens.append(2)
        return tokens


class _FakeRuntime:
    """在无 GPU 环境下模拟中立 replay 的最小 runtime。"""

    def __init__(self) -> None:
        self.ops: list[tuple[str, str]] = []
        self.nodes_by_digest: dict[str, dict[str, object]] = {}
        self.nodes_by_id: dict[int, dict[str, object]] = {}
        self.node_order: list[int] = []
        self.next_node_id = 1
        self.last_pos_by_label: dict[str, int] = {}
        self.shutdown_called = False

    def census(
        self,
        label: str,
        *,
        ordinal: int,
        request: object,
        previous: dict[str, object] | None,
    ) -> dict[str, object]:
        self.ops.append(("census", label))
        previous_ids = (
            set(previous["resident_ids"]) if previous is not None else set()
        )
        resident_ids = list(self.node_order)
        added = [
            node_id for node_id in resident_ids if node_id not in previous_ids
        ]
        rows = [
            {
                "node_id": node_id,
                "slots": (node_id,),
                "token_position": self.nodes_by_id[node_id]["token_pos"],
            }
            for node_id in resident_ids
        ]
        return {
            "resident_ids": resident_ids,
            "mamba_node_count": len(rows),
            "resident_mamba_nodes": rows,
            "added_mamba_node_ids": added,
            "removed_mamba_node_ids": [],
            "changed_existing_mamba_node_ids": [],
            "full_device_node_ids": resident_ids,
            "native_mamba_capacity_eviction_inferred": False,
            "fa_kv_cascade_eviction_inferred": False,
            "mamba_available_slots": 28 - len(rows),
            "mamba_evictable_slots": 28 - len(rows),
            "mamba_protected_slots": 0,
            "full_evictable_tokens": 0,
            "full_protected_tokens": 0,
            "tree_node_count": len(rows) + 1,
        }

    def execute(self, request: dict[str, object], ordinal: int) -> dict[str, object]:
        label = str(request["workflow_label"])
        input_ids = request["input_ids"]
        token_pos = len(input_ids)
        previous_pos = self.last_pos_by_label.get(label, 0)
        digest = token_digest(input_ids[:token_pos])
        node_id = self.next_node_id
        self.next_node_id += 1
        node = {
            "node_id": node_id,
            "token_pos": token_pos,
            "digest": digest,
        }
        self.nodes_by_digest[digest] = node
        self.nodes_by_id[node_id] = node
        self.node_order.append(node_id)
        self.last_pos_by_label[label] = token_pos
        self.ops.append(("execute", str(request["rid"])))
        return {
            "request_completed": True,
            "token_count_exact": True,
            "runtime_metrics_valid": True,
            "e": previous_pos,
            "h": previous_pos,
            "g": 0,
            "oom": False,
            "error": None,
            "rid": str(request["rid"]),
        }

    def inspect_checkpoint(
        self,
        probe_id: str,
        token_ids: object,
    ) -> dict[str, object]:
        digest = token_digest(token_ids)
        node = self.nodes_by_digest[digest]
        self.ops.append(("inspect_checkpoint", probe_id))
        return {
            "after": {
                "path": {
                    "node_id": node["node_id"],
                    "prefix_tokens": node["token_pos"],
                    "prefix_sha256": digest,
                    "target_full_present": True,
                    "path_full_all_present": True,
                    "target_mamba_present": True,
                    "target_mamba_slots": (node["node_id"],),
                }
            }
        }

    def inspect_fa_frontier(
        self,
        token_ids: object,
        *,
        nonce: str,
    ) -> dict[str, object]:
        self.ops.append(("fa_frontier", nonce))
        return {
            "state_equal": True,
            "scope_before": "S",
            "scope_after": "S",
            "resident_fa_frontier": len(token_ids),
            "traversed_node_ids": [1],
        }

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.ops.append(("shutdown", ""))


def test_session_ordering_is_seed_deterministic() -> None:
    """相同 seed 的排序必须完全一致，不同 seed 必须改变排序。"""

    sessions = _session_ids(64)
    first = order_sessions_by_digest(sessions)
    second = order_sessions_by_digest(list(reversed(sessions)))
    assert first == second
    assert tuple(sorted(first)) == tuple(sorted(sessions))
    other_seed = order_sessions_by_digest(sessions, seed=20260904)
    assert other_seed != first
    expected = session_order_digest(sessions[0])
    assert expected == session_order_digest(
        sessions[0],
        seed=20260903,
        protocol_version="rq3-openhands-v1",
    )
    manual = order_sessions_by_digest(sessions)
    assert manual == tuple(
        sorted(sessions, key=lambda value: session_order_digest(value))
    )


def test_group_assignment_and_labeling_are_deterministic() -> None:
    """连续 4 会话成组、组间不重叠、组内 A/B/C/D 顺序冻结。"""

    ordered = order_sessions_by_digest(_session_ids(1200))
    groups = build_workflow_groups(ordered)
    assert len(groups) == 300
    flat = [session for group in groups for session in group.session_ids]
    assert flat == list(ordered[: 300 * 4])
    first = groups[0]
    assert first.session_by_label == {
        "A": first.session_ids[0],
        "B": first.session_ids[1],
        "C": first.session_ids[2],
        "D": first.session_ids[3],
    }
    assert groups[0].session_ids != groups[1].session_ids


def test_allocation_round_mapping_is_balanced() -> None:
    """ordinal 轮转必须把 200 个 Main groups 均分到 round 2/3/4/5。"""

    assert allocation_round_for_ordinal(0) == 2
    assert allocation_round_for_ordinal(1) == 3
    assert allocation_round_for_ordinal(2) == 4
    assert allocation_round_for_ordinal(3) == 5
    assert allocation_round_for_ordinal(4) == 2
    groups = build_workflow_groups(order_sessions_by_digest(_session_ids(1200)))
    main = designated_main_groups(groups)
    assert len(main) == MAIN_GROUP_COUNT
    distribution = {round_id: 0 for round_id in (2, 3, 4, 5)}
    for group in main:
        distribution[group.allocation_round] += 1
    assert distribution == {2: 50, 3: 50, 4: 50, 5: 50}
    preflight_rounds = [group.allocation_round for group in main[:4]]
    assert preflight_rounds == [2, 3, 4, 5]


def test_main_groups_have_no_duplicate_sessions() -> None:
    """Main groups 之间 session 不得重复，且与 sensitivity 不重叠。"""

    groups = build_workflow_groups(order_sessions_by_digest(_session_ids(1200)))
    main = designated_main_groups(groups)
    sensitivity = reserved_sensitivity_groups(groups)
    assert len(sensitivity) == SENSITIVITY_GROUP_COUNT
    main_sessions = {
        session for group in main for session in group.session_ids
    }
    assert len(main_sessions) == MAIN_GROUP_COUNT * 4
    sensitivity_sessions = {
        session for group in sensitivity for session in group.session_ids
    }
    assert main_sessions.isdisjoint(sensitivity_sessions)


def test_materialize_stops_before_r_plus_2() -> None:
    """request 物化不得消费 pending 输出或 r+2 消息。"""

    group = _group(0, 3)
    messages = _fake_messages(8)
    messages[7]["content"] = "SENTINEL 第三轮助手输出"
    messages[9]["content"] = "SENTINEL 第四轮助手输出"
    tokenizer = _FakeTokenizer()
    requests, audits = materialize_group_requests(
        tokenizer,
        {label: list(messages) for label in ("A", "B", "C", "D")},
        group=group,
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
    )
    assert sorted(requests) == [(label, turn) for label in ("A", "B", "C", "D") for turn in (1, 2, 3, 4)]
    assert tokenizer.calls == 16
    for request in requests.values():
        assert 999 not in request["input_ids"]
    for audit in audits:
        assert audit["pending_turn_output_read"] is False
        assert audit["r_plus_2_message_consumed"] is False
        assert audit["r_plus_2_request_materialized"] is False
        assert audit["raw_items_iterated_through_pending_marker"] < len(messages)


def test_lfu_provenance_matches_frozen_frequency(tmp_path) -> None:
    """provenance 事件数必须与 snapshot 冻结的 access_frequency 完全一致。"""

    trace = _trace(0, 4)
    result = assemble_group_snapshot(trace, checkpoint_size_bytes=1024)
    assert result.status == "ELIGIBLE"
    snapshot = result.snapshot
    assert snapshot is not None
    frozen = {
        item.checkpoint_id: item.access_frequency
        for item in snapshot.lfu_access_frequency
    }
    assert len(result.lfu_provenance) == len(frozen)
    for row in result.lfu_provenance:
        checkpoint_id = str(row["checkpoint_id"])
        assert int(row["contributing_event_count"]) == frozen[checkpoint_id]
        assert int(row["access_frequency"]) == frozen[checkpoint_id]
        assert row["matches_frozen_frequency"] is True
        assert row["frequency_observed_through_epoch"] == 4
    turn_one = next(
        row
        for row in result.lfu_provenance
        if row["checkpoint_id"].endswith("TURN_001")
    )
    assert int(turn_one["access_frequency"]) == 4
    turn_four = next(
        row
        for row in result.lfu_provenance
        if row["checkpoint_id"].endswith("TURN_004")
    )
    assert int(turn_four["access_frequency"]) == 1


def test_candidate_ordering_is_deterministic() -> None:
    """候选注册顺序不同但内容相同时 snapshot 必须一致。"""

    trace = _trace(0, 2)
    shuffled = GroupReplayTrace(
        **{
            **trace.__dict__,
            "checkpoints": tuple(reversed(trace.checkpoints)),
        }
    )
    first = assemble_group_snapshot(trace, checkpoint_size_bytes=1024)
    second = assemble_group_snapshot(shuffled, checkpoint_size_bytes=1024)
    assert first.snapshot is not None and second.snapshot is not None
    ordered_ids = [
        item.checkpoint_id for item in first.snapshot.eligible_candidates
    ]
    assert ordered_ids == sorted(ordered_ids)
    assert first.snapshot.content_digest() == second.snapshot.content_digest()


def test_snapshot_digest_is_identical_across_rebuilds(tmp_path) -> None:
    """同一 trace 重复装配 digest 必须相同，artifact 写读往返一致。"""

    trace = _trace(0, 2)
    first = assemble_group_snapshot(trace, checkpoint_size_bytes=1024)
    second = assemble_group_snapshot(trace, checkpoint_size_bytes=1024)
    assert first.snapshot is not None and second.snapshot is not None
    assert first.snapshot.content_digest() == second.snapshot.content_digest()
    group = _group(0, 2)
    path = write_snapshot_artifact(tmp_path, group, first.snapshot)
    reloaded = load_allocation_snapshot(path)
    assert reloaded.content_digest() == first.snapshot.content_digest()
    assert reloaded.canonical_serialization() == (
        first.snapshot.canonical_serialization()
    )


def test_build_failed_never_substitutes_group() -> None:
    """BUILD_FAILED 只记录原 group，不得用后续 group 补位。"""

    groups = [_group(ordinal, 2 + ordinal % 4) for ordinal in range(6)]
    attempted: list[int] = []

    def collect_one(group: WorkflowGroup) -> dict[str, object]:
        attempted.append(group.group_ordinal)
        if group.group_ordinal == 2:
            return {
                "group_ordinal": group.group_ordinal,
                "allocation_round": group.allocation_round,
                "session_ids": list(group.session_ids),
                "status": "BUILD_FAILED",
                "primary_reason": "native_mamba_eviction",
            }
        return {
            "group_ordinal": group.group_ordinal,
            "allocation_round": group.allocation_round,
            "session_ids": list(group.session_ids),
            "status": "ELIGIBLE",
            "snapshot_digest": f"{group.group_ordinal:064x}",
        }

    verdicts = run_designated_groups(groups, collect_one)
    assert attempted == [0, 1, 2, 3, 4, 5]
    assert [int(row["group_ordinal"]) for row in verdicts] == [0, 1, 2, 3, 4, 5]
    failed = [row for row in verdicts if row["status"] == "BUILD_FAILED"]
    assert len(failed) == 1 and failed[0]["group_ordinal"] == 2
    summary = summarize_population(verdicts)
    assert summary["attempted_groups"] == 6
    assert summary["eligible_count"] == 5
    assert summary["failure_reason_distribution"] == {
        "native_mamba_eviction": 1
    }


def test_collector_never_calls_policy_selectors(tmp_path, monkeypatch) -> None:
    """selector 全部置为报错时 collector 全流程仍必须成功。"""

    import evaluation.rq3_openhands_neutral_collector as collector_module
    import evaluation.controlled_multiworkflow_v1.policies as policies_module
    import evaluation.sota_policies as sota_module
    import flowstate.optimizer as optimizer_module

    def forbidden(*args, **kwargs):
        raise AssertionError("collector 不得调用任何 policy selector")

    monkeypatch.setattr(policies_module, "select_global_lru", forbidden)
    monkeypatch.setattr(
        "evaluation.rq3_frozen_snapshot_evaluator.select_lfu",
        forbidden,
    )
    monkeypatch.setattr(
        "evaluation.rq3_frozen_snapshot_evaluator.select_exact_opt",
        forbidden,
    )
    monkeypatch.setattr(sota_module.MarconiStylePolicy, "select", forbidden)
    monkeypatch.setattr(optimizer_module.GlobalOptimizer, "select", forbidden)
    for name in (
        "select_global_lru",
        "select_lfu",
        "select_exact_opt",
        "MarconiStylePolicy",
        "GlobalOptimizer",
        "StateController",
    ):
        assert not hasattr(collector_module, name)

    group = _group(0, 2)
    runtime = _FakeRuntime()
    tokenizer = _FakeTokenizer()
    messages = _fake_messages(6)
    directories = prepare_artifact_directories(tmp_path)
    verdict = collect_group(
        group,
        runtime_factory=lambda: runtime,
        messages_by_label={
            label: list(messages) for label in ("A", "B", "C", "D")
        },
        tokenizer=tokenizer,
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
        checkpoint_size_bytes=1024,
        directories=directories,
    )
    assert verdict["status"] == "ELIGIBLE"
    assert runtime.shutdown_called is True


def test_collector_never_executes_logical_eviction(tmp_path) -> None:
    """collector 的 runtime 操作日志中不得出现任何驱逐动作。"""

    group = _group(0, 3)
    runtime = _FakeRuntime()
    tokenizer = _FakeTokenizer()
    messages = _fake_messages(6)
    directories = prepare_artifact_directories(tmp_path)
    verdict = collect_group(
        group,
        runtime_factory=lambda: runtime,
        messages_by_label={
            label: list(messages) for label in ("A", "B", "C", "D")
        },
        tokenizer=tokenizer,
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
        checkpoint_size_bytes=1024,
        directories=directories,
    )
    assert verdict["status"] == "ELIGIBLE"
    op_names = {name for name, _ in runtime.ops}
    assert op_names <= {
        "execute",
        "census",
        "inspect_checkpoint",
        "fa_frontier",
        "shutdown",
    }
    assert not any("evict" in name for name in op_names)


def test_full_collection_path_writes_consistent_artifacts(tmp_path) -> None:
    """端到端 fake 采集必须产出一致的 snapshot、provenance 与 correctness artifact。"""

    group = _group(0, 2)
    runtime = _FakeRuntime()
    tokenizer = _FakeTokenizer()
    messages = _fake_messages(6)
    directories = prepare_artifact_directories(tmp_path)
    verdict = collect_group(
        group,
        runtime_factory=lambda: runtime,
        messages_by_label={
            label: list(messages) for label in ("A", "B", "C", "D")
        },
        tokenizer=tokenizer,
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
        checkpoint_size_bytes=1024,
        directories=directories,
    )
    assert verdict["status"] == "ELIGIBLE"
    snapshot_files = list(directories.snapshots.glob("g000_*.json"))
    provenance_files = list(directories.lfu_provenance.glob("g000_*.json"))
    correctness_files = list(directories.runtime_correctness.glob("g000.json"))
    assert len(snapshot_files) == 1
    assert len(provenance_files) == 1
    assert len(correctness_files) == 1
    reloaded = load_allocation_snapshot(snapshot_files[0])
    provenance = json.loads(provenance_files[0].read_text(encoding="utf-8"))
    frozen = {
        item.checkpoint_id: item.access_frequency
        for item in reloaded.lfu_access_frequency
    }
    assert provenance["snapshot_digest"] == reloaded.content_digest()
    assert all(row["matches_frozen_frequency"] for row in provenance["rows"])
    for row in provenance["rows"]:
        assert int(row["contributing_event_count"]) == frozen[
            row["checkpoint_id"]
        ]
    correctness = json.loads(correctness_files[0].read_text(encoding="utf-8"))
    assert correctness["boundary_audit"]
    assert len(correctness["request_rows"]) == 8
    expected_frequency = {"TURN_001": 2, "TURN_002": 1}
    for row in provenance["rows"]:
        turn_key = row["checkpoint_id"].rsplit("_", 2)[-2] + "_" + row["checkpoint_id"].rsplit("_", 1)[-1]
        assert frozen[row["checkpoint_id"]] == expected_frequency[turn_key]


def test_failure_reason_is_deterministic() -> None:
    """同一失败输入必须产生完全相同的失败主原因与诊断。"""

    base = _trace(0, 2)
    insufficient = GroupReplayTrace(
        **{
            **base.__dict__,
            "checkpoints": tuple(
                item for item in base.checkpoints if item.turn == 1
            ),
        }
    )
    first = assemble_group_snapshot(insufficient, checkpoint_size_bytes=1024)
    second = assemble_group_snapshot(insufficient, checkpoint_size_bytes=1024)
    assert first.status == "BUILD_FAILED"
    assert first.primary_reason == "candidate_count_below_8"
    assert first.primary_reason == second.primary_reason
    assert first.diagnostics == second.diagnostics

    not_resident = _trace(0, 2, resident=False)
    third = assemble_group_snapshot(not_resident, checkpoint_size_bytes=1024)
    assert third.status == "BUILD_FAILED"
    assert third.primary_reason == "checkpoint_not_resident_at_barrier"

    side_effect = _trace(0, 2, fa_clean=False)
    fourth = assemble_group_snapshot(side_effect, checkpoint_size_bytes=1024)
    assert fourth.status == "BUILD_FAILED"
    assert fourth.primary_reason == "fa_frontier_query_side_effect"


def test_k_sweep_and_reference_budget_follow_protocol() -> None:
    """K sweep 必须按冻结 ratio 与 K=2 去重，且全部小于 |C_t|。"""

    assert k_sweep_for_candidate_count(8) == (2, 4, 6)
    assert k_sweep_for_candidate_count(12) == (2, 3, 6, 9)
    assert k_sweep_for_candidate_count(20) == (2, 5, 10, 15)
    assert reference_budget_for_candidate_count(8) == 6
    assert reference_budget_for_candidate_count(20) == 15
    with pytest.raises(ValueError):
        reference_budget_for_candidate_count(7)
    with pytest.raises(ValueError):
        k_sweep_for_candidate_count(2)


def test_session_eligibility_rules() -> None:
    """dataset-level eligibility 必须按冻结条件逐项判定。"""

    eligible = evaluate_session_eligibility("s", 60, 6, [100] * 6)
    assert eligible.eligible is True and eligible.reason == "eligible"
    low_turns = evaluate_session_eligibility("s", 59, 100, [100] * 6)
    assert low_turns.eligible is False
    assert low_turns.reason == "n_turns_below_60"
    few_assistant = evaluate_session_eligibility("s", 60, 5, ())
    assert few_assistant.eligible is False
    assert few_assistant.reason == "assistant_turns_insufficient"
    too_long = evaluate_session_eligibility(
        "s", 60, 6, [100, 100, 100, 100, 100, 131_073]
    )
    assert too_long.eligible is False
    assert too_long.reason == "replay_input_exceeds_131072"
    boundary = evaluate_session_eligibility(
        "s", 60, 6, [131_072] * 6
    )
    assert boundary.eligible is True


def test_replay_driver_registers_expected_trace() -> None:
    """fake runtime 驱动 replay 后 trace 必须包含全部 checkpoint 与 pending。"""

    group = _group(0, 2)
    runtime = _FakeRuntime()
    tokenizer = _FakeTokenizer()
    messages = _fake_messages(6)
    requests, audits = materialize_group_requests(
        tokenizer,
        {label: list(messages) for label in ("A", "B", "C", "D")},
        group=group,
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
    )
    trace = replay_group_to_barrier(runtime, group, requests)
    assert len(trace.checkpoints) == 8
    assert len(trace.pendings) == 4
    frequencies = {
        item.checkpoint_id: len(item.contributing_request_ids)
        for item in trace.checkpoints
    }
    assert all(value >= 1 for value in frequencies.values())
    turn_one = [
        value
        for key, value in frequencies.items()
        if key.endswith("TURN_001")
    ]
    assert turn_one == [2, 2, 2, 2]
    turn_two = [
        value
        for key, value in frequencies.items()
        if key.endswith("TURN_002")
    ]
    assert turn_two == [1, 1, 1, 1]
    for pending in trace.pendings:
        assert pending.query_state_equal is True
        assert pending.resident_fa_frontier == pending.anchor_pos


class _EvictingFakeRuntime(_FakeRuntime):
    """在指定 ordinal 的 census 中模拟原生 Mamba 驱逐。"""

    def __init__(self, fail_ordinal: int) -> None:
        super().__init__()
        self._fail_ordinal = fail_ordinal

    def census(self, label, *, ordinal, request, previous):
        result = super().census(
            label, ordinal=ordinal, request=request, previous=previous
        )
        if ordinal == self._fail_ordinal:
            result["native_mamba_capacity_eviction_inferred"] = True
            result["removed_mamba_node_ids"] = [result["resident_ids"][0]]
        return result


class _ShrinkingFakeRuntime(_FakeRuntime):
    """让指定 workflow 的后继 checkpoint 位置小于前一个。"""

    def execute(self, request, ordinal):
        if str(request["workflow_label"]) == "A" and int(request["turn"]) == 2:
            label = str(request["workflow_label"])
            digest = token_digest(request["input_ids"][:1])
            node_id = self.next_node_id
            self.next_node_id += 1
            node = {"node_id": node_id, "token_pos": 1, "digest": digest}
            self.nodes_by_digest[digest] = node
            self.nodes_by_id[node_id] = node
            self.node_order.append(node_id)
            self.last_pos_by_label[label] = 1
            self.ops.append(("execute", str(request["rid"])))
            return {
                "request_completed": True,
                "token_count_exact": True,
                "runtime_metrics_valid": True,
                "e": self.last_pos_by_label.get(label, 0),
                "h": 0,
                "g": 0,
                "oom": False,
                "error": None,
                "rid": str(request["rid"]),
            }
        return super().execute(request, ordinal)


def test_worker_crash_does_not_kill_parent(tmp_path) -> None:
    """worker 子进程自杀式崩溃时 parent 必须存活并记录确定性失败。"""

    import sys as _sys

    directories = prepare_artifact_directories(tmp_path)
    crash_command = [_sys.executable, "-c", "import os; os._exit(1)"]
    first = run_group_via_worker(
        _group(0, 2),
        directories=directories,
        gpu_wait_record={"stable": True},
        worker_command=crash_command,
        worker_timeout_s=60,
    )
    second = run_group_via_worker(
        _group(1, 3),
        directories=directories,
        gpu_wait_record={"stable": True},
        worker_command=crash_command,
        worker_timeout_s=60,
    )
    assert first["status"] == "BUILD_FAILED"
    assert first["primary_reason"] == "worker_process_died"
    assert first["worker_lifecycle"]["timed_out"] is False
    assert second["status"] == "BUILD_FAILED"
    assert second["primary_reason"] == "worker_process_died"
    assert first["group_ordinal"] == 0
    assert second["group_ordinal"] == 1


def test_worker_timeout_is_deterministically_reaped(tmp_path) -> None:
    """worker 超时必须被确定性回收并生成 worker_timeout 失败。"""

    import sys as _sys

    directories = prepare_artifact_directories(tmp_path)
    sleep_command = [_sys.executable, "-c", "import time; time.sleep(120)"]
    verdict = run_group_via_worker(
        _group(0, 2),
        directories=directories,
        gpu_wait_record={"stable": True},
        worker_command=sleep_command,
        worker_timeout_s=1.0,
    )
    assert verdict["status"] == "BUILD_FAILED"
    assert verdict["primary_reason"] == "worker_timeout"
    assert verdict["worker_lifecycle"]["timed_out"] is True
    failure_files = list(directories.failures.glob("g000.json"))
    assert len(failure_files) == 1
    payload = json.loads(failure_files[0].read_text(encoding="utf-8"))
    assert payload["primary_reason"] == "worker_timeout"


def test_worker_failure_artifact_is_consumed(tmp_path) -> None:
    """worker 已写失败 artifact 后退出时 parent 必须采用其原因。"""

    import sys as _sys

    directories = prepare_artifact_directories(tmp_path)
    failure_path = directories.failures / "g000.json"
    stub_code = (
        "import json, pathlib;"
        f"pathlib.Path(r'{failure_path}').write_text("
        "json.dumps({'primary_reason': 'engine_startup_failed',"
        " 'diagnostics': {'stage': 'startup'}}))"
    )
    verdict = run_group_via_worker(
        _group(0, 2),
        directories=directories,
        gpu_wait_record={"stable": True},
        worker_command=[_sys.executable, "-c", stub_code],
        worker_timeout_s=60,
    )
    assert verdict["status"] == "BUILD_FAILED"
    assert verdict["primary_reason"] == "engine_startup_failed"


def test_crash_never_substitutes_next_group(tmp_path) -> None:
    """worker 崩溃后下一 attempt 必须仍是冻结顺序的下一个 ordinal。"""

    import sys as _sys

    directories = prepare_artifact_directories(tmp_path)
    crash_command = [_sys.executable, "-c", "import os; os._exit(1)"]
    groups = [_group(ordinal, 2 + ordinal % 4) for ordinal in range(3)]

    def collect_one(group):
        return run_group_via_worker(
            group,
            directories=directories,
            gpu_wait_record={"stable": True},
            worker_command=crash_command,
            worker_timeout_s=60,
        )

    verdicts = run_designated_groups(groups, collect_one)
    assert [int(row["group_ordinal"]) for row in verdicts] == [0, 1, 2]
    assert all(row["status"] == "BUILD_FAILED" for row in verdicts)


def test_non_monotonic_positions_are_never_auto_corrected(tmp_path) -> None:
    """位置非递增的 checkpoint 必须原样失败，不得被 collector 修正。"""

    directories = prepare_artifact_directories(tmp_path)
    group = _group(0, 2)
    runtime = _ShrinkingFakeRuntime()
    verdict = collect_group(
        group,
        runtime_factory=lambda: runtime,
        messages_by_label={
            label: _fake_messages(6) for label in ("A", "B", "C", "D")
        },
        tokenizer=_FakeTokenizer(),
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
        checkpoint_size_bytes=1024,
        directories=directories,
    )
    assert verdict["status"] == "BUILD_FAILED"
    assert verdict["primary_reason"] == "non_monotonic_inputs"
    failure_files = list(directories.failures.glob("g000.json"))
    assert len(failure_files) == 1
    payload = json.loads(failure_files[0].read_text(encoding="utf-8"))
    assert payload["primary_reason"] == "non_monotonic_inputs"
    assert not list(directories.snapshots.glob("g*.json"))


def test_collector_never_modifies_frozen_runtime_configuration(tmp_path) -> None:
    """采集全流程不得改动冻结 engine 配置的任何字段。"""

    import copy

    from evaluation.openhands_common_barrier_snapshot_gate import (
        ENGINE_CONFIGURATION_COMMON_BARRIER,
    )

    frozen = copy.deepcopy(ENGINE_CONFIGURATION_COMMON_BARRIER)
    directories = prepare_artifact_directories(tmp_path)
    verdict = collect_group(
        _group(0, 2),
        runtime_factory=lambda: _FakeRuntime(),
        messages_by_label={
            label: _fake_messages(6) for label in ("A", "B", "C", "D")
        },
        tokenizer=_FakeTokenizer(),
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
        checkpoint_size_bytes=1024,
        directories=directories,
    )
    assert verdict["status"] == "ELIGIBLE"
    assert ENGINE_CONFIGURATION_COMMON_BARRIER == frozen
    assert frozen["max_mamba_cache_size"] == 28
    assert frozen["mem_fraction_static"] == 0.40
    assert frozen["context_length"] == 131200


def test_interrupted_root_is_never_formal_population(tmp_path) -> None:
    """标记 DIAGNOSTIC_ONLY 的 root 不得被识别为正式 population。"""

    from evaluation.rq3_openhands_neutral_collector import (
        assert_formal_restart_root,
        read_population_root_status,
        write_interrupted_marker,
    )

    write_interrupted_marker(
        tmp_path,
        attempted_groups=30,
        eligible_groups=25,
        failed_groups=5,
        failure_reason_distribution={"non_monotonic_inputs": 4},
        collector_versions=["test-version"],
        reason="测试中断",
    )
    assert read_population_root_status(tmp_path) == "diagnostic_only"
    with pytest.raises(RuntimeError, match="DIAGNOSTIC_ONLY"):
        assert_formal_restart_root(tmp_path)


def test_formal_restart_requires_fresh_empty_root(tmp_path) -> None:
    """正式重跑必须从空 root 开始，已有 snapshots 的 root 必须被拒绝。"""

    from evaluation.rq3_openhands_neutral_collector import (
        assert_formal_restart_root,
        read_population_root_status,
    )

    empty_root = tmp_path / "fresh"
    assert_formal_restart_root(empty_root)
    assert read_population_root_status(empty_root) == "empty"
    used_root = tmp_path / "used"
    (used_root / "snapshots").mkdir(parents=True)
    (used_root / "snapshots" / "g000_ab12.json").write_text("{}")
    assert read_population_root_status(used_root) == "unmarked_partial"
    with pytest.raises(RuntimeError, match="空 artifact root"):
        assert_formal_restart_root(used_root)


def test_restart_ordinal_starts_from_group_zero(tmp_path) -> None:
    """正式重跑的第一个 attempt 必须是 group ordinal 0。"""

    import sys as _sys

    directories = prepare_artifact_directories(tmp_path)
    attempted: list[int] = []
    groups = build_workflow_groups(
        order_sessions_by_digest(_session_ids(16))
    )[:3]

    def collect_one(group):
        attempted.append(group.group_ordinal)
        return run_group_via_worker(
            group,
            directories=directories,
            gpu_wait_record={"stable": True},
            worker_command=[_sys.executable, "-c", "import os; os._exit(1)"],
            worker_timeout_s=60,
        )

    run_designated_groups(groups, collect_one)
    assert attempted[0] == 0
    assert attempted == [0, 1, 2]


def test_partial_diagnostics_survive_group_abort(tmp_path) -> None:
    """group 中途 abort 时必须保留已取得的 request 与 census 诊断行。"""

    directories = prepare_artifact_directories(tmp_path)
    verdict = collect_group(
        _group(0, 2),
        runtime_factory=lambda: _EvictingFakeRuntime(fail_ordinal=5),
        messages_by_label={
            label: _fake_messages(6) for label in ("A", "B", "C", "D")
        },
        tokenizer=_FakeTokenizer(),
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
        checkpoint_size_bytes=1024,
        directories=directories,
    )
    assert verdict["status"] == "BUILD_FAILED"
    assert verdict["primary_reason"] == "native_mamba_eviction"
    correctness = json.loads(
        (directories.runtime_correctness / "g000.json").read_text(
            encoding="utf-8"
        )
    )
    assert correctness["abort_reason"] == "native_mamba_eviction"
    assert len(correctness["partial_request_rows"]) == 5
    assert len(correctness["partial_census_rows"]) == 6


def test_wait_gpu_stable_requires_consecutive_clean(monkeypatch) -> None:
    """GPU 稳定等待必须连续多次观测干净才放行，否则超时。"""

    import evaluation.rq3_openhands_neutral_collector as collector_module

    memory_sequence = iter([30000, 30000, 1200, 900, 800, 700, 600])
    monkeypatch.setattr(
        collector_module,
        "query_gpu_memory_used_mib",
        lambda gpu_index=0: next(memory_sequence),
    )
    monkeypatch.setattr(
        collector_module,
        "query_gpu_compute_processes",
        lambda gpu_index=0: [],
    )
    record = collector_module.wait_gpu_stable(
        gpu_index=0,
        threshold_mib=4096,
        required_observations=3,
        interval_s=0.01,
        timeout_s=30.0,
    )
    assert record["stable"] is True
    assert len(record["observations"]) == 5
    assert record["final_used_mib"] == 800

    monkeypatch.setattr(
        collector_module,
        "query_gpu_memory_used_mib",
        lambda gpu_index=0: 30000,
    )
    with pytest.raises(RuntimeError, match="超时"):
        collector_module.wait_gpu_stable(
            gpu_index=0,
            threshold_mib=4096,
            required_observations=3,
            interval_s=0.01,
            timeout_s=0.05,
        )
