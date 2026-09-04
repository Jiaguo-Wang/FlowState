"""验证正式 RQ3 冻结快照、公共目标与五策略统一入口。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from evaluation.rq3_frozen_snapshot_evaluator import (
    AllocationSnapshot,
    FrozenCheckpointRuntimeEvidence,
    FrozenOnlineInformationBoundary,
    ObjectiveEvaluation,
    PolicyEvaluation,
    build_allocation_snapshot,
    evaluate_allocation_snapshot,
    evaluate_objective,
    greedy_exact_metrics,
    select_exact_opt,
    select_lfu,
)
from evaluation.sota_metadata import CONTROLLED_MARCONI_ALPHA
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


CHECKPOINT_BYTES = 1024


def _digest(character: str) -> str:
    """构造测试使用的稳定 SHA-256 形状摘要。"""

    return character * 64


def _pending() -> list[PendingContinuation]:
    """构造两个不同恢复压力的待续请求。"""

    return [
        PendingContinuation(
            continuation_id="P1",
            workflow_id="W1",
            lineage_path=("W1", "NEXT"),
            anchor_pos=8192,
            resident_fa_frontier=8192,
        ),
        PendingContinuation(
            continuation_id="P2",
            workflow_id="W2",
            lineage_path=("W2", "NEXT"),
            anchor_pos=1024,
            resident_fa_frontier=1024,
        ),
    ]


def _candidates() -> list[CheckpointCandidate]:
    """构造三个全部满足正式 eligibility 的候选。"""

    return [
        CheckpointCandidate(
            checkpoint_id="A",
            workflow_id="W1",
            lineage_path=("W1",),
            token_pos=8192,
            memory_bytes=CHECKPOINT_BYTES,
        ),
        CheckpointCandidate(
            checkpoint_id="B",
            workflow_id="W1",
            lineage_path=("W1",),
            token_pos=4096,
            memory_bytes=CHECKPOINT_BYTES,
        ),
        CheckpointCandidate(
            checkpoint_id="C",
            workflow_id="W2",
            lineage_path=("W2",),
            token_pos=1024,
            memory_bytes=CHECKPOINT_BYTES,
        ),
    ]


def _snapshot(
    *,
    budget_k: int = 1,
    candidates: list[CheckpointCandidate] | None = None,
    creation: dict[str, int] | None = None,
    access: dict[str, int] | None = None,
    flop: dict[str, float] | None = None,
    frequency: dict[str, int] | None = None,
    frequency_observed_epoch: int = 2,
    evidence_node_offset: int = 0,
    evidence_digest_char: str = "a",
) -> AllocationSnapshot:
    """用可变输入构造一个完整 canonical allocation snapshot。"""

    active_candidates = candidates or _candidates()
    checkpoint_ids = [item.checkpoint_id for item in active_candidates]
    active_creation = creation or {
        checkpoint_id: index
        for index, checkpoint_id in enumerate(checkpoint_ids, start=1)
    }
    active_access = access or dict(active_creation)
    active_flop = flop or {
        "A": 4096.0,
        "B": 4096.0,
        "C": 1024.0,
    }
    active_frequency = frequency or {
        checkpoint_id: 1 for checkpoint_id in checkpoint_ids
    }
    node_by_checkpoint = {
        checkpoint_id: index + evidence_node_offset
        for index, checkpoint_id in enumerate(sorted(checkpoint_ids), start=1)
    }
    evidence = [
        FrozenCheckpointRuntimeEvidence(
            checkpoint_id=checkpoint_id,
            node_id=node_by_checkpoint[checkpoint_id],
            runtime_identity_digest=_digest(evidence_digest_char),
            checkpoint_handle_digest=_digest("b"),
        )
        for checkpoint_id in checkpoint_ids
    ]
    pending = _pending()
    return build_allocation_snapshot(
        allocation_epoch=2,
        snapshot_id="snapshot-2",
        pending_continuations=pending,
        eligible_candidates=active_candidates,
        creation_order_by_checkpoint=active_creation,
        last_access_order_by_checkpoint=active_access,
        marconi_flop_saved_by_checkpoint=active_flop,
        access_frequency_by_checkpoint=active_frequency,
        frequency_observed_through_epoch=frequency_observed_epoch,
        marconi_alpha=CONTROLLED_MARCONI_ALPHA,
        logical_budget_k=budget_k,
        budget_bytes=budget_k * CHECKPOINT_BYTES,
        runtime_evidence=evidence,
        residency_snapshot_digest=_digest("c"),
        online_boundary=FrozenOnlineInformationBoundary(
            materialized_through_epoch=2,
            visible_continuation_ids=("P2", "P1"),
        ),
    )


def _policy_evaluation(
    *,
    name: str,
    cost: float,
    empty: float,
    benefit: float,
) -> PolicyEvaluation:
    """构造零分母指标测试所需的最小策略结果。"""

    return PolicyEvaluation(
        policy_name=name,
        selected_checkpoint_ids=(),
        selector_wall_time_ms=0.0,
        total_recovery_cost_ms=cost,
        empty_selection_cost_ms=empty,
        total_benefit_ms=benefit,
        per_continuation=(),
        candidate_count=0,
        pending_count=0,
        selector_internal_evaluations=0,
        final_common_scoring_evaluations=1,
        snapshot_digest_before=_digest("d"),
        snapshot_digest_after=_digest("d"),
    )


def test_identical_content_has_identical_digest() -> None:
    """输入顺序不同但语义相同时必须得到相同摘要。"""

    creation = {"A": 1, "B": 2, "C": 3}
    access = {"A": 1, "B": 2, "C": 3}
    first = _snapshot(creation=creation, access=access)
    reversed_candidates = list(reversed(_candidates()))
    second = _snapshot(
        candidates=reversed_candidates,
        creation=creation,
        access=access,
    )
    assert first.canonical_serialization() == second.canonical_serialization()
    assert first.content_digest() == second.content_digest()
    assert json.loads(first.canonical_serialization())["snapshot_id"] == "snapshot-2"


def test_any_key_field_change_changes_digest() -> None:
    """各类关键字段变化均必须进入 snapshot digest。"""

    snapshot = _snapshot()
    changes = (
        replace(snapshot, allocation_epoch=3),
        replace(snapshot, snapshot_id="snapshot-other"),
        replace(
            snapshot,
            pending_continuations=(
                replace(snapshot.pending_continuations[0], anchor_pos=8193),
                snapshot.pending_continuations[1],
            ),
        ),
        replace(
            snapshot,
            eligible_candidates=(
                replace(snapshot.eligible_candidates[0], token_pos=8191),
                *snapshot.eligible_candidates[1:],
            ),
        ),
        replace(
            snapshot,
            candidate_metadata=(
                replace(snapshot.candidate_metadata[0], last_access_order=99),
                *snapshot.candidate_metadata[1:],
            ),
        ),
        replace(snapshot, logical_budget_k=2),
        replace(
            snapshot,
            runtime_evidence=(
                replace(snapshot.runtime_evidence[0], node_id=99),
                *snapshot.runtime_evidence[1:],
            ),
        ),
        replace(snapshot, residency_snapshot_digest=_digest("e")),
        replace(
            snapshot,
            online_boundary=replace(
                snapshot.online_boundary,
                materialized_through_epoch=1,
            ),
        ),
    )
    original = snapshot.content_digest()
    assert all(item.content_digest() != original for item in changes)


def test_snapshot_is_deeply_immutable_and_detached_from_inputs() -> None:
    """构造后的嵌套字段不得受原可变 metadata 影响。"""

    creation = {"A": 1, "B": 2, "C": 3}
    access = {"A": 1, "B": 2, "C": 3}
    flop = {"A": 4096.0, "B": 4096.0, "C": 1024.0}
    snapshot = _snapshot(creation=creation, access=access, flop=flop)
    original_digest = snapshot.content_digest()
    creation["A"] = 999
    access["A"] = 999
    flop["A"] = 999.0
    assert snapshot.content_digest() == original_digest
    with pytest.raises(FrozenInstanceError):
        snapshot.logical_budget_k = 9
    with pytest.raises(TypeError):
        snapshot.pending_continuations[0].lineage_path[0] = "MUTATED"


def test_all_policies_preserve_snapshot_digest() -> None:
    """五种方法选择和统一评分前后均不得改变 snapshot。"""

    snapshot = _snapshot()
    result = evaluate_allocation_snapshot(snapshot)
    assert tuple(item.policy_name for item in result.policy_results) == (
        "LRU",
        "LFU",
        "Marconi",
        "FlowState",
        "Exact OPT",
    )
    assert all(
        item.snapshot_digest_before == item.snapshot_digest_after
        == snapshot.content_digest()
        for item in result.policy_results
    )


def test_all_final_selections_use_shared_objective() -> None:
    """Exact 内部集合与五个最终集合必须经过同一注入评分器。"""

    calls: list[tuple[str, ...]] = []

    def tracking(
        snapshot: AllocationSnapshot,
        selected: list[str] | tuple[str, ...],
    ) -> ObjectiveEvaluation:
        calls.append(tuple(sorted(selected)))
        return evaluate_objective(snapshot, selected)

    result = evaluate_allocation_snapshot(
        _snapshot(),
        objective_function=tracking,
    )
    exact = next(
        item for item in result.policy_results if item.policy_name == "Exact OPT"
    )
    final_selected = [item.selected_checkpoint_ids for item in result.policy_results]
    assert all(selected in calls for selected in final_selected)
    assert len(calls) == exact.selector_internal_evaluations + 5
    assert all(item.final_common_scoring_evaluations == 1 for item in result.policy_results)


def test_benefit_equals_empty_cost_minus_selected_cost() -> None:
    """公共评分结果必须严格满足冻结的 F(S) 定义。"""

    objective = evaluate_objective(_snapshot(), ("A",))
    assert objective.total_benefit_ms == pytest.approx(
        objective.empty_selection_cost_ms - objective.total_recovery_cost_ms
    )
    assert objective.objective_evaluation_count == 1


def test_exact_opt_matches_manual_small_case() -> None:
    """K=1 时恢复压力较大的 W1 深检查点必须成为手工最优解。"""

    exact = select_exact_opt(_snapshot())
    assert exact.selected_checkpoint_ids == ("A",)
    assert exact.selector_internal_evaluations == 4


def test_flowstate_cost_is_not_lower_than_exact_cost() -> None:
    """greedy 统一评分成本不得低于 combination Exact OPT。"""

    result = evaluate_allocation_snapshot(_snapshot())
    by_name = {item.policy_name: item for item in result.policy_results}
    assert by_name["FlowState"].total_recovery_cost_ms >= (
        by_name["Exact OPT"].total_recovery_cost_ms
    )
    assert by_name["LRU"].selector_internal_evaluations == 0
    assert by_name["LFU"].selector_internal_evaluations == 0
    assert by_name["Marconi"].selector_internal_evaluations == 0
    assert by_name["FlowState"].selector_internal_evaluations == 4
    assert by_name["Exact OPT"].selector_internal_evaluations == 4


def test_equal_greedy_and_opt_metrics_are_zero() -> None:
    """greedy 与 Exact 成本相同时差距必须为零。"""

    result = evaluate_allocation_snapshot(_snapshot())
    metrics = result.flowstate_vs_exact
    assert metrics.absolute_cost_gap_ms == 0.0
    assert metrics.relative_cost_gap == 0.0
    assert metrics.benefit_ratio == pytest.approx(1.0)


def test_zero_optimum_cost_avoids_division_by_zero() -> None:
    """最优成本为零时按冻结规则返回零或空比例。"""

    equal = greedy_exact_metrics(
        _policy_evaluation(name="FlowState", cost=0.0, empty=10.0, benefit=10.0),
        _policy_evaluation(name="Exact OPT", cost=0.0, empty=10.0, benefit=10.0),
    )
    positive = greedy_exact_metrics(
        _policy_evaluation(name="FlowState", cost=2.0, empty=10.0, benefit=8.0),
        _policy_evaluation(name="Exact OPT", cost=0.0, empty=10.0, benefit=10.0),
    )
    no_benefit = greedy_exact_metrics(
        _policy_evaluation(name="FlowState", cost=10.0, empty=10.0, benefit=0.0),
        _policy_evaluation(name="Exact OPT", cost=10.0, empty=10.0, benefit=0.0),
    )
    assert equal.absolute_cost_gap_ms == 0.0
    assert equal.relative_cost_gap == 0.0
    assert positive.absolute_cost_gap_ms == 2.0
    assert positive.relative_cost_gap is None
    assert no_benefit.benefit_ratio is None


def test_budget_larger_than_candidate_count_is_supported() -> None:
    """K 大于候选数时所有方法仍需返回合法集合。"""

    snapshot = _snapshot(budget_k=8)
    result = evaluate_allocation_snapshot(snapshot)
    assert all(
        len(item.selected_checkpoint_ids) <= len(snapshot.eligible_candidates)
        for item in result.policy_results
    )
    exact = next(
        item for item in result.policy_results if item.policy_name == "Exact OPT"
    )
    assert exact.selector_internal_evaluations == 8


def test_zero_budget_selects_empty_set() -> None:
    """K=0 时五种方法都必须选择空集合并正常评分。"""

    result = evaluate_allocation_snapshot(_snapshot(budget_k=0))
    assert all(not item.selected_checkpoint_ids for item in result.policy_results)
    exact = next(
        item for item in result.policy_results if item.policy_name == "Exact OPT"
    )
    flowstate = next(
        item for item in result.policy_results if item.policy_name == "FlowState"
    )
    assert exact.selector_internal_evaluations == 1
    assert flowstate.selector_internal_evaluations == 1


@pytest.mark.parametrize(
    ("recurrent_resident", "fa_resident"),
    ((False, True), (True, False)),
)
def test_invalid_candidate_eligibility_fails_construction(
    recurrent_resident: bool,
    fa_resident: bool,
) -> None:
    """非正式 eligible 候选不得被静默过滤。"""

    candidates = _candidates()
    candidates[0] = replace(
        candidates[0],
        recurrent_resident=recurrent_resident,
        fa_resident=fa_resident,
    )
    with pytest.raises(ValueError, match="未驻留"):
        _snapshot(candidates=candidates)


def test_exact_tie_break_is_deterministic() -> None:
    """目标相同时必须稳定选择 checkpoint ID 字典序较小的集合。"""

    candidates = [
        CheckpointCandidate(
            checkpoint_id=checkpoint_id,
            workflow_id="W1",
            lineage_path=("W1",),
            token_pos=8192,
            memory_bytes=CHECKPOINT_BYTES,
        )
        for checkpoint_id in ("Z", "A")
    ]
    snapshot = _snapshot(
        candidates=candidates,
        creation={"A": 1, "Z": 2},
        access={"A": 1, "Z": 2},
        flop={"A": 8192.0, "Z": 8192.0},
    )
    observed = {
        select_exact_opt(snapshot).selected_checkpoint_ids for _ in range(10)
    }
    assert observed == {("A",)}


def test_future_information_and_metadata_mismatch_fail_construction() -> None:
    """未来信息或不完整 metadata 必须使 snapshot 构造明确失败。"""

    candidates = _candidates()
    common = {
        "allocation_epoch": 2,
        "snapshot_id": "invalid",
        "pending_continuations": _pending(),
        "eligible_candidates": candidates,
        "creation_order_by_checkpoint": {"A": 1, "B": 2, "C": 3},
        "last_access_order_by_checkpoint": {"A": 1, "B": 2, "C": 3},
        "marconi_flop_saved_by_checkpoint": {
            "A": 4096.0,
            "B": 4096.0,
            "C": 1024.0,
        },
        "access_frequency_by_checkpoint": {"A": 1, "B": 1, "C": 1},
        "frequency_observed_through_epoch": 2,
        "marconi_alpha": 1.0,
        "logical_budget_k": 1,
        "budget_bytes": CHECKPOINT_BYTES,
        "runtime_evidence": [
            FrozenCheckpointRuntimeEvidence(
                checkpoint_id=item.checkpoint_id,
                node_id=index,
                runtime_identity_digest=_digest("a"),
                checkpoint_handle_digest=_digest("b"),
            )
            for index, item in enumerate(candidates, start=1)
        ],
        "residency_snapshot_digest": _digest("c"),
    }
    with pytest.raises(ValueError, match="未来信息"):
        build_allocation_snapshot(
            **common,
            online_boundary=FrozenOnlineInformationBoundary(
                materialized_through_epoch=2,
                visible_continuation_ids=("P1", "P2"),
                future_latency_included=True,
            ),
        )
    broken = dict(common)
    broken["creation_order_by_checkpoint"] = {"A": 1, "B": 2}
    with pytest.raises(ValueError, match="不一致"):
        build_allocation_snapshot(
            **broken,
            online_boundary=FrozenOnlineInformationBoundary(
                materialized_through_epoch=2,
                visible_continuation_ids=("P1", "P2"),
            ),
        )


def test_same_frequency_metadata_gives_same_digest() -> None:
    """相同 frequency metadata（无论输入顺序）必须得到相同 digest。"""

    first = _snapshot(frequency={"A": 3, "B": 1, "C": 2})
    second = _snapshot(frequency={"C": 2, "B": 1, "A": 3})
    assert first.content_digest() == second.content_digest()
    serialized = json.loads(first.canonical_serialization())
    assert [
        (item["checkpoint_id"], item["access_frequency"])
        for item in serialized["lfu_access_frequency"]
    ] == [("A", 3), ("B", 1), ("C", 2)]
    assert serialized["frequency_observed_through_epoch"] == 2


def test_any_frequency_change_changes_digest() -> None:
    """修改任一 checkpoint frequency 或观测时点必须改变 digest。"""

    snapshot = _snapshot(frequency={"A": 3, "B": 1, "C": 2})
    original = snapshot.content_digest()
    changed_frequency = replace(
        snapshot,
        lfu_access_frequency=(
            replace(snapshot.lfu_access_frequency[0], access_frequency=4),
            *snapshot.lfu_access_frequency[1:],
        ),
    )
    changed_epoch = replace(snapshot, frequency_observed_through_epoch=1)
    assert changed_frequency.content_digest() != original
    assert changed_epoch.content_digest() != original


def test_frequency_metadata_is_deeply_immutable() -> None:
    """frequency metadata 必须深度不可变且与构造输入脱钩。"""

    frequency = {"A": 3, "B": 1, "C": 2}
    snapshot = _snapshot(frequency=frequency)
    original = snapshot.content_digest()
    frequency["A"] = 999
    assert snapshot.content_digest() == original
    with pytest.raises(FrozenInstanceError):
        snapshot.lfu_access_frequency[0].access_frequency = 7
    with pytest.raises(FrozenInstanceError):
        snapshot.frequency_observed_through_epoch = 1
    with pytest.raises(TypeError):
        snapshot.lfu_access_frequency[0] = snapshot.lfu_access_frequency[1]


def test_lfu_selector_preserves_snapshot_digest() -> None:
    """LFU 选择与公共评分前后 snapshot digest 必须不变。"""

    snapshot = _snapshot(frequency={"A": 3, "B": 1, "C": 2})
    result = evaluate_allocation_snapshot(snapshot)
    lfu = next(
        item for item in result.policy_results if item.policy_name == "LFU"
    )
    assert lfu.snapshot_digest_before == snapshot.content_digest()
    assert lfu.snapshot_digest_after == snapshot.content_digest()


def test_lfu_selection_independent_of_runtime_evidence() -> None:
    """LFU 选择不得依赖 runtime evidence，只消费冻结 metadata。"""

    frequency = {"A": 3, "B": 1, "C": 2}
    baseline = _snapshot(frequency=frequency)
    altered = _snapshot(
        frequency=frequency,
        evidence_node_offset=100,
        evidence_digest_char="f",
    )
    assert baseline.content_digest() != altered.content_digest()
    first = evaluate_allocation_snapshot(baseline)
    second = evaluate_allocation_snapshot(altered)
    first_lfu = next(
        item for item in first.policy_results if item.policy_name == "LFU"
    )
    second_lfu = next(
        item for item in second.policy_results if item.policy_name == "LFU"
    )
    assert first_lfu.selected_checkpoint_ids == second_lfu.selected_checkpoint_ids


def test_lfu_final_selection_scored_by_shared_objective() -> None:
    """LFU 最终集合必须经公共 C(S)/F(S) 评分且内部 objective 计数为零。"""

    calls: list[tuple[str, ...]] = []

    def tracking(
        snapshot: AllocationSnapshot,
        selected: list[str] | tuple[str, ...],
    ) -> ObjectiveEvaluation:
        calls.append(tuple(sorted(selected)))
        return evaluate_objective(snapshot, selected)

    snapshot = _snapshot(frequency={"A": 3, "B": 1, "C": 2})
    result = evaluate_allocation_snapshot(snapshot, objective_function=tracking)
    lfu = next(
        item for item in result.policy_results if item.policy_name == "LFU"
    )
    assert lfu.selected_checkpoint_ids == ("A",)
    assert tuple(sorted(lfu.selected_checkpoint_ids)) in calls
    assert lfu.selector_internal_evaluations == 0
    assert lfu.final_common_scoring_evaluations == 1
    direct = evaluate_objective(snapshot, lfu.selected_checkpoint_ids)
    assert lfu.total_recovery_cost_ms == direct.total_recovery_cost_ms
    assert lfu.total_benefit_ms == direct.total_benefit_ms
    assert lfu.per_continuation == direct.per_continuation


def test_lfu_prefers_higher_frequency() -> None:
    """高 frequency 候选必须优先于低 frequency 候选被保留。"""

    snapshot = _snapshot(
        budget_k=2,
        frequency={"A": 1, "B": 5, "C": 2},
    )
    result = evaluate_allocation_snapshot(snapshot)
    lfu = next(
        item for item in result.policy_results if item.policy_name == "LFU"
    )
    assert lfu.selected_checkpoint_ids == ("B", "C")


def test_lfu_frequency_tie_uses_frozen_deterministic_tiebreak() -> None:
    """frequency 并列时按最近访问降序、再按 checkpoint ID 升序稳定tie-break。"""

    snapshot = _snapshot(
        budget_k=2,
        frequency={"A": 4, "B": 4, "C": 4},
        access={"A": 1, "B": 3, "C": 2},
    )
    result = evaluate_allocation_snapshot(snapshot)
    lfu = next(
        item for item in result.policy_results if item.policy_name == "LFU"
    )
    assert lfu.selected_checkpoint_ids == ("B", "C")
    fully_tied = _snapshot(
        budget_k=1,
        frequency={"A": 4, "B": 4, "C": 4},
        access={"A": 7, "B": 7, "C": 7},
    )
    tied_result = evaluate_allocation_snapshot(fully_tied)
    tied_lfu = next(
        item for item in tied_result.policy_results if item.policy_name == "LFU"
    )
    assert tied_lfu.selected_checkpoint_ids == ("A",)
    observed = {
        select_lfu(
            fully_tied.core_candidates(),
            {"A": 4, "B": 4, "C": 4},
            {"A": 7, "B": 7, "C": 7},
            1,
        )
        for _ in range(10)
    }
    assert observed == {("A",)}


def test_lfu_repeated_runs_are_identical() -> None:
    """同一快照上重复运行 LFU 结果必须完全一致。"""

    snapshot = _snapshot(frequency={"A": 3, "B": 1, "C": 2})
    results = [evaluate_allocation_snapshot(snapshot) for _ in range(5)]
    selections = {
        next(
            item.selected_checkpoint_ids
            for item in result.policy_results
            if item.policy_name == "LFU"
        )
        for result in results
    }
    costs = {
        next(
            item.total_recovery_cost_ms
            for item in result.policy_results
            if item.policy_name == "LFU"
        )
        for result in results
    }
    assert selections == {("A",)}
    assert len(costs) == 1


def test_lfu_zero_budget_selects_empty() -> None:
    """K=0 时 LFU 必须选择空集合。"""

    snapshot = _snapshot(budget_k=0, frequency={"A": 3, "B": 1, "C": 2})
    result = evaluate_allocation_snapshot(snapshot)
    lfu = next(
        item for item in result.policy_results if item.policy_name == "LFU"
    )
    assert lfu.selected_checkpoint_ids == ()


def test_lfu_budget_above_candidate_count_selects_all() -> None:
    """K 大于候选数时 LFU 必须保留全部候选且不越界。"""

    snapshot = _snapshot(budget_k=8, frequency={"A": 3, "B": 1, "C": 2})
    result = evaluate_allocation_snapshot(snapshot)
    lfu = next(
        item for item in result.policy_results if item.policy_name == "LFU"
    )
    assert lfu.selected_checkpoint_ids == ("A", "B", "C")
    assert set(lfu.selected_checkpoint_ids) == {"A", "B", "C"}


def test_incomplete_or_invalid_frequency_metadata_fails() -> None:
    """frequency 缺项、多项、为零、为负或类型非法时必须明确失败。"""

    with pytest.raises(ValueError, match="不一致"):
        _snapshot(frequency={"A": 1, "B": 1})
    with pytest.raises(ValueError, match="不一致"):
        _snapshot(frequency={"A": 1, "B": 1, "C": 1, "D": 1})
    with pytest.raises(ValueError, match="至少为 1"):
        _snapshot(frequency={"A": 0, "B": 1, "C": 1})
    with pytest.raises(ValueError, match="非负整数"):
        _snapshot(frequency={"A": -1, "B": 1, "C": 1})
    with pytest.raises(ValueError, match="非负整数"):
        _snapshot(frequency={"A": True, "B": 1, "C": 1})
    with pytest.raises(ValueError, match="访问频率"):
        select_lfu(
            _snapshot().core_candidates(),
            {"A": 1},
            {"A": 1, "B": 1, "C": 1},
            1,
        )
    with pytest.raises(ValueError, match="最近访问"):
        select_lfu(
            _snapshot().core_candidates(),
            {"A": 1, "B": 1, "C": 1},
            {"A": 1},
            1,
        )


def test_future_derived_frequency_fails_boundary_validation() -> None:
    """观测时点越过 online boundary 的 frequency 必须被拒绝。"""

    with pytest.raises(ValueError, match="未来信息"):
        _snapshot(frequency_observed_epoch=3)
    boundary = FrozenOnlineInformationBoundary(
        materialized_through_epoch=1,
        visible_continuation_ids=("P1", "P2"),
    )
    candidates = _candidates()
    with pytest.raises(ValueError, match="物化时点"):
        build_allocation_snapshot(
            allocation_epoch=2,
            snapshot_id="future-frequency",
            pending_continuations=_pending(),
            eligible_candidates=candidates,
            creation_order_by_checkpoint={"A": 1, "B": 2, "C": 3},
            last_access_order_by_checkpoint={"A": 1, "B": 2, "C": 3},
            marconi_flop_saved_by_checkpoint={
                "A": 4096.0,
                "B": 4096.0,
                "C": 1024.0,
            },
            access_frequency_by_checkpoint={"A": 1, "B": 1, "C": 1},
            frequency_observed_through_epoch=2,
            marconi_alpha=CONTROLLED_MARCONI_ALPHA,
            logical_budget_k=1,
            budget_bytes=CHECKPOINT_BYTES,
            runtime_evidence=[
                FrozenCheckpointRuntimeEvidence(
                    checkpoint_id=item.checkpoint_id,
                    node_id=index,
                    runtime_identity_digest=_digest("a"),
                    checkpoint_handle_digest=_digest("b"),
                )
                for index, item in enumerate(candidates, start=1)
            ],
            residency_snapshot_digest=_digest("c"),
            online_boundary=boundary,
        )
