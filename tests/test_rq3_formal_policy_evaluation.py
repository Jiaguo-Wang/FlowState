"""验证 Step 13F Formal RQ3 Same-Snapshot Policy Evaluation runner。

覆盖 Section 27 要求的 17 项 validation gate：
1. budget ratio → K 映射规则；2. trivial K 排除；3. 重复 K 折叠；
4. Exact OPT 搜索空间计算；5. 本地 snapshot loader digest 校验；
6. snapshot 排序；7. budget variant 不修改原对象；8. policy result 字段完整性；
9. FlowState greedy 机制复现一致性；10. 运行后原 snapshot digest 不变；
11. Exact OPT 阈值门禁；12. FlowState 不低于 Exact OPT；
13. normalized cost 范围；14. paired reduction 定义正确性；
15. bootstrap CI 基本性质；16. determinism rerun；17. source digest 变化检测。
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.rq3_formal_policy_evaluation import (
    _bootstrap_ci,
    _load_allocation_snapshot,
    _run_determinism_on_snapshots,
    aggregate_results,
    compute_budget_ks,
    compute_source_digest,
    create_budget_variant,
    evaluate_snapshot_at_ks,
    load_eligible_snapshots,
    search_space_size,
)
from evaluation.rq3_frozen_snapshot_evaluator import (
    AllocationSnapshot,
    build_allocation_snapshot,
    evaluate_objective,
)
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


CHECKPOINT_BYTES = 1024


def _digest(character: str) -> str:
    """构造测试使用的稳定 SHA-256 形状摘要（十六进制字符）。"""

    hex_char = hex(ord(character) % 16)[2:]
    return hex_char * 64


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


def _candidates(n: int = 3) -> list[CheckpointCandidate]:
    """构造 n 个全部满足正式 eligibility 的候选。"""

    candidates = [
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
    if n <= 3:
        return candidates[:n]

    # 扩展候选到 n 个，保持 lineage 兼容
    extra: list[CheckpointCandidate] = []
    for idx in range(n - 3):
        workflow = "W1" if idx % 2 == 0 else "W2"
        extra.append(
            CheckpointCandidate(
                checkpoint_id=f"X{idx:02d}",
                workflow_id=workflow,
                lineage_path=(workflow,),
                token_pos=512 + idx * 64,
                memory_bytes=CHECKPOINT_BYTES,
            )
        )
    return candidates + extra


def _snapshot(
    *,
    budget_k: int = 1,
    candidate_count: int = 3,
) -> AllocationSnapshot:
    """用可变输入构造一个完整 canonical allocation snapshot。"""

    active_candidates = _candidates(candidate_count)
    checkpoint_ids = [item.checkpoint_id for item in active_candidates]
    creation = {
        checkpoint_id: index for index, checkpoint_id in enumerate(checkpoint_ids, start=1)
    }
    access = dict(creation)
    flop = {
        checkpoint_id: float(4096 if item.workflow_id == "W1" else 1024)
        for checkpoint_id, item in zip(checkpoint_ids, active_candidates)
    }
    frequency = {checkpoint_id: 1 for checkpoint_id in checkpoint_ids}
    from evaluation.rq3_frozen_snapshot_evaluator import (
        FrozenCheckpointRuntimeEvidence,
        FrozenOnlineInformationBoundary,
    )

    evidence = [
        FrozenCheckpointRuntimeEvidence(
            checkpoint_id=checkpoint_id,
            node_id=index,
            runtime_identity_digest=_digest("r"),
            checkpoint_handle_digest=_digest("h"),
        )
        for index, checkpoint_id in enumerate(sorted(checkpoint_ids), start=1)
    ]
    boundary = FrozenOnlineInformationBoundary(
        materialized_through_epoch=2,
        visible_continuation_ids=tuple(
            item.continuation_id for item in _pending()
        ),
    )
    return build_allocation_snapshot(
        allocation_epoch=2,
        snapshot_id=f"test-snapshot-c{candidate_count}-k{budget_k}",
        pending_continuations=_pending(),
        eligible_candidates=active_candidates,
        creation_order_by_checkpoint=creation,
        last_access_order_by_checkpoint=access,
        marconi_flop_saved_by_checkpoint=flop,
        access_frequency_by_checkpoint=frequency,
        frequency_observed_through_epoch=2,
        marconi_alpha=1.0,
        logical_budget_k=budget_k,
        budget_bytes=budget_k * CHECKPOINT_BYTES,
        runtime_evidence=evidence,
        residency_snapshot_digest=_digest("s"),
        online_boundary=boundary,
    )


@pytest.fixture(scope="module")
def formal_root() -> Path:
    """返回 Step 13E-F 正式 population root。"""

    return Path(
        "evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017"
    )


@pytest.fixture(scope="module")
def first_snapshot(formal_root: Path) -> AllocationSnapshot:
    """返回第一个 eligible snapshot 用于快速测试。"""

    snapshots = load_eligible_snapshots(formal_root)
    assert snapshots
    return snapshots[0]


# ---------------------------------------------------------------------------
# 1–4. Budget K 规则与 Exact OPT 搜索空间
# ---------------------------------------------------------------------------


def test_compute_budget_ks_floor_and_start_at_one() -> None:
    """K = max(1, floor(r * |C|))。"""

    ks = compute_budget_ks(10)
    assert ks == [(0.25, 2), (0.50, 5), (0.75, 7)]


def test_compute_budget_ks_skip_trivial_and_collapse() -> None:
    """K >= |C| 跳过；重复 K 折叠。"""

    # |C|=4：0.25→1, 0.50→2, 0.75→3（3 < 4 保留）
    assert compute_budget_ks(4) == [(0.25, 1), (0.50, 2), (0.75, 3)]
    # |C|=3：0.25→0→max1=1, 0.50→1, 0.75→2（2<3 保留），折叠重复 1
    assert compute_budget_ks(3) == [(0.25, 1), (0.75, 2)]
    # |C|=2：0.50→1, 0.75→1.5→1，且 0.75 的 K=2 == |C| 跳过；只剩一个 1
    assert compute_budget_ks(2) == [(0.25, 1)]


def test_search_space_size_matches_combinatorial_definition() -> None:
    """搜索空间 = sum_{i=0}^{K} C(|C|, i)。"""

    n, k = 10, 2
    expected = sum(math.comb(n, i) for i in range(k + 1))
    assert search_space_size(n, k) == expected
    assert search_space_size(20, 6) <= 100_000
    assert search_space_size(20, 7) > 100_000


# ---------------------------------------------------------------------------
# 5–6. Snapshot loader 与排序
# ---------------------------------------------------------------------------


def test_load_allocation_snapshot_digest_mismatch_raises(formal_root: Path, tmp_path: Path) -> None:
    """篡改 digest 后 loader 必须报错。"""

    snapshot_path = next(formal_root.glob("snapshots/g*.json"))
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["snapshot_digest"] = "0" * 64
    fake_path = tmp_path / "fake_test_snapshot.json"
    fake_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="digest 不一致"):
        _load_allocation_snapshot(fake_path)


def test_load_eligible_snapshots_sorted(formal_root: Path) -> None:
    """snapshots 按 group ordinal 升序。"""

    snapshots = load_eligible_snapshots(formal_root)
    ordinals = [
        int(s.snapshot_id.split("-")[-2].replace("g", ""))
        for s in snapshots
    ]
    assert ordinals == sorted(ordinals)
    assert len(snapshots) == 168


# ---------------------------------------------------------------------------
# 7. Budget variant 不可变性
# ---------------------------------------------------------------------------


def test_create_budget_variant_does_not_mutate_original() -> None:
    """variant 修改 logical_budget_k 与 budget_bytes，原 snapshot 不变。"""

    snapshot = _snapshot(budget_k=1, candidate_count=10)
    original_digest = snapshot.content_digest()
    variant = create_budget_variant(snapshot, 3)
    assert variant.logical_budget_k == 3
    assert variant.budget_bytes == 3 * CHECKPOINT_BYTES
    assert snapshot.logical_budget_k == 1
    assert snapshot.content_digest() == original_digest


# ---------------------------------------------------------------------------
# 8–10. Policy evaluation 结果结构、greedy 复现、digest 不变
# ---------------------------------------------------------------------------


def test_evaluate_snapshot_at_ks_result_structure(first_snapshot: AllocationSnapshot) -> None:
    """每个 snapshot×K 结果包含要求的 policy、exact_opt、paired、normalized 字段。"""

    results = evaluate_snapshot_at_ks(first_snapshot, 0)
    assert results
    for row in results:
        assert set(row["policies"].keys()) == {"LRU", "LFU", "Marconi", "FlowState"}
        assert "tractable" in row["exact_opt"]
        assert set(row["paired"].keys()) == {"LRU", "LFU", "Marconi"}
        assert set(row["normalized_cost"].keys()) == {"LRU", "LFU", "Marconi", "FlowState"}
        assert row["snapshot_digest_before"] == row["snapshot_digest_after"]
        assert "mechanism_diagnostics" in row


def test_flowstate_greedy_trace_matches_frozen_selector(first_snapshot: AllocationSnapshot) -> None:
    """机制复现产生的 final_selected 与 frozen FlowState selector 一致。"""

    from evaluation.rq3_formal_policy_evaluation import (
        _flowstate_greedy_trace,
        _select_policy,
    )

    for ratio, k in compute_budget_ks(len(first_snapshot.eligible_candidates)):
        variant = create_budget_variant(first_snapshot, k)
        frozen_selected, _ = _select_policy("FlowState", variant, evaluate_objective)
        trace = _flowstate_greedy_trace(variant, k)
        assert tuple(trace["final_selected"]) == frozen_selected


def test_snapshot_digest_immutable_after_evaluation(first_snapshot: AllocationSnapshot) -> None:
    """evaluate_snapshot_at_ks 不修改原 snapshot。"""

    original_digest = first_snapshot.content_digest()
    evaluate_snapshot_at_ks(first_snapshot, 0)
    assert first_snapshot.content_digest() == original_digest


# ---------------------------------------------------------------------------
# 11–14. Exact OPT 阈值、最优性、normalized cost、paired reduction
# ---------------------------------------------------------------------------


def test_exact_opt_threshold_respected_by_runner(first_snapshot: AllocationSnapshot) -> None:
    """搜索空间超过阈值时 exact_opt.tractable = False。"""

    results = evaluate_snapshot_at_ks(first_snapshot, 0, exact_threshold=10)
    for row in results:
        if search_space_size(row["candidate_count"], row["k"]) > 10:
            assert row["exact_opt"]["tractable"] is False
            assert row["exact_opt"]["reason"] == "search_space_exceeds_threshold"


def test_flowstate_cost_not_below_exact_on_tractable(formal_root: Path) -> None:
    """所有 tractable Exact OPT case 中 FlowState C(S) 不低于最优。"""

    snapshots = load_eligible_snapshots(formal_root)[:10]
    for snapshot in snapshots:
        for row in evaluate_snapshot_at_ks(snapshot, 0):
            exact = row["exact_opt"]
            if exact.get("tractable"):
                fs_cost = row["policies"]["FlowState"]["total_recovery_cost_ms"]
                assert fs_cost + 1e-6 >= exact["total_recovery_cost_ms"]
                assert exact["flowstate_vs_exact"]["absolute_cost_gap_ms"] >= 0.0


def test_normalized_cost_within_valid_range(first_snapshot: AllocationSnapshot) -> None:
    """normalized cost = C(S) / C(空集) ∈ [0, 1]（选择不会增加恢复成本）。"""

    for row in evaluate_snapshot_at_ks(first_snapshot, 0):
        for name in ("LRU", "LFU", "Marconi", "FlowState"):
            norm = row["normalized_cost"][name]
            assert norm is not None
            assert 0.0 <= norm <= 1.0 + 1e-9


def test_paired_reduction_consistent_with_costs(first_snapshot: AllocationSnapshot) -> None:
    """relative_reduction = (baseline - FlowState) / baseline。"""

    for row in evaluate_snapshot_at_ks(first_snapshot, 0):
        fs_cost = row["policies"]["FlowState"]["total_recovery_cost_ms"]
        for baseline in ("LRU", "LFU", "Marconi"):
            base_cost = row["policies"][baseline]["total_recovery_cost_ms"]
            expected = (
                (base_cost - fs_cost) / base_cost if base_cost > 1e-9 else None
            )
            assert abs(row["paired"][baseline]["absolute_difference_ms"] - (base_cost - fs_cost)) < 1e-9
            got = row["paired"][baseline]["relative_reduction"]
            if expected is None:
                assert got is None
            else:
                assert abs(got - expected) < 1e-9


# ---------------------------------------------------------------------------
# 15. Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_basic_properties() -> None:
    """有效数据返回 n、mean、ci_low、ci_high；空数据返回 None。"""

    reductions = [0.1, 0.15, 0.12, 0.18, None, 0.11]
    ci = _bootstrap_ci(reductions, n_iterations=1000, seed=42)
    assert ci["n"] == 5
    assert ci["mean"] is not None
    assert ci["ci_low"] is not None
    assert ci["ci_high"] is not None
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]

    empty_ci = _bootstrap_ci([])
    assert empty_ci["n"] == 0
    assert empty_ci["mean"] is None


# ---------------------------------------------------------------------------
# 16. 聚合与 determinism
# ---------------------------------------------------------------------------


def test_aggregate_results_keys(first_snapshot: AllocationSnapshot) -> None:
    """aggregate_results 包含三个 ratio 与 exact_opt 入口。"""

    results = evaluate_snapshot_at_ks(first_snapshot, 0)
    aggregate = aggregate_results(results)
    for ratio in (0.25, 0.50, 0.75):
        assert ratio in aggregate
        for policy in ("LRU", "LFU", "Marconi", "FlowState"):
            assert f"C_{policy}" in aggregate[ratio]
            assert f"C_hat_{policy}" in aggregate[ratio]
        for baseline in ("LRU", "LFU", "Marconi"):
            key = f"FlowState_vs_{baseline}"
            assert key in aggregate[ratio]
            assert "bootstrap_ci" in aggregate[ratio][key]
    assert "exact_opt" in aggregate


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_determinism_rerun_passes_on_subset(formal_root: Path) -> None:
    """对前 20 个 snapshot 做两次运行，selected sets 完全一致。"""

    snapshots = load_eligible_snapshots(formal_root)[:20]
    report = _run_determinism_on_snapshots(snapshots)
    assert report["pass"]
    assert report["mismatch_count"] == 0


# ---------------------------------------------------------------------------
# 17. Source digest
# ---------------------------------------------------------------------------


def test_compute_source_digest_detects_modification(tmp_path: Path) -> None:
    """源码 digest 在文件内容变化后必须变化。"""

    file_a = tmp_path / "a.py"
    file_b = tmp_path / "b.py"
    file_a.write_text("print(1)", encoding="utf-8")
    file_b.write_text("print(2)", encoding="utf-8")

    digest_before = compute_source_digest([file_a, file_b])
    file_a.write_text("print(3)", encoding="utf-8")
    digest_after = compute_source_digest([file_a, file_b])
    assert digest_before != digest_after

    digest_reverted = compute_source_digest([file_a, file_b])
    assert digest_after == digest_reverted
