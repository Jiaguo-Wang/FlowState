"""Step 13G-C AgentX 正式 CPU 策略评估测试。"""

from __future__ import annotations

import inspect
from functools import lru_cache

import pytest

from evaluation.rq3_agentx_formal_policy_evaluation import (
    EXPECTED_CANDIDATES,
    EXPECTED_FORMAL_MAX_DEGREE,
    EXPECTED_SNAPSHOTS,
    _EXACT_OPT_SEARCH_SPACE_THRESHOLD,
    _FROZEN_BUDGET_RATIOS,
    _aggregate_agentx,
    build_agentx_formal_snapshots,
    compute_budget_ks,
    encode_usable_compatibility,
    evaluate_agentx_snapshots,
)
from evaluation.rq3_formal_policy_evaluation import (
    _EXACT_OPT_SEARCH_SPACE_THRESHOLD as OPENHANDS_EXACT_THRESHOLD,
)
from evaluation.rq3_formal_policy_evaluation import (
    _FROZEN_BUDGET_RATIOS as OPENHANDS_BUDGET_RATIOS,
)
from flowstate.state_catalog import CheckpointCandidate, is_compatible
from flowstate.workflow import PendingContinuation


@lru_cache(maxsize=1)
def _formal_inputs():
    """缓存真实冻结输入，避免重复扫描 corpus。"""
    return build_agentx_formal_snapshots()


def test_budget_and_exact_protocol_are_identical_to_openhands():
    assert _FROZEN_BUDGET_RATIOS == OPENHANDS_BUDGET_RATIOS == (0.25, 0.5, 0.75)
    assert _EXACT_OPT_SEARCH_SPACE_THRESHOLD == OPENHANDS_EXACT_THRESHOLD == 100_000
    assert compute_budget_ks(10) == [(0.25, 2), (0.5, 5), (0.75, 7)]


def test_compatibility_encoding_preserves_exact_usable_sets():
    compatible = {
        "c0": ("p0", "p1"),
        "c1": ("p0",),
        "c2": (),
    }
    depths = {"c0": 10, "c1": 20, "c2": 5}
    targets = {"p0": 30, "p1": 15}
    candidate_paths, candidate_workflows, pending_paths, pending_workflows, usable = (
        encode_usable_compatibility(compatible, depths, targets, "trace")
    )
    candidates = [
        CheckpointCandidate(
            checkpoint_id=identifier,
            workflow_id=candidate_workflows[identifier],
            lineage_path=candidate_paths[identifier],
            token_pos=depths[identifier],
            memory_bytes=1,
        )
        for identifier in compatible
    ]
    pendings = [
        PendingContinuation(
            continuation_id=identifier,
            workflow_id=pending_workflows[identifier],
            lineage_path=pending_paths[identifier],
            anchor_pos=targets[identifier],
            resident_fa_frontier=targets[identifier],
        )
        for identifier in targets
    ]
    observed = {
        candidate.checkpoint_id: tuple(
            sorted(
                pending.continuation_id
                for pending in pendings
                if is_compatible(candidate, pending)
            )
        )
        for candidate in candidates
    }
    assert observed == usable == {"c0": ("p0", "p1"), "c1": ("p0",), "c2": ()}


def test_non_laminar_compatibility_is_rejected():
    with pytest.raises(RuntimeError, match="不是层叠集合"):
        encode_usable_compatibility(
            {"c0": ("p0", "p1"), "c1": ("p1", "p2")},
            {"c0": 1, "c1": 1},
            {"p0": 2, "p1": 2, "p2": 2},
            "trace",
        )


def test_real_formal_snapshot_assembly_has_frozen_counts():
    snapshots, audit = _formal_inputs()
    assert len(snapshots) == EXPECTED_SNAPSHOTS
    assert audit["candidate_universe_total"] == EXPECTED_CANDIDATES
    assert audit["candidate_pending_universe_exact_match"] == "23/23"
    assert audit["formal_max_d_t_c"] == EXPECTED_FORMAL_MAX_DEGREE
    assert audit["b2_raw_compatibility_used"] is False
    assert sum(
        len(values) > 1
        for item in snapshots
        for values in item.formal_compatible_pending_ids.values()
    ) == 31


def test_builder_does_not_read_b2_raw_compatibility_field():
    source = inspect.getsource(build_agentx_formal_snapshots)
    assert 'row["compatible_pending_ids"]' not in source
    assert "raw_compatible_pending_ids" not in source


def test_small_real_snapshot_evaluates_three_budgets_without_mutation():
    snapshots, _audit = _formal_inputs()
    formal = next(item for item in snapshots if len(item.snapshot.eligible_candidates) == 6)
    digest = formal.snapshot.content_digest()
    rows = evaluate_agentx_snapshots([formal])
    assert len(rows) == 3
    assert {row["budget_ratio"] for row in rows} == set(_FROZEN_BUDGET_RATIOS)
    assert all(set(row["policies"]) == {"LRU", "LFU", "Marconi", "FlowState"} for row in rows)
    assert all(row["common_input_verified"] for row in rows)
    assert all(len(set(row["policy_input_digests"].values())) == 1 for row in rows)
    assert formal.snapshot.content_digest() == digest


def test_agentx_aggregate_contains_exact_shared_and_standalone_metrics():
    snapshots, _audit = _formal_inputs()
    formal = next(item for item in snapshots if len(item.snapshot.eligible_candidates) == 6)
    rows = evaluate_agentx_snapshots([formal])
    aggregate, exact, shared = _aggregate_agentx(rows, [formal])
    assert set(_FROZEN_BUDGET_RATIOS) <= set(aggregate)
    assert exact["tractable_cases"] == 3
    assert "flowstate_vs_standalone_topk" in shared
    assert "marginal_effect" in shared
