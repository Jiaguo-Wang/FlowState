from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.agentx_runtime_final_correctness_audit import (
    EMPTY_FULL_TREE_SHA256,
    EMPTY_MAMBA_TREE_SHA256,
    EMPTY_STRUCTURE_SHA256,
    FORBIDDEN_B2_ROOT,
    _classify_raw_extra,
    _collector_lifecycle_evidence,
    _snapshot_runtime_state_checks,
    logical_compatible_pending_ids,
    run_audit,
)
from evaluation.agentx_runtime_snapshot_collection import __file__ as collector_file


def _conversation(identifier, path, inherited=True):
    return SimpleNamespace(
        conversation_id=identifier,
        lineage_path=path,
        is_inherited=inherited,
        request_hash_token_positions=[100],
    )


def test_logical_compatibility_only_accepts_strict_inherited_descendants():
    conversations = [
        _conversation("root", ("root",), False),
        _conversation("child", ("root", "child"), True),
        _conversation("spawn", ("root", "spawn"), False),
        _conversation("other", ("other", "child"), True),
    ]
    actual = logical_compatible_pending_ids(
        0,
        50,
        {0: 100, 1: 100, 2: 100, 3: 100},
        conversations,
    )
    assert actual == ["child::pending0000"]


def test_logical_compatibility_honors_active_target():
    conversations = [
        _conversation("root", ("root",), False),
        _conversation("child", ("root", "child"), True),
    ]
    assert logical_compatible_pending_ids(0, 101, {1: 100}, conversations) == []
    assert logical_compatible_pending_ids(0, 50, {1: 0}, conversations) == []


def test_raw_extra_classification_distinguishes_invalid_relations():
    conversations = {
        "root": _conversation("root", ("root",), False),
        "fork": _conversation("fork", ("root", "fork"), True),
        "spawn": _conversation("spawn", ("root", "spawn"), False),
        "other": _conversation("other", ("other", "fork"), True),
    }
    candidate = {"conversation_id": "root", "lineage_path": ["root"]}
    assert _classify_raw_extra(candidate, "root::pending0000", conversations) == "same_conversation_self_match"
    assert _classify_raw_extra(candidate, "spawn::pending0000", conversations) == "spawn_or_noninherited_physical_prefix_match"
    assert _classify_raw_extra(candidate, "other::pending0000", conversations) == "nonancestor_physical_prefix_match"


def _clean_snapshot(trace_id="trace"):
    return {
        "census_rows": [
            {
                "nonce": f"agentx:{trace_id}:baseline",
                "accounting": {
                    "mamba_available": 192,
                    "mamba_schedulable_available": 192,
                    "mamba_evictable": 0,
                    "mamba_protected": 0,
                    "mamba_free_slots": list(range(1, 193)),
                    "full_allocator": {"available": 1000},
                },
                "scope": {
                    "scheduler_fully_idle": True,
                    "waiting_requests": 0,
                    "running_requests": 0,
                    "chunked_request_present": False,
                },
                "tree": {
                    "node_count": 1,
                    "mamba_node_count": 0,
                    "mamba_rows": [],
                    "structure_sha256": EMPTY_STRUCTURE_SHA256,
                    "full_tree_sha256": EMPTY_FULL_TREE_SHA256,
                    "mamba_tree_sha256": EMPTY_MAMBA_TREE_SHA256,
                },
            }
        ],
        "request_rows": [
            {
                "request_id": f"{trace_id}::req0000",
                "runtime_metrics": {
                    "physical_fa_hit": 0,
                    "executable_prefix": 0,
                    "replay_gap": 0,
                    "mamba_host_hit_length": 0,
                },
            }
        ],
        "checkpoints": [{"checkpoint_id": f"{trace_id}::req0000"}],
        "pendings": [{"continuation_id": f"{trace_id}::pending0000"}],
    }


def test_clean_baseline_proves_runtime_state_isolation():
    result = _snapshot_runtime_state_checks(_clean_snapshot(), "trace")
    assert result["runtime_state_isolation_pass"] is True
    assert result["recurrent_handles_or_residency_inherited"] is False
    assert result["radix_or_fa_cache_inherited"] is False


def test_inherited_radix_state_fails_isolation():
    snapshot = _clean_snapshot()
    snapshot["census_rows"][0]["tree"]["node_count"] = 2
    result = _snapshot_runtime_state_checks(snapshot, "trace")
    assert result["runtime_state_isolation_pass"] is False
    assert result["radix_or_fa_cache_inherited"] is True


def test_collector_lifecycle_has_fresh_constructor_and_shutdown():
    result = _collector_lifecycle_evidence(__import__("pathlib").Path(collector_file))
    assert result["collect_snapshot_calls_start_runtime"] is True
    assert result["collect_snapshot_calls_shutdown"] is True
    assert result["run_collection_calls_collect_snapshot"] is True
    assert result["start_runtime_constructs_engine"] is True


def test_invalid_formal_run_is_rejected_before_reading(tmp_path):
    with pytest.raises(RuntimeError, match="禁止使用"):
        run_audit(tmp_path / "output", b2_root=FORBIDDEN_B2_ROOT)
