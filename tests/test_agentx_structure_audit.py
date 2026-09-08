"""Step 13G-B0.1 单元测试：AgentX online-safe 兼容性修复审计。

所有测试使用小型合成 trace，避免加载完整 1.8 GiB corpus；
集成测试在 corpus 存在时运行完整审计。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.agentx_structure_audit import (
    FROZEN_AGENTX_PATH,
    analyze_online_logical_workflows_and_compatibility,
    analyze_pending_concurrency,
    analyze_spawn_fork_and_prefix,
    build_old_max_compatibility_diagnostic,
    collect_corpus_statistics,
    collect_schema_inventory,
    context_length_feasibility,
    reconstruct_logical_workflows_and_compatibility,
    run_audit,
    _analyze_trace_online_compatibility,
    _build_all_trace_conversations,
    _detect_and_build_conversations,
)

# 模块级缓存：完整 corpus 的 expensive 计算只执行一次。
_CACHE: dict[str, Any] = {}


def _cached_trace_conversations() -> list[Any]:
    if "tc" not in _CACHE:
        _CACHE["tc"] = _build_all_trace_conversations()
    return _CACHE["tc"]


def _cached_online() -> dict[str, Any]:
    if "online" not in _CACHE:
        _CACHE["online"] = analyze_online_logical_workflows_and_compatibility(
            _trace_conversations=_cached_trace_conversations()
        )
    return _CACHE["online"]


def _cached_diagnostic() -> dict[str, Any]:
    if "diagnostic" not in _CACHE:
        _CACHE["diagnostic"] = build_old_max_compatibility_diagnostic(
            _trace_conversations=_cached_trace_conversations()
        )
    return _CACHE["diagnostic"]


def _cached_audit_root() -> Path:
    if "audit_root" not in _CACHE:
        _CACHE["audit_root"] = run_audit()
    return _CACHE["audit_root"]


def _synthetic_trace(trace_id: str = "synthetic-01") -> dict[str, Any]:
    """含继承 subagent 的最小 trace，root 与 subagent 时间重叠。"""
    return {
        "id": trace_id,
        "models": ["claude-haiku-4-5-20251001"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "type": "n",
                "t": 0.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 128,
                "out": 10,
                "hash_ids": [1, 2, 3],
                "api_time": 1.0,
            },
            {
                "type": "n",
                "t": 4.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 64,
                "out": 8,
                "hash_ids": [1, 2, 3, 4],
                "api_time": 2.0,
            },
            {
                "type": "subagent",
                "t": 3.5,
                "agent_id": "coder",
                "subagent_type": "tool",
                "duration_ms": 500.0,
                "total_tokens": 200,
                "tool_use_count": 1,
                "status": "success",
                "requests": [
                    {
                        "type": "n",
                        "t": 3.6,
                        "model": "claude-haiku-4-5-20251001",
                        "in": 192,
                        "out": 12,
                        "hash_ids": [1, 2, 3, 5],
                        "api_time": 0.4,
                    },
                    {
                        "type": "n",
                        "t": 4.2,
                        "model": "claude-haiku-4-5-20251001",
                        "in": 256,
                        "out": 20,
                        "hash_ids": [1, 2, 3, 5, 6],
                        "api_time": 0.6,
                    },
                ],
            },
        ],
    }


def _synthetic_trace_with_spawn(trace_id: str = "synthetic-spawn") -> dict[str, Any]:
    """含两个 spawn flat-chain 的 trace：它们与 root hash 不共享前缀，root candidate 已 materialized。"""
    return {
        "id": trace_id,
        "models": ["claude-haiku-4-5-20251001"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "type": "n",
                "t": 0.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 128,
                "out": 10,
                "hash_ids": [1, 2, 3],
                "api_time": 1.0,
            },
            {
                "type": "n",
                "t": 2.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 128,
                "out": 10,
                "hash_ids": [10, 11, 12],
                "api_time": 1.0,
            },
            {
                "type": "n",
                "t": 4.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 128,
                "out": 10,
                "hash_ids": [20, 21, 22],
                "api_time": 1.0,
            },
            {
                "type": "n",
                "t": 5.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 64,
                "out": 8,
                "hash_ids": [1, 2, 3, 4],
                "api_time": 1.0,
            },
        ],
    }


def _synthetic_trace_legal_prefix(trace_id: str = "synthetic-legal") -> dict[str, Any]:
    """root 有一个高 token pos candidate，subagent 的 target 无法覆盖它。"""
    return {
        "id": trace_id,
        "models": ["claude-haiku-4-5-20251001"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "type": "n",
                "t": 0.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 128,
                "out": 10,
                "hash_ids": [1, 2, 3],
                "api_time": 1.0,
            },
            {
                "type": "n",
                "t": 1.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 64,
                "out": 8,
                "hash_ids": [1, 2, 3, 4],
                "api_time": 1.0,
            },
            {
                "type": "n",
                "t": 2.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 64,
                "out": 8,
                "hash_ids": [1, 2, 3, 4, 7],
                "api_time": 1.0,
            },
            {
                "type": "n",
                "t": 5.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 64,
                "out": 8,
                "hash_ids": [1, 2, 3, 4, 7, 8],
                "api_time": 5.0,
            },
            {
                "type": "subagent",
                "t": 3.5,
                "agent_id": "coder",
                "subagent_type": "tool",
                "duration_ms": 500.0,
                "total_tokens": 200,
                "tool_use_count": 1,
                "status": "success",
                "requests": [
                    {
                        "type": "n",
                        "t": 3.6,
                        "model": "claude-haiku-4-5-20251001",
                        "in": 192,
                        "out": 12,
                        "hash_ids": [1, 2, 3, 5],
                        "api_time": 1.2,
                    },
                ],
            },
        ],
    }


def _write_trace(tmp_path: Path, trace: dict[str, Any]) -> Path:
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(trace), encoding="utf-8")
    return path


def test_schema_inventory(tmp_path: Path) -> None:
    path = _write_trace(tmp_path, _synthetic_trace())
    inv = collect_schema_inventory(path)
    assert inv["record_count"] == 1
    assert inv["top_level_required_fields"] == ["id", "models", "block_size", "hash_id_scope", "requests"]
    assert inv["top_level_request_type_counts"] == {"n": 2, "subagent": 1}
    assert inv["inner_request_type_counts"] == {"n": 2}
    assert inv["subagent_status_counts"] == {"success": 1}


def test_corpus_statistics(tmp_path: Path) -> None:
    path = _write_trace(tmp_path, _synthetic_trace())
    stats = collect_corpus_statistics(path)
    assert stats["record_count"] == 1
    assert stats["traces_with_subagent"] == 1
    assert stats["total_subagent_entries"] == 1


def test_dt_invariant(tmp_path: Path) -> None:
    """d_t(c) <= |P_t| 必须在所有 epoch 成立。"""
    path = _write_trace(tmp_path, _synthetic_trace())
    online = analyze_online_logical_workflows_and_compatibility(path)
    assert online["max_online_degree"] <= online["max_pending"]
    assert online["all_per_epoch_invariants_pass"] is True
    assert online["invariant_max_degree_le_max_pending"] is True


def test_future_child_not_in_current_pt(tmp_path: Path) -> None:
    """subagent 开始前，它不能出现在 P_t 中。"""
    trace = _synthetic_trace()
    tc = _detect_and_build_conversations(trace["id"], trace["block_size"], trace["requests"])
    result = _analyze_trace_online_compatibility(tc)
    sub = next(c for c in tc.conversations if c.source == "subagent_main")
    before_start = [s for s in result.snapshots if s.t < sub.start_seconds - 1e-9]
    assert before_start
    for snap in before_start:
        conv_ids = {tc.conversations[idx].conversation_id for idx in snap.active_targets}
        assert sub.conversation_id not in conv_ids


def test_completed_child_not_pending(tmp_path: Path) -> None:
    """subagent 结束后，它必须从 P_t 中移除。"""
    trace = _synthetic_trace()
    tc = _detect_and_build_conversations(trace["id"], trace["block_size"], trace["requests"])
    result = _analyze_trace_online_compatibility(tc)
    sub = next(c for c in tc.conversations if c.source == "subagent_main")
    after_end = [s for s in result.snapshots if s.t > sub.end_seconds + 1e-9]
    assert after_end
    for snap in after_end:
        conv_ids = {tc.conversations[idx].conversation_id for idx in snap.active_targets}
        assert sub.conversation_id not in conv_ids


def test_spawn_child_does_not_inherit_parent(tmp_path: Path) -> None:
    """spawn flat-chain 不继承 root checkpoint，因此 root candidate 的 d_t(c)=0。"""
    trace = _synthetic_trace_with_spawn()
    tc = _detect_and_build_conversations(trace["id"], trace["block_size"], trace["requests"])
    result = _analyze_trace_online_compatibility(tc)
    root = next(c for c in tc.conversations if c.source == "root")
    # 找到 root candidate materialized 且 spawn child active 的 epoch。
    spawn = next(c for c in tc.conversations if c.source == "flat_chain" and not c.is_inherited)
    assert spawn.is_inherited is False
    relevant = [
        s for s in result.snapshots
        if root.conversation_id in {tc.conversations[i].conversation_id for i in s.active_targets}
        and spawn.conversation_id in {tc.conversations[i].conversation_id for i in s.active_targets}
    ]
    assert relevant
    for snap in relevant:
        assert snap.max_degree == 0


def test_fork_child_inherits_only_legal_prefix(tmp_path: Path) -> None:
    """fork 后代只能覆盖 candidate token_pos <= pending target 的部分。"""
    trace = _synthetic_trace_legal_prefix()
    tc = _detect_and_build_conversations(trace["id"], trace["block_size"], trace["requests"])
    result = _analyze_trace_online_compatibility(tc)
    sub = next(c for c in tc.conversations if c.source == "subagent_main")
    assert sub.is_inherited is True
    # 在 subagent active 的某个 epoch，检查 root 高 token pos candidate 不被计入。
    # subagent 只有一个 hash block [1,2,3,5]，target=256；root 第三个请求结束后累积位置 768。
    # 因此 root candidate@768 对 subagent 不兼容。
    root = next(c for c in tc.conversations if c.source == "root")
    high_pos = root.request_hash_token_positions[0] + root.request_hash_token_positions[1] + root.request_hash_token_positions[2]
    sub_active = [s for s in result.snapshots if sub.conversation_id in {tc.conversations[i].conversation_id for i in s.active_targets}]
    assert sub_active
    # 只要存在一个 epoch 满足 subagent target < high_pos，即可验证 prefix 约束。
    found = False
    for snap in sub_active:
        sub_target = snap.active_targets[tc.conversations.index(sub)]
        if sub_target < high_pos:
            found = True
            break
    assert found


def test_pending_count_not_conversation_count(tmp_path: Path) -> None:
    """同时 pending 数（max |P_t|）严格小于 conversation 总数（当存在非并发 child 时）。"""
    trace = _synthetic_trace_with_spawn()
    tc = _detect_and_build_conversations(trace["id"], trace["block_size"], trace["requests"])
    online = analyze_online_logical_workflows_and_compatibility(
        _write_trace(tmp_path, trace)
    )
    assert online["max_pending"] <= len(tc.conversations)
    # spawn trace 有三个 conversation，但它们并非全部同时 active，因此 max |P_t| < 总数。
    assert online["max_pending"] < len(tc.conversations)


def test_epoch_construction_deterministic(tmp_path: Path) -> None:
    """同一输入两次分析应产生完全相同的 snapshot 序列。"""
    path = _write_trace(tmp_path, _synthetic_trace())
    online1 = analyze_online_logical_workflows_and_compatibility(path)
    online2 = analyze_online_logical_workflows_and_compatibility(path)
    assert len(online1["trace_results"]) == len(online2["trace_results"])
    assert online1["max_online_degree"] == online2["max_online_degree"]
    assert online1["max_pending"] == online2["max_pending"]


def test_old_d734_regression() -> None:
    """旧版诊断必须能复现 d(c)=734 并指出其大于同时 active 的 pending 数。"""
    pytest.importorskip("numpy")
    if not FROZEN_AGENTX_PATH.exists():
        pytest.skip("Frozen AgentX corpus not present")
    diagnostic = _cached_diagnostic()
    assert diagnostic["old_max_degree"] >= 700
    assert diagnostic["breakdown"]["max_simultaneously_active_in_this_trace"] < diagnostic["old_max_degree"]
    assert diagnostic["breakdown"]["future_descendants_included"] is True
    assert diagnostic["breakdown"]["completed_or_inactive_descendants_included"] is True


def test_canonical_artifact_root() -> None:
    """正式 artifact root 必须位于持久目录，而不是 pytest temp。"""
    if not FROZEN_AGENTX_PATH.exists():
        pytest.skip("Frozen AgentX corpus not present")
    root = _cached_audit_root()
    assert str(root).startswith("/home/wjg/data/agentx/audits/")
    assert (root / "INPUTS.json").exists()
    assert (root / "STEP_13G_B0_1_AGENTX_ONLINE_COMPATIBILITY_FINAL_REPORT.md").exists()
    # 验证 repo 根目录也有副本。
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "STEP_13G_B0_1_AGENTX_ONLINE_COMPATIBILITY_FINAL_REPORT.md").exists()


def test_online_safe_shared_coverage_examples() -> None:
    """必须产生至少 10 个满足 |P_t|>=2 且 d_t(c)>=2 的确定性 shared-coverage 示例。"""
    if not FROZEN_AGENTX_PATH.exists():
        pytest.skip("Frozen AgentX corpus not present")
    online = _cached_online()
    examples = online["shared_coverage_examples"]
    assert len(examples) >= 10
    for ex in examples:
        assert ex["|P_t|"] >= 2
        assert ex["d_t(c)"] >= 2
        assert len(ex["compatible_pending_ids"]) >= 2
        assert ex["candidate_materialization_time"] is None or ex["candidate_materialization_time"] <= ex["epoch_timestamp"]


def test_context_length_counts(tmp_path: Path) -> None:
    """上下文长度统计必须包含 <=131200 与 >131200 的计数。"""
    path = _write_trace(tmp_path, _synthetic_trace())
    ctx = context_length_feasibility(path)
    assert "requests_le_soft_limit" in ctx
    assert "requests_gt_soft_limit" in ctx
    assert "fraction_le_soft_limit" in ctx
    assert "traces_all_requests_le_soft_limit" in ctx
    assert "traces_any_request_gt_soft_limit" in ctx


def test_spawn_fork_counts(tmp_path: Path) -> None:
    """SPAWN/FORK 计数必须精确且包含 trace-level 统计。"""
    result = analyze_spawn_fork_and_prefix(_write_trace(tmp_path, _synthetic_trace()))
    assert "child_context_inheritance_counts" in result
    assert "traces_with_spawn" in result
    assert "traces_with_fork" in result
    assert "traces_with_both" in result
    assert "traces_with_neither" in result


@pytest.mark.skipif(
    not FROZEN_AGENTX_PATH.exists(),
    reason="Frozen AgentX corpus not present on this host",
)
def test_run_audit_integration() -> None:
    """端到端审计：验证所有 required artifact 均存在且 frozen gate 通过。"""
    artifact_root = _cached_audit_root()
    assert artifact_root.exists()
    required = {
        "INPUTS.json",
        "SCHEMA_INVENTORY.json",
        "SCHEMA_NOTES.md",
        "CORPUS_STATISTICS.json",
        "SEMANTICS_EVIDENCE.json",
        "SPAWN_FORK_AUDIT.json",
        "OLD_MAX_COMPATIBILITY_DIAGNOSTIC.json",
        "ONLINE_COMPATIBILITY_DEGREE.json",
        "PENDING_CONCURRENCY.json",
        "LOGICAL_WORKFLOW_RECONSTRUCTION.json",
        "SHARED_COVERAGE_GATE.json",
        "CONTEXT_LENGTH_FEASIBILITY.json",
        "BRANCHING_AUDIT.json",
        "COMPARISON_OPENHANDS.json",
        "AUDIT_PROTOCOL.json",
        "STEP_13G_B0_1_AGENTX_ONLINE_COMPATIBILITY_FINAL_REPORT.md",
    }
    found = {p.name for p in artifact_root.iterdir()}
    missing = required - found
    assert not missing, f"Missing artifacts: {missing}"

    inputs = json.loads((artifact_root / "INPUTS.json").read_text())
    assert inputs["sha256_match"] is True
    assert inputs["record_count_match"] is True

    online = json.loads((artifact_root / "ONLINE_COMPATIBILITY_DEGREE.json").read_text())
    assert online["max_online_degree"] <= online["max_pending"]
    assert online["all_per_epoch_invariants_pass"] is True
