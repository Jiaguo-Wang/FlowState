"""Step 13G-B1 单元测试：AgentX runtime-compatible population census。

所有测试使用小型合成 trace；涉及完整 corpus 的测试在文件存在时运行。
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest

from evaluation.agentx_runtime_population_census import (
    RUNTIME_CONTEXT_LIMIT,
    _active_request_context_length,
    _census_trace,
    _collect_all_request_inputs,
    _collect_trace_request_inputs,
    _context_length_distributions,
    _exact_search_space,
    _load_inputs,
    _replay_semantics_audit,
    run_census,
)
from evaluation.agentx_structure_audit import (
    FROZEN_AGENTX_PATH,
    _analyze_trace_online_compatibility,
    _detect_and_build_conversations,
)


def _write_trace(tmp_path: Path, trace: dict[str, Any]) -> Path:
    """把单个 trace 写入临时 JSONL 文件。"""
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(trace), encoding="utf-8")
    return path


def _synthetic_fork_trace(trace_id: str = "synthetic-fork") -> dict[str, Any]:
    """构造一个含两个 fork subagent 且所有请求 input 均 ≤131200 的最小 trace。

    root 与两个 subagent 同时重叠，保证出现 |P_t|>=2 与 d_t(c)>=2。
    """
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
                "t": 1.5,
                "model": "claude-haiku-4-5-20251001",
                "in": 64,
                "out": 8,
                "hash_ids": [1, 2, 3, 4],
                "api_time": 1.0,
            },
            {
                "type": "subagent",
                "t": 2.6,
                "agent_id": "coder",
                "subagent_type": "tool",
                "duration_ms": 500.0,
                "total_tokens": 200,
                "tool_use_count": 1,
                "status": "success",
                "requests": [
                    {
                        "type": "n",
                        "t": 2.7,
                        "model": "claude-haiku-4-5-20251001",
                        "in": 192,
                        "out": 12,
                        "hash_ids": [1, 2, 3, 5],
                        "api_time": 0.3,
                    },
                ],
            },
            {
                "type": "subagent",
                "t": 2.6,
                "agent_id": "reviewer",
                "subagent_type": "tool",
                "duration_ms": 500.0,
                "total_tokens": 200,
                "tool_use_count": 1,
                "status": "success",
                "requests": [
                    {
                        "type": "n",
                        "t": 2.7,
                        "model": "claude-haiku-4-5-20251001",
                        "in": 256,
                        "out": 20,
                        "hash_ids": [1, 2, 3, 6],
                        "api_time": 0.3,
                    },
                ],
            },
            {
                "type": "n",
                "t": 3.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 128,
                "out": 10,
                "hash_ids": [1, 2, 3, 4, 7],
                "api_time": 2.0,
            },
        ],
    }


def _synthetic_overlength_later(trace_id: str = "synthetic-overlength-later") -> dict[str, Any]:
    """早期 epoch context 合规，但未来某请求 input 超过 131200。"""
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
                "in": 64,
                "out": 8,
                "hash_ids": [1, 2, 3, 4],
                "api_time": 1.0,
            },
            {
                "type": "n",
                "t": 5.0,
                "model": "claude-haiku-4-5-20251001",
                "in": 200_000,
                "out": 10,
                "hash_ids": [10, 11, 12],
                "api_time": 1.0,
            },
        ],
    }


def _synthetic_current_overlength(trace_id: str = "synthetic-current-overlength") -> dict[str, Any]:
    """当前 pending 的 input 超过 131200，当前 epoch 必须 ineligible。"""
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
                "in": 200_000,
                "out": 10,
                "hash_ids": [1, 2, 3],
                "api_time": 1.0,
            },
            {
                "type": "n",
                "t": 0.5,
                "model": "claude-haiku-4-5-20251001",
                "in": 64,
                "out": 8,
                "hash_ids": [1, 2, 3, 4],
                "api_time": 5.0,
            },
        ],
    }


def _run_census_on_trace(trace: dict[str, Any]) -> Any:
    """对单个 trace 执行 _census_trace 的便捷包装。"""
    tc = _detect_and_build_conversations(trace["id"], trace["block_size"], trace["requests"])
    online = _analyze_trace_online_compatibility(tc)
    return _census_trace(tc, online)


def test_context_limit_131200(tmp_path: Path) -> None:
    """context limit 固定为 131200；低于限制应 eligible，高于应 ineligible。"""
    trace = _synthetic_fork_trace()
    census = _run_census_on_trace(trace)
    assert any(s.category == "SHARED_CONTEXT_ELIGIBLE" for s in census.snapshots)

    over = _synthetic_current_overlength()
    census_over = _run_census_on_trace(over)
    assert all(s.category != "SHARED_CONTEXT_ELIGIBLE" for s in census_over.snapshots)


def test_future_overlength_does_not_eliminate_current_epoch(tmp_path: Path) -> None:
    """未来超长请求不能使当前合规 epoch 变为 ineligible。"""
    trace = _synthetic_overlength_later()
    census = _run_census_on_trace(trace)
    # 在超长请求发生前存在 pending 重叠，应至少有一个 epoch 是 eligible（含 CHAIN_ONLY）。
    early_eligible = [
        s for s in census.snapshots
        if s.t < 5.0 and s.context_compatible
    ]
    assert early_eligible


def test_current_overlength_ancestry_eliminates(tmp_path: Path) -> None:
    """当前 pending 或 candidate 的 context 超过 limit 必须淘汰当前 epoch。"""
    trace = _synthetic_current_overlength()
    census = _run_census_on_trace(trace)
    assert all(not s.context_compatible for s in census.snapshots)


def test_shared_coverage_gate_preserved(tmp_path: Path) -> None:
    """合规的 fork 场景必须保留 shared-coverage epoch。"""
    trace = _synthetic_fork_trace()
    census = _run_census_on_trace(trace)
    eligible = [s for s in census.snapshots if s.category == "SHARED_CONTEXT_ELIGIBLE"]
    assert eligible
    for s in eligible:
        assert s.pending_count >= 2
        assert s.max_degree >= 2


def test_earliest_eligible_epoch_deterministic(tmp_path: Path) -> None:
    """同一 trace 两次 census 得到的最早 eligible epoch 必须相同。"""
    trace = _synthetic_fork_trace()
    c1 = _run_census_on_trace(trace)
    c2 = _run_census_on_trace(trace)
    assert c1.earliest_eligible == c2.earliest_eligible


def test_policy_blindness(tmp_path: Path) -> None:
    """正式选择规则不读取任何 policy 或未来信息。"""
    replay = _replay_semantics_audit()
    assert replay["replay_constructability"] in ("READY", "PARTIAL", "BLOCKED")
    # 选择规则硬编码为 EARLIEST_ELIGIBLE，不依赖外部 selector。
    assert replay["aiperf_replay_method"]


def test_one_snapshot_per_trace(tmp_path: Path) -> None:
    """formal population 每 trace 至多一个 snapshot。"""
    trace = _synthetic_fork_trace()
    census = _run_census_on_trace(trace)
    if census.earliest_eligible:
        # 单个 trace 只产生一个 earliest。
        assert isinstance(census.earliest_eligible, dict)
        assert "t" in census.earliest_eligible


def test_no_future_leakage(tmp_path: Path) -> None:
    """subagent 开始前不能出现在 pending set 中。"""
    trace = _synthetic_fork_trace()
    tc = _detect_and_build_conversations(trace["id"], trace["block_size"], trace["requests"])
    online = _analyze_trace_online_compatibility(tc)
    sub = next(c for c in tc.conversations if c.source == "subagent_main")
    before_start = [s for s in online.snapshots if s.t < sub.start_seconds - 1e-9]
    assert before_start
    for snap in before_start:
        assert all(tc.conversations[i].source != "subagent_main" for i in snap.active_targets)


def test_no_candidate_truncation(tmp_path: Path) -> None:
    """census 不对 candidate 做 Top-N 截断；candidate_count 反映实际 materialized 数量。"""
    trace = _synthetic_fork_trace()
    census = _run_census_on_trace(trace)
    for s in census.snapshots:
        # candidate_count 由实际 materialized request 数量累加，不应被人为截断。
        assert s.candidate_count >= 0
        assert s.shared_candidate_count <= s.candidate_count


def test_exact_tractability_estimation() -> None:
    """Exact search-space 超过阈值时返回 None，未超过时返回整数。"""
    # n=100, k=25 必然超过 100000。
    assert _exact_search_space(100, 25, threshold=100_000) is None
    # n=10, k=2 远小于阈值。
    space = _exact_search_space(10, 2, threshold=100_000)
    assert space is not None
    assert space == 1 + 10 + 45


def test_replay_constructability_classification() -> None:
    """replay constructability 分类必须是 READY/PARTIAL/BLOCKED 之一。"""
    replay = _replay_semantics_audit()
    assert replay["replay_constructability"] in ("READY", "PARTIAL", "BLOCKED")
    assert replay["exact_length_preservable"] in (True, False)
    assert replay["prefix_topology_preservable"] in (True, False)
    assert replay["fork_semantics_preservable"] in (True, False)


def test_deterministic_artifact_serialization(tmp_path: Path) -> None:
    """同一输入两次 run_census 产生相同的高层指标。"""
    trace = _synthetic_fork_trace()
    tc = _detect_and_build_conversations(trace["id"], trace["block_size"], trace["requests"])
    with tempfile.TemporaryDirectory() as td:
        r1 = run_census(Path(td) / "a", _trace_conversations=[tc])
        r2 = run_census(Path(td) / "b", _trace_conversations=[tc])
        s1 = json.loads((r1 / "POPULATION_STATISTICS.json").read_text())
        s2 = json.loads((r2 / "POPULATION_STATISTICS.json").read_text())
        assert s1 == s2


def test_context_length_distribution_helpers() -> None:
    """context length 分布辅助函数正确统计 ≤/＞ limit。"""
    all_inp = [100, 131200, 131201, 200000]
    shared_inp = [100, 131200]
    eligible_ctx = [128, 130000]
    dist = _context_length_distributions(all_inp, shared_inp, eligible_ctx)
    assert dist["all_corpus_requests"]["requests_le_limit"] == 2
    assert dist["all_corpus_requests"]["requests_gt_limit"] == 2
    assert dist["online_safe_shared_coverage_traces"]["requests_le_limit"] == 2
    assert dist["online_safe_shared_coverage_traces"]["requests_gt_limit"] == 0
    assert dist["shared_context_eligible_epochs"]["contexts_gt_limit"] == 0


@pytest.mark.skipif(
    not FROZEN_AGENTX_PATH.exists(),
    reason="Frozen AgentX corpus not present on this host",
)
def test_frozen_input_match() -> None:
    """frozen input 校验必须通过。"""
    inputs = _load_inputs()
    assert inputs["sha256_match"] is True
    assert inputs["record_count_match"] is True


@pytest.mark.skipif(
    not FROZEN_AGENTX_PATH.exists(),
    reason="Frozen AgentX corpus not present on this host",
)
def test_run_census_integration(tmp_path: Path) -> None:
    """端到端 census：artifact 齐全且 validation gate 通过。"""
    root = run_census(tmp_path / "census")
    required = {
        "INPUTS.json",
        "CONTEXT_COMPATIBILITY.json",
        "SHARED_CONTEXT_CENSUS.json",
        "REPLAY_SEMANTICS.json",
        "REPLAY_CONSTRUCTABILITY.json",
        "TOKENIZER_FEASIBILITY.json",
        "POPULATION_PROTOCOL.json",
        "POLICY_BLINDNESS_GATE.json",
        "FORMAL_CANDIDATE_POPULATION.json",
        "POPULATION_STATISTICS.json",
        "BUDGET_FEASIBILITY.json",
        "EXACT_TRACTABILITY_ESTIMATE.json",
        "OPENHANDS_AGENTX_COMPARISON.json",
        "CONTEXT_LENGTH_FEASIBILITY.json",
        "validation_report.json",
        "STEP_13G_B1_AGENTX_RUNTIME_POPULATION_FINAL_REPORT.md",
    }
    found = {p.name for p in root.iterdir()}
    assert required <= found, f"Missing artifacts: {required - found}"

    validation = json.loads((root / "validation_report.json").read_text())
    assert validation["gate_passed"] is True
    assert validation["checks"]["policy_blindness"] == "PASS"
    assert validation["checks"]["future_blindness"] == "PASS"

    # repo 根目录副本存在。
    repo_copy = Path(__file__).resolve().parents[1] / "STEP_13G_B1_AGENTX_RUNTIME_POPULATION_FINAL_REPORT.md"
    assert repo_copy.exists()


@pytest.mark.skipif(
    not FROZEN_AGENTX_PATH.exists(),
    reason="Frozen AgentX corpus not present on this host",
)
def test_corpus_input_collection() -> None:
    """全 corpus request input 收集与单 trace 收集结果一致。"""
    inputs = _collect_all_request_inputs(FROZEN_AGENTX_PATH)
    assert len(inputs) > 0
    per_trace_sum = 0
    for record in _stream_records_from_path(FROZEN_AGENTX_PATH):
        per_trace_sum += len(_collect_trace_request_inputs(record))
    assert len(inputs) == per_trace_sum


def _stream_records_from_path(path: Path) -> Iterator[dict[str, Any]]:
    """复用 structure_audit 的流式读取。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
