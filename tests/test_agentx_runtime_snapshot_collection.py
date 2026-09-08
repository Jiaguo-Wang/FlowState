"""Step 13G-B2 单元测试：AgentX neutral runtime snapshot collection。

纯 CPU；通过 mock runtime 验证 collection 流程、gate 判定与 artifact 输出。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.agentx_runtime_snapshot_collection import (
    CollectionAbort,
    FAILURE_REASONS,
    SnapshotResult,
    _SnapshotReplayPlan,
    _MaterializedRequest,
    _PendingRequest,
    _CheckpointObservation,
    _PendingObservation,
    _detect_census_anomaly,
    _wait_gpu_stable,
    _choose_gpu,
    _collect_snapshot,
    run_collection,
)


class _FakeRuntime:
    """Mock SGLang runtime adapter，维护极简 tree 状态。"""

    def __init__(
        self,
        *,
        fail_execute: Exception | None = None,
        frontier: int | None = None,
        checkpoint_not_resident: bool = False,
        fail_checkpoint: bool = False,
    ) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail_execute = fail_execute
        self.frontier = frontier  # None 表示返回完整长度
        self.checkpoint_not_resident = checkpoint_not_resident
        self.fail_checkpoint = fail_checkpoint
        self.executed = 0

    def _tree_state(self) -> dict[str, Any]:
        # root node id 0，seg_len 0。
        structure_rows: list[list[Any]] = [[0, None, [], 0]]
        mamba_rows: list[list[Any]] = []
        for i in range(1, self.executed + 1):
            structure_rows.append([i, 0, [], i * 64])
            mamba_rows.append([i, []])
        return {
            "node_count": len(structure_rows),
            "structure_rows": structure_rows,
            "structure_sha256": "fake",
            "mamba_rows": mamba_rows,
            "mamba_tree_sha256": "fake",
            "mamba_node_count": len(mamba_rows),
        }

    def census(self, nonce: str) -> dict[str, Any]:
        self.calls.append(("census", (nonce,)))
        return {"tree": self._tree_state()}

    def execute(self, request_id: str, token_ids: tuple[int, ...]) -> dict[str, Any]:
        self.calls.append(("execute", (request_id,)))
        if self.fail_execute is not None:
            raise self.fail_execute
        self.executed += 1
        return {"completion_tokens": 1, "num_retractions": 0}

    def query_runtime_metrics(self, request_id: str) -> dict[str, Any]:
        self.calls.append(("query_runtime_metrics", (request_id,)))
        return {
            "request_id": request_id,
            "physical_fa_hit": 0,
            "executable_prefix": 0,
        }

    def inspect_checkpoint(
        self, checkpoint_id: str, token_ids: tuple[int, ...]
    ) -> dict[str, Any]:
        self.calls.append(("inspect_checkpoint", (checkpoint_id,)))
        if self.fail_checkpoint:
            raise RuntimeError("radix segment boundary mismatch")
        resident = not self.checkpoint_not_resident
        return {
            "compact": {
                "node_id": 7,
                "token_pos": len(token_ids),
                "fa_resident": resident,
                "mamba_resident": resident,
            }
        }

    def inspect_checkpoint_recurrent(
        self,
        checkpoint_id: str,
        token_ids: Sequence[int],
        token_pos: int,
    ) -> dict[str, Any]:
        self.calls.append(("inspect_checkpoint_recurrent", (checkpoint_id,)))
        if self.fail_checkpoint:
            raise RuntimeError("radix segment boundary mismatch")
        resident = not self.checkpoint_not_resident
        return {
            "compact": {
                "node_id": 7,
                "token_pos": len(token_ids),
                "fa_resident": resident,
                "mamba_resident": resident,
            },
            "prefix_token_ids": tuple(token_ids),
            "resolution_source": "exact",
        }

    def inspect_fa_frontier(
        self, token_ids: tuple[int, ...], *, nonce: str
    ) -> dict[str, Any]:
        self.calls.append(("inspect_fa_frontier", (nonce,)))
        return {
            "resident_fa_frontier": (
                self.frontier if self.frontier is not None else len(token_ids)
            ),
            "traversed_node_ids": [],
        }

    def shutdown(self) -> None:
        self.calls.append(("shutdown", ()))


def _fake_plan(
    materialized: list[_MaterializedRequest] | None = None,
    pendings: list[_PendingRequest] | None = None,
) -> _SnapshotReplayPlan:
    materialized = materialized or []
    pendings = pendings or []
    return _SnapshotReplayPlan(
        trace_id="trace_0",
        block_size=64,
        t=1.0,
        pending_count=len(pendings),
        candidate_count=len(materialized),
        shared_candidate_count=0,
        max_degree=2,
        max_fork_depth_blocks=1,
        materialized=materialized,
        pendings=pendings,
        active_targets={},
        expected_candidate_ids={m.request_id for m in materialized},
        expected_pending_ids={p.request_id for p in pendings},
    )


def _mat(
    token_ids: list[int], *, rid: str = "r1", end: float = 1.0
) -> _MaterializedRequest:
    return _MaterializedRequest(
        conversation_id="c0",
        request_index=0,
        token_pos=len(token_ids),
        token_ids=token_ids,
        end_seconds=end,
        request_id=rid,
    )


def _pend(token_ids: list[int], *, rid: str = "p1") -> _PendingRequest:
    return _PendingRequest(
        conversation_id="c0",
        request_index=1,
        token_pos=len(token_ids),
        token_ids=token_ids,
        request_id=rid,
    )


def test_failure_reasons_are_known() -> None:
    """所有 CollectionAbort 使用的原因必须预定义。"""
    for reason in FAILURE_REASONS:
        CollectionAbort(reason, "test")
    with pytest.raises(ValueError):
        CollectionAbort("not_a_reason", "test")


def test_detect_census_anomaly_no_anomaly() -> None:
    """节点数量稳定时无异常。"""
    prev = {"tree": {"mamba_rows": [[1, []], [2, []]], "node_count": 10}}
    cur = {"tree": {"mamba_rows": [[1, []], [2, []], [3, []]], "node_count": 11}}
    anomalies = _detect_census_anomaly(prev, cur)
    assert not anomalies["native_mamba_eviction"]
    assert not anomalies["fa_cascade"]


def test_detect_census_anomaly_eviction() -> None:
    """mamba node 数量减少即判定 native eviction。"""
    prev = {"tree": {"mamba_rows": [[1, []], [2, []], [3, []]], "node_count": 10}}
    cur = {"tree": {"mamba_rows": [[1, []], [2, []]], "node_count": 10}}
    anomalies = _detect_census_anomaly(prev, cur)
    assert anomalies["native_mamba_eviction"]
    assert not anomalies["fa_cascade"]


def test_detect_census_anomaly_cascade() -> None:
    """tree_node_count 骤降判定 FA cascade。"""
    prev = {"tree": {"mamba_rows": [[1, []], [2, []], [3, []]], "node_count": 100}}
    cur = {"tree": {"mamba_rows": [[1, []], [2, []], [3, []]], "node_count": 80}}
    anomalies = _detect_census_anomaly(prev, cur)
    assert not anomalies["native_mamba_eviction"]
    assert anomalies["fa_cascade"]


def test_wait_gpu_stable_clean(monkeypatch: Any) -> None:
    """GPU 干净时 wait_gpu_stable 立即返回。"""
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._query_gpu_memory_used_mib",
        lambda idx: 100,
    )
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._query_gpu_compute_processes",
        lambda idx: [],
    )
    res = _wait_gpu_stable(0)
    assert res["stable"] is True
    assert res["final_used_mib"] == 100


def test_wait_gpu_stable_timeout(monkeypatch: Any) -> None:
    """GPU 持续不干净时返回 stable=False。"""
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._query_gpu_memory_used_mib",
        lambda idx: 100_000,
    )
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._query_gpu_compute_processes",
        lambda idx: [{"pid": 123, "used_mib": 100_000}],
    )
    # 缩短超时以加速测试。
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection.GPU_STABLE_TIMEOUT_S", 0.01
    )
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection.GPU_STABLE_INTERVAL_S", 0.001
    )
    res = _wait_gpu_stable(0)
    assert res["stable"] is False


def test_choose_gpu(monkeypatch: Any) -> None:
    """选择显存使用最小的 GPU。"""
    usage = {0: 8000, 1: 2000, 2: 5000}

    def fake_used(idx: int) -> int:
        if idx not in usage:
            raise RuntimeError("no gpu")
        return usage[idx]

    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._query_gpu_memory_used_mib",
        fake_used,
    )
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._query_gpu_compute_processes",
        lambda idx: [],
    )
    assert _choose_gpu() == 1


def test_collect_snapshot_eligible(monkeypatch: Any) -> None:
    """正常路径返回 ELIGIBLE 并记录 checkpoint / pending 观测。"""
    fake = _FakeRuntime()
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._start_runtime",
        lambda gpu=0: fake,
    )
    plan = _fake_plan(
        materialized=[_mat([10] * 64, rid="r0"), _mat([20] * 128, rid="r1")],
        pendings=[_pend([30] * 96, rid="p0")],
    )
    result = _collect_snapshot(plan, gpu_index=0)
    assert result.status == "ELIGIBLE"
    assert len(result.checkpoints) == 2
    assert len(result.pendings) == 1
    assert result.pendings[0].resident_fa_frontier == 96
    assert result.checkpoints[0].fa_resident is True
    assert result.checkpoints[0].recurrent_resident is True
    assert any(c[0] == "shutdown" for c in fake.calls)


def test_collect_snapshot_checkpoint_inspect_failure(monkeypatch: Any) -> None:
    """inspect_checkpoint_recurrent 失败时标记 INFRASTRUCTURE_FAILED，不再回退到执行证据。"""
    fake = _FakeRuntime(fail_checkpoint=True)
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._start_runtime",
        lambda gpu=0: fake,
    )
    plan = _fake_plan(materialized=[_mat([10] * 64)])
    result = _collect_snapshot(plan, gpu_index=0)
    assert result.status == "INFRASTRUCTURE_FAILED(checkpoint_inspect_failed)"


def test_collect_snapshot_b1_universe_mismatch(monkeypatch: Any) -> None:
    """replay 实际执行的 request 集合与 B1 universe 不一致时标记 BUILD_FAILED。"""
    fake = _FakeRuntime()
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._start_runtime",
        lambda gpu=0: fake,
    )
    plan = _fake_plan(materialized=[_mat([10] * 64, rid="expected")])
    # _FakeRuntime.execute 把 request_id 原样返回在调用记录中，因此实际执行的是 expected。
    # 这里人为把 plan 的期望集合改成另一个 id，触发 mismatch。
    plan.expected_candidate_ids = {"other"}
    result = _collect_snapshot(plan, gpu_index=0)
    assert result.status == "BUILD_FAILED(b1_universe_mismatch)"
    assert result.primary_reason == "b1_universe_mismatch"


def test_collect_snapshot_fa_cascade(monkeypatch: Any) -> None:
    """num_retractions > 0 触发 FA_KV_CASCADE 基础设施失败。"""

    class FakeWithCascade(_FakeRuntime):
        def execute(self, request_id: str, token_ids: tuple[int, ...]) -> dict[str, Any]:
            return {"completion_tokens": 1, "num_retractions": 1}

    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._start_runtime",
        lambda gpu=0: FakeWithCascade(),
    )
    plan = _fake_plan(materialized=[_mat([10] * 64)])
    result = _collect_snapshot(plan, gpu_index=0)
    assert result.status == "INFRASTRUCTURE_FAILED(fa_kv_cascade)"
    assert result.primary_reason == "fa_kv_cascade"


def test_collect_snapshot_request_failed(monkeypatch: Any) -> None:
    """execute 异常应产生 BUILD_FAILED(request_failed)。"""
    fake = _FakeRuntime(fail_execute=RuntimeError("engine boom"))
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._start_runtime",
        lambda gpu=0: fake,
    )
    plan = _fake_plan(materialized=[_mat([10] * 64)])
    result = _collect_snapshot(plan, gpu_index=0)
    assert result.status == "BUILD_FAILED(request_failed)"
    assert result.primary_reason == "request_failed"


def test_collect_snapshot_checkpoint_not_resident(monkeypatch: Any) -> None:
    """checkpoint 显式未 resident 时标记基础设施失败。"""
    fake = _FakeRuntime(checkpoint_not_resident=True)
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._start_runtime",
        lambda gpu=0: fake,
    )
    plan = _fake_plan(materialized=[_mat([10] * 64)])
    result = _collect_snapshot(plan, gpu_index=0)
    assert result.status == "INFRASTRUCTURE_FAILED(checkpoint_not_resident_at_barrier)"
    assert result.primary_reason == "checkpoint_not_resident_at_barrier"


def test_collect_snapshot_context_truncation(monkeypatch: Any) -> None:
    """输入超过 RUNTIME_CONTEXT_LIMIT 时标记 BUILD_FAILED(context_truncation)。"""
    from evaluation.agentx_qwen_replay_preflight import RUNTIME_CONTEXT_LIMIT

    fake = _FakeRuntime()
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._start_runtime",
        lambda gpu=0: fake,
    )
    plan = _fake_plan(
        materialized=[_mat([0] * (RUNTIME_CONTEXT_LIMIT + 1), rid="toolong")]
    )
    result = _collect_snapshot(plan, gpu_index=0)
    assert result.status == "BUILD_FAILED(context_truncation)"
    assert result.primary_reason == "context_truncation"


def test_run_collection_dry_run(monkeypatch: Any, tmp_path: Path) -> None:
    """dry_run 模式可生成正式 artifact 且 status 标记为 BUILD_FAILED(dry_run)。"""
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._load_tokenizer",
        lambda path: object(),
    )
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._load_formal_population",
        lambda path: [{"trace_id": "t0", "t": 1.0}],
    )
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._build_conversations_for_population",
        lambda path, pop: [type("TC", (), {"trace_id": "t0", "conversations": []})],
    )
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._build_replay_plan",
        lambda tc, entry, tok: _fake_plan(
            materialized=[_mat([1] * 64)], pendings=[_pend([2] * 64)]
        ),
    )
    monkeypatch.setattr(
        "evaluation.agentx_runtime_snapshot_collection._choose_gpu", lambda: 0
    )

    root = run_collection(tmp_path / "b2", dry_run=True)
    assert root.exists()
    assert (root / "manifest.json").exists()
    assert (root / "eligibility.json").exists()
    assert (root / "runtime_snapshots.jsonl").exists()
    assert (root / "runtime_correctness.json").exists()
    assert (root / "gpu_lifecycle.json").exists()
    assert (root / "final_report.md").exists()

    correctness = json.loads((root / "runtime_correctness.json").read_text())
    assert correctness["attempted"] == 1
    assert correctness["eligible"] == 0
    assert correctness["build_failed"] == 1
    assert correctness["infrastructure_failed"] == 0
    assert "dry_run" in correctness["ineligible_reasons"]

    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["status"] == "BUILD_FAILED(dry_run)"

    eligibility = json.loads((root / "eligibility.json").read_text())
    assert eligibility[0]["status"] == "BUILD_FAILED(dry_run)"

    # per-snapshot artifact 也应被写入。
    snapshot_dir = root / "per_snapshot"
    assert snapshot_dir.is_dir()
    subdirs = [d for d in snapshot_dir.iterdir() if d.is_dir()]
    assert len(subdirs) == 1
    assert (subdirs[0] / "snapshot.json").exists()
