"""验证 RQ6-E 批量控制的消息边界、只读视图与等价门禁。"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.rq6_batched_control import (
    BatchedControlClient,
    _handle_rows,
    _record_base,
    _source_manifest,
    _state_from_view,
)


def _candidate(checkpoint_id: str) -> SimpleNamespace:
    """构造最小候选对象。"""
    return SimpleNamespace(checkpoint_id=checkpoint_id)


def _handle(checkpoint_id: str, node_id: int) -> SimpleNamespace:
    """构造最小运行时句柄。"""
    return SimpleNamespace(
        checkpoint_id=checkpoint_id,
        token_ids=(node_id, node_id + 1),
        extra_key=None,
        expected_node_id=node_id,
        expected_prefix_digest=f"摘要-{checkpoint_id}",
    )


def test_handle_rows_preserve_candidate_order_and_identity():
    """批量请求必须按共同候选顺序携带完整精确身份。"""
    candidates = (_candidate("甲"), _candidate("乙"))
    handles = {"甲": _handle("甲", 3), "乙": _handle("乙", 7)}
    rows = _handle_rows(candidates, handles)
    assert [row["checkpoint_id"] for row in rows] == ["甲", "乙"]
    assert rows[1]["token_ids"] == [7, 8]
    assert rows[1]["expected_node_id"] == 7
    assert rows[1]["expected_prefix_digest"] == "摘要-乙"


def test_batched_client_uses_exactly_two_control_messages():
    """一个 epoch 的读取与协调必须各占一次同步 RPC。"""
    calls = []

    class Delegate:
        def _call(self, request):
            calls.append(request)
            return {"ok": True}

    candidates = (_candidate("甲"), _candidate("乙"))
    handles = {"甲": _handle("甲", 3), "乙": _handle("乙", 7)}
    client = BatchedControlClient(Delegate())
    client.introspect(nonce="读", candidates=candidates, handles=handles)
    client.reconcile(
        nonce="改", candidates=candidates, handles=handles,
        selected_ids=("乙",), expected_view_digest="视图摘要",
    )
    assert client.rpc_count == 2
    assert [row["action"] for row in calls] == [
        "flowstate_batch_introspection", "flowstate_batch_reconciliation"
    ]
    assert calls[1]["selected_ids"] == ["乙"]
    assert calls[1]["expected_view_digest"] == "视图摘要"


def test_batched_client_rejects_missing_handle_before_rpc():
    """候选缺少运行时句柄时必须在发送前失败。"""
    client = BatchedControlClient(SimpleNamespace(_call=lambda request: request))
    with pytest.raises(KeyError):
        client.introspect(
            nonce="读", candidates=(_candidate("甲"),), handles={}
        )
    assert client.rpc_count == 1


def test_state_from_view_reuses_existing_compact_semantics():
    """批量路径必须产生与原逐候选门禁相同的候选状态字段。"""
    path = {
        "node_id": 8, "prefix_tokens": 2, "prefix_sha256": "摘要",
        "path_node_ids": [2, 8], "path_full_sha256": "FA路径",
        "target_full_present": True, "path_full_all_present": True,
        "target_mamba_present": True,
    }
    view = {
        "paths": {"甲": path},
        "tree": {"structure_sha256": "结构", "full_tree_sha256": "FA树"},
        "accounting": {"full_allocator": {"available": 9}},
    }
    state = _state_from_view(view, ("甲",))["甲"]
    assert state["node_id"] == 8
    assert state["fa_resident"] is True
    assert state["recurrent_resident"] is True
    assert state["full_tree_digest"] == "FA树"


def test_record_base_is_fail_closed_and_keeps_reference_selection():
    """未完成记录默认必须无效且保留冻结选择。"""
    plan = {
        "run_id": "运行", "snapshot_id": "快照", "snapshot_digest": "摘要",
        "logical_k": 2, "candidate_count": 8, "repetition": 1,
        "selected_candidate_ids": ["甲", "乙"],
    }
    record = _record_base(plan)
    assert record["status"] == "INVALID"
    assert record["reference_selected_checkpoint_ids"] == ["甲", "乙"]
    assert all(value is None for value in record["timings_ns"].values())


def test_source_manifest_covers_frozen_core_and_batch_sources():
    """正式采集前后摘要必须同时覆盖冻结核心与两个批量入口。"""
    files = _source_manifest()["files"]
    assert any(path.endswith("flowstate/optimizer.py") for path in files)
    assert any(path.endswith("evaluation/rq6_batched_control.py") for path in files)
    assert any(path.endswith("tests/runtime/rq6_batched_control_transport.py") for path in files)


def _load_transport_functions(*names: str) -> dict[str, object]:
    """仅载入无 SGLang 初始化需求的传输层纯函数。"""
    path = Path(__file__).parent / "runtime/rq6_batched_control_transport.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]

    class Probe:
        @staticmethod
        def _changed_mamba_nodes(before, after):
            before_map = {row[0]: row[1] for row in before}
            after_map = {row[0]: row[1] for row in after}
            return sorted(key for key in set(before_map) | set(after_map)
                          if before_map.get(key) != after_map.get(key))

    namespace = {"hashlib": hashlib, "json": json, "_probe": Probe(),
                 "RuntimeCheckpointHandle": SimpleNamespace}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_canonical_view_digest_is_key_order_independent():
    """相同状态视图不得因字典插入顺序产生不同摘要。"""
    function = _load_transport_functions("_canonical_digest")["_canonical_digest"]
    assert function({"甲": 1, "乙": 2}) == function({"乙": 2, "甲": 1})


def test_unified_proof_accepts_only_expected_recurrent_change():
    """统一验证应接受仅目标循环状态消失且 FA 完全不变。"""
    function = _load_transport_functions("_proof")["_proof"]
    path_selected = {
        "node_id": 3, "prefix_sha256": "甲", "path_node_ids": [3],
        "path_full_sha256": "FA甲", "target_full_present": True,
        "path_full_all_present": True, "target_mamba_present": True,
        "target_mamba_slots": [1],
    }
    path_evicted = {
        "node_id": 7, "prefix_sha256": "乙", "path_node_ids": [7],
        "path_full_sha256": "FA乙", "target_full_present": True,
        "path_full_all_present": True, "target_mamba_present": True,
        "target_mamba_slots": [2],
    }
    before = {
        "paths": {"甲": path_selected, "乙": path_evicted},
        "tree": {"mamba_rows": [[3, [1]], [7, [2]]], "mamba_node_count": 2,
                 "full_tree_sha256": "FA树", "structure_sha256": "结构"},
        "accounting": {"mamba_available": 4, "full_allocator": "FA分配器"},
    }
    after = json.loads(json.dumps(before, ensure_ascii=False))
    after["paths"]["乙"]["target_mamba_present"] = False
    after["paths"]["乙"]["target_mamba_slots"] = []
    after["tree"]["mamba_rows"] = [[3, [1]]]
    after["tree"]["mamba_node_count"] = 1
    after["accounting"]["mamba_available"] = 5
    proof = function(before, after, ("甲",), ("乙",))
    assert proof["status"] == "PASS"
    assert proof["fa_preserved"] is True
    assert proof["native_recurrent_eviction"] is False


def test_unified_proof_rejects_fa_change():
    """任何 FA 摘要变化都必须令批量协调失败。"""
    function = _load_transport_functions("_proof")["_proof"]
    path = {
        "node_id": 3, "prefix_sha256": "甲", "path_node_ids": [3],
        "path_full_sha256": "FA甲", "target_full_present": True,
        "path_full_all_present": True, "target_mamba_present": True,
        "target_mamba_slots": [1],
    }
    before = {"paths": {"甲": path}, "tree": {"mamba_rows": [[3, [1]]],
        "mamba_node_count": 1, "full_tree_sha256": "原", "structure_sha256": "结构"},
        "accounting": {"mamba_available": 4, "full_allocator": "FA分配器"}}
    after = json.loads(json.dumps(before, ensure_ascii=False))
    after["tree"]["full_tree_sha256"] = "改变"
    proof = function(before, after, ("甲",), ())
    assert proof["status"] == "FAIL"
    assert proof["fa_cascade"] is True


def test_transport_has_no_s0_s4_trace_actions():
    """正式批量路径不得保留逐驱逐 S0 至 S4 全候选追踪。"""
    path = Path(__file__).parent / "runtime/rq6_batched_control_transport.py"
    source = path.read_text(encoding="utf-8")
    for boundary in ("S0", "S1", "S2", "S3", "S4"):
        assert f'"{boundary}"' not in source
    assert source.count("adapter.evict_mamba_only") == 1
    assert '"post_validation_count": 1' in source
