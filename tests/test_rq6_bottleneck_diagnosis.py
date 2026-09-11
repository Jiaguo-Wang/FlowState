"""验证瓶颈诊断的计时边界、只读门禁与确定性样本选择。"""

import ast
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

import pytest

from evaluation.rq6_bottleneck_diagnosis import (
    MeasuredClient, assert_read_only, diagnostic_plan, timing_record,
)


def test_timing_decomposition_reconciles():
    """服务端与往返剩余耗时必须精确相加，不能重复累计。"""
    response = {"diagnostic_timing": {"server_total_ns": 30,
        "server_started_ns": 120, "server_ended_ns": 150}}
    row = timing_record({"op": "census"}, response, 100, 180)
    assert row["server_ns"] + row["outside_server_ns"] == row["roundtrip_ns"] == 80
    assert row["arrival_wait_ns"] == 20
    assert row["return_ns"] == 30
    with pytest.raises(ValueError):
        timing_record({"op": "census"}, response, 100, 110)


@pytest.mark.parametrize("field", ["tree", "accounting"])
def test_readonly_rejects_mutation(field):
    """树或分配器任一改变都必须阻断只读结论。"""
    before = {"tree": {"resident": 3}, "accounting": {"available": 7}}
    assert_read_only(before, before)
    after = {**before, field: {"changed": True}}
    with pytest.raises(RuntimeError, match="状态变更"):
        assert_read_only(before, after)


def test_client_keeps_unmeasured_requests_unchanged():
    """重放阶段不能附加计时标志或额外请求。"""
    calls = []

    class Client:
        def _call(self, request):
            calls.append(request)
            return {"ok": True}

    client = MeasuredClient(Client())
    request = {"op": "census", "nonce": "测试"}
    assert client._call(request) == {"ok": True}
    assert calls == [request]
    assert client.rows == []


def test_client_preserves_formal_prefix_digest_validation():
    """诊断客户端必须保留正式客户端的令牌规范化及摘要校验字段。"""
    from array import array
    import hashlib
    calls = []

    class Client:
        def _call(self, request):
            calls.append(request)
            return {"ok": True}

    MeasuredClient(Client()).checkpoint_control(nonce="测试", label="测试", action="inspect", token_ids=(1, 2))
    assert calls[0]["token_ids"] == [1, 2]
    assert calls[0]["extra_key"] is None
    assert calls[0]["expected_prefix_sha256"] == hashlib.sha256(array("q", [1, 2]).tobytes()).hexdigest()


def test_plan_uses_three_scales_without_latency_selection():
    """每个规模恰有两个条件，且选样不读取延迟字段。"""
    plan = diagnostic_plan()
    assert plan == diagnostic_plan()
    assert len(plan) == 6
    assert {r["candidate_count"] for r in plan} == {8, 12, 20}
    for count in (8, 12, 20):
        rows = [r for r in plan if r["candidate_count"] == count]
        assert {r["mode"] for r in rows} == {"one", "multiple"}
        assert rows[0]["snapshot_digest"] == rows[1]["snapshot_digest"]


def test_profiler_restores_after_exception():
    """直接载入无设备依赖的探针定义，验证嵌套计时及异常恢复。"""
    path = Path(__file__).parents[1] / "tests/runtime/rq6_bottleneck_transport.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in {"_CallProfiler", "_profile_probe_calls"}]
    namespace = {"defaultdict": defaultdict, "contextmanager": contextmanager,
                 "perf_counter_ns": perf_counter_ns, "Callable": Callable}

    class Probe:
        pass

    probe = Probe()
    names = ("_validate_runtime_scope", "_path_snapshot", "_accounting_snapshot",
             "_global_maps", "_find_exact_node", "_tensor_sha256", "_tensor_ids")
    originals = {}
    for name in names:
        originals[name] = lambda: 42
        setattr(probe, name, originals[name])
    namespace["_probe"] = probe
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    profiler = namespace["_CallProfiler"]()
    with pytest.raises(RuntimeError):
        with namespace["_profile_probe_calls"](profiler):
            assert probe._path_snapshot() == 42
            raise RuntimeError("测试异常")
    assert all(getattr(probe, name) is original for name, original in originals.items())
    assert profiler.counts["_path_snapshot"] == 1
    assert profiler.inclusive_ns["_path_snapshot"] >= 0


def test_actual_eviction_count_includes_unused_budget():
    """预算未用满时，消息数必须使用实际驱逐数十六而非十五。"""
    from evaluation.rq6_bottleneck_analysis import message_counts

    counts = message_counts(20, 16)
    assert counts["controller_rpc"] == 74
    assert counts["probe_path_snapshot_calls"] == 1712
    assert message_counts(8, 0)["s4_snapshot_rpc"] == 0
    with pytest.raises(ValueError):
        message_counts(8, 9)


def test_exclusive_decomposition_closes_without_double_counting():
    """父函数内的嵌套摘要时间不能再次加入总耗时。"""
    from evaluation.rq6_bottleneck_analysis import exclusive_breakdown

    rows = [{"roundtrip_ns": 100, "server_ns": 40, "outside_server_ns": 60,
        "arrival_wait_ns": 50, "return_ns": 10, "profile": {"inclusive_ns": {
            "_path_snapshot": 20, "_path_snapshot:_tensor_sha256": 10}}}]
    result = exclusive_breakdown(rows, 120)
    assert result["closure_error_ns"] == 0
    assert result["worker_other_ms"] == 20 / 1e6
    assert result["controller_outside_rpc_ms"] == 20 / 1e6


def test_wait_probe_preserves_submit_arguments_and_result():
    """队列探针必须透传原有请求和超时，只添加时间字段。"""
    from types import SimpleNamespace
    from evaluation.rq6_wait_diagnosis import install_wait_timing

    calls = []

    class State:
        def submit(self, request, timeout_s=180):
            calls.append((request, timeout_s))
            return {"ok": True, "diagnostic_timing": {"server_total_ns": 1}}

    probe = SimpleNamespace(ProbeState=State)
    install_wait_timing(probe, MeasuredClient)
    request = {"nonce": "队列测试"}
    result = State().submit(request, timeout_s=99)
    assert calls == [(request, 99)]
    assert result["ok"] is True
    assert result["diagnostic_timing"]["server_total_ns"] == 1
    assert result["diagnostic_timing"]["submit_exit_ns"] >= result["diagnostic_timing"]["submit_enter_ns"]
