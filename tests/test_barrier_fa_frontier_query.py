from __future__ import annotations

import inspect
from types import SimpleNamespace

from evaluation.barrier_fa_frontier_control import (
    BarrierFAControlClient,
    effective_request_prefix_limit,
    semantic_snapshot_differences,
    traverse_device_fa_frontier,
)
from evaluation.barrier_fa_frontier_query_gate import (
    ENGINE_CONFIGURATION_BARRIER_QUERY,
    build_cases,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K


FULL = 0


class FakeKey:
    """实现测试所需的最小 RadixKey 行为。"""

    def __init__(self, values, extra_key=None):
        self.values = tuple(values)
        self.extra_key = extra_key

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return FakeKey(self.values[item], self.extra_key)

    def page_aligned(self, page_size):
        length = len(self) // page_size * page_size
        return self[:length]

    def child_key(self, page_size):
        return self.extra_key, self.values[:page_size]

    def match(self, other, page_size=1):
        matched = 0
        for left, right in zip(self.values, other.values):
            if left != right:
                break
            matched += 1
        return matched // page_size * page_size


class FakeNode:
    """保存只读 traversal 使用的最小节点字段。"""

    def __init__(self, node_id, values=(), resident=True):
        self.id = node_id
        self.key = FakeKey(values)
        self.children = {}
        self.component_data = {
            FULL: SimpleNamespace(value=object() if resident else None)
        }

    def add(self, child):
        self.children[child.key.child_key(1)] = child
        return child


def query(root, values):
    """使用固定 page size 执行纯内存查询。"""
    return traverse_device_fa_frontier(
        root_node=root,
        key=FakeKey(values),
        page_size=1,
        full_component_type=FULL,
    )


def test_effective_limit_matches_plain_generation_request() -> None:
    assert effective_request_prefix_limit(0) == 0
    assert effective_request_prefix_limit(1) == 0
    assert effective_request_prefix_limit(101) == 100
    assert effective_request_prefix_limit(101, 48) == 48
    assert effective_request_prefix_limit(101, 500) == 101


def test_gate_reuses_formal_128k_configuration_without_override() -> None:
    assert ENGINE_CONFIGURATION_BARRIER_QUERY == ENGINE_CONFIGURATION_128K
    assert ENGINE_CONFIGURATION_BARRIER_QUERY is not ENGINE_CONFIGURATION_128K


def test_gate_cases_freeze_boundary_and_partial_expectations() -> None:
    boundary, partial = build_cases()
    assert boundary["expected_h"] == 320
    assert boundary["expected_partial_match"] is False
    assert partial["expected_h"] == 333
    assert partial["expected_partial_match"] is True
    assert boundary["setup_input_ids"][0] != partial["setup_input_ids"][0]


def test_node_boundary_query_returns_complete_resident_path() -> None:
    root = FakeNode(0)
    first = root.add(FakeNode(1, (10, 11, 12)))
    first.add(FakeNode(2, (20, 21)))

    result = query(root, (10, 11, 12, 99))

    assert result["resident_fa_frontier"] == 3
    assert result["traversed_node_ids"] == [1]
    assert result["partial_match"] is False
    assert result["stop_reason"] == "没有匹配子节点"


def test_partial_node_query_does_not_split_or_touch_recency() -> None:
    root = FakeNode(0)
    child = root.add(FakeNode(7, (30, 31, 32, 33)))
    child.last_access_time = 123.5
    structure_before = (tuple(root.children), child.id, child.key.values)

    result = query(root, (30, 31, 88, 89))

    assert result["resident_fa_frontier"] == 2
    assert result["traversed_node_ids"] == [7]
    assert result["partial_match"] is True
    assert result["stop_reason"] == "在节点分段内部结束或分叉"
    assert (tuple(root.children), child.id, child.key.values) == structure_before
    assert child.last_access_time == 123.5


def test_full_residency_is_independent_of_recurrent_fields() -> None:
    root = FakeNode(0)
    root.add(FakeNode(9, (40, 41, 42), resident=True))
    result = query(root, (40, 41, 42))
    assert result["resident_fa_frontier"] == 3


def test_nonresident_full_component_stops_before_segment() -> None:
    root = FakeNode(0)
    root.add(FakeNode(11, (50, 51), resident=False))
    result = query(root, (50, 51))
    assert result["resident_fa_frontier"] == 0
    assert result["stop_reason"] == "Full组件不在设备上"


def test_query_implementation_avoids_mutating_cache_paths() -> None:
    source = inspect.getsource(traverse_device_fa_frontier)
    forbidden = (
        "match_prefix(",
        "_match_prefix_helper",
        "_match_post_processor",
        "_split_node",
        "refresh_lru",
        "inc_lock_ref",
        "dec_lock_ref",
        ".alloc(",
        ".free(",
    )
    assert all(name not in source for name in forbidden)


def test_semantic_difference_reports_nested_field() -> None:
    before = {"tree": {"rows": [[1, 2]]}, "lru": [3, 2]}
    after = {"tree": {"rows": [[1, 4]]}, "lru": [3, 2]}
    assert semantic_snapshot_differences(before, after) == [
        "tree.rows[0][1]"
    ]


def test_client_helper_sends_only_current_pending_request() -> None:
    calls = []
    raw_client = SimpleNamespace(_call=lambda request: calls.append(request) or request)
    client = BarrierFAControlClient(raw_client)
    response = client.inspect_fa_frontier(
        [1, 2, 3],
        limit=2,
        nonce="测试查询",
    )
    assert response["action"] == "inspect_fa_frontier"
    assert response["token_ids"] == [1, 2, 3]
    assert response["limit"] == 2
    assert set(response) == {
        "op",
        "action",
        "nonce",
        "token_ids",
        "extra_key",
        "limit",
    }
