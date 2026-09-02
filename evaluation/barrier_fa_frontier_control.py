"""提供 barrier 时刻 FA 驻留前缀的最小只读查询工具。"""

from __future__ import annotations

from array import array
from typing import Mapping, Sequence


_INSPECT_FA_FRONTIER_ACTION = "inspect_fa_frontier"


def effective_request_prefix_limit(
    input_token_count: int,
    limit: int | None = None,
) -> int:
    """返回普通生成请求实际允许匹配的前缀长度。"""
    if isinstance(input_token_count, bool) or input_token_count < 0:
        raise ValueError("input_token_count 必须是非负整数")
    if limit is None:
        return max(input_token_count - 1, 0)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit 必须是非负整数或空值")
    return min(limit, input_token_count)


def traverse_device_fa_frontier(
    *,
    root_node: object,
    key: object,
    page_size: int,
    full_component_type: object,
) -> dict[str, object]:
    """只读遍历 radix tree，并返回连续驻留的 Full-KV 前缀。"""
    if page_size <= 0:
        raise ValueError("page_size 必须为正整数")

    remaining = key.page_aligned(page_size)
    node = root_node
    consumed = 0
    traversed_node_ids: list[int] = []
    matched_segments: list[dict[str, object]] = []
    partial_match = False
    stop_reason = "达到有效前缀上限"

    while len(remaining) > 0:
        child_key = remaining.child_key(page_size)
        child = node.children.get(child_key)
        if child is None:
            stop_reason = "没有匹配子节点"
            break

        node_id = int(child.id)
        traversed_node_ids.append(node_id)
        full_value = child.component_data[full_component_type].value
        if full_value is None:
            matched_segments.append(
                {
                    "node_id": node_id,
                    "segment_length": len(child.key),
                    "matched_length": 0,
                    "full_device_resident": False,
                }
            )
            stop_reason = "Full组件不在设备上"
            break

        matched = int(child.key.match(remaining, page_size=page_size))
        segment_length = len(child.key)
        matched_segments.append(
            {
                "node_id": node_id,
                "segment_length": segment_length,
                "matched_length": matched,
                "full_device_resident": True,
            }
        )
        if matched < segment_length:
            consumed += matched
            partial_match = True
            stop_reason = "在节点分段内部结束或分叉"
            break

        consumed += segment_length
        remaining = remaining[segment_length:]
        node = child

    return {
        "resident_fa_frontier": consumed,
        "traversed_node_ids": traversed_node_ids,
        "partial_match": partial_match,
        "matched_segments": matched_segments,
        "stop_reason": stop_reason,
    }


def inspect_resident_fa_frontier(
    cache: object,
    token_ids: Sequence[int],
    *,
    extra_key: str | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    """使用正式 RadixKey 语义执行一次只读 FA 前缀查询。"""
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    normalized = tuple(int(token_id) for token_id in token_ids)
    if any(token_id < 0 for token_id in normalized):
        raise ValueError("token_ids 必须只包含非负整数")
    effective_limit = effective_request_prefix_limit(len(normalized), limit)
    key = RadixKey(
        array("q", normalized),
        extra_key,
        limit=effective_limit,
    )
    result = traverse_device_fa_frontier(
        root_node=cache.tree_core.root_node,
        key=key,
        page_size=int(cache.tree_core.page_size),
        full_component_type=ComponentType.FULL,
    )
    return {
        **result,
        "input_token_count": len(normalized),
        "effective_lookup_limit": effective_limit,
        "extra_key": extra_key,
        "page_size": int(cache.tree_core.page_size),
    }


def lru_node_order(lru: object) -> list[int]:
    """读取链表从 MRU 到 LRU 的真实节点顺序。"""
    point = int(lru._pt)
    current = lru.head.lru_next[point]
    result: list[int] = []
    maximum_steps = len(lru.cache) + 4
    steps = 0
    while current is not None and current is not lru.tail:
        if int(current.id) in lru.cache:
            result.append(int(current.id))
        current = current.lru_next[point]
        steps += 1
        if steps > maximum_steps:
            raise RuntimeError("LRU 链表出现环或哨兵结构异常")
    if current is not lru.tail:
        raise RuntimeError("LRU 链表没有到达尾部哨兵")
    if set(result) != {int(node_id) for node_id in lru.cache}:
        raise RuntimeError("LRU 链表与节点索引不一致")
    return result


def semantic_cache_snapshot(
    cache: object,
    *,
    component_type: object,
    tree_snapshot: Mapping[str, object],
    accounting_snapshot: Mapping[str, object],
) -> dict[str, object]:
    """冻结会影响后续 residency、驱逐或调度的稳定缓存状态。"""
    core = cache.tree_core
    arena = getattr(core, "_node_arena", None)
    if arena is None:
        raise RuntimeError("UnifiedTreeCore 缺少节点索引")
    nodes = sorted(
        (
            (int(node_id), node)
            for node_id, node in arena.items()
            if node is not None
        ),
        key=lambda item: item[0],
    )

    recency_rows = []
    reference_rows = []
    for node_id, node in nodes:
        recency_rows.append(
            [
                node_id,
                float(node.last_access_time),
                float(node.creation_time),
                int(node.hit_count),
            ]
        )
        full_data = node.component_data[component_type.FULL]
        mamba_data = node.component_data[component_type.MAMBA]
        reference_rows.append(
            [
                node_id,
                int(full_data.lock_ref),
                int(full_data.host_lock_ref),
                int(full_data.session_ref),
                int(mamba_data.lock_ref),
                int(mamba_data.host_lock_ref),
                int(mamba_data.session_ref),
            ]
        )

    mamba_lru = core.lru_lists[component_type.MAMBA]
    return {
        "tree": dict(tree_snapshot),
        "accounting": dict(accounting_snapshot),
        "recency_rows": recency_rows,
        "mamba_lru_order_mru_to_lru": lru_node_order(mamba_lru),
        "reference_rows": reference_rows,
        "full_evictable_leaf_ids": sorted(
            int(node.id) for node in core.evictable_device_leaves
        ),
    }


def semantic_snapshot_differences(
    before: object,
    after: object,
) -> list[str]:
    """返回两个稳定状态快照中不一致的字段路径。"""
    differences: list[str] = []

    def compare(left: object, right: object, path: str) -> None:
        if type(left) is not type(right):
            differences.append(path or "<root>")
            return
        if isinstance(left, Mapping):
            keys = sorted(set(left) | set(right), key=str)
            for key in keys:
                next_path = f"{path}.{key}" if path else str(key)
                if key not in left or key not in right:
                    differences.append(next_path)
                else:
                    compare(left[key], right[key], next_path)
            return
        if isinstance(left, list):
            if len(left) != len(right):
                differences.append(f"{path}.length")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if left != right:
            differences.append(path or "<root>")

    compare(before, after, "")
    return differences


class BarrierFAControlClient:
    """为既有 ControlClient 增加 FA frontier 查询。"""

    def __init__(self, client: object) -> None:
        self._client = client

    def inspect_fa_frontier(
        self,
        token_ids: Sequence[int],
        *,
        extra_key: str | None = None,
        limit: int | None = None,
        nonce: str = "barrier_fa_frontier",
    ) -> dict[str, object]:
        """在 scheduler idle safe point 查询设备 FA 驻留前缀。"""
        return self._client._call(
            {
                "op": "checkpoint_control",
                "action": _INSPECT_FA_FRONTIER_ACTION,
                "nonce": nonce,
                "token_ids": [int(token_id) for token_id in token_ids],
                "extra_key": extra_key,
                "limit": limit,
            }
        )
