"""FlowState 与 SGLang 运行时之间的最小适配边界。"""

from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Sequence

from ..state_catalog import CheckpointCandidate


@dataclass(frozen=True)
class RuntimeCheckpointHandle:
    """保存 FlowState 检查点在 SGLang 运行时中的定位信息。"""

    checkpoint_id: str
    token_ids: tuple[int, ...]
    extra_key: str | None = None
    expected_node_id: int | None = None
    expected_prefix_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("检查点标识不能为空")
        if not isinstance(self.token_ids, tuple) or not self.token_ids:
            raise ValueError("token_ids 必须是非空元组")
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in self.token_ids
        ):
            raise ValueError("token_ids 必须只包含非负整数")
        if self.extra_key is not None and not isinstance(self.extra_key, str):
            raise ValueError("extra_key 必须是字符串或空值")
        if self.expected_node_id is not None and (
            not isinstance(self.expected_node_id, int)
            or isinstance(self.expected_node_id, bool)
            or self.expected_node_id < 0
        ):
            raise ValueError("预期节点标识必须是非负整数或空值")
        if self.expected_prefix_digest is not None and (
            not isinstance(self.expected_prefix_digest, str)
            or not self.expected_prefix_digest
        ):
            raise ValueError("预期前缀摘要必须是非空字符串或空值")


@dataclass(frozen=True)
class _FAAllocatorSnapshot:
    """保存 FA 分配器已确认可靠的计数状态。"""

    allocator_type: str
    available_size: int
    allocated_count: int | None = None


@dataclass(frozen=True)
class _TargetSnapshot:
    """保存单节点驱逐所需的最小前置状态。"""

    node: object
    node_id: int
    path_node_ids: tuple[int, ...]
    path_structure: tuple[tuple[object, ...], ...]
    path_full_values: tuple[object, ...]
    tree_node_count: int
    fa_allocator_snapshot: _FAAllocatorSnapshot


class SGLangAdapter:
    """在调度器安全时点访问 SGLang 检查点运行时。"""

    def __init__(self, cache: object | None = None) -> None:
        self._cache = cache

    def inspect_checkpoint(
        self,
        handle: RuntimeCheckpointHandle,
    ) -> CheckpointCandidate:
        """检查运行时目标，并返回对应的核心检查点候选。"""
        self._validate_handle(handle)
        raise NotImplementedError(
            "仅凭运行时句柄无法构造 workflow_id 与 lineage_path"
        )

    def evict_mamba_only(
        self,
        handle: RuntimeCheckpointHandle,
    ) -> None:
        """仅驱逐目标检查点的 Mamba 设备状态，并保留 FA-KV。"""
        self._validate_handle(handle)
        node, path = self._find_exact_node(handle.token_ids, handle.extra_key)
        self._validate_target(handle, node, path)
        before = self._capture_target_snapshot(node, path)

        self._evict_mamba_component_only(node)

        after_node, after_path = self._find_exact_node(
            handle.token_ids,
            handle.extra_key,
        )
        self._validate_postconditions(before, after_node, after_path)

    def _find_exact_node(
        self,
        token_ids: tuple[int, ...],
        extra_key: str | None,
    ) -> tuple[object, tuple[object, ...]]:
        """按完整基数树分段定位精确节点，并返回从根后开始的路径。"""
        cache = self._require_cache()
        core = cache.tree_core
        remaining = self._make_radix_key(token_ids, extra_key)
        node = core.root_node
        path = []
        consumed = 0

        while len(remaining) > 0:
            child_key = remaining.child_key(core.page_size)
            child = node.children.get(child_key)
            if child is None:
                raise RuntimeError(
                    "找不到精确前缀节点："
                    f"已匹配 {consumed}/{len(token_ids)} 个令牌"
                )

            matched = child.key.match(remaining, page_size=core.page_size)
            if matched != len(child.key):
                raise RuntimeError(
                    "目标前缀终止于基数树分段内部，拒绝近似匹配："
                    f"已匹配 {consumed}，当前匹配 {matched}，"
                    f"分段长度 {len(child.key)}"
                )

            consumed += matched
            path.append(child)
            node = child
            remaining = remaining[matched:]

        if consumed != len(token_ids):
            raise RuntimeError(
                f"目标前缀长度不一致：{consumed} != {len(token_ids)}"
            )
        return node, tuple(path)

    def _validate_target(
        self,
        handle: RuntimeCheckpointHandle,
        node: object,
        path: Sequence[object] | None = None,
    ) -> None:
        """状态变更前重新验证节点标识、前缀摘要和状态驻留情况。"""
        cache = self._require_cache()
        core = cache.tree_core
        component_type = self._component_type()

        if path is None:
            resolved_node, resolved_path = self._find_exact_node(
                handle.token_ids,
                handle.extra_key,
            )
            if resolved_node is not node:
                raise RuntimeError("目标节点已发生变化")
            path = resolved_path

        try:
            current_node = core.node_by_id(int(node.id))
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("目标节点已不在当前基数树中") from error
        if current_node is not node:
            raise RuntimeError("节点标识已指向另一个运行时节点")

        if handle.expected_node_id is not None and (
            int(node.id) != handle.expected_node_id
        ):
            raise RuntimeError(
                "目标节点标识与预期不一致："
                f"{int(node.id)} != {handle.expected_node_id}"
            )

        prefix_length = sum(len(current.key) for current in path)
        if prefix_length != len(handle.token_ids):
            raise RuntimeError(
                "运行时前缀长度与句柄不一致："
                f"{prefix_length} != {len(handle.token_ids)}"
            )

        actual_digest = self._token_digest(handle.token_ids)
        if (
            handle.expected_prefix_digest is not None
            and actual_digest != handle.expected_prefix_digest
        ):
            raise RuntimeError(
                "目标前缀摘要与预期不一致："
                f"{actual_digest} != {handle.expected_prefix_digest}"
            )

        if node is core.root_node:
            raise RuntimeError("根节点不能作为循环状态检查点被驱逐")

        full_data = node.component_data[component_type.FULL]
        mamba_data = node.component_data[component_type.MAMBA]
        if full_data.value is None:
            raise RuntimeError("目标 FA-KV 当前不在设备上")
        if any(
            current.component_data[component_type.FULL].value is None
            for current in path
        ):
            raise RuntimeError("目标前缀的 FA-KV 路径不完整")
        if mamba_data.value is None:
            raise RuntimeError("目标 Mamba 状态当前不在设备上")

        for name, data in (("FA-KV", full_data), ("Mamba", mamba_data)):
            if int(data.lock_ref) != 0:
                raise RuntimeError(f"目标 {name} 仍有设备引用")
            if int(data.host_lock_ref) != 0:
                raise RuntimeError(f"目标 {name} 仍有主机引用")
            if int(data.session_ref) != 0:
                raise RuntimeError(f"目标 {name} 仍有会话引用")

        if mamba_data.host_value is not None:
            raise RuntimeError("目标 Mamba 状态存在未处理的主机副本")

        mamba_lru = core.lru_lists[component_type.MAMBA]
        if not mamba_lru.in_list(node):
            raise RuntimeError("目标 Mamba 状态不在设备 LRU 中")

        self._validate_runtime_scope(cache, core)

    def _evict_mamba_component_only(self, node: object) -> None:
        """仅分离并释放目标 Mamba 组件，不执行级联驱逐。"""
        cache = self._require_cache()
        core = cache.tree_core
        component_type = self._component_type()
        evict_layer = self._evict_layer()
        mamba = cache.components[component_type.MAMBA]
        mamba_value = node.component_data[component_type.MAMBA].value
        tracker = {item: 0 for item in cache.tree_components}
        device_frees = defaultdict(list)
        host_frees = defaultdict(list)

        device_freed = 0
        host_freed = 0
        mamba_frees: tuple[object, ...] = ()
        unexpected_device_free = False
        unexpected_host_free = False
        try:
            device_freed, host_freed = core._evict_component_and_detach_lru(
                node,
                mamba,
                target=evict_layer.DEVICE,
                tracker=tracker,
                device_frees=device_frees,
                host_frees=host_frees,
            )
            core._update_evictable_leaf_sets(node)
            mamba_frees = tuple(device_frees.get(component_type.MAMBA, ()))
            unexpected_device_free = any(
                values
                for item, values in device_frees.items()
                if item != component_type.MAMBA
            )
            unexpected_host_free = any(host_frees.values())
        finally:
            cache._free_values(device_frees, host_frees)

        expected_freed = len(mamba_value)
        if (
            device_freed != expected_freed
            or host_freed != 0
            or len(mamba_frees) != 1
            or mamba_frees[0] is not mamba_value
        ):
            raise RuntimeError(
                "仅 Mamba 驱逐的释放范围异常："
                f"设备释放 {device_freed}，主机释放 {host_freed}"
            )
        if unexpected_device_free:
            raise RuntimeError("仅 Mamba 驱逐意外释放了其他设备组件")
        if unexpected_host_free:
            raise RuntimeError("仅 Mamba 驱逐意外释放了主机组件")

        cache.sanity_check()

    def _capture_target_snapshot(
        self,
        node: object,
        path: Sequence[object],
    ) -> _TargetSnapshot:
        """采集前后条件需要的局部结构和分配器计数。"""
        cache = self._require_cache()
        component_type = self._component_type()
        return _TargetSnapshot(
            node=node,
            node_id=int(node.id),
            path_node_ids=tuple(int(current.id) for current in path),
            path_structure=self._path_structure(path),
            path_full_values=tuple(
                current.component_data[component_type.FULL].value
                for current in path
            ),
            tree_node_count=self._tree_node_count(cache.tree_core),
            fa_allocator_snapshot=self._fa_allocator_snapshot(cache),
        )

    def _validate_postconditions(
        self,
        before: _TargetSnapshot,
        node: object,
        path: Sequence[object],
    ) -> None:
        """强制检查单节点驱逐后的最小安全条件。"""
        cache = self._require_cache()
        component_type = self._component_type()

        if int(node.id) != before.node_id or node is not before.node:
            raise RuntimeError("驱逐后目标节点发生变化")
        if node.component_data[component_type.MAMBA].value is not None:
            raise RuntimeError("驱逐后目标 Mamba 状态仍然存在")
        if node.component_data[component_type.FULL].value is None:
            raise RuntimeError("驱逐后目标 FA-KV 不再驻留")

        path_node_ids = tuple(int(current.id) for current in path)
        if path_node_ids != before.path_node_ids:
            raise RuntimeError("驱逐后目标前缀路径发生变化")
        if self._path_structure(path) != before.path_structure:
            raise RuntimeError("驱逐后基数树局部结构发生变化")
        if self._tree_node_count(cache.tree_core) != before.tree_node_count:
            raise RuntimeError("驱逐后基数树节点数量发生变化")

        path_full_values = tuple(
            current.component_data[component_type.FULL].value
            for current in path
        )
        if len(path_full_values) != len(before.path_full_values) or any(
            current is not previous
            for current, previous in zip(
                path_full_values,
                before.path_full_values,
            )
        ):
            raise RuntimeError("驱逐后 FA-KV 路径值发生变化")

        if self._fa_allocator_snapshot(cache) != before.fa_allocator_snapshot:
            raise RuntimeError("驱逐后 FA 分配器计数发生变化")

    def _validate_runtime_scope(self, cache: object, core: object) -> None:
        """拒绝与插入、传输或其他 Mamba 驱逐并发的状态变更。"""
        if core.has_ongoing_insert():
            raise RuntimeError("基数树当前存在未完成的插入")
        if getattr(core, "enable_hicache", False):
            raise RuntimeError("当前版本不支持在 HiCache 开启时执行该操作")
        if getattr(core, "enable_session_radix_cache", False):
            raise RuntimeError("当前版本不支持在会话基数缓存开启时执行该操作")
        if getattr(cache.req_to_token_pool, "mamba_ckpt_pool", None) is not None:
            raise RuntimeError("当前版本不支持 int8 Mamba 检查点池")

        for name in (
            "ongoing_write_through",
            "ongoing_load_back",
            "ongoing_prefetch",
            "ongoing_backup",
        ):
            if getattr(cache, name, None):
                raise RuntimeError(f"当前存在未完成的运行时传输：{name}")

        component_type = self._component_type()
        mamba = cache.components[component_type.MAMBA]
        if getattr(mamba, "is_evict_device_ongoing", False):
            raise RuntimeError("当前已有 Mamba 设备驱逐正在执行")

    @staticmethod
    def _path_structure(
        path: Sequence[object],
    ) -> tuple[tuple[object, ...], ...]:
        """记录目标路径上的节点关系，不扫描整棵树。"""
        return tuple(
            (
                int(node.id),
                None if node.parent is None else int(node.parent.id),
                tuple(sorted(int(child.id) for child in node.children.values())),
                len(node.key),
            )
            for node in path
        )

    @staticmethod
    def _tree_node_count(core: object) -> int:
        """读取节点注册表中的当前有效节点数量。"""
        arena = getattr(core, "_node_arena", None)
        if not isinstance(arena, dict):
            raise RuntimeError("运行时未暴露可验证的节点注册表")
        return sum(node is not None for node in arena.values())

    @staticmethod
    def _fa_allocator_snapshot(
        cache: object,
    ) -> _FAAllocatorSnapshot:
        """读取 FA 分配器的权威可用量及可选已分配量。"""
        allocator = cache.token_to_kv_pool_allocator
        owner = getattr(allocator, "full_attn_allocator", None) or allocator
        available_size = getattr(owner, "available_size", None)
        if not callable(available_size):
            raise RuntimeError("运行时未提供 FA 分配器 available_size 接口")

        allocated_count = getattr(owner, "allocated_count", None)
        return _FAAllocatorSnapshot(
            allocator_type=type(owner).__name__,
            available_size=int(available_size()),
            allocated_count=(
                int(allocated_count()) if callable(allocated_count) else None
            ),
        )

    @staticmethod
    def _token_digest(token_ids: tuple[int, ...]) -> str:
        """计算与冻结 WP3B 运行时一致的令牌前缀摘要。"""
        values = array("q", [int(token_id) for token_id in token_ids])
        return hashlib.sha256(values.tobytes()).hexdigest()

    @staticmethod
    def _make_radix_key(
        token_ids: tuple[int, ...],
        extra_key: str | None,
    ) -> object:
        """惰性构造 SGLang 的 RadixKey，避免普通单元测试依赖 SGLang。"""
        from sglang.srt.mem_cache.radix_cache import RadixKey

        return RadixKey(array("q", token_ids), extra_key)

    @staticmethod
    def _component_type() -> object:
        """惰性取得 SGLang 组件类型。"""
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        return ComponentType

    @staticmethod
    def _evict_layer() -> object:
        """惰性取得 SGLang 设备驱逐层标识。"""
        from sglang.srt.mem_cache.unified_cache.components.tree_component import (
            EvictLayer,
        )

        return EvictLayer

    def _require_cache(self) -> object:
        """返回已绑定的运行时缓存。"""
        if self._cache is None:
            raise RuntimeError("SGLangAdapter 尚未绑定运行时缓存")
        return self._cache

    @staticmethod
    def _validate_handle(handle: RuntimeCheckpointHandle) -> None:
        """验证公开接口接收到的是正式运行时句柄。"""
        if not isinstance(handle, RuntimeCheckpointHandle):
            raise TypeError("handle 必须是 RuntimeCheckpointHandle")
