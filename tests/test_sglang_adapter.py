from __future__ import annotations

from array import array
import hashlib
from types import SimpleNamespace

import pytest

from flowstate.adapters.sglang import RuntimeCheckpointHandle, SGLangAdapter


class FakeComponentType:
    FULL = "full"
    MAMBA = "mamba"


class FakeEvictLayer:
    DEVICE = 1


class FakeValue:
    def __init__(self, *values: int) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)


class FakeComponentData:
    def __init__(self, value: object | None = None) -> None:
        self.value = value
        self.lock_ref = 0
        self.host_value = None
        self.host_lock_ref = 0
        self.session_ref = 0


class FakeRadixKey:
    def __init__(
        self,
        token_ids: tuple[int, ...],
        extra_key: str | None,
    ) -> None:
        self.token_ids = token_ids
        self.extra_key = extra_key

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, index: slice) -> FakeRadixKey:
        return FakeRadixKey(self.token_ids[index], self.extra_key)

    def child_key(self, page_size: int) -> object:
        assert page_size == 1
        first_token = self.token_ids[0]
        if self.extra_key is None:
            return first_token
        return self.extra_key, first_token

    def match(self, other: FakeRadixKey, page_size: int) -> int:
        assert page_size == 1
        assert self.extra_key == other.extra_key
        matched = 0
        for current, expected in zip(self.token_ids, other.token_ids):
            if current != expected:
                break
            matched += 1
        return matched


class FakeNode:
    def __init__(
        self,
        node_id: int,
        key: FakeRadixKey,
        parent: FakeNode | None,
        full_value: object | None,
        mamba_value: object | None,
    ) -> None:
        self.id = node_id
        self.key = key
        self.parent = parent
        self.children: dict[object, FakeNode] = {}
        self.component_data = {
            FakeComponentType.FULL: FakeComponentData(full_value),
            FakeComponentType.MAMBA: FakeComponentData(mamba_value),
        }


class FakeLRU:
    def __init__(self, nodes: tuple[FakeNode, ...]) -> None:
        self.nodes = set(nodes)

    def in_list(self, node: FakeNode) -> bool:
        return node in self.nodes

    def remove_node(self, node: FakeNode) -> None:
        self.nodes.remove(node)


class FakeFullAllocator:
    def __init__(self) -> None:
        self.available = 100
        self.allocated = 50

    def available_size(self) -> int:
        return self.available

    def allocated_count(self) -> int:
        return self.allocated


class FakeAvailableOnlyAllocator:
    def __init__(self) -> None:
        self.available = 100

    def available_size(self) -> int:
        return self.available


class FakeMambaComponent:
    def __init__(self) -> None:
        self.is_evict_device_ongoing = False


class FakeTreeCore:
    def __init__(
        self,
        root: FakeNode,
        nodes: tuple[FakeNode, ...],
        target: FakeNode,
    ) -> None:
        self.page_size = 1
        self.root_node = root
        self._node_arena = {node.id: node for node in nodes}
        self.enable_hicache = False
        self.enable_session_radix_cache = False
        self.lru_lists = {
            FakeComponentType.MAMBA: FakeLRU((target,)),
        }
        self.leaf_update_count = 0
        self.cascade_called = False
        self.ongoing_insert = False

    def node_by_id(self, node_id: int) -> FakeNode:
        return self._node_arena[node_id]

    def has_ongoing_insert(self) -> bool:
        return self.ongoing_insert

    def _evict_component_and_detach_lru(
        self,
        node: FakeNode,
        component: FakeMambaComponent,
        *,
        target: int,
        tracker,
        device_frees,
        host_frees,
    ) -> tuple[int, int]:
        assert component is not None
        assert target == FakeEvictLayer.DEVICE
        assert not host_frees
        data = node.component_data[FakeComponentType.MAMBA]
        value = data.value
        device_frees[FakeComponentType.MAMBA].append(value)
        data.value = None
        tracker[FakeComponentType.MAMBA] += len(value)
        self.lru_lists[FakeComponentType.MAMBA].remove_node(node)
        return len(value), 0

    def _update_evictable_leaf_sets(self, node: FakeNode) -> None:
        assert node.id in self._node_arena
        self.leaf_update_count += 1

    def _cascade_evict(self, *args, **kwargs) -> None:
        self.cascade_called = True
        raise AssertionError("正式原语不得调用级联驱逐")


class FakeCache:
    def __init__(self) -> None:
        extra_key = "workflow-key"
        root = FakeNode(
            0,
            FakeRadixKey((), None),
            None,
            FakeValue(),
            None,
        )
        middle = FakeNode(
            3,
            FakeRadixKey((11, 22), extra_key),
            root,
            FakeValue(1, 2),
            FakeValue(81),
        )
        target = FakeNode(
            7,
            FakeRadixKey((33,), extra_key),
            middle,
            FakeValue(3),
            FakeValue(91),
        )
        root.children[(extra_key, 11)] = middle
        middle.children[(extra_key, 33)] = target

        self.target = target
        self.tree_core = FakeTreeCore(root, (root, middle, target), target)
        self.tree_components = (
            FakeComponentType.FULL,
            FakeComponentType.MAMBA,
        )
        self.components = {
            FakeComponentType.MAMBA: FakeMambaComponent(),
        }
        self.req_to_token_pool = SimpleNamespace(mamba_ckpt_pool=None)
        self.token_to_kv_pool_allocator = SimpleNamespace(
            full_attn_allocator=FakeFullAllocator()
        )
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        self.freed_device_values = {}
        self.freed_host_values = {}
        self.sanity_check_count = 0

    def _free_values(self, device_frees, host_frees) -> None:
        self.freed_device_values = {
            item: tuple(values) for item, values in device_frees.items()
        }
        self.freed_host_values = {
            item: tuple(values) for item, values in host_frees.items()
        }
        device_frees.clear()
        host_frees.clear()

    def sanity_check(self) -> None:
        self.sanity_check_count += 1


class FakeSGLangAdapter(SGLangAdapter):
    @staticmethod
    def _make_radix_key(
        token_ids: tuple[int, ...],
        extra_key: str | None,
    ) -> FakeRadixKey:
        return FakeRadixKey(token_ids, extra_key)

    @staticmethod
    def _component_type() -> object:
        return FakeComponentType

    @staticmethod
    def _evict_layer() -> object:
        return FakeEvictLayer


def token_digest(token_ids: tuple[int, ...]) -> str:
    values = array("q", token_ids)
    return hashlib.sha256(values.tobytes()).hexdigest()


def make_handle() -> RuntimeCheckpointHandle:
    token_ids = (11, 22, 33)
    return RuntimeCheckpointHandle(
        checkpoint_id="P1",
        token_ids=token_ids,
        extra_key="workflow-key",
        expected_node_id=7,
        expected_prefix_digest=token_digest(token_ids),
    )


def test_runtime_checkpoint_handle_keeps_runtime_location() -> None:
    handle = make_handle()

    assert handle.checkpoint_id == "P1"
    assert handle.token_ids == (11, 22, 33)
    assert handle.extra_key == "workflow-key"
    assert handle.expected_node_id == 7
    assert handle.expected_prefix_digest == token_digest(handle.token_ids)


@pytest.mark.parametrize(
    "values",
    [
        {"checkpoint_id": "", "token_ids": (1,)},
        {"checkpoint_id": "P1", "token_ids": ()},
        {"checkpoint_id": "P1", "token_ids": (1, -1)},
        {"checkpoint_id": "P1", "token_ids": (True,)},
        {
            "checkpoint_id": "P1",
            "token_ids": (1,),
            "expected_node_id": -1,
        },
        {
            "checkpoint_id": "P1",
            "token_ids": (1,),
            "expected_prefix_digest": "",
        },
    ],
)
def test_runtime_checkpoint_handle_rejects_invalid_values(values) -> None:
    with pytest.raises(ValueError):
        RuntimeCheckpointHandle(**values)


def test_find_exact_node_returns_target_and_full_path() -> None:
    cache = FakeCache()
    adapter = FakeSGLangAdapter(cache)

    node, path = adapter._find_exact_node(
        (11, 22, 33),
        "workflow-key",
    )

    assert node is cache.target
    assert tuple(current.id for current in path) == (3, 7)


def test_find_exact_node_rejects_approximate_segment_match() -> None:
    adapter = FakeSGLangAdapter(FakeCache())

    with pytest.raises(RuntimeError, match="拒绝近似匹配"):
        adapter._find_exact_node((11,), "workflow-key")


def test_find_exact_node_rejects_missing_path() -> None:
    adapter = FakeSGLangAdapter(FakeCache())

    with pytest.raises(RuntimeError, match="找不到精确前缀节点"):
        adapter._find_exact_node((99,), "workflow-key")


def test_validate_target_rejects_node_identity_drift() -> None:
    cache = FakeCache()
    adapter = FakeSGLangAdapter(cache)
    node, path = adapter._find_exact_node((11, 22, 33), "workflow-key")
    handle = RuntimeCheckpointHandle(
        checkpoint_id="P1",
        token_ids=(11, 22, 33),
        extra_key="workflow-key",
        expected_node_id=8,
    )

    with pytest.raises(RuntimeError, match="节点标识与预期不一致"):
        adapter._validate_target(handle, node, path)


def test_validate_target_rejects_prefix_digest_drift() -> None:
    cache = FakeCache()
    adapter = FakeSGLangAdapter(cache)
    node, path = adapter._find_exact_node((11, 22, 33), "workflow-key")
    handle = RuntimeCheckpointHandle(
        checkpoint_id="P1",
        token_ids=(11, 22, 33),
        extra_key="workflow-key",
        expected_prefix_digest="摘要不一致",
    )

    with pytest.raises(RuntimeError, match="前缀摘要与预期不一致"):
        adapter._validate_target(handle, node, path)


def test_validate_target_rejects_live_reference() -> None:
    cache = FakeCache()
    cache.target.component_data[FakeComponentType.MAMBA].lock_ref = 1
    adapter = FakeSGLangAdapter(cache)
    node, path = adapter._find_exact_node((11, 22, 33), "workflow-key")

    with pytest.raises(RuntimeError, match="仍有设备引用"):
        adapter._validate_target(make_handle(), node, path)


def test_validate_target_rejects_ongoing_transfer() -> None:
    cache = FakeCache()
    cache.ongoing_load_back[1] = object()
    adapter = FakeSGLangAdapter(cache)
    node, path = adapter._find_exact_node((11, 22, 33), "workflow-key")

    with pytest.raises(RuntimeError, match="未完成的运行时传输"):
        adapter._validate_target(make_handle(), node, path)


def test_evict_mamba_only_preserves_fa_and_tree() -> None:
    cache = FakeCache()
    adapter = FakeSGLangAdapter(cache)
    full_value = cache.target.component_data[FakeComponentType.FULL].value
    mamba_value = cache.target.component_data[FakeComponentType.MAMBA].value
    node_count = len(cache.tree_core._node_arena)
    allocator_state = adapter._fa_allocator_snapshot(cache)

    adapter.evict_mamba_only(make_handle())

    node, path = adapter._find_exact_node((11, 22, 33), "workflow-key")
    assert node is cache.target
    assert tuple(current.id for current in path) == (3, 7)
    assert node.component_data[FakeComponentType.FULL].value is full_value
    assert node.component_data[FakeComponentType.MAMBA].value is None
    assert len(cache.tree_core._node_arena) == node_count
    assert adapter._fa_allocator_snapshot(cache) == allocator_state
    assert cache.freed_device_values == {
        FakeComponentType.MAMBA: (mamba_value,)
    }
    assert cache.freed_host_values == {}
    assert cache.tree_core.leaf_update_count == 1
    assert cache.tree_core.cascade_called is False
    assert cache.sanity_check_count == 1


def test_fa_allocator_can_expose_available_count_only() -> None:
    cache = FakeCache()
    allocator = FakeAvailableOnlyAllocator()
    cache.token_to_kv_pool_allocator.full_attn_allocator = allocator
    adapter = FakeSGLangAdapter(cache)

    adapter.evict_mamba_only(make_handle())

    snapshot = adapter._fa_allocator_snapshot(cache)
    assert snapshot.allocator_type == "FakeAvailableOnlyAllocator"
    assert snapshot.available_size == 100
    assert snapshot.allocated_count is None


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("available", "FA 分配器计数发生变化"),
        ("allocated", "FA 分配器计数发生变化"),
    ],
)
def test_fa_allocator_validates_every_available_counter(
    attribute: str,
    message: str,
) -> None:
    cache = FakeCache()
    allocator = cache.token_to_kv_pool_allocator.full_attn_allocator
    original_free_values = cache._free_values

    def change_counter_after_free(device_frees, host_frees) -> None:
        original_free_values(device_frees, host_frees)
        setattr(allocator, attribute, getattr(allocator, attribute) - 1)

    cache._free_values = change_counter_after_free
    adapter = FakeSGLangAdapter(cache)

    with pytest.raises(RuntimeError, match=message):
        adapter.evict_mamba_only(make_handle())


def test_fa_allocator_with_two_counters_records_both() -> None:
    cache = FakeCache()
    adapter = FakeSGLangAdapter(cache)

    snapshot = adapter._fa_allocator_snapshot(cache)

    assert snapshot.allocator_type == "FakeFullAllocator"
    assert snapshot.available_size == 100
    assert snapshot.allocated_count == 50


def test_inspect_checkpoint_stays_unimplemented_without_logical_metadata() -> None:
    adapter = FakeSGLangAdapter(FakeCache())

    with pytest.raises(NotImplementedError, match="workflow_id"):
        adapter.inspect_checkpoint(make_handle())


def test_adapter_rejects_non_handle_arguments() -> None:
    adapter = SGLangAdapter()

    with pytest.raises(TypeError, match="RuntimeCheckpointHandle"):
        adapter.evict_mamba_only(object())
