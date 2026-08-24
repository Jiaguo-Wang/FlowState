#!/usr/bin/env python3
"""Scheduler-thread control probe for WP3A direct Mamba eviction.

The TCP thread never reads or mutates the radix tree.  It only enqueues a
command.  ``Scheduler.on_idle`` drains one command on the scheduler thread,
after the ordinary request pipeline is fully idle.

The eviction path intentionally reuses SGLang's component lifecycle:

  MambaComponent.evict_component
    -> UnifiedTreeCore LRU detach/cascade
    -> UnifiedRadixCache component-aware allocator free

Targets must be internal radix nodes.  This is important: SGLang gives all
components equal priority on a leaf, while an internal Mamba component has
lower eviction priority than Full KV.  The internal-node gate therefore makes
the intervention remove only Mamba while retaining the exact-prefix Full KV.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import socket
import threading
import time
import traceback
import uuid
from array import array
from collections import defaultdict


LOG = logging.getLogger(__name__)


def _tensor_ids(value):
    if value is None:
        return []
    try:
        return [int(x) for x in value.detach().reshape(-1).tolist()]
    except Exception:
        return [int(x) for x in list(value)]


def _tensor_sha256(value) -> str | None:
    if value is None:
        return None
    try:
        raw = value.detach().reshape(-1).cpu().numpy().tobytes()
    except Exception:
        raw = array("q", _tensor_ids(value)).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _token_sha256(token_ids) -> str:
    return hashlib.sha256(array("q", [int(x) for x in token_ids]).tobytes()).hexdigest()


def _queue_lengths(scheduler) -> dict:
    waiting = len(getattr(scheduler, "waiting_queue", []) or [])
    running_batch = getattr(scheduler, "running_batch", None)
    try:
        running = len(running_batch.reqs) if running_batch is not None else 0
    except Exception:
        running = 0
    return {
        "scheduler_fully_idle": bool(scheduler.is_fully_idle()),
        "waiting_requests": int(waiting),
        "running_requests": int(running),
        "chunked_request_present": getattr(scheduler, "chunked_req", None) is not None,
    }


def _full_allocator_snapshot(cache) -> dict:
    alloc = cache.token_to_kv_pool_allocator
    fa = getattr(alloc, "full_attn_allocator", None)
    owner = fa if fa is not None else alloc

    def maybe_int(name):
        fn = getattr(owner, name, None)
        if fn is None:
            return None
        try:
            return int(fn())
        except Exception:
            return None

    return {
        "allocator_type": type(owner).__name__,
        "available": maybe_int("available_size"),
        "allocated": maybe_int("allocated_count"),
    }


def _tree_items(core):
    arena = getattr(core, "_node_arena", None)
    if arena is None:
        return []
    return sorted(
        ((int(node_id), node) for node_id, node in arena.items() if node is not None),
        key=lambda item: item[0],
    )


def _global_maps(cache) -> dict:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    full_rows = []
    mamba_rows = []
    structure_rows = []
    for node_id, node in _tree_items(cache.tree_core):
        fcd = node.component_data[ComponentType.FULL]
        mcd = node.component_data[ComponentType.MAMBA]
        parent_id = None if node.parent is None else int(node.parent.id)
        children = sorted(int(child.id) for child in node.children.values())
        structure_rows.append(
            [node_id, parent_id, children, 0 if node.key is None else len(node.key)]
        )
        full_rows.append(
            [node_id, None if fcd.value is None else len(fcd.value), _tensor_sha256(fcd.value)]
        )
        if mcd.value is not None:
            mamba_rows.append([node_id, _tensor_ids(mcd.value)])

    def digest(rows):
        return hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    return {
        "node_count": len(structure_rows),
        "structure_rows": structure_rows,
        "structure_sha256": digest(structure_rows),
        "full_rows": full_rows,
        "full_tree_sha256": digest(full_rows),
        "mamba_rows": mamba_rows,
        "mamba_tree_sha256": digest(mamba_rows),
        "mamba_node_count": len(mamba_rows),
    }


def _find_exact_node(cache, token_ids, extra_key):
    from sglang.srt.mem_cache.radix_cache import RadixKey

    core = cache.tree_core
    remaining = RadixKey(array("q", [int(x) for x in token_ids]), extra_key)
    node = core.root_node
    path = []
    consumed = 0
    while len(remaining) > 0:
        child_key = remaining.child_key(core.page_size)
        child = node.children.get(child_key)
        if child is None:
            raise AssertionError(
                f"exact prefix path missing after {consumed}/{len(token_ids)} tokens"
            )
        matched = child.key.match(remaining, page_size=core.page_size)
        if matched != len(child.key):
            raise AssertionError(
                "target ends within a radix segment; refusing to split or mutate: "
                f"consumed={consumed}, matched={matched}, segment={len(child.key)}"
            )
        consumed += matched
        path.append(child)
        node = child
        remaining = remaining[matched:]

    if consumed != len(token_ids):
        raise AssertionError(f"target length mismatch: {consumed} != {len(token_ids)}")
    return node, path


def _path_snapshot(cache, token_ids, extra_key) -> dict:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    node, path = _find_exact_node(cache, token_ids, extra_key)
    core = cache.tree_core
    path_full_hasher = hashlib.sha256()
    path_full_rows = 0
    path_full_all_present = True
    path_mamba_positions = []
    position = 0
    segment_lengths = []
    path_node_ids = []
    for current in path:
        seg_len = len(current.key)
        position += seg_len
        segment_lengths.append(seg_len)
        path_node_ids.append(int(current.id))
        fcd = current.component_data[ComponentType.FULL]
        mcd = current.component_data[ComponentType.MAMBA]
        if fcd.value is None:
            path_full_all_present = False
        else:
            path_full_rows += len(fcd.value)
            path_full_hasher.update(
                bytes.fromhex(_tensor_sha256(fcd.value))
            )
        if mcd.value is not None:
            path_mamba_positions.append(
                {"position": position, "node_id": int(current.id), "slots": _tensor_ids(mcd.value)}
            )

    fcd = node.component_data[ComponentType.FULL]
    mcd = node.component_data[ComponentType.MAMBA]
    mamba_lru = core.lru_lists[ComponentType.MAMBA]
    return {
        "prefix_tokens": len(token_ids),
        "prefix_sha256": _token_sha256(token_ids),
        "extra_key": extra_key,
        "node_id": int(node.id),
        "parent_node_id": None if node.parent is None else int(node.parent.id),
        "child_node_ids": sorted(int(child.id) for child in node.children.values()),
        "n_children": len(node.children),
        "is_device_leaf": bool(node in core.evictable_device_leaves),
        "segment_lengths": segment_lengths,
        "path_node_ids": path_node_ids,
        "path_full_all_present": path_full_all_present,
        "path_full_rows": int(path_full_rows),
        "path_full_sha256": path_full_hasher.hexdigest(),
        "path_mamba_positions": path_mamba_positions,
        "target_full_present": fcd.value is not None,
        "target_full_value_len": None if fcd.value is None else len(fcd.value),
        "target_full_value_sha256": _tensor_sha256(fcd.value),
        "target_full_lock_ref": int(fcd.lock_ref),
        "target_mamba_present": mcd.value is not None,
        "target_mamba_slots": _tensor_ids(mcd.value),
        "target_mamba_lock_ref": int(mcd.lock_ref),
        "target_mamba_host_lock_ref": int(mcd.host_lock_ref),
        "target_mamba_session_ref": int(mcd.session_ref),
        "target_mamba_host_present": mcd.host_value is not None,
        "target_mamba_in_lru": bool(mamba_lru.in_list(node)),
    }


def _accounting_snapshot(cache) -> dict:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    core = cache.tree_core
    alloc = cache.req_to_token_pool.mamba_allocator
    free_slots = sorted(_tensor_ids(alloc.free_slots))
    return {
        "mamba_allocator_type": type(alloc).__name__,
        "mamba_available": int(alloc.available_size()),
        "mamba_schedulable_available": int(alloc.schedulable_available_size()),
        "mamba_free_slots": free_slots,
        "mamba_free_slots_sha256": hashlib.sha256(
            array("q", free_slots).tobytes()
        ).hexdigest(),
        "mamba_evictable": int(core.component_evictable_size_[ComponentType.MAMBA]),
        "mamba_protected": int(core.component_protected_size_[ComponentType.MAMBA]),
        "full_evictable": int(core.component_evictable_size_[ComponentType.FULL]),
        "full_protected": int(core.component_protected_size_[ComponentType.FULL]),
        "full_allocator": _full_allocator_snapshot(cache),
    }


def _snapshot(cache, token_ids, extra_key) -> dict:
    return {
        "path": _path_snapshot(cache, token_ids, extra_key),
        "accounting": _accounting_snapshot(cache),
        "tree": _global_maps(cache),
    }


def _changed_mamba_nodes(before_rows, after_rows):
    before = {int(node_id): slots for node_id, slots in before_rows}
    after = {int(node_id): slots for node_id, slots in after_rows}
    return sorted(
        node_id
        for node_id in set(before) | set(after)
        if before.get(node_id) != after.get(node_id)
    )


def _validate_runtime_scope(scheduler, cache) -> dict:
    idle = _queue_lengths(scheduler)
    if not idle["scheduler_fully_idle"]:
        raise AssertionError(f"scheduler is not fully idle: {idle}")
    if type(cache).__name__ != "UnifiedRadixCache":
        raise AssertionError(f"unexpected cache type: {type(cache).__name__}")
    core = cache.tree_core
    if getattr(core, "enable_hicache", False):
        raise AssertionError("HiCache must be disabled")
    if getattr(core, "enable_session_radix_cache", False):
        raise AssertionError("session radix cache must be disabled")
    if getattr(cache.req_to_token_pool, "mamba_ckpt_pool", None) is not None:
        raise AssertionError("int8 Mamba checkpoint pool must be disabled")
    if core.has_ongoing_insert():
        raise AssertionError("tree has an ongoing insert")
    server_args = scheduler.server_args
    for name in ("tp_size", "pp_size", "dp_size"):
        value = int(getattr(server_args, name, 1))
        if value != 1:
            raise AssertionError(f"{name}={value}, expected 1")
    if bool(getattr(scheduler, "enable_overlap", False)):
        raise AssertionError("overlap scheduling must be disabled")
    return {
        **idle,
        "cache_type": type(cache).__name__,
        "tree_core_type": type(core).__name__,
        "page_size": int(core.page_size),
        "tp_size": int(getattr(server_args, "tp_size", 1)),
        "pp_size": int(getattr(server_args, "pp_size", 1)),
        "dp_size": int(getattr(server_args, "dp_size", 1)),
        "enable_hicache": bool(getattr(core, "enable_hicache", False)),
        "enable_session_radix_cache": bool(
            getattr(core, "enable_session_radix_cache", False)
        ),
    }


def _run_checkpoint_control(scheduler, request: dict) -> dict:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.components.tree_component import EvictLayer

    cache = scheduler.tree_cache
    scope = _validate_runtime_scope(scheduler, cache)
    action = request.get("action")
    if action not in ("inspect", "evict_mamba"):
        raise AssertionError(f"unsupported action {action!r}")
    token_ids = request.get("token_ids") or []
    extra_key = request.get("extra_key")
    expected_sha = request.get("expected_prefix_sha256")
    actual_sha = _token_sha256(token_ids)
    if expected_sha and expected_sha != actual_sha:
        raise AssertionError(f"prefix digest mismatch: {expected_sha} != {actual_sha}")

    before = _snapshot(cache, token_ids, extra_key)
    path = before["path"]
    if path["node_id"] == int(cache.tree_core.root_node.id):
        raise AssertionError("root cannot be controlled")
    if not path["target_full_present"] or not path["path_full_all_present"]:
        raise AssertionError("exact-prefix Full KV is not fully device resident")
    if action == "evict_mamba":
        if not path["target_mamba_present"]:
            raise AssertionError("target Mamba checkpoint is already absent")
        if len(path["target_mamba_slots"]) != 1:
            raise AssertionError(
                f"expected one Mamba slot: {path['target_mamba_slots']}"
            )
        if path["n_children"] < 1 or path["is_device_leaf"]:
            raise AssertionError(
                "target must be an internal radix node before component eviction"
            )
        if not path["target_mamba_in_lru"]:
            raise AssertionError("target Mamba checkpoint is not in the device LRU")
        if any(
            path[name] != 0
            for name in (
                "target_full_lock_ref",
                "target_mamba_lock_ref",
                "target_mamba_host_lock_ref",
                "target_mamba_session_ref",
            )
        ):
            raise AssertionError(f"target has a live reference: {path}")
        if path["target_mamba_host_present"]:
            raise AssertionError("unexpected host Mamba value")

    mutation = {
        "action": action,
        "device_freed": 0,
        "host_freed": 0,
        "tracker_mamba": 0,
        "tracker_full": 0,
    }
    if action == "evict_mamba":
        core = cache.tree_core
        node = core.node_by_id(path["node_id"])
        mamba = cache.components[ComponentType.MAMBA]
        tracker = {ct: 0 for ct in cache.tree_components}
        device_frees = defaultdict(list)
        host_frees = defaultdict(list)
        try:
            device_freed, host_freed = core._evict_component_and_detach_lru(
                node,
                mamba,
                target=EvictLayer.DEVICE,
                tracker=tracker,
                device_frees=device_frees,
                host_frees=host_frees,
            )
            core._cascade_evict(
                node,
                mamba,
                tracker,
                device_frees=device_frees,
                host_frees=host_frees,
                target=EvictLayer.DEVICE,
            )
        finally:
            cache._free_values(device_frees, host_frees)
        mutation = {
            "action": action,
            "device_freed": int(device_freed),
            "host_freed": int(host_freed),
            "tracker_mamba": int(tracker.get(ComponentType.MAMBA, 0)),
            "tracker_full": int(tracker.get(ComponentType.FULL, 0)),
        }
        cache.sanity_check()

    after = _snapshot(cache, token_ids, extra_key)
    changed_mamba_nodes = _changed_mamba_nodes(
        before["tree"]["mamba_rows"], after["tree"]["mamba_rows"]
    )
    result = {
        "ok": True,
        "nonce": request.get("nonce"),
        "label": request.get("label"),
        "scope": scope,
        "before": before,
        "after": after,
        "mutation": mutation,
        "proof": {
            "prefix_digest_verified": expected_sha in (None, actual_sha),
            "same_node_id": before["path"]["node_id"] == after["path"]["node_id"],
            "structure_unchanged": before["tree"]["structure_sha256"]
            == after["tree"]["structure_sha256"],
            "full_tree_unchanged": before["tree"]["full_tree_sha256"]
            == after["tree"]["full_tree_sha256"],
            "full_path_unchanged": before["path"]["path_full_sha256"]
            == after["path"]["path_full_sha256"],
            "full_allocator_unchanged": before["accounting"]["full_allocator"]
            == after["accounting"]["full_allocator"],
            "changed_mamba_node_ids": changed_mamba_nodes,
            "only_target_mamba_changed": changed_mamba_nodes
            in ([], [before["path"]["node_id"]]),
            "mamba_available_delta": after["accounting"]["mamba_available"]
            - before["accounting"]["mamba_available"],
            "mamba_evictable_delta": after["accounting"]["mamba_evictable"]
            - before["accounting"]["mamba_evictable"],
            "mamba_node_count_delta": after["tree"]["mamba_node_count"]
            - before["tree"]["mamba_node_count"],
            "sanity_check_passed": True,
        },
    }
    print(
        "[FSWP3D] control "
        f"nonce={result['nonce']} label={result['label']} action={action} "
        f"node={path['node_id']} internal={path['n_children'] > 0} "
        f"mamba_before={path['target_mamba_present']} "
        f"mamba_after={after['path']['target_mamba_present']} "
        f"fa_before={path['path_full_all_present']} "
        f"fa_after={after['path']['path_full_all_present']} "
        f"freed={mutation['tracker_mamba']} "
        f"full_freed={mutation['tracker_full']} "
        f"allocator_delta={result['proof']['mamba_available_delta']} "
        f"other_unchanged={result['proof']['only_target_mamba_changed']}",
        flush=True,
    )
    return result


def _run_census(scheduler, request: dict) -> dict:
    cache = scheduler.tree_cache
    return {
        "ok": True,
        "nonce": request.get("nonce"),
        "scope": _validate_runtime_scope(scheduler, cache),
        "accounting": _accounting_snapshot(cache),
        "tree": _global_maps(cache),
    }


class ProbeState:
    def __init__(self):
        self.commands = queue.Queue()
        self.results = {}
        self.pending = {}
        self.lock = threading.Lock()

    def submit(self, request: dict, timeout_s: float = 180.0) -> dict:
        nonce = str(request.get("nonce") or uuid.uuid4())
        request = {**request, "nonce": nonce}
        with self.lock:
            if nonce in self.results:
                return self.results[nonce]
            event = self.pending.get(nonce)
            if event is None:
                event = threading.Event()
                self.pending[nonce] = event
                self.commands.put(request)
        if not event.wait(timeout_s):
            raise TimeoutError(f"scheduler control timed out: {nonce}")
        with self.lock:
            return self.results[nonce]

    def drain_one(self, scheduler) -> bool:
        try:
            request = self.commands.get_nowait()
        except queue.Empty:
            return False
        nonce = request["nonce"]
        try:
            if request.get("op") == "checkpoint_control":
                result = _run_checkpoint_control(scheduler, request)
            elif request.get("op") == "census":
                result = _run_census(scheduler, request)
            else:
                raise AssertionError(f"unknown queued op {request.get('op')!r}")
        except Exception as exc:
            result = {
                "ok": False,
                "nonce": nonce,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        with self.lock:
            self.results[nonce] = result
            event = self.pending.pop(nonce)
            event.set()
        return True


def install_control_server(scheduler, port: int) -> int:
    if getattr(scheduler, "_wp3d_probe_server", None) is not None:
        return port
    state = ProbeState()
    scheduler._wp3d_probe_state = state

    def serve():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(32)
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            with connection:
                stream = connection.makefile("rwb")
                for line in stream:
                    try:
                        request = json.loads(line.decode())
                        if request.get("op") == "ping":
                            response = {
                                "ok": True,
                                "cache_type": type(scheduler.tree_cache).__name__,
                            }
                        elif request.get("op") == "get_result":
                            with state.lock:
                                response = state.results.get(
                                    str(request.get("nonce")),
                                    {"ok": False, "pending_or_unknown": True},
                                )
                        else:
                            response = state.submit(request)
                    except Exception as exc:
                        response = {"ok": False, "error": repr(exc)}
                    stream.write((json.dumps(response, sort_keys=True) + "\n").encode())
                    stream.flush()

    threading.Thread(target=serve, daemon=True).start()
    scheduler._wp3d_probe_server = True
    return port


class ControlClient:
    def __init__(self, port: int, timeout_s: float = 240.0):
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def _call(self, request: dict) -> dict:
        with socket.create_connection(
            ("127.0.0.1", self.port), timeout=self.timeout_s
        ) as sock:
            sock.settimeout(self.timeout_s)
            sock.sendall((json.dumps(request) + "\n").encode())
            line = sock.makefile("rb").readline()
        if not line:
            raise RuntimeError(f"empty probe response for {request.get('op')}")
        response = json.loads(line.decode())
        if not response.get("ok"):
            raise RuntimeError(f"probe operation failed: {response}")
        return response

    def ping(self):
        return self._call({"op": "ping"})

    def census(self, nonce: str):
        return self._call({"op": "census", "nonce": nonce})

    def checkpoint_control(
        self,
        *,
        nonce: str,
        label: str,
        action: str,
        token_ids,
        extra_key=None,
    ):
        return self._call(
            {
                "op": "checkpoint_control",
                "nonce": nonce,
                "label": label,
                "action": action,
                "token_ids": [int(x) for x in token_ids],
                "extra_key": extra_key,
                "expected_prefix_sha256": _token_sha256(token_ids),
            }
        )
