#!/usr/bin/env python3
"""执行一次 WP3B 自动策略到真实 sibling 请求的端到端验证。"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIRECTORY = (
    _REPOSITORY_ROOT
    / "motivation"
    / "artifacts"
    / "wp3b_gate_20260820"
)
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_ARTIFACT_DIRECTORY))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import StateController
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


VOCAB_SIZE = 248_320
PARENT_LENGTH = 32_768
BRANCH_SUFFIX_LENGTH = 63
CHILD_LENGTH = 32_832
CHECKPOINT_SIZE_BYTES = 51_511_296
PARENT_SEEDS = (51_001, 91_003, 131_009, 171_017)
CHILD_A_SEEDS = (201_001, 211_003, 221_009, 231_017)
CHILD_B_SEEDS = tuple(241_019 + index * 10_007 for index in range(4))
SAMPLING_PARAMETERS = {
    "max_new_tokens": 1,
    "temperature": 0,
    "ignore_eos": True,
}
ENGINE_CONFIGURATION = {
    "model_path": "/model",
    "context_length": 45_056,
    "mem_fraction_static": 0.40,
    "tp_size": 1,
    "chunked_prefill_size": 45_056,
    "max_mamba_cache_size": 16,
    "mamba_radix_cache_strategy": "extra_buffer",
    "disable_cuda_graph": True,
    "disable_overlap_schedule": True,
    "enable_request_time_stats_logging": True,
    "stream_interval": 1,
    "log_level": "info",
}


@dataclass
class WorkflowRuntime:
    """保存单个 workflow 在真实运行时中构造出的请求材料。"""

    workflow_id: str
    index: int
    parent: tuple[int, ...]
    parent_output: int
    child_a: tuple[int, ...]
    child_a_suffix_first: int


class SchedulerRuntimeAdapter:
    """把 controller 动作投递到 scheduler 内的正式适配器。"""

    def __init__(self, client: ControlClient) -> None:
        self._client = client
        self.evicted_checkpoint_ids: list[str] = []
        self.eviction_responses: list[dict] = []

    def evict_mamba_only(
        self,
        handle: RuntimeCheckpointHandle,
    ) -> None:
        """在调度器安全时点调用正式 Mamba-only 驱逐接口。"""
        response = self._client._call(
            {
                "op": "checkpoint_control",
                "nonce": f"flowstate_step5d:evict:{handle.checkpoint_id}",
                "label": f"flowstate_step5d:{handle.checkpoint_id}",
                "action": "flowstate_evict_mamba_only",
                "checkpoint_id": handle.checkpoint_id,
                "token_ids": list(handle.token_ids),
                "extra_key": handle.extra_key,
                "expected_node_id": handle.expected_node_id,
                "expected_prefix_sha256": handle.expected_prefix_digest,
            }
        )
        self.evicted_checkpoint_ids.append(handle.checkpoint_id)
        self.eviction_responses.append(response)


def make_tokens(seed: int, count: int) -> tuple[int, ...]:
    """构造确定且彼此易于区分的令牌序列。"""
    return tuple(
        (seed + index * 7_919) % VOCAB_SIZE
        for index in range(count)
    )


def token_digest(token_ids: tuple[int, ...]) -> str:
    """计算正式运行时句柄使用的令牌摘要。"""
    return hashlib.sha256(array("q", token_ids).tobytes()).hexdigest()


def wait_for_transport(
    client: ControlClient,
    timeout_seconds: float = 300.0,
) -> None:
    """等待 scheduler 测试传输层完成初始化。"""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if client.ping().get("ok"):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("等待 Step 5D 测试传输层超时")


def generate(
    engine: FormalEndToEndGateEngine,
    request_id: str,
    token_ids: tuple[int, ...],
) -> tuple[int, dict]:
    """发送一次确定性请求，并返回生成令牌与服务端元数据。"""
    result = engine.generate(
        input_ids=list(token_ids),
        sampling_params=SAMPLING_PARAMETERS,
        rid=request_id,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"请求 {request_id} 未返回对象")
    output_ids = result.get("output_ids") or []
    metadata = result.get("meta_info") or {}
    if len(output_ids) != 1:
        raise RuntimeError(
            f"请求 {request_id} 的输出令牌数量异常：{output_ids}"
        )
    if int(metadata.get("completion_tokens", 1)) != 1:
        raise RuntimeError(f"请求 {request_id} 的完成长度异常")
    if int(metadata.get("num_retractions", 0) or 0) != 0:
        raise RuntimeError(f"请求 {request_id} 发生了意外回撤")
    return int(output_ids[0]), metadata


def inspect_checkpoint(
    client: ControlClient,
    checkpoint_id: str,
    token_ids: tuple[int, ...],
) -> dict:
    """在 scheduler 空闲时点读取精确前缀节点状态。"""
    return client.checkpoint_control(
        nonce=f"flowstate_step5d:inspect:{checkpoint_id}:{time.monotonic_ns()}",
        label=f"flowstate_step5d:{checkpoint_id}",
        action="inspect",
        token_ids=token_ids,
    )


def query_runtime_metrics(
    client: ControlClient,
    request_id: str,
) -> dict:
    """读取请求匹配阶段记录的物理命中与可执行前缀。"""
    response = client._call(
        {
            "op": "checkpoint_control",
            "nonce": f"flowstate_step5d:metrics:{request_id}",
            "action": "flowstate_runtime_metrics",
            "request_id": request_id,
        }
    )
    return response["metrics"]


def path_state(response: dict) -> dict:
    """取得 inspect 响应中的目标路径状态。"""
    return response["after"]["path"]


def compact_state(path: dict) -> dict:
    """保留结果报告需要的驻留与节点字段。"""
    return {
        "node_id": int(path["node_id"]),
        "fa_resident": bool(path["target_full_present"]),
        "mamba_resident": bool(path["target_mamba_present"]),
    }


def mamba_rows_by_node(snapshot: dict) -> dict[int, object]:
    """把全树 Mamba 行转换为按节点标识索引的映射。"""
    return {
        int(node_id): slots
        for node_id, slots in snapshot["tree"]["mamba_rows"]
    }


def changed_mamba_nodes(before: dict, after: dict) -> set[int]:
    """返回两个全树快照之间发生变化的 Mamba 节点。"""
    before_rows = mamba_rows_by_node(before)
    after_rows = mamba_rows_by_node(after)
    node_ids = set(before_rows) | set(after_rows)
    return {
        node_id
        for node_id in node_ids
        if before_rows.get(node_id) != after_rows.get(node_id)
    }


def optional_latency_ms(metadata: dict, field: str) -> float | None:
    """读取可选的秒级延迟，并在可用时转换为毫秒。"""
    value = metadata.get(field)
    if value is None:
        return None
    try:
        latency_ms = float(value) * 1_000.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        return None
    return latency_ms


def validate_sibling_observation(
    workflow_id: str,
    metrics: dict,
    metadata: dict,
) -> dict:
    """强制验证可执行状态，并附加不影响结论的时间观测。"""
    physical_hit = int(metrics["physical_fa_hit"])
    executable_prefix = int(metrics["executable_prefix"])
    gap = int(metrics["replay_gap"])
    if physical_hit != 32_769:
        raise RuntimeError(f"{workflow_id} 的物理 FA 命中不是 32769")
    if executable_prefix != PARENT_LENGTH:
        raise RuntimeError(f"{workflow_id} 的可执行前缀不是 32768")
    if gap != 1:
        raise RuntimeError(f"{workflow_id} 的真实恢复间隔不是 1")
    return {
        "physical_hit": physical_hit,
        "executable_prefix": executable_prefix,
        "gap": gap,
        "ttft_ms": optional_latency_ms(
            metadata,
            "first_token_latency",
        ),
        "request_e2e_ms": optional_latency_ms(metadata, "e2e_latency"),
    }


def build_workflows(
    engine: FormalEndToEndGateEngine,
) -> tuple[WorkflowRuntime, ...]:
    """在真实运行时中构造四个独立 Parent 与 Child-A。"""
    parent_material = []
    first_tokens = set()
    for index, seed in enumerate(PARENT_SEEDS, start=1):
        parent = make_tokens(seed, PARENT_LENGTH)
        if parent[0] in first_tokens:
            raise RuntimeError("不同 workflow 的首令牌发生碰撞")
        first_tokens.add(parent[0])
        parent_output, _ = generate(
            engine,
            f"flowstate_step5d_w{index}_parent",
            parent,
        )
        parent_material.append((index, parent, parent_output))
        print(f"[STEP5D-GATE] W{index} Parent 已构造", flush=True)

    workflows = []
    for index, parent, parent_output in parent_material:
        suffix = list(make_tokens(CHILD_A_SEEDS[index - 1], BRANCH_SUFFIX_LENGTH))
        if suffix[0] == parent_output:
            suffix[0] = (suffix[0] + 1) % VOCAB_SIZE
        child_a = parent + (parent_output,) + tuple(suffix)
        if len(child_a) != CHILD_LENGTH:
            raise RuntimeError(f"W{index} 的 Child-A 长度异常")
        generate(
            engine,
            f"flowstate_step5d_w{index}_child_a",
            child_a,
        )
        workflows.append(
            WorkflowRuntime(
                workflow_id=f"W{index}",
                index=index,
                parent=parent,
                parent_output=parent_output,
                child_a=child_a,
                child_a_suffix_first=int(suffix[0]),
            )
        )
        print(f"[STEP5D-GATE] W{index} Child-A 已构造", flush=True)
    return tuple(workflows)


def build_logical_inputs(
    client: ControlClient,
    workflows: tuple[WorkflowRuntime, ...],
) -> tuple[
    tuple[PendingContinuation, ...],
    tuple[CheckpointCandidate, ...],
    dict[str, RuntimeCheckpointHandle],
    dict[str, dict],
    dict[str, dict],
]:
    """把 workflow metadata 与真实运行时句柄分别构造。"""
    continuations = []
    parents = []
    children = []
    handles = {}
    before_paths = {}
    before_states = {}

    for workflow in workflows:
        parent_id = f"P{workflow.index}"
        child_id = f"C{workflow.index}"
        parent_response = inspect_checkpoint(
            client,
            parent_id,
            workflow.parent,
        )
        child_response = inspect_checkpoint(
            client,
            child_id,
            workflow.child_a,
        )
        for checkpoint_id, token_ids, response in (
            (parent_id, workflow.parent, parent_response),
            (child_id, workflow.child_a, child_response),
        ):
            path = path_state(response)
            if path["prefix_sha256"] != token_digest(token_ids):
                raise RuntimeError(f"{checkpoint_id} 的前缀摘要不一致")
            if not path["target_full_present"]:
                raise RuntimeError(f"{checkpoint_id} 的 FA-KV 未驻留")
            if not path["target_mamba_present"]:
                raise RuntimeError(f"{checkpoint_id} 的 Mamba 状态未驻留")
            handles[checkpoint_id] = RuntimeCheckpointHandle(
                checkpoint_id=checkpoint_id,
                token_ids=token_ids,
                expected_node_id=int(path["node_id"]),
                expected_prefix_digest=str(path["prefix_sha256"]),
            )
            before_paths[checkpoint_id] = path
            before_states[checkpoint_id] = compact_state(path)

        continuations.append(
            PendingContinuation(
                continuation_id=f"B{workflow.index}",
                workflow_id=workflow.workflow_id,
                lineage_path=("P", "B"),
                anchor_pos=PARENT_LENGTH,
                resident_fa_frontier=PARENT_LENGTH,
            )
        )
        parents.append(
            CheckpointCandidate(
                checkpoint_id=parent_id,
                workflow_id=workflow.workflow_id,
                lineage_path=("P",),
                token_pos=PARENT_LENGTH,
                memory_bytes=CHECKPOINT_SIZE_BYTES,
            )
        )
        children.append(
            CheckpointCandidate(
                checkpoint_id=child_id,
                workflow_id=workflow.workflow_id,
                lineage_path=("P", "A"),
                token_pos=CHILD_LENGTH,
                memory_bytes=CHECKPOINT_SIZE_BYTES,
            )
        )

    return (
        tuple(continuations),
        tuple(children + parents),
        handles,
        before_paths,
        before_states,
    )


def inspect_after_allocation(
    client: ControlClient,
    workflows: tuple[WorkflowRuntime, ...],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """读取 allocation 后八个检查点的精确状态。"""
    paths = {}
    states = {}
    for workflow in workflows:
        for checkpoint_id, token_ids in (
            (f"P{workflow.index}", workflow.parent),
            (f"C{workflow.index}", workflow.child_a),
        ):
            path = path_state(
                inspect_checkpoint(client, checkpoint_id, token_ids)
            )
            paths[checkpoint_id] = path
            states[checkpoint_id] = compact_state(path)
    return paths, states


def validate_siblings(
    engine: FormalEndToEndGateEngine,
    client: ControlClient,
    workflows: tuple[WorkflowRuntime, ...],
) -> dict[str, dict]:
    """发送四个 sibling continuation 并验证真实可执行状态。"""
    results = {}
    for workflow in workflows:
        suffix = list(
            make_tokens(
                CHILD_B_SEEDS[workflow.index - 1],
                BRANCH_SUFFIX_LENGTH,
            )
        )
        forbidden = {
            workflow.child_a_suffix_first,
            workflow.parent_output,
        }
        while suffix[0] in forbidden:
            suffix[0] = (suffix[0] + 1) % VOCAB_SIZE
        child_b = (
            workflow.parent
            + (workflow.parent_output,)
            + tuple(suffix)
        )
        request_id = f"flowstate_step5d_w{workflow.index}_child_b"
        _, metadata = generate(engine, request_id, child_b)
        metrics = query_runtime_metrics(client, request_id)
        results[workflow.workflow_id] = validate_sibling_observation(
            workflow.workflow_id,
            metrics,
            metadata,
        )
        print(
            f"[STEP5D-GATE] {workflow.workflow_id} sibling 验证通过",
            flush=True,
        )
    return results


def main() -> int:
    """执行一次完整的自动选择、真实驱逐与 sibling 验证。"""
    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import (
        FormalEndToEndGateEngine,
        requested_control_port,
    )

    engine = None
    stage = "初始化"
    report: dict[str, object] = {
        "status": "FAIL",
        "failure_stage": stage,
        "decision_hardcoded": False,
    }
    try:
        checkpoint_size_value = 49.125 * 1024 * 1024
        if (
            not checkpoint_size_value.is_integer()
            or int(checkpoint_size_value) != CHECKPOINT_SIZE_BYTES
        ):
            raise RuntimeError("检查点字节数计算不精确")

        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        engine.flush_cache()

        stage = "构造 Parent 与 Child-A"
        workflows = build_workflows(engine)

        stage = "构造逻辑候选与运行时句柄"
        (
            continuations,
            candidates,
            handles,
            before_paths,
            before_states,
        ) = build_logical_inputs(client, workflows)
        report["before_allocation"] = before_states

        stage = "GlobalOptimizer 与 StateController"
        runtime_adapter = SchedulerRuntimeAdapter(client)
        controller = StateController(
            GlobalOptimizer(RecoveryCostModel()),
            runtime_adapter,
        )
        allocation = controller.reconcile(
            continuations,
            candidates,
            handles,
            4 * CHECKPOINT_SIZE_BYTES,
        )
        selected_ids = tuple(
            candidate.checkpoint_id
            for candidate in allocation.selected
        )
        evicted_ids = tuple(runtime_adapter.evicted_checkpoint_ids)
        expected_selected = tuple(f"P{index}" for index in range(1, 5))
        expected_evicted = tuple(f"C{index}" for index in range(1, 5))
        if selected_ids != expected_selected:
            raise RuntimeError(f"optimizer 选择异常：{selected_ids}")
        if evicted_ids != expected_evicted:
            raise RuntimeError(f"controller 驱逐异常：{evicted_ids}")
        if len(runtime_adapter.eviction_responses) != 4:
            raise RuntimeError("正式 mutation 次数不是四次")
        report["optimizer_selected_ids"] = selected_ids
        report["controller_evicted_ids"] = evicted_ids

        stage = "allocation 后状态与全局安全校验"
        after_paths, after_states = inspect_after_allocation(
            client,
            workflows,
        )
        report["after_allocation"] = after_states

        responses = runtime_adapter.eviction_responses
        global_before = responses[0]["before"]
        global_after = responses[-1]["after"]
        allocator_before = global_before["accounting"]["full_allocator"]
        allocator_after = global_after["accounting"]["full_allocator"]
        available_before = int(allocator_before["available"])
        available_after = int(allocator_after["available"])
        allocator_unchanged = available_before == available_after

        path_unchanged = all(
            before_paths[checkpoint_id]["path_node_ids"]
            == after_paths[checkpoint_id]["path_node_ids"]
            and before_paths[checkpoint_id]["segment_lengths"]
            == after_paths[checkpoint_id]["segment_lengths"]
            and before_paths[checkpoint_id]["prefix_sha256"]
            == after_paths[checkpoint_id]["prefix_sha256"]
            for checkpoint_id in before_paths
        )
        tree_structure_unchanged = (
            global_before["tree"]["structure_sha256"]
            == global_after["tree"]["structure_sha256"]
        )
        full_tree_unchanged = (
            global_before["tree"]["full_tree_sha256"]
            == global_after["tree"]["full_tree_sha256"]
        )
        parent_mamba_preserved = all(
            after_states[f"P{index}"]["mamba_resident"]
            for index in range(1, 5)
        )
        child_mamba_removed = all(
            not after_states[f"C{index}"]["mamba_resident"]
            for index in range(1, 5)
        )
        fa_preserved = all(
            state["fa_resident"] for state in after_states.values()
        ) and full_tree_unchanged
        sanity_check = all(
            response["proof"]["sanity_check"]
            for response in responses
        )
        fa_identity_unchanged = all(
            response["proof"]["fa_identity_unchanged"]
            for response in responses
        )
        cascade_called = any(
            response["proof"]["cascade_called"]
            for response in responses
        )
        expected_changed_nodes = {
            int(before_paths[f"C{index}"]["node_id"])
            for index in range(1, 5)
        }
        only_expected_mamba_changed = (
            changed_mamba_nodes(global_before, global_after)
            == expected_changed_nodes
        )

        safety = {
            "tree_structure_unchanged": tree_structure_unchanged,
            "fa_preserved": fa_preserved,
            "parent_mamba_preserved": parent_mamba_preserved,
            "child_mamba_removed": child_mamba_removed,
            "path_unchanged": path_unchanged,
            "allocator_unchanged": allocator_unchanged,
            "fa_identity_unchanged": fa_identity_unchanged,
            "only_expected_mamba_changed": only_expected_mamba_changed,
            "sanity_check": sanity_check,
            "cascade_called": cascade_called,
        }
        report["fa_allocator"] = {
            "before_available_size": available_before,
            "after_available_size": available_after,
            "unchanged": allocator_unchanged,
        }
        report["safety"] = safety
        if not all(
            (
                tree_structure_unchanged,
                fa_preserved,
                parent_mamba_preserved,
                child_mamba_removed,
                path_unchanged,
                allocator_unchanged,
                fa_identity_unchanged,
                only_expected_mamba_changed,
                sanity_check,
                not cascade_called,
            )
        ):
            raise RuntimeError(f"allocation 后安全条件失败：{safety}")

        formal_primitives = {
            response["formal_primitive"] for response in responses
        }
        expected_primitive = (
            "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
        )
        if formal_primitives != {expected_primitive}:
            raise RuntimeError(
                f"正式 mutation primitive 异常：{formal_primitives}"
            )
        report["formal_mutation_primitive"] = expected_primitive

        stage = "真实 Child-B sibling 请求"
        sibling_results = validate_siblings(
            engine,
            client,
            workflows,
        )

        report.update(
            {
                "status": "PASS",
                "failure_stage": None,
                "sibling_runtime_validation": sibling_results,
            }
        )
        print(
            "[STEP5D-GATE] RESULT="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )
        return 0
    except Exception as error:
        if "runtime_adapter" in locals():
            report["controller_evicted_ids"] = tuple(
                runtime_adapter.evicted_checkpoint_ids
            )
        if "allocation" in locals():
            report["optimizer_selected_ids"] = tuple(
                candidate.checkpoint_id
                for candidate in allocation.selected
            )
        report["status"] = "FAIL"
        report["failure_stage"] = stage
        report["error"] = repr(error)
        traceback.print_exc()
        print(
            "[STEP5D-GATE] RESULT="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
