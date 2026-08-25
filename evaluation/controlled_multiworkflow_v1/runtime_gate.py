#!/usr/bin/env python3
"""执行一次受控多工作流真实运行时可行性验证。"""

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
_TEST_RUNTIME_DIRECTORY = _REPOSITORY_ROOT / "tests" / "runtime"
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_ARTIFACT_DIRECTORY))
sys.path.insert(0, str(_TEST_RUNTIME_DIRECTORY))

from evaluation.controlled_multiworkflow_v1.scenario import (
    ControlledScenario,
    WorkflowSpec,
    build_scenario,
)
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import StateController
from flowstate.executable_state import recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel


VOCAB_SIZE = 248_320
PENDING_SUFFIX_LENGTH = 63
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
    "mamba_track_interval": 256,
    "mamba_max_states_per_path": -1,
    "disable_cuda_graph": True,
    "disable_overlap_schedule": True,
    "enable_request_time_stats_logging": True,
    "stream_interval": 1,
    "log_level": "info",
}


class RuntimeRepresentationMismatch(RuntimeError):
    """表示逻辑候选无法映射为稳定且独立的真实检查点。"""


@dataclass
class RuntimeWorkflow:
    """保存一个 workflow 在真实运行时中构造出的前缀材料。"""

    spec: WorkflowSpec
    anchor_tokens: tuple[int, ...]
    anchor_output: int
    candidate_tokens: dict[str, tuple[int, ...]]


class SchedulerRuntimeAdapter:
    """把 Controller 的驱逐动作投递到 scheduler 安全时点。"""

    def __init__(self, client: ControlClient) -> None:
        self._client = client
        self.evicted_checkpoint_ids: list[str] = []
        self.eviction_responses: list[dict] = []

    def evict_mamba_only(
        self,
        handle: RuntimeCheckpointHandle,
    ) -> None:
        """通过测试传输层调用正式 SGLangAdapter 驱逐接口。"""
        response = self._client._call(
            {
                "op": "checkpoint_control",
                "nonce": f"flowstate_step7b:evict:{handle.checkpoint_id}",
                "label": f"flowstate_step7b:{handle.checkpoint_id}",
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
    """构造确定且不依赖分词器的令牌序列。"""
    return tuple(
        (seed + index * 7_919) % VOCAB_SIZE
        for index in range(count)
    )


def token_digest(token_ids: tuple[int, ...]) -> str:
    """计算正式运行时句柄使用的令牌摘要。"""
    return hashlib.sha256(array("q", token_ids).tobytes()).hexdigest()


def optional_latency_ms(metadata: dict, field: str) -> float | None:
    """读取可选秒级延迟，并在有效时转换为毫秒。"""
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


def wait_for_transport(
    client: ControlClient,
    timeout_seconds: float = 300.0,
) -> None:
    """等待 scheduler 内的测试传输层完成初始化。"""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if client.ping().get("ok"):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("等待受控多工作流测试传输层超时")


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
        raise RuntimeError(f"请求 {request_id} 的输出令牌数量异常")
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
    """在 scheduler 安全时点读取精确前缀节点状态。"""
    return client.checkpoint_control(
        nonce=(
            f"flowstate_step7b:inspect:{checkpoint_id}:"
            f"{time.monotonic_ns()}"
        ),
        label=f"flowstate_step7b:{checkpoint_id}",
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
            "nonce": f"flowstate_step7b:metrics:{request_id}",
            "action": "flowstate_runtime_metrics",
            "request_id": request_id,
        }
    )
    return response["metrics"]


def path_state(response: dict) -> dict:
    """取得 inspect 响应中的精确目标路径。"""
    return response["after"]["path"]


def compact_state(path: dict) -> dict:
    """提取结果报告需要的节点与驻留字段。"""
    return {
        "node_id": int(path["node_id"]),
        "token_pos": int(path["prefix_tokens"]),
        "fa_resident": bool(path["target_full_present"]),
        "mamba_resident": bool(path["target_mamba_present"]),
    }


def build_runtime_workflows(
    engine: FormalEndToEndGateEngine,
    scenario: ControlledScenario,
) -> tuple[tuple[RuntimeWorkflow, ...], dict[str, tuple[int, ...]]]:
    """按 scenario 候选位置递增构造真实同 lineage 检查点。"""
    candidates_by_workflow = {
        workflow.workflow_id: sorted(
            (
                candidate
                for candidate in scenario.candidates
                if candidate.workflow_id == workflow.workflow_id
            ),
            key=lambda candidate: (
                candidate.token_pos,
                candidate.checkpoint_id,
            ),
        )
        for workflow in scenario.metadata.workflows
    }
    first_tokens = set()
    runtime_workflows = []
    all_candidate_tokens = {}

    for workflow_index, workflow in enumerate(scenario.metadata.workflows):
        candidates = candidates_by_workflow[workflow.workflow_id]
        if not candidates:
            raise RuntimeError(f"{workflow.workflow_id} 没有候选检查点")
        if candidates[-1].token_pos != workflow.anchor_pos:
            raise RuntimeError(
                f"{workflow.workflow_id} 的最深候选与 anchor 不一致"
            )

        first_candidate = candidates[0]
        current_tokens = make_tokens(
            51_001 + workflow_index * 40_003,
            first_candidate.token_pos,
        )
        if current_tokens[0] in first_tokens:
            raise RuntimeError("不同 workflow 的首令牌发生碰撞")
        first_tokens.add(current_tokens[0])

        current_output, _ = generate(
            engine,
            (
                f"flowstate_step7b_{workflow.workflow_id.lower()}_"
                f"{first_candidate.checkpoint_id.lower()}"
            ),
            current_tokens,
        )
        candidate_tokens = {
            first_candidate.checkpoint_id: current_tokens,
        }

        for depth, candidate in enumerate(candidates[1:], start=1):
            extension_length = (
                candidate.token_pos - len(current_tokens) - 1
            )
            if extension_length < 0:
                raise RuntimeError(
                    f"{candidate.checkpoint_id} 无法沿前一候选继续延伸"
                )
            extension = make_tokens(
                201_001 + workflow_index * 20_003 + depth * 10_007,
                extension_length,
            )
            current_tokens = (
                current_tokens + (current_output,) + extension
            )
            if len(current_tokens) != candidate.token_pos:
                raise RuntimeError(
                    f"{candidate.checkpoint_id} 的运行时前缀长度异常"
                )
            current_output, _ = generate(
                engine,
                (
                    f"flowstate_step7b_{workflow.workflow_id.lower()}_"
                    f"{candidate.checkpoint_id.lower()}"
                ),
                current_tokens,
            )
            candidate_tokens[candidate.checkpoint_id] = current_tokens

        all_candidate_tokens.update(candidate_tokens)
        runtime_workflows.append(
            RuntimeWorkflow(
                spec=workflow,
                anchor_tokens=current_tokens,
                anchor_output=current_output,
                candidate_tokens=candidate_tokens,
            )
        )
        print(
            f"[STEP7B-GATE] {workflow.workflow_id} checkpoints 已构造",
            flush=True,
        )

    if len(first_tokens) != len(scenario.metadata.workflows):
        raise RuntimeError("workflow token namespace 未完全隔离")
    return tuple(runtime_workflows), all_candidate_tokens


def build_runtime_handles(
    client: ControlClient,
    scenario: ControlledScenario,
    candidate_tokens: dict[str, tuple[int, ...]],
) -> tuple[
    dict[str, RuntimeCheckpointHandle],
    dict[str, dict],
    dict[str, dict],
]:
    """把 scenario 候选与真实 exact node 显式映射。"""
    handles = {}
    paths = {}
    states = {}
    candidates_by_workflow: dict[str, list] = {}

    for candidate in scenario.candidates:
        token_ids = candidate_tokens.get(candidate.checkpoint_id)
        if token_ids is None:
            raise RuntimeRepresentationMismatch(
                f"{candidate.checkpoint_id} 缺少真实 token 前缀"
            )
        if len(token_ids) != candidate.token_pos:
            raise RuntimeRepresentationMismatch(
                f"{candidate.checkpoint_id} 的逻辑位置与真实长度不一致"
            )
        try:
            response = inspect_checkpoint(
                client,
                candidate.checkpoint_id,
                token_ids,
            )
        except Exception as error:
            raise RuntimeRepresentationMismatch(
                f"{candidate.checkpoint_id} 无法定位为 exact runtime node：{error}"
            ) from error

        path = path_state(response)
        state = compact_state(path)
        if path["prefix_sha256"] != token_digest(token_ids):
            raise RuntimeRepresentationMismatch(
                f"{candidate.checkpoint_id} 的真实前缀摘要不一致"
            )
        if not state["fa_resident"] or not state["mamba_resident"]:
            raise RuntimeRepresentationMismatch(
                f"{candidate.checkpoint_id} 未同时保持 FA 与 Mamba 驻留："
                f"{state}"
            )

        handles[candidate.checkpoint_id] = RuntimeCheckpointHandle(
            checkpoint_id=candidate.checkpoint_id,
            token_ids=token_ids,
            expected_node_id=state["node_id"],
            expected_prefix_digest=str(path["prefix_sha256"]),
        )
        paths[candidate.checkpoint_id] = path
        states[candidate.checkpoint_id] = state
        candidates_by_workflow.setdefault(candidate.workflow_id, []).append(
            candidate
        )

    for workflow_id, workflow_candidates in candidates_by_workflow.items():
        ordered = sorted(
            workflow_candidates,
            key=lambda candidate: candidate.token_pos,
        )
        for shallow, deep in zip(ordered, ordered[1:]):
            shallow_node_id = int(paths[shallow.checkpoint_id]["node_id"])
            deep_path = tuple(
                int(node_id)
                for node_id in paths[deep.checkpoint_id]["path_node_ids"]
            )
            if (
                shallow_node_id == int(paths[deep.checkpoint_id]["node_id"])
                or shallow_node_id not in deep_path
            ):
                raise RuntimeRepresentationMismatch(
                    f"{workflow_id} 的同 lineage 候选无法独立控制："
                    f"{shallow.checkpoint_id} node={shallow_node_id}，"
                    f"{deep.checkpoint_id} path={deep_path}"
                )

    return handles, paths, states


def inspect_after_allocation(
    client: ControlClient,
    candidate_tokens: dict[str, tuple[int, ...]],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """读取 allocation 后所有候选的真实状态。"""
    paths = {}
    states = {}
    for checkpoint_id, token_ids in candidate_tokens.items():
        path = path_state(
            inspect_checkpoint(client, checkpoint_id, token_ids)
        )
        paths[checkpoint_id] = path
        states[checkpoint_id] = compact_state(path)
    return paths, states


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


def validate_runtime_observation(
    *,
    anchor_pos: int,
    planning_gap: int,
    metrics: dict,
) -> dict:
    """按相对 anchor 关系验证运行时前缀，不预设 exact hit。"""
    physical_hit = int(metrics["physical_fa_hit"])
    executable_prefix = int(metrics["executable_prefix"])
    actual_gap = int(metrics["replay_gap"])

    if actual_gap != physical_hit - executable_prefix:
        raise RuntimeError("运行时 recovery gap 与 H-E 不一致")
    if actual_gap < 0 or executable_prefix < 0:
        raise RuntimeError("运行时前缀指标不能为负")
    if abs(physical_hit - anchor_pos) > 1:
        raise RuntimeError(
            f"物理命中未落在 anchor 边界附近：H={physical_hit}，"
            f"anchor={anchor_pos}"
        )

    if planning_gap == 0:
        if abs(executable_prefix - anchor_pos) > 1 or actual_gap > 1:
            raise RuntimeError(
                "保留检查点未把 executable frontier 推进到 anchor 附近："
                f"E={executable_prefix}，G={actual_gap}，anchor={anchor_pos}"
            )
    else:
        if abs(actual_gap - planning_gap) > 1 or executable_prefix > 1:
            raise RuntimeError(
                "已逐出检查点未产生预期深度的恢复间隔："
                f"E={executable_prefix}，G={actual_gap}，"
                f"planning_gap={planning_gap}"
            )

    return {
        "physical_hit": physical_hit,
        "executable_prefix": executable_prefix,
        "recovery_gap": actual_gap,
    }


def aggregate_observations(observations: dict[str, dict]) -> dict:
    """汇总所有 pending continuation 的 H、E、G 与前缀比例。"""
    physical_tokens = sum(
        int(observation["physical_hit"])
        for observation in observations.values()
    )
    executable_tokens = sum(
        int(observation["executable_prefix"])
        for observation in observations.values()
    )
    recovery_gap_tokens = sum(
        int(observation["recovery_gap"])
        for observation in observations.values()
    )
    ratio = (
        executable_tokens / physical_tokens
        if physical_tokens > 0
        else 0.0
    )
    return {
        "physical_hit_tokens": physical_tokens,
        "executable_hit_tokens": executable_tokens,
        "recovery_gap_tokens": recovery_gap_tokens,
        "executable_prefix_ratio": ratio,
    }


def send_pending_continuations(
    engine: FormalEndToEndGateEngine,
    client: ControlClient,
    scenario: ControlledScenario,
    runtime_workflows: tuple[RuntimeWorkflow, ...],
    selected: tuple,
) -> dict[str, dict]:
    """发送全部七个待续分支并验证真实恢复间隔。"""
    runtime_by_workflow = {
        workflow.spec.workflow_id: workflow
        for workflow in runtime_workflows
    }
    metadata_by_workflow = {
        workflow.workflow_id: workflow
        for workflow in scenario.metadata.workflows
    }
    used_branch_tokens: dict[str, set[int]] = {}
    observations = {}

    for request_index, continuation in enumerate(scenario.continuations):
        runtime_workflow = runtime_by_workflow[continuation.workflow_id]
        workflow_metadata = metadata_by_workflow[continuation.workflow_id]
        suffix = list(
            make_tokens(
                241_019 + request_index * 10_007,
                PENDING_SUFFIX_LENGTH,
            )
        )
        used = used_branch_tokens.setdefault(continuation.workflow_id, set())
        while suffix[0] in used:
            suffix[0] = (suffix[0] + 1) % VOCAB_SIZE
        used.add(suffix[0])

        pending_tokens = (
            runtime_workflow.anchor_tokens
            + (runtime_workflow.anchor_output,)
            + tuple(suffix)
        )
        request_id = (
            "flowstate_step7b_pending_"
            + continuation.continuation_id.lower()
        )
        _, metadata = generate(engine, request_id, pending_tokens)
        metrics = query_runtime_metrics(client, request_id)
        planning_gap = recovery_gap(continuation, selected)
        observation = validate_runtime_observation(
            anchor_pos=continuation.planning_target,
            planning_gap=planning_gap,
            metrics=metrics,
        )
        display_id = (
            continuation.workflow_id
            if workflow_metadata.pending_fanout == 1
            else continuation.continuation_id
        )
        observations[display_id] = {
            "continuation_id": continuation.continuation_id,
            "workflow_id": continuation.workflow_id,
            **observation,
            "planning_gap": planning_gap,
            "request_e2e_ms": optional_latency_ms(
                metadata,
                "e2e_latency",
            ),
            "ttft_ms": optional_latency_ms(
                metadata,
                "first_token_latency",
            ),
        }
        print(
            f"[STEP7B-GATE] {display_id} runtime 前缀验证通过",
            flush=True,
        )

    return observations


def main() -> int:
    """执行一次逻辑 scenario 到真实 runtime 的完整闭环。"""
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
        "runtime_representation_mismatch": None,
    }
    try:
        scenario = build_scenario()
        if scenario.budget_bytes != 154_533_888:
            raise RuntimeError("scenario 的共享预算不是 154533888 bytes")

        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        engine.flush_cache()

        stage = "构造真实多工作流 checkpoints"
        runtime_workflows, candidate_tokens = build_runtime_workflows(
            engine,
            scenario,
        )

        stage = "验证 runtime representation"
        handles, before_paths, before_states = build_runtime_handles(
            client,
            scenario,
            candidate_tokens,
        )
        report["runtime_checkpoints_before_allocation"] = before_states

        stage = "GlobalOptimizer 与 StateController"
        runtime_adapter = SchedulerRuntimeAdapter(client)
        controller = StateController(
            GlobalOptimizer(RecoveryCostModel()),
            runtime_adapter,
        )
        allocation = controller.reconcile(
            scenario.continuations,
            scenario.candidates,
            handles,
            scenario.budget_bytes,
        )
        selected_ids = tuple(
            candidate.checkpoint_id
            for candidate in allocation.selected
        )
        evicted_ids = tuple(runtime_adapter.evicted_checkpoint_ids)
        expected_selected = (
            "W1_PARENT",
            "W2_PARENT",
            "W4_PARENT",
        )
        expected_evicted = ("W1_SHALLOW", "W3_PARENT")
        if selected_ids != expected_selected:
            raise RuntimeError(f"optimizer 选择异常：{selected_ids}")
        if evicted_ids != expected_evicted:
            raise RuntimeError(f"controller 驱逐异常：{evicted_ids}")
        report["optimizer_selected"] = selected_ids
        report["controller_evicted"] = evicted_ids

        stage = "allocation 后状态与安全校验"
        after_paths, after_states = inspect_after_allocation(
            client,
            candidate_tokens,
        )
        report["runtime_checkpoints_after_allocation"] = after_states

        responses = runtime_adapter.eviction_responses
        if len(responses) != len(evicted_ids):
            raise RuntimeError("正式 runtime mutation 数量异常")
        global_before = responses[0]["before"]
        global_after = responses[-1]["after"]
        allocator_before = int(
            global_before["accounting"]["full_allocator"]["available"]
        )
        allocator_after = int(
            global_after["accounting"]["full_allocator"]["available"]
        )
        allocator_unchanged = allocator_before == allocator_after
        tree_unchanged = (
            global_before["tree"]["structure_sha256"]
            == global_after["tree"]["structure_sha256"]
        )
        full_tree_unchanged = (
            global_before["tree"]["full_tree_sha256"]
            == global_after["tree"]["full_tree_sha256"]
        )
        path_unchanged = all(
            before_paths[checkpoint_id]["path_node_ids"]
            == after_paths[checkpoint_id]["path_node_ids"]
            and before_paths[checkpoint_id]["segment_lengths"]
            == after_paths[checkpoint_id]["segment_lengths"]
            and before_paths[checkpoint_id]["prefix_sha256"]
            == after_paths[checkpoint_id]["prefix_sha256"]
            for checkpoint_id in before_paths
        )
        fa_preserved = all(
            state["fa_resident"] for state in after_states.values()
        ) and full_tree_unchanged
        retained_mamba_preserved = all(
            after_states[checkpoint_id]["mamba_resident"]
            for checkpoint_id in selected_ids
        )
        evicted_mamba_removed = all(
            not after_states[checkpoint_id]["mamba_resident"]
            for checkpoint_id in evicted_ids
        )
        sanity_check = all(
            response["proof"]["sanity_check"]
            for response in responses
        )
        cascade_called = any(
            response["proof"]["cascade_called"]
            for response in responses
        )
        fa_identity_unchanged = all(
            response["proof"]["fa_identity_unchanged"]
            for response in responses
        )
        expected_changed_nodes = {
            int(before_paths[checkpoint_id]["node_id"])
            for checkpoint_id in evicted_ids
        }
        only_expected_mamba_changed = (
            changed_mamba_nodes(global_before, global_after)
            == expected_changed_nodes
        )

        safety = {
            "fa_allocator_unchanged": allocator_unchanged,
            "tree_unchanged": tree_unchanged,
            "path_unchanged": path_unchanged,
            "fa_preserved": fa_preserved,
            "fa_identity_unchanged": fa_identity_unchanged,
            "retained_mamba_preserved": retained_mamba_preserved,
            "evicted_mamba_removed": evicted_mamba_removed,
            "only_expected_mamba_changed": only_expected_mamba_changed,
            "sanity_check": sanity_check,
            "cascade_called": cascade_called,
        }
        report["fa_allocator"] = {
            "before_available_size": allocator_before,
            "after_available_size": allocator_after,
            "unchanged": allocator_unchanged,
        }
        report["safety"] = safety
        if not all(
            (
                allocator_unchanged,
                tree_unchanged,
                path_unchanged,
                fa_preserved,
                fa_identity_unchanged,
                retained_mamba_preserved,
                evicted_mamba_removed,
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

        stage = "发送七个 pending continuation"
        observations = send_pending_continuations(
            engine,
            client,
            scenario,
            runtime_workflows,
            allocation.selected,
        )
        report["pending_continuations"] = observations
        report["aggregate"] = aggregate_observations(observations)

        report["status"] = "PASS"
        report["failure_stage"] = None
        print(
            "[STEP7B-GATE] RESULT="
            + json.dumps(report, sort_keys=True),
            flush=True,
        )
        return 0
    except Exception as error:
        if isinstance(error, RuntimeRepresentationMismatch):
            report["runtime_representation_mismatch"] = str(error)
        if "runtime_adapter" in locals():
            report["controller_evicted"] = tuple(
                runtime_adapter.evicted_checkpoint_ids
            )
        if "allocation" in locals():
            report["optimizer_selected"] = tuple(
                candidate.checkpoint_id
                for candidate in allocation.selected
            )
        report["status"] = "FAIL"
        report["failure_stage"] = stage
        report["error"] = repr(error)
        traceback.print_exc()
        print(
            "[STEP7B-GATE] RESULT="
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
