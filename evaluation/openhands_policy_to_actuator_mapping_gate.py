#!/usr/bin/env python3
"""验证冻结策略选择到循环状态单独驱逐执行结果的精确映射。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    SchedulerRuntimeAdapter,
    inspect_checkpoint,
    wait_for_transport,
)
from evaluation.openhands_4workflow_occupancy_calibration import (
    WORKFLOWS,
    _environment,
    _failure_record,
    compact_census,
    execute_request,
)
from evaluation.openhands_common_barrier_snapshot_gate import (
    BUDGET_BYTES,
    ENGINE_CONFIGURATION_COMMON_BARRIER,
    PENDING_TURN,
    SCHEDULE,
    build_pending_set,
    load_barrier_requests,
    locate_materialized_candidate,
    validate_candidate_at_barrier,
)
from evaluation.openhands_recurrent_only_causal_gate import (
    validate_eviction_response,
)
from evaluation.openhands_single_workflow_baseline10 import (
    ArtifactLogCapture,
    _append_jsonl,
    _write_json,
)
from evaluation.openhands_single_workflow_smoke import (
    DATASET_PATH,
    TOKENIZER_PATH,
)
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import StateController
from flowstate.optimizer import AllocationResult
from flowstate.state_catalog import (
    CheckpointCandidate,
    validate_unique_checkpoint_ids,
)
from flowstate.workflow import PendingContinuation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
ENGINE_CONFIGURATION_ACTUATOR_MAPPING = dict(
    ENGINE_CONFIGURATION_COMMON_BARRIER
)
SELECTED_SETS = {
    "LM": (
        "OPENHANDS_BARRIER_C_TURN_001",
        "OPENHANDS_BARRIER_D_TURN_001",
    ),
    "F": (
        "OPENHANDS_BARRIER_A_TURN_001",
        "OPENHANDS_BARRIER_C_TURN_001",
    ),
}
EXPECTED_EVICTIONS = {
    "LM": (
        "OPENHANDS_BARRIER_A_TURN_001",
        "OPENHANDS_BARRIER_B_TURN_001",
    ),
    "F": (
        "OPENHANDS_BARRIER_B_TURN_001",
        "OPENHANDS_BARRIER_D_TURN_001",
    ),
}


def _artifact_directory() -> Path:
    """创建不会覆盖既有结果的独立产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_policy_to_actuator_mapping_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """对仓库内路径返回相对表示。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


class FrozenSelectedSetOptimizer:
    """把已经冻结的合法 selected set 交给真实控制器执行。"""

    def __init__(self, selected_checkpoint_ids: Sequence[str]) -> None:
        self._selected_checkpoint_ids = tuple(selected_checkpoint_ids)

    def select(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        budget_bytes: int,
    ) -> AllocationResult:
        """验证冻结集合后构造控制器所需的 AllocationResult。"""
        del continuations
        validate_unique_checkpoint_ids(candidates)
        if budget_bytes < 0:
            raise ValueError("内存预算必须大于等于零")
        if len(set(self._selected_checkpoint_ids)) != len(
            self._selected_checkpoint_ids
        ):
            raise ValueError("冻结 selected set 含有重复检查点")
        by_id = {
            candidate.checkpoint_id: candidate for candidate in candidates
        }
        missing = tuple(
            checkpoint_id
            for checkpoint_id in self._selected_checkpoint_ids
            if checkpoint_id not in by_id
        )
        if missing:
            raise ValueError(f"冻结 selected set 含非候选检查点：{missing}")
        selected = tuple(
            by_id[checkpoint_id]
            for checkpoint_id in self._selected_checkpoint_ids
        )
        if any(not candidate.recurrent_resident for candidate in selected):
            raise ValueError("冻结 selected set 含非驻留循环状态")
        used_bytes = sum(candidate.memory_bytes for candidate in selected)
        if used_bytes > budget_bytes:
            raise ValueError(
                f"冻结 selected set 超出预算：{used_bytes} > {budget_bytes}"
            )
        return AllocationResult(
            selected=selected,
            total_benefit_ms=0.0,
            recovery_cost_before_ms=0.0,
            recovery_cost_after_ms=0.0,
            used_bytes=used_bytes,
        )


class RecordingRuntimeAdapter:
    """记录控制器动作并委托既有 scheduler 安全时点适配器。"""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.requested_checkpoint_ids: list[str] = []
        self.successful_checkpoint_ids: list[str] = []
        self.failed_evictions: list[dict[str, str]] = []
        self.responses: list[dict[str, object]] = []

    def evict_mamba_only(
        self,
        handle: RuntimeCheckpointHandle,
    ) -> None:
        """记录一次请求，并把状态变更交给冻结的正式执行路径。"""
        checkpoint_id = handle.checkpoint_id
        self.requested_checkpoint_ids.append(checkpoint_id)
        try:
            self._delegate.evict_mamba_only(handle)
        except Exception as error:
            self.failed_evictions.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "error": repr(error),
                }
            )
            raise
        self.successful_checkpoint_ids.append(checkpoint_id)
        responses = getattr(self._delegate, "eviction_responses", ())
        if responses:
            self.responses.append(dict(responses[-1]))


def build_controller_report(
    *,
    allocation: AllocationResult | None,
    adapter: RecordingRuntimeAdapter,
) -> dict[str, object]:
    """按现有控制器返回值和实际 actuator 调用构造审计报告。"""
    return {
        "controller_return_type": (
            None if allocation is None else type(allocation).__name__
        ),
        "selected_checkpoint_ids": (
            []
            if allocation is None
            else [item.checkpoint_id for item in allocation.selected]
        ),
        "requested_eviction_ids": list(adapter.requested_checkpoint_ids),
        "successful_eviction_ids": list(adapter.successful_checkpoint_ids),
        "failed_evictions": list(adapter.failed_evictions),
        "formal_responses": list(adapter.responses),
    }


def compact_checkpoint_state(
    response: Mapping[str, object],
) -> dict[str, object]:
    """提取候选节点和全局 FA 不变量所需的稳定字段。"""
    snapshot = response["after"]
    if not isinstance(snapshot, Mapping):
        raise TypeError("checkpoint inspect 缺少 after 快照")
    path = snapshot["path"]
    tree = snapshot["tree"]
    accounting = snapshot["accounting"]
    if not all(
        isinstance(item, Mapping) for item in (path, tree, accounting)
    ):
        raise TypeError("checkpoint inspect 快照结构异常")
    return {
        "node_id": int(path["node_id"]),
        "token_pos": int(path["prefix_tokens"]),
        "prefix_digest": str(path["prefix_sha256"]),
        "path_node_ids": [int(value) for value in path["path_node_ids"]],
        "path_full_digest": str(path["path_full_sha256"]),
        "fa_resident": bool(
            path["target_full_present"] and path["path_full_all_present"]
        ),
        "recurrent_resident": bool(path["target_mamba_present"]),
        "tree_structure_digest": str(tree["structure_sha256"]),
        "full_tree_digest": str(tree["full_tree_sha256"]),
        "full_allocator": accounting["full_allocator"],
    }


def inspect_candidate_states(
    client: object,
    candidates: Sequence[CheckpointCandidate],
    handles: Mapping[str, RuntimeCheckpointHandle],
    *,
    phase: str,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    """在 scheduler 安全时点读取全部候选的精确节点状态。"""
    states: dict[str, dict[str, object]] = {}
    responses: list[dict[str, object]] = []
    for candidate in candidates:
        handle = handles[candidate.checkpoint_id]
        response = inspect_checkpoint(
            client,
            f"{candidate.checkpoint_id}_{phase}",
            handle.token_ids,
        )
        state = compact_checkpoint_state(response)
        if (
            state["node_id"] != handle.expected_node_id
            or state["token_pos"] != candidate.token_pos
            or state["prefix_digest"] != handle.expected_prefix_digest
        ):
            raise RuntimeError(
                f"{candidate.checkpoint_id} 的 runtime identity 不一致"
            )
        states[candidate.checkpoint_id] = state
        responses.append(dict(response))
    return states, responses


def evaluate_mapping_invariants(
    *,
    candidate_ids: Sequence[str],
    selected_ids: Sequence[str],
    expected_evicted_ids: Sequence[str],
    handles: Mapping[str, RuntimeCheckpointHandle],
    before_states: Mapping[str, Mapping[str, object]],
    after_states: Mapping[str, Mapping[str, object]],
    before_census: Mapping[str, object],
    after_census: Mapping[str, object],
    controller_report: Mapping[str, object],
) -> dict[str, object]:
    """验证 selected set、运行时驻留和 recurrent-only 不变量。"""
    candidate_set = set(candidate_ids)
    selected_set = set(selected_ids)
    expected_evicted_set = set(expected_evicted_ids)
    actual_retained = sorted(
        checkpoint_id
        for checkpoint_id in candidate_ids
        if after_states[checkpoint_id]["recurrent_resident"]
    )
    actual_evicted = sorted(
        checkpoint_id
        for checkpoint_id in candidate_ids
        if before_states[checkpoint_id]["recurrent_resident"]
        and not after_states[checkpoint_id]["recurrent_resident"]
    )
    expected_removed_node_ids = sorted(
        int(handles[checkpoint_id].expected_node_id)
        for checkpoint_id in expected_evicted_ids
    )
    removed_node_ids = sorted(
        int(value) for value in after_census["removed_mamba_node_ids"]
    )
    changed_node_ids = sorted(
        int(value)
        for value in after_census["changed_existing_mamba_node_ids"]
    )
    expected_released_slots = sum(
        len(item["slots"])
        for item in before_census["resident_mamba_nodes"]
        if int(item["node_id"]) in expected_removed_node_ids
    )
    available_delta = (
        int(after_census["mamba_available_slots"])
        - int(before_census["mamba_available_slots"])
    )
    resident_delta = (
        int(before_census["device_resident_mamba_slots"])
        - int(after_census["device_resident_mamba_slots"])
    )
    node_count_delta = (
        int(before_census["mamba_node_count"])
        - int(after_census["mamba_node_count"])
    )
    tree_unchanged = all(
        before_states[checkpoint_id]["tree_structure_digest"]
        == after_states[checkpoint_id]["tree_structure_digest"]
        for checkpoint_id in candidate_ids
    )
    fa_preserved = all(
        before_states[checkpoint_id]["fa_resident"]
        and after_states[checkpoint_id]["fa_resident"]
        and before_states[checkpoint_id]["path_full_digest"]
        == after_states[checkpoint_id]["path_full_digest"]
        and before_states[checkpoint_id]["full_tree_digest"]
        == after_states[checkpoint_id]["full_tree_digest"]
        for checkpoint_id in candidate_ids
    )
    fa_allocator_unchanged = all(
        before_states[checkpoint_id]["full_allocator"]
        == after_states[checkpoint_id]["full_allocator"]
        for checkpoint_id in candidate_ids
    )
    node_identity_unchanged = all(
        before_states[checkpoint_id]["node_id"]
        == after_states[checkpoint_id]["node_id"]
        and before_states[checkpoint_id]["path_node_ids"]
        == after_states[checkpoint_id]["path_node_ids"]
        for checkpoint_id in candidate_ids
    )
    prefix_digest_unchanged = all(
        before_states[checkpoint_id]["prefix_digest"]
        == after_states[checkpoint_id]["prefix_digest"]
        for checkpoint_id in candidate_ids
    )
    recurrent_change_exact = bool(
        removed_node_ids == expected_removed_node_ids
        and not changed_node_ids
        and available_delta == expected_released_slots
        and resident_delta == expected_released_slots
        and node_count_delta == len(expected_removed_node_ids)
    )
    native_eviction = bool(
        removed_node_ids != expected_removed_node_ids or changed_node_ids
    )
    fa_cascade = bool(
        before_census["full_device_node_ids"]
        != after_census["full_device_node_ids"]
        or before_census["structure_node_ids"]
        != after_census["structure_node_ids"]
        or after_census["removed_full_device_node_ids"]
        or after_census["removed_structure_node_ids"]
    )
    selected_residency_exact = bool(
        selected_set.issubset(candidate_set)
        and expected_evicted_set == candidate_set - selected_set
        and set(actual_retained) == selected_set
        and set(actual_evicted) == expected_evicted_set
    )
    controller_exact = bool(
        set(controller_report["selected_checkpoint_ids"]) == selected_set
        and set(controller_report["requested_eviction_ids"])
        == expected_evicted_set
        and set(controller_report["successful_eviction_ids"])
        == expected_evicted_set
        and not controller_report["failed_evictions"]
    )
    response_correctness = all(
        validate_eviction_response(response)
        for response in controller_report["formal_responses"]
    ) and len(controller_report["formal_responses"]) == len(
        expected_evicted_ids
    )
    passed = bool(
        selected_residency_exact
        and controller_exact
        and response_correctness
        and tree_unchanged
        and fa_preserved
        and fa_allocator_unchanged
        and node_identity_unchanged
        and prefix_digest_unchanged
        and recurrent_change_exact
        and not native_eviction
        and not fa_cascade
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "candidate_ids": list(candidate_ids),
        "selected_ids": list(selected_ids),
        "expected_evicted_ids": list(expected_evicted_ids),
        "actual_retained_ids": actual_retained,
        "actual_evicted_ids": actual_evicted,
        "selected_residency_exact": selected_residency_exact,
        "controller_exact": controller_exact,
        "actuator_responses_valid": response_correctness,
        "tree_unchanged": tree_unchanged,
        "fa_residency_preserved": fa_preserved,
        "fa_allocator_unchanged": fa_allocator_unchanged,
        "node_identity_unchanged": node_identity_unchanged,
        "prefix_digest_unchanged": prefix_digest_unchanged,
        "expected_removed_node_ids": expected_removed_node_ids,
        "observed_removed_node_ids": removed_node_ids,
        "observed_changed_node_ids": changed_node_ids,
        "expected_released_slots": expected_released_slots,
        "mamba_available_slot_delta": available_delta,
        "resident_mamba_slot_delta": resident_delta,
        "mamba_node_count_delta": node_count_delta,
        "recurrent_change_exact": recurrent_change_exact,
        "native_mamba_capacity_eviction": native_eviction,
        "fa_kv_cascade": fa_cascade,
    }


@dataclass(frozen=True)
class RunPaths:
    """保存一次独立 Engine 条件的产物路径。"""

    requests: Path
    censuses: Path
    controller_reports: Path
    runs: Path


def run_condition(
    *,
    condition: str,
    engine_ordinal: int,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    continuations_audit: Sequence[Mapping[str, object]],
    paths: RunPaths,
) -> dict[str, object]:
    """用一个全新 Engine 建立 barrier 并执行一次 controller reconcile。"""
    selected_ids = SELECTED_SETS[condition]
    expected_evicted_ids = EXPECTED_EVICTIONS[condition]
    engine = None
    records: list[dict[str, object]] = []
    censuses: list[dict[str, object]] = []
    candidates: list[CheckpointCandidate] = []
    candidate_rows: list[dict[str, object]] = []
    handles: dict[str, RuntimeCheckpointHandle] = {}
    controller_report: dict[str, object] | None = None
    invariants: dict[str, object] | None = None
    fatal_error = None
    shutdown_error = None
    try:
        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(
            **ENGINE_CONFIGURATION_ACTUATOR_MAPPING
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        baseline = compact_census(
            client.census(f"openhands-actuator:{condition}:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        baseline["condition"] = condition
        baseline["event"] = "baseline"
        censuses.append(baseline)
        _append_jsonl(paths.censuses, baseline)
        if int(baseline["mamba_node_count"]) != 0:
            raise RuntimeError("全新 Engine 初始 census 含有循环检查点")

        previous = baseline
        for ordinal, (label, turn) in enumerate(SCHEDULE, start=1):
            request = requests[(label, turn)]
            try:
                record = execute_request(engine, client, request, ordinal)
                census = compact_census(
                    client.census(
                        f"openhands-actuator:{condition}:after:{label}{turn}"
                    ),
                    ordinal=ordinal,
                    request=request,
                    previous=previous,
                )
                census["condition"] = condition
                census["event"] = f"after_{label}{turn}"
                candidate, handle, row = locate_materialized_candidate(
                    client,
                    request,
                    census,
                    event_order=ordinal,
                )
            except Exception as error:
                record = _failure_record(request, ordinal, error)
                record["condition"] = condition
                records.append(record)
                _append_jsonl(paths.requests, record)
                raise
            record["condition"] = condition
            records.append(record)
            censuses.append(census)
            candidates.append(candidate)
            candidate_rows.append(row)
            handles[candidate.checkpoint_id] = handle
            _append_jsonl(paths.requests, record)
            _append_jsonl(paths.censuses, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{condition} 的 {label}{turn} 请求失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                raise RuntimeError(f"{condition} 建态时发生原生 Mamba 驱逐")
            if census["fa_kv_cascade_eviction_inferred"]:
                raise RuntimeError(f"{condition} 建态时发生 FA-KV 级联")

        validate_unique_checkpoint_ids(candidates)
        for candidate in candidates:
            validation = validate_candidate_at_barrier(
                client,
                candidate,
                handles[candidate.checkpoint_id],
            )
            if not validation["consistent"]:
                raise RuntimeError(
                    f"{condition} 的 {candidate.checkpoint_id} barrier 状态不一致"
                )

        barrier_client = __import__(
            "evaluation.barrier_fa_frontier_control",
            fromlist=["BarrierFAControlClient"],
        ).BarrierFAControlClient(client)
        continuations, _ = build_pending_set(barrier_client, requests)
        before_census = compact_census(
            client.census(f"openhands-actuator:{condition}:before-reconcile"),
            ordinal=len(SCHEDULE),
            request=None,
            previous=previous,
        )
        before_census["condition"] = condition
        before_census["event"] = "before_reconcile"
        censuses.append(before_census)
        _append_jsonl(paths.censuses, before_census)
        before_states, before_responses = inspect_candidate_states(
            client,
            candidates,
            handles,
            phase=f"{condition}_BEFORE",
        )
        if not all(
            state["fa_resident"] and state["recurrent_resident"]
            for state in before_states.values()
        ):
            raise RuntimeError(f"{condition} reconcile 前候选未全部驻留")

        delegate = SchedulerRuntimeAdapter(client)
        recording_adapter = RecordingRuntimeAdapter(delegate)
        controller = StateController(
            FrozenSelectedSetOptimizer(selected_ids),
            recording_adapter,
        )
        allocation = None
        try:
            allocation = controller.reconcile(
                continuations,
                tuple(candidates),
                handles,
                BUDGET_BYTES,
            )
        finally:
            controller_report = build_controller_report(
                allocation=allocation,
                adapter=recording_adapter,
            )
            controller_report["condition"] = condition
            _append_jsonl(paths.controller_reports, controller_report)

        after_census = compact_census(
            client.census(f"openhands-actuator:{condition}:after-reconcile"),
            ordinal=len(SCHEDULE),
            request=None,
            previous=before_census,
        )
        after_census["condition"] = condition
        after_census["event"] = "after_reconcile"
        censuses.append(after_census)
        _append_jsonl(paths.censuses, after_census)
        after_states, after_responses = inspect_candidate_states(
            client,
            candidates,
            handles,
            phase=f"{condition}_AFTER",
        )
        invariants = evaluate_mapping_invariants(
            candidate_ids=[item.checkpoint_id for item in candidates],
            selected_ids=selected_ids,
            expected_evicted_ids=expected_evicted_ids,
            handles=handles,
            before_states=before_states,
            after_states=after_states,
            before_census=before_census,
            after_census=after_census,
            controller_report=controller_report,
        )
        if invariants["status"] != "PASS":
            raise RuntimeError(
                f"{condition} selected-set 到 actuator 映射不一致"
            )
        condition_snapshot = {
            "candidate_rows": candidate_rows,
            "before_inspections": before_responses,
            "after_inspections": after_responses,
        }
    except Exception as error:
        fatal_error = repr(error)
        condition_snapshot = {}
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                shutdown_error = repr(error)

    future_information_used = any(
        item.get("r_plus_2_message_consumed") is not False
        or item.get("r_plus_2_request_materialized") is not False
        or item.get("pending_assistant_output_read") is not False
        for item in continuations_audit
    )
    passed = bool(
        fatal_error is None
        and shutdown_error is None
        and len(records) == len(SCHEDULE)
        and all(item.get("status") == "PASS" for item in records)
        and invariants is not None
        and invariants.get("status") == "PASS"
        and not future_information_used
    )
    result = {
        "condition": condition,
        "engine_lifecycle": "independent_fresh",
        "engine_ordinal": engine_ordinal,
        "status": "PASS" if passed else "FAIL",
        "selected_ids": list(selected_ids),
        "expected_evicted_ids": list(expected_evicted_ids),
        "executed_schedule": [f"{label}{turn}" for label, turn in SCHEDULE],
        "pending_requests_executed": False,
        "future_information_used": future_information_used,
        "controller_report": controller_report,
        "invariants": invariants,
        "snapshot": condition_snapshot,
        "fatal_error": fatal_error,
        "engine_shutdown_error": shutdown_error,
    }
    _append_jsonl(paths.runs, result)
    return result


def build_summary(
    *,
    artifact: Path,
    runs: Sequence[Mapping[str, object]],
    boundary_audit: Sequence[Mapping[str, object]],
    environment: Mapping[str, object] | None,
) -> dict[str, object]:
    """汇总两个独立 Engine 条件的映射正确性门禁。"""
    by_condition = {
        str(run["condition"]): run for run in runs
    }
    future_information_used = any(
        item.get("r_plus_2_message_consumed") is not False
        or item.get("r_plus_2_request_materialized") is not False
        or item.get("pending_assistant_output_read") is not False
        for item in boundary_audit
    )
    both_present = set(by_condition) == set(SELECTED_SETS)
    complete_results = bool(
        both_present
        and all(
            isinstance(run.get("invariants"), Mapping)
            and isinstance(run.get("controller_report"), Mapping)
            for run in by_condition.values()
        )
    )
    different_residency = bool(
        complete_results
        and set(
            by_condition["LM"]["invariants"]["actual_retained_ids"]
        )
        != set(by_condition["F"]["invariants"]["actual_retained_ids"])
    )
    same_actuator_path = bool(
        complete_results
        and all(
            all(
                response.get("formal_primitive")
                == "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
                for response in run["controller_report"]["formal_responses"]
            )
            for run in by_condition.values()
        )
    )
    passed = bool(
        complete_results
        and all(run.get("status") == "PASS" for run in runs)
        and same_actuator_path
        and different_residency
        and not future_information_used
    )
    return {
        "schema_version": "flowstate.openhands_policy_to_actuator_mapping.v1",
        "status": "PASS" if passed else "FAIL",
        "verdict": "READY" if passed else "PARTIAL",
        "artifact": _display_path(artifact),
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_ACTUATOR_MAPPING,
        "engine_lifecycle_count": len(runs),
        "independent_engine_lifecycles": all(
            run.get("engine_lifecycle") == "independent_fresh"
            for run in runs
        ),
        "workflows": WORKFLOWS,
        "logical_k": 2,
        "budget_bytes": BUDGET_BYTES,
        "selected_sets": {
            key: list(value) for key, value in SELECTED_SETS.items()
        },
        "runs": list(runs),
        "same_actuator_path": same_actuator_path,
        "different_selected_sets_produced_corresponding_residency": (
            different_residency
        ),
        "pending_requests_executed": False,
        "future_information_used": future_information_used,
        "future_leakage": future_information_used,
        "online_information_boundary": list(boundary_audit),
        "environment": dict(environment) if environment is not None else None,
    }


def _run(artifact: Path) -> dict[str, object]:
    """顺序执行两个独立 Engine 条件，首个失败后立即停止。"""
    paths = RunPaths(
        requests=artifact / "requests.jsonl",
        censuses=artifact / "census.jsonl",
        controller_reports=artifact / "controller_reports.jsonl",
        runs=artifact / "runs.jsonl",
    )
    for path in (
        paths.requests,
        paths.censuses,
        paths.controller_reports,
        paths.runs,
    ):
        path.write_text("", encoding="utf-8")
    environment = _environment()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )
    requests, boundary_audit = load_barrier_requests(tokenizer)
    _write_json(
        artifact / "config.json",
        {
            "workflows": WORKFLOWS,
            "run_order": ["LM", "F"],
            "selected_sets": {
                key: list(value) for key, value in SELECTED_SETS.items()
            },
            "expected_evictions": {
                key: list(value) for key, value in EXPECTED_EVICTIONS.items()
            },
            "engine_configuration": ENGINE_CONFIGURATION_ACTUATOR_MAPPING,
            "sampling_parameters": SAMPLING_PARAMETERS,
            "dataset_path": str(DATASET_PATH),
            "tokenizer_path": str(TOKENIZER_PATH),
            "logical_k": 2,
            "budget_bytes": BUDGET_BYTES,
            "executed_schedule_per_run": [
                f"{label}{turn}" for label, turn in SCHEDULE
            ],
            "pending_schedule": [f"{label}2" for label in WORKFLOWS],
            "pending_requests_executed": False,
            "online_information_boundary": boundary_audit,
            "environment": environment,
        },
    )
    runs = []
    for ordinal, condition in enumerate(("LM", "F"), start=1):
        result = run_condition(
            condition=condition,
            engine_ordinal=ordinal,
            requests=requests,
            continuations_audit=boundary_audit,
            paths=paths,
        )
        runs.append(result)
        if result["status"] != "PASS":
            break
    summary = build_summary(
        artifact=artifact,
        runs=runs,
        boundary_audit=boundary_audit,
        environment=environment,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行唯一一次 selected-set 到 actuator 门禁。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
