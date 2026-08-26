#!/usr/bin/env python3
"""在冻结 H100 runtime 中执行独立 Recovery Profiler v2。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
import json
import random
from pathlib import Path
import time
import traceback
from typing import Mapping, Sequence

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    PENDING_SUFFIX_LENGTH,
    build_runtime_handles,
    build_runtime_workflows,
    compact_state,
    inspect_after_allocation,
    inspect_checkpoint,
    make_tokens,
    path_state,
    query_runtime_metrics,
    wait_for_transport,
)
from evaluation.controlled_multiworkflow_v1.scenario import (
    CHECKPOINT_SIZE_BYTES,
    CheckpointRecency,
    ControlledScenario,
    WorkflowSpec,
    WorkloadMetadata,
)
from evaluation.controlled_multiworkflow_v1.snapshot_runtime import (
    SnapshotSchedulerRuntimeAdapter,
    allocation_safety_snapshot,
    validate_clean_cache,
)
from evaluation.recovery_profiler_v2.analyze import (
    ALL_GAPS,
    CALIBRATION_GAPS,
    MEASURED_REPETITIONS,
    VALIDATION_GAPS,
    WARMUP_REPETITIONS,
    analyze_records,
    write_analysis_artifacts,
)
from evaluation.sota_latency_runtime import measure_streaming_request
from evaluation.sota_runtime_correctness import STEP8E_ENGINE_CONFIGURATION
from flowstate.controller import StateController
from flowstate.optimizer import AllocationResult
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
RAW_SAMPLES_PATH = OUTPUT_DIRECTORY / "raw_samples.jsonl"
RUN_METADATA_PATH = OUTPUT_DIRECTORY / "run_metadata.json"
ORDER_SEED = 20_260_826
ANCHOR_POS = 32_768
PROFILE_WORKFLOW_ID = "RECOVERY_PROFILE"
PROFILE_ROOT_LINEAGE = "PROFILE_ROOT"
PROFILE_CONTINUATION_ID = "RECOVERY_PROFILE-B"
DEEP_CHECKPOINT_ID = "PROFILE_DEEP"
SHALLOW_CHECKPOINT_ID = "PROFILE_SHALLOW"
_FORMAL_PRIMITIVE = (
    "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
)


@dataclass(frozen=True)
class ProfileCase:
    """描述一次 fresh-snapshot recovery profiling case。"""

    case_id: str
    target_gap: int
    target_frontier: int
    repetition: int
    is_warmup: bool
    gap_order_position: int
    execution_order_position: int


class FixedSelectionOptimizer:
    """只执行 profiler 已冻结的单个 gap 状态选择。"""

    def __init__(self, selected_ids: Sequence[str]) -> None:
        self._selected_ids = tuple(selected_ids)

    def select(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        budget_bytes: int,
    ) -> AllocationResult:
        """返回预注册的 retain set，不计算或读取 recovery cost。"""
        del continuations
        candidates_by_id = {
            candidate.checkpoint_id: candidate for candidate in candidates
        }
        if any(
            checkpoint_id not in candidates_by_id
            for checkpoint_id in self._selected_ids
        ):
            raise ValueError("profiler selected ID 不在 candidate set 中")
        selected = tuple(
            candidates_by_id[checkpoint_id]
            for checkpoint_id in self._selected_ids
        )
        used_bytes = sum(candidate.memory_bytes for candidate in selected)
        if used_bytes > budget_bytes:
            raise ValueError("profiler selected set 超过冻结预算")
        return AllocationResult(
            selected=selected,
            total_benefit_ms=0.0,
            recovery_cost_before_ms=0.0,
            recovery_cost_after_ms=0.0,
            used_bytes=used_bytes,
        )


def build_profile_schedule(
    seed: int = ORDER_SEED,
) -> tuple[ProfileCase, ...]:
    """构建固定 seed 的循环 gap 顺序。"""
    base_order = list(ALL_GAPS)
    random.Random(seed).shuffle(base_order)
    if tuple(base_order) == ALL_GAPS:
        raise RuntimeError("gap 基序列不能退化为递增顺序")

    cases = []
    execution_position = 0
    cycle_index = 0
    for is_warmup, repetitions in (
        (True, WARMUP_REPETITIONS),
        (False, MEASURED_REPETITIONS),
    ):
        for repetition in range(repetitions):
            offset = cycle_index % len(base_order)
            order = base_order[offset:] + base_order[:offset]
            for gap_order_position, gap in enumerate(order):
                phase = "warmup" if is_warmup else "measured"
                cases.append(
                    ProfileCase(
                        case_id=(
                            f"profile_{phase}_r{repetition:02d}_"
                            f"p{gap_order_position:02d}_g{gap}"
                        ),
                        target_gap=gap,
                        target_frontier=ANCHOR_POS - gap,
                        repetition=repetition,
                        is_warmup=is_warmup,
                        gap_order_position=gap_order_position,
                        execution_order_position=execution_position,
                    )
                )
                execution_position += 1
            cycle_index += 1
    return tuple(cases)


def build_profile_scenario(
    target_gap: int,
) -> tuple[ControlledScenario, tuple[str, ...]]:
    """构造固定 H、仅改变 E 的单工作流 profiler snapshot。"""
    if target_gap not in ALL_GAPS:
        raise ValueError(f"未冻结的 profiling gap：{target_gap}")
    target_frontier = ANCHOR_POS - target_gap
    workflow = WorkflowSpec(
        workflow_id=PROFILE_WORKFLOW_ID,
        root_lineage=PROFILE_ROOT_LINEAGE,
        anchor_pos=ANCHOR_POS,
        pending_branches=("B",),
    )
    continuation = PendingContinuation(
        continuation_id=PROFILE_CONTINUATION_ID,
        workflow_id=PROFILE_WORKFLOW_ID,
        lineage_path=(PROFILE_ROOT_LINEAGE, "B"),
        anchor_pos=ANCHOR_POS,
        resident_fa_frontier=ANCHOR_POS,
    )
    candidates = []
    recency = []
    if 0 < target_frontier < ANCHOR_POS:
        candidates.append(
            CheckpointCandidate(
                checkpoint_id=SHALLOW_CHECKPOINT_ID,
                workflow_id=PROFILE_WORKFLOW_ID,
                lineage_path=(PROFILE_ROOT_LINEAGE,),
                token_pos=target_frontier,
                memory_bytes=CHECKPOINT_SIZE_BYTES,
            )
        )
        recency.append(
            CheckpointRecency(
                checkpoint_id=SHALLOW_CHECKPOINT_ID,
                creation_order=1,
                last_access_order=1,
            )
        )
    candidates.append(
        CheckpointCandidate(
            checkpoint_id=DEEP_CHECKPOINT_ID,
            workflow_id=PROFILE_WORKFLOW_ID,
            lineage_path=(PROFILE_ROOT_LINEAGE,),
            token_pos=ANCHOR_POS,
            memory_bytes=CHECKPOINT_SIZE_BYTES,
        )
    )
    recency.append(
        CheckpointRecency(
            checkpoint_id=DEEP_CHECKPOINT_ID,
            creation_order=2,
            last_access_order=2,
        )
    )

    if target_gap == 0:
        selected_ids = (DEEP_CHECKPOINT_ID,)
    elif target_frontier > 0:
        selected_ids = (SHALLOW_CHECKPOINT_ID,)
    else:
        selected_ids = ()
    budget_checkpoints = len(selected_ids)
    metadata = WorkloadMetadata(
        workflows=(workflow,),
        checkpoint_recency=tuple(recency),
        workflow_order=(PROFILE_WORKFLOW_ID,),
        checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
        budget_checkpoints=budget_checkpoints,
    )
    return (
        ControlledScenario(
            continuations=(continuation,),
            candidates=tuple(candidates),
            budget_bytes=budget_checkpoints * CHECKPOINT_SIZE_BYTES,
            metadata=metadata,
        ),
        selected_ids,
    )


def validate_runtime_gap(
    target_gap: int,
    metrics: Mapping[str, object],
) -> dict[str, int]:
    """严格验证真实 H/E/G 与冻结 profiling target 完全一致。"""
    runtime_h = int(metrics["physical_fa_hit"])
    runtime_e = int(metrics["executable_prefix"])
    runtime_g = int(metrics["replay_gap"])
    expected_e = ANCHOR_POS - target_gap
    if runtime_h - runtime_e != runtime_g:
        raise RuntimeError("runtime recovery gap 不等于 H-E")
    if runtime_h != ANCHOR_POS:
        raise RuntimeError(
            f"runtime H 不匹配：{runtime_h} != {ANCHOR_POS}"
        )
    if runtime_e != expected_e:
        raise RuntimeError(f"runtime E 不匹配：{runtime_e} != {expected_e}")
    if runtime_g != target_gap:
        raise RuntimeError(
            f"runtime G 不匹配：{runtime_g} != {target_gap}"
        )
    return {
        "runtime_H": runtime_h,
        "runtime_E": runtime_e,
        "runtime_G": runtime_g,
    }


def execute_profile_case(
    case: ProfileCase,
    *,
    engine: object,
    client: object,
) -> dict[str, object]:
    """执行一个独立 snapshot，并返回完整 TTFT 与正确性记录。"""
    scenario, expected_selected_ids = build_profile_scenario(case.target_gap)
    namespace = f"flowstate_step9d_{case.case_id}"
    record: dict[str, object] = {
        "case_id": case.case_id,
        "target_gap": case.target_gap,
        "target_H": ANCHOR_POS,
        "target_E": case.target_frontier,
        "split": (
            "calibration"
            if case.target_gap in CALIBRATION_GAPS
            else "held_out_validation"
        ),
        "repetition": case.repetition,
        "is_warmup": case.is_warmup,
        "gap_order_position": case.gap_order_position,
        "execution_order_position": case.execution_order_position,
        "status": "FAIL",
        "correctness_pass": False,
        "gap_match": False,
        "step9b_data_used_for_fitting": False,
    }
    stage = "刷新并验证空缓存"
    try:
        started = time.perf_counter_ns()
        engine.flush_cache()
        clean_census = client.census(f"{namespace}:after_flush")
        record["clean_cache"] = validate_clean_cache(clean_census)
        record["flush_ms"] = _elapsed_ms(started)

        stage = "构建相同 request shape 的 recurrent checkpoints"
        started = time.perf_counter_ns()
        runtime_workflows, candidate_tokens = build_runtime_workflows(
            engine,
            scenario,
            request_namespace=f"{namespace}_build",
        )
        handles, before_paths, before_states = build_runtime_handles(
            client,
            scenario,
            candidate_tokens,
        )
        before_census = client.census(f"{namespace}:before_allocation")
        record["snapshot_build_ms"] = _elapsed_ms(started)
        record["checkpoints_before_allocation"] = before_states

        stage = "执行预注册 recurrent retain/evict 状态"
        runtime_adapter = SnapshotSchedulerRuntimeAdapter(
            client,
            case.case_id,
            "flowstate_step9d",
        )
        started = time.perf_counter_ns()
        allocation = StateController(
            FixedSelectionOptimizer(expected_selected_ids),
            runtime_adapter,
        ).reconcile(
            scenario.continuations,
            scenario.candidates,
            handles,
            scenario.budget_bytes,
        )
        record["reconcile_ms"] = _elapsed_ms(started)
        selected_ids = tuple(
            candidate.checkpoint_id for candidate in allocation.selected
        )
        evicted_ids = tuple(runtime_adapter.evicted_checkpoint_ids)
        if selected_ids != expected_selected_ids:
            raise RuntimeError("profiler runtime selected set 偏离预注册状态")
        record["selected_checkpoint_ids"] = list(selected_ids)
        record["evicted_checkpoint_ids"] = list(evicted_ids)

        stage = "验证 allocation 后状态与安全条件"
        started = time.perf_counter_ns()
        after_paths, after_states = inspect_after_allocation(
            client,
            candidate_tokens,
        )
        after_census = client.census(f"{namespace}:after_allocation")
        if evicted_ids:
            safety, allocator = allocation_safety_snapshot(
                before_paths=before_paths,
                after_paths=after_paths,
                after_states=after_states,
                selected_ids=selected_ids,
                evicted_ids=evicted_ids,
                eviction_responses=runtime_adapter.eviction_responses,
            )
        else:
            safety, allocator = _no_mutation_safety(
                client=client,
                candidate_tokens=candidate_tokens,
                before_paths=before_paths,
                before_states=before_states,
                after_paths=after_paths,
                after_states=after_states,
                before_census=before_census,
                after_census=after_census,
                selected_ids=selected_ids,
            )
        record["safety_validation_ms"] = _elapsed_ms(started)
        record["checkpoints_after_allocation"] = after_states
        record["safety"] = safety
        record["safety_pass"] = all(safety.values())
        record["fa_allocator"] = allocator
        record["formal_mutation_primitive"] = (
            _FORMAL_PRIMITIVE if evicted_ids else None
        )
        if not record["safety_pass"]:
            raise RuntimeError(f"profiler 安全条件失败：{safety}")

        stage = "确认 scheduler runtime ready"
        if client.ping().get("ok") is not True:
            raise RuntimeError("scheduler transport 未就绪")
        record["runtime_ready"] = True

        stage = "发送唯一 target request 并测量 TTFT"
        runtime_workflow = runtime_workflows[0]
        request_tokens = (
            runtime_workflow.anchor_tokens
            + (runtime_workflow.anchor_output,)
            + make_tokens(541_019, PENDING_SUFFIX_LENGTH)
        )
        request_id = f"{namespace}_target"
        timing = measure_streaming_request(
            engine,
            request_id=request_id,
            token_ids=request_tokens,
        )
        record["request_id"] = request_id
        record["request_input_tokens"] = len(request_tokens)
        record["ttft_ms"] = timing["ttft_ms"]
        record["request_latency_ms"] = timing["request_latency_ms"]
        record["server_metadata"] = timing["server_metadata"]

        stage = "读取真实 H/E/G 并执行严格正确性门禁"
        metrics = query_runtime_metrics(client, request_id)
        runtime = validate_runtime_gap(case.target_gap, metrics)
        record.update(runtime)
        record["H_match"] = runtime["runtime_H"] == ANCHOR_POS
        record["E_match"] = runtime["runtime_E"] == case.target_frontier
        record["gap_match"] = runtime["runtime_G"] == case.target_gap
        record["correctness_pass"] = bool(
            record["H_match"]
            and record["E_match"]
            and record["gap_match"]
            and record["safety_pass"]
        )
        if not record["correctness_pass"]:
            raise RuntimeError("profiler case 未通过正确性门禁")
        record["status"] = "PASS"
        record["failure_stage"] = None
        return record
    except Exception as error:
        record["failure_stage"] = stage
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
        return record


def _no_mutation_safety(
    *,
    client: object,
    candidate_tokens: Mapping[str, tuple[int, ...]],
    before_paths: Mapping[str, Mapping[str, object]],
    before_states: Mapping[str, Mapping[str, object]],
    after_paths: Mapping[str, Mapping[str, object]],
    after_states: Mapping[str, Mapping[str, object]],
    before_census: Mapping[str, object],
    after_census: Mapping[str, object],
    selected_ids: Sequence[str],
) -> tuple[dict[str, bool], dict[str, int]]:
    """验证 G=0 case 未执行 mutation 时状态完全保持。"""
    final_inspection = inspect_checkpoint(
        client,
        DEEP_CHECKPOINT_ID,
        candidate_tokens[DEEP_CHECKPOINT_ID],
    )
    final_path = path_state(final_inspection)
    final_state = compact_state(final_path)
    before_allocator = int(
        before_census["accounting"]["full_allocator"]["available"]
    )
    after_allocator = int(
        after_census["accounting"]["full_allocator"]["available"]
    )
    safety = {
        "selected_mamba_resident": all(
            after_states[checkpoint_id]["mamba_resident"]
            for checkpoint_id in selected_ids
        ),
        "unselected_mamba_evicted": True,
        "fa_preserved": all(
            state["fa_resident"] for state in after_states.values()
        ),
        "allocator_invariant": before_allocator == after_allocator,
        "tree_invariant": (
            before_census["tree"]["structure_sha256"]
            == after_census["tree"]["structure_sha256"]
        ),
        "path_invariant": all(
            before_paths[checkpoint_id]["path_node_ids"]
            == after_paths[checkpoint_id]["path_node_ids"]
            and before_paths[checkpoint_id]["prefix_sha256"]
            == after_paths[checkpoint_id]["prefix_sha256"]
            for checkpoint_id in before_paths
        ),
        "fa_identity_invariant": (
            before_paths[DEEP_CHECKPOINT_ID]["path_full_sha256"]
            == final_path["path_full_sha256"]
        ),
        "only_expected_mamba_changed": (
            before_census["tree"]["mamba_rows"]
            == after_census["tree"]["mamba_rows"]
            and before_states == after_states
        ),
        "sanity_check": bool(
            final_inspection["proof"]["sanity_check_passed"]
        ),
        "cascade_not_called": True,
        "formal_primitive": True,
        "target_fa_and_mamba_present": bool(
            final_state["fa_resident"] and final_state["mamba_resident"]
        ),
    }
    return safety, {
        "before_available_size": before_allocator,
        "after_available_size": after_allocator,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """稳定写出 UTF-8 JSON。"""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _append_raw_sample(record: Mapping[str, object]) -> None:
    """在每个 case 后立即持久化原始记录。"""
    with RAW_SAMPLES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        )


def _elapsed_ms(started_ns: int) -> float:
    """返回单调时钟经过的毫秒数。"""
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _ensure_fresh_output() -> None:
    """拒绝覆盖既有 profiler 数据 artifacts。"""
    generated_paths = (
        RAW_SAMPLES_PATH,
        RUN_METADATA_PATH,
        OUTPUT_DIRECTORY / "calibration_summary.csv",
        OUTPUT_DIRECTORY / "validation_summary.csv",
        OUTPUT_DIRECTORY / "model_comparison.json",
    )
    existing = tuple(path.name for path in generated_paths if path.exists())
    if existing:
        raise RuntimeError(f"拒绝覆盖既有 profiler artifacts：{existing}")


def main() -> int:
    """启动一次 GPU runtime 并完成全部 126 个 profiling cases。"""
    _ensure_fresh_output()
    schedule = build_profile_schedule()
    expected_cases = len(ALL_GAPS) * (
        WARMUP_REPETITIONS + MEASURED_REPETITIONS
    )
    if len(schedule) != expected_cases:
        raise RuntimeError("profiler schedule 数量异常")

    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import (
        FormalEndToEndGateEngine,
        requested_control_port,
    )

    started_at = datetime.now().astimezone().isoformat()
    base_order = tuple(
        case.target_gap
        for case in schedule[: len(ALL_GAPS)]
    )
    metadata: dict[str, object] = {
        "schema_version": "flowstate.recovery_profiler_v2.metadata.v1",
        "status": "RUNNING",
        "environment": "冻结 SGLang Docker runtime",
        "model": "Qwen3.5-9B",
        "sglang_version": importlib.metadata.version("sglang"),
        "gpu": "NVIDIA H100 PCIe 80 GiB",
        "tp": 1,
        "seed": ORDER_SEED,
        "base_gap_order": list(base_order),
        "calibration_gaps": list(CALIBRATION_GAPS),
        "held_out_validation_gaps": list(VALIDATION_GAPS),
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "total_expected_cases": expected_cases,
        "engine_configuration": STEP8E_ENGINE_CONFIGURATION,
        "ttft_boundary": "复用 Step 9B 首个流式 token client-side 计时",
        "step9b_data_used_for_fitting": False,
        "start_time": started_at,
        "end_time": None,
    }
    _write_json(RUN_METADATA_PATH, metadata)

    engine = None
    records = []
    fatal_error = None
    started = time.perf_counter_ns()
    try:
        engine = FormalEndToEndGateEngine(**STEP8E_ENGINE_CONFIGURATION)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        for index, case in enumerate(schedule, start=1):
            phase = "WARMUP" if case.is_warmup else "MEASURED"
            print(
                f"[STEP9D] {index}/{len(schedule)} {phase} "
                f"gap={case.target_gap}",
                flush=True,
            )
            record = execute_profile_case(
                case,
                engine=engine,
                client=client,
            )
            records.append(record)
            _append_raw_sample(record)
            if record["status"] != "PASS":
                fatal_error = (
                    f"case {case.case_id} 失败："
                    f"{record.get('failure_stage')} {record.get('error')}"
                )
                break
    except Exception as error:
        fatal_error = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                if fatal_error is None:
                    fatal_error = f"关闭 runtime 失败：{error!r}"

    total_runtime_ms = _elapsed_ms(started)
    completed = sum(record.get("status") == "PASS" for record in records)
    gap_mismatch = sum(record.get("gap_match") is False for record in records)
    status = (
        "PASS"
        if fatal_error is None
        and completed == expected_cases
        and gap_mismatch == 0
        else "FAIL"
    )
    if status == "PASS":
        analysis = analyze_records(records)
        write_analysis_artifacts(analysis, OUTPUT_DIRECTORY)
    metadata.update(
        {
            "status": status,
            "end_time": datetime.now().astimezone().isoformat(),
            "total_runtime_ms": total_runtime_ms,
            "completed_cases": completed,
            "raw_record_count": len(records),
            "gap_mismatch": gap_mismatch,
            "fatal_error": fatal_error,
        }
    )
    _write_json(RUN_METADATA_PATH, metadata)
    print(
        json.dumps(
            {
                "status": status,
                "completed": completed,
                "expected": expected_cases,
                "gap_mismatch": gap_mismatch,
                "fatal_error": fatal_error,
                "output_directory": str(OUTPUT_DIRECTORY),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
