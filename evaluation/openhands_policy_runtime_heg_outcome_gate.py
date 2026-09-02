#!/usr/bin/env python3
"""验证 K=2 不同循环状态选择产生的真实 H/E/G 结果。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.barrier_fa_frontier_control import BarrierFAControlClient
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    SchedulerRuntimeAdapter,
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
    PENDING_TURN,
    SCHEDULE,
    build_pending_set,
    load_barrier_requests,
    locate_materialized_candidate,
    token_digest,
    validate_candidate_at_barrier,
)
from evaluation.openhands_policy_to_actuator_mapping_gate import (
    ENGINE_CONFIGURATION_ACTUATOR_MAPPING,
    EXPECTED_EVICTIONS,
    SELECTED_SETS,
    FrozenSelectedSetOptimizer,
    RecordingRuntimeAdapter,
    build_controller_report,
    evaluate_mapping_invariants,
    inspect_candidate_states,
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
from flowstate.state_catalog import (
    CheckpointCandidate,
    validate_unique_checkpoint_ids,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
ENGINE_CONFIGURATION_HEG_OUTCOME = dict(
    ENGINE_CONFIGURATION_ACTUATOR_MAPPING
)
PENDING_SCHEDULE = tuple((label, PENDING_TURN) for label in WORKFLOWS)
RUN_ORDER = ("LM", "F")


def _artifact_directory() -> Path:
    """创建不会覆盖既有结果的时间戳产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_policy_runtime_heg_outcome_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """对仓库内路径返回相对表示。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def prepare_outcome_requests(
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    condition: str,
) -> tuple[dict[str, object], ...]:
    """为一个条件构造严格 A2、B2、C2、D2 串行请求。"""
    if condition not in SELECTED_SETS:
        raise ValueError(f"未知条件：{condition}")
    result = []
    for label, turn in PENDING_SCHEDULE:
        source = requests[(label, turn)]
        input_ids = source["input_ids"]
        if not isinstance(input_ids, list):
            raise TypeError("pending input_ids 必须是列表")
        result.append(
            {
                **source,
                "condition": condition,
                "rid": (
                    f"openhands-heg-{condition.lower()}-"
                    f"{label.lower()}-turn-002"
                ),
                "input_ids": list(input_ids),
                "input_token_digest": token_digest(input_ids),
            }
        )
    return tuple(result)


def attach_barrier_prediction(
    record: Mapping[str, object],
    prediction: Mapping[str, object],
) -> dict[str, object]:
    """把 barrier 预测与请求到达时实际 H/E/G 严格区分。"""
    label = str(record["workflow"])
    h_actual = record.get("h")
    h_pred = int(prediction["resident_fa_frontier"])
    anchor_pos = int(prediction["anchor_pos"])
    planning_target = int(prediction["planning_target"])
    delta = None if h_actual is None else int(h_actual) - h_pred
    return {
        **record,
        "h_pred_barrier": h_pred,
        "anchor_pos_barrier": anchor_pos,
        "planning_target_barrier": planning_target,
        "h_actual_minus_h_pred": delta,
        "barrier_prediction_equality_required": label == "A",
        "barrier_prediction_exact": (
            None if label != "A" or h_actual is None else int(h_actual) == h_pred
        ),
        "sequential_runtime_evolution_allowed": label != "A",
    }


def validate_condition_outcomes(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """验证一个条件的四个串行 outcome 结构与 A2 预测一致性。"""
    labels = [str(record.get("workflow")) for record in records]
    metrics_valid = bool(
        len(records) == len(PENDING_SCHEDULE)
        and labels == list(WORKFLOWS)
        and all(
            int(record.get("turn", -1)) == PENDING_TURN
            and record.get("status") == "PASS"
            and record.get("request_completed") is True
            and record.get("token_count_exact") is True
            and record.get("runtime_metrics_valid") is True
            and record.get("h") is not None
            and record.get("e") is not None
            and record.get("g") is not None
            and 0 <= int(record["e"]) <= int(record["h"])
            <= int(record["offline_input_tokens"])
            and int(record["g"]) == int(record["h"]) - int(record["e"])
            for record in records
        )
    )
    a_record = next(
        (record for record in records if record.get("workflow") == "A"),
        None,
    )
    a_prediction_exact = bool(
        a_record is not None
        and a_record.get("barrier_prediction_equality_required") is True
        and a_record.get("barrier_prediction_exact") is True
    )
    passed = metrics_valid and a_prediction_exact
    return {
        "status": "PASS" if passed else "FAIL",
        "schedule_exact": labels == list(WORKFLOWS),
        "runtime_metrics_valid": metrics_valid,
        "a2_h_pred_equals_h_actual": a_prediction_exact,
        "sequential_h_deltas": {
            str(record["workflow"]): record.get("h_actual_minus_h_pred")
            for record in records
        },
    }


def compare_policy_outcomes(
    lm_run: Mapping[str, object],
    flowstate_run: Mapping[str, object],
) -> dict[str, object]:
    """按 retained/evicted 语义比较 A2 与 D2 的真实 E/G。"""
    lm_records = {
        str(record["workflow"]): record
        for record in lm_run["outcome_records"]
    }
    flow_records = {
        str(record["workflow"]): record
        for record in flowstate_run["outcome_records"]
    }
    lm_positions = {
        str(row["workflow_label"]): int(row["token_pos"])
        for row in lm_run["candidate_rows"]
    }
    flow_positions = {
        str(row["workflow_label"]): int(row["token_pos"])
        for row in flowstate_run["candidate_rows"]
    }
    same_positions = lm_positions == flow_positions
    request_digests_equal = all(
        lm_records[label]["input_token_digest"]
        == flow_records[label]["input_token_digest"]
        for label in WORKFLOWS
    )
    a_position = lm_positions.get("A")
    d_position = lm_positions.get("D")
    a_signal = bool(
        a_position is not None
        and int(lm_records["A"]["e"]) < a_position
        and int(flow_records["A"]["e"]) >= a_position
        and (
            int(lm_records["A"]["e"]),
            int(lm_records["A"]["g"]),
        )
        != (
            int(flow_records["A"]["e"]),
            int(flow_records["A"]["g"]),
        )
    )
    d_signal = bool(
        d_position is not None
        and int(lm_records["D"]["e"]) >= d_position
        and int(flow_records["D"]["e"]) < d_position
        and (
            int(lm_records["D"]["e"]),
            int(lm_records["D"]["g"]),
        )
        != (
            int(flow_records["D"]["e"]),
            int(flow_records["D"]["g"]),
        )
    )
    c_retained_reference = bool(
        int(lm_records["C"]["e"]) >= lm_positions["C"]
        and int(flow_records["C"]["e"]) >= flow_positions["C"]
    )
    eg_differences = {
        label: (
            int(lm_records[label]["e"]),
            int(lm_records[label]["g"]),
        )
        != (
            int(flow_records[label]["e"]),
            int(flow_records[label]["g"]),
        )
        for label in WORKFLOWS
    }
    heg_differences = {
        label: (
            int(lm_records[label]["h"]),
            int(lm_records[label]["e"]),
            int(lm_records[label]["g"]),
        )
        != (
            int(flow_records[label]["h"]),
            int(flow_records[label]["e"]),
            int(flow_records[label]["g"]),
        )
        for label in WORKFLOWS
    }
    passed = bool(same_positions and request_digests_equal)
    return {
        "status": "PASS" if passed else "FAIL",
        "candidate_positions_equal": same_positions,
        "request_input_ids_equal": request_digests_equal,
        "a2_selected_set_signal": a_signal,
        "d2_selected_set_signal": d_signal,
        "c2_retained_reference_valid": c_retained_reference,
        "eg_differences": eg_differences,
        "heg_differences": heg_differences,
        "at_least_one_runtime_heg_outcome_differs": any(heg_differences.values()),
        "outcome_difference_required_for_correctness": False,
        "a2": {
            "LM": {name: lm_records["A"][name] for name in ("h", "e", "g")},
            "F": {
                name: flow_records["A"][name] for name in ("h", "e", "g")
            },
        },
        "d2": {
            "LM": {name: lm_records["D"][name] for name in ("h", "e", "g")},
            "F": {
                name: flow_records["D"][name] for name in ("h", "e", "g")
            },
        },
        "common_reference": {
            label: {
                "LM": {
                    name: lm_records[label][name] for name in ("h", "e", "g")
                },
                "F": {
                    name: flow_records[label][name]
                    for name in ("h", "e", "g")
                },
            }
            for label in ("B", "C")
        },
    }


@dataclass(frozen=True)
class RunPaths:
    """保存 H/E/G outcome gate 的产物路径。"""

    setup_requests: Path
    outcomes: Path
    censuses: Path
    predictions: Path
    controller_reports: Path
    runs: Path


def run_condition(
    *,
    condition: str,
    engine_ordinal: int,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    boundary_audit: Sequence[Mapping[str, object]],
    paths: RunPaths,
) -> dict[str, object]:
    """用一个全新 Engine 完成 barrier、reconcile 与四个 outcome 请求。"""
    engine = None
    setup_records: list[dict[str, object]] = []
    outcome_records: list[dict[str, object]] = []
    candidates: list[CheckpointCandidate] = []
    candidate_rows: list[dict[str, object]] = []
    handles: dict[str, RuntimeCheckpointHandle] = {}
    controller_report = None
    mapping_invariants = None
    outcome_validation = None
    native_eviction = False
    fa_cascade = False
    fatal_error = None
    shutdown_error = None
    try:
        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION_HEG_OUTCOME)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        baseline = compact_census(
            client.census(f"openhands-heg:{condition}:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        baseline["condition"] = condition
        baseline["event"] = "baseline"
        _append_jsonl(paths.censuses, baseline)
        if int(baseline["mamba_node_count"]) != 0:
            raise RuntimeError("全新 Engine 初始 census 含循环检查点")

        previous = baseline
        for ordinal, (label, turn) in enumerate(SCHEDULE, start=1):
            request = requests[(label, turn)]
            try:
                record = execute_request(engine, client, request, ordinal)
                census = compact_census(
                    client.census(
                        f"openhands-heg:{condition}:setup:{label}{turn}"
                    ),
                    ordinal=ordinal,
                    request=request,
                    previous=previous,
                )
                census["condition"] = condition
                census["event"] = f"setup_{label}{turn}"
                candidate, handle, row = locate_materialized_candidate(
                    client,
                    request,
                    census,
                    event_order=ordinal,
                )
            except Exception as error:
                record = _failure_record(request, ordinal, error)
                record["condition"] = condition
                setup_records.append(record)
                _append_jsonl(paths.setup_requests, record)
                raise
            record["condition"] = condition
            setup_records.append(record)
            candidates.append(candidate)
            candidate_rows.append(row)
            handles[candidate.checkpoint_id] = handle
            _append_jsonl(paths.setup_requests, record)
            _append_jsonl(paths.censuses, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{condition} 的 {label}{turn} 建态失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                native_eviction = True
                raise RuntimeError(f"{condition} 建态时发生原生 Mamba 驱逐")
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
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

        continuations, prediction_rows = build_pending_set(
            BarrierFAControlClient(client),
            requests,
        )
        prediction_by_label = {
            str(row["workflow_label"]): row for row in prediction_rows
        }
        for row in prediction_rows:
            _append_jsonl(
                paths.predictions,
                {**row, "condition": condition},
            )
        before_census = compact_census(
            client.census(f"openhands-heg:{condition}:before-reconcile"),
            ordinal=len(SCHEDULE),
            request=None,
            previous=previous,
        )
        before_census["condition"] = condition
        before_census["event"] = "before_reconcile"
        _append_jsonl(paths.censuses, before_census)
        before_states, _ = inspect_candidate_states(
            client,
            candidates,
            handles,
            phase=f"{condition}_HEG_BEFORE",
        )

        delegate = SchedulerRuntimeAdapter(client)
        recording_adapter = RecordingRuntimeAdapter(delegate)
        controller = StateController(
            FrozenSelectedSetOptimizer(SELECTED_SETS[condition]),
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
            client.census(f"openhands-heg:{condition}:after-reconcile"),
            ordinal=len(SCHEDULE),
            request=None,
            previous=before_census,
        )
        after_census["condition"] = condition
        after_census["event"] = "after_reconcile"
        _append_jsonl(paths.censuses, after_census)
        after_states, _ = inspect_candidate_states(
            client,
            candidates,
            handles,
            phase=f"{condition}_HEG_AFTER",
        )
        mapping_invariants = evaluate_mapping_invariants(
            candidate_ids=[item.checkpoint_id for item in candidates],
            selected_ids=SELECTED_SETS[condition],
            expected_evicted_ids=EXPECTED_EVICTIONS[condition],
            handles=handles,
            before_states=before_states,
            after_states=after_states,
            before_census=before_census,
            after_census=after_census,
            controller_report=controller_report,
        )
        if mapping_invariants["status"] != "PASS":
            raise RuntimeError(f"{condition} reconcile correctness 失败")
        previous = after_census

        outcome_requests = prepare_outcome_requests(requests, condition)
        for offset, request in enumerate(outcome_requests, start=1):
            ordinal = len(SCHEDULE) + offset
            label = str(request["workflow_label"])
            try:
                raw_record = execute_request(engine, client, request, ordinal)
                record = attach_barrier_prediction(
                    raw_record,
                    prediction_by_label[label],
                )
                record["condition"] = condition
                record["input_token_digest"] = request[
                    "input_token_digest"
                ]
                census = compact_census(
                    client.census(
                        f"openhands-heg:{condition}:outcome:{label}2"
                    ),
                    ordinal=ordinal,
                    request=request,
                    previous=previous,
                )
                census["condition"] = condition
                census["event"] = f"outcome_{label}2"
            except Exception as error:
                record = _failure_record(request, ordinal, error)
                record["condition"] = condition
                record["input_token_digest"] = request[
                    "input_token_digest"
                ]
                outcome_records.append(record)
                _append_jsonl(paths.outcomes, record)
                raise
            outcome_records.append(record)
            _append_jsonl(paths.outcomes, record)
            _append_jsonl(paths.censuses, census)
            previous = census
            if census["native_mamba_capacity_eviction_inferred"]:
                native_eviction = True
                raise RuntimeError(
                    f"{condition} 的 {label}2 发生原生 Mamba 驱逐"
                )
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
                raise RuntimeError(
                    f"{condition} 的 {label}2 发生 FA-KV 级联"
                )

        outcome_validation = validate_condition_outcomes(outcome_records)
        if outcome_validation["status"] != "PASS":
            raise RuntimeError(f"{condition} 的 H/E/G outcome 校验失败")
    except Exception as error:
        fatal_error = repr(error)
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
        for item in boundary_audit
    )
    passed = bool(
        fatal_error is None
        and shutdown_error is None
        and mapping_invariants is not None
        and mapping_invariants.get("status") == "PASS"
        and outcome_validation is not None
        and outcome_validation.get("status") == "PASS"
        and not native_eviction
        and not fa_cascade
        and not future_information_used
    )
    result = {
        "condition": condition,
        "engine_lifecycle": "independent_fresh",
        "engine_ordinal": engine_ordinal,
        "status": "PASS" if passed else "FAIL",
        "selected_ids": list(SELECTED_SETS[condition]),
        "expected_evicted_ids": list(EXPECTED_EVICTIONS[condition]),
        "setup_schedule": [f"{label}{turn}" for label, turn in SCHEDULE],
        "outcome_schedule": [
            f"{label}{turn}" for label, turn in PENDING_SCHEDULE
        ],
        "candidate_rows": candidate_rows,
        "barrier_predictions": (
            [] if "prediction_rows" not in locals() else prediction_rows
        ),
        "controller_report": controller_report,
        "mapping_invariants": mapping_invariants,
        "outcome_records": outcome_records,
        "outcome_validation": outcome_validation,
        "native_mamba_capacity_eviction": native_eviction,
        "fa_kv_cascade": fa_cascade,
        "future_information_used": future_information_used,
        "ttft_telemetry_saved_only": True,
        "ttft_compared": False,
        "performance_claim_made": False,
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
    """汇总两个独立 selected set 的 runtime H/E/G correctness。"""
    by_condition = {str(run["condition"]): run for run in runs}
    complete = bool(
        set(by_condition) == set(RUN_ORDER)
        and all(run.get("status") == "PASS" for run in runs)
    )
    comparison = None
    if complete:
        comparison = compare_policy_outcomes(
            by_condition["LM"],
            by_condition["F"],
        )
    future_information_used = any(
        item.get("r_plus_2_message_consumed") is not False
        or item.get("r_plus_2_request_materialized") is not False
        or item.get("pending_assistant_output_read") is not False
        for item in boundary_audit
    )
    passed = bool(
        complete
        and comparison is not None
        and comparison.get("status") == "PASS"
        and not future_information_used
    )
    return {
        "schema_version": "flowstate.openhands_policy_runtime_heg_outcome.v1",
        "status": "PASS" if passed else "FAIL",
        "verdict": "READY" if passed else "PARTIAL",
        "artifact": _display_path(artifact),
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_HEG_OUTCOME,
        "engine_lifecycle_count": len(runs),
        "independent_engine_lifecycles": all(
            run.get("engine_lifecycle") == "independent_fresh"
            for run in runs
        ),
        "logical_k": 2,
        "budget_bytes": BUDGET_BYTES,
        "run_order": list(RUN_ORDER),
        "setup_schedule": [f"{label}{turn}" for label, turn in SCHEDULE],
        "outcome_schedule": [
            f"{label}{turn}" for label, turn in PENDING_SCHEDULE
        ],
        "runs": list(runs),
        "policy_outcome_comparison": comparison,
        "future_information_used": future_information_used,
        "future_leakage": future_information_used,
        "online_information_boundary": list(boundary_audit),
        "ttft_telemetry_saved_only": True,
        "ttft_compared": False,
        "statistical_test_performed": False,
        "performance_claim_made": False,
        "environment": dict(environment) if environment is not None else None,
    }


def _run(artifact: Path) -> dict[str, object]:
    """顺序执行两个 fresh Engine，任一条件失败后立即停止。"""
    paths = RunPaths(
        setup_requests=artifact / "setup_requests.jsonl",
        outcomes=artifact / "outcomes.jsonl",
        censuses=artifact / "census.jsonl",
        predictions=artifact / "barrier_predictions.jsonl",
        controller_reports=artifact / "controller_reports.jsonl",
        runs=artifact / "runs.jsonl",
    )
    for path in (
        paths.setup_requests,
        paths.outcomes,
        paths.censuses,
        paths.predictions,
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
            "run_order": list(RUN_ORDER),
            "selected_sets": {
                key: list(value) for key, value in SELECTED_SETS.items()
            },
            "expected_evictions": {
                key: list(value) for key, value in EXPECTED_EVICTIONS.items()
            },
            "engine_configuration": ENGINE_CONFIGURATION_HEG_OUTCOME,
            "sampling_parameters": SAMPLING_PARAMETERS,
            "dataset_path": str(DATASET_PATH),
            "tokenizer_path": str(TOKENIZER_PATH),
            "setup_schedule": [
                f"{label}{turn}" for label, turn in SCHEDULE
            ],
            "outcome_schedule": [
                f"{label}{turn}" for label, turn in PENDING_SCHEDULE
            ],
            "barrier_h_pred_source": "inspect_fa_frontier",
            "a2_prediction_equality_required": True,
            "later_prediction_equality_required": False,
            "logical_k": 2,
            "budget_bytes": BUDGET_BYTES,
            "online_information_boundary": boundary_audit,
            "ttft_telemetry_saved_only": True,
            "ttft_compared": False,
            "environment": environment,
        },
    )
    runs = []
    for ordinal, condition in enumerate(RUN_ORDER, start=1):
        result = run_condition(
            condition=condition,
            engine_ordinal=ordinal,
            requests=requests,
            boundary_audit=boundary_audit,
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
    """保存完整日志并执行唯一一次 H/E/G outcome 门禁。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
