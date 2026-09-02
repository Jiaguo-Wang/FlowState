#!/usr/bin/env python3
"""对五组独立 Engine 配对试验执行 K=2 policy TTFT 门禁。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
)
from evaluation.openhands_4workflow_occupancy_calibration import (
    WORKFLOWS,
    _environment,
)
from evaluation.openhands_common_barrier_snapshot_gate import (
    BUDGET_BYTES,
    PENDING_TURN,
    SCHEDULE,
    load_barrier_requests,
)
from evaluation.openhands_policy_runtime_heg_outcome_gate import (
    ENGINE_CONFIGURATION_HEG_OUTCOME,
    PENDING_SCHEDULE,
    RunPaths,
    run_condition,
)
from evaluation.openhands_policy_to_actuator_mapping_gate import (
    EXPECTED_EVICTIONS,
    SELECTED_SETS,
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
TRIAL_COUNT = 5
POLICY_ORDER = ("LM", "F")
ENGINE_LIFECYCLE_COUNT = TRIAL_COUNT * len(POLICY_ORDER)
ENGINE_CONFIGURATION_PAIRED_TTFT = dict(
    ENGINE_CONFIGURATION_HEG_OUTCOME
)


def _artifact_directory() -> Path:
    """创建不会覆盖既有结果的配对 TTFT 产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_policy_paired_ttft_runtime_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """对仓库内路径返回相对表示。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _trial_paths(directory: Path) -> RunPaths:
    """创建一个配对 trial 独占的底层运行产物文件。"""
    directory.mkdir(parents=True, exist_ok=False)
    paths = RunPaths(
        setup_requests=directory / "setup_requests.jsonl",
        outcomes=directory / "outcomes.jsonl",
        censuses=directory / "census.jsonl",
        predictions=directory / "barrier_predictions.jsonl",
        controller_reports=directory / "controller_reports.jsonl",
        runs=directory / "runs.jsonl",
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
    return paths


def _records_by_workflow(
    run: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    """按 workflow 标记索引四个正式测量请求。"""
    records = run.get("outcome_records")
    if not isinstance(records, Sequence):
        raise TypeError("outcome_records 必须是序列")
    indexed = {
        str(record["workflow"]): record
        for record in records
        if isinstance(record, Mapping)
    }
    if set(indexed) != set(WORKFLOWS):
        raise ValueError("正式测量请求必须完整覆盖 A2、B2、C2、D2")
    return indexed


def _positions_by_workflow(
    run: Mapping[str, object],
) -> dict[str, int]:
    """读取 Round 1 四个候选检查点的 token 位置。"""
    rows = run.get("candidate_rows")
    if not isinstance(rows, Sequence):
        raise TypeError("candidate_rows 必须是序列")
    positions = {
        str(row["workflow_label"]): int(row["token_pos"])
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(positions) != set(WORKFLOWS):
        raise ValueError("候选位置必须完整覆盖 A1、B1、C1、D1")
    return positions


def summarize_policy_run(run: Mapping[str, object]) -> dict[str, object]:
    """汇总一个 policy 的四个正式测量请求。"""
    records = _records_by_workflow(run)
    positions = _positions_by_workflow(run)
    ttft_by_workflow = {
        label: float(records[label]["ttft_ms"]) for label in WORKFLOWS
    }
    latency_by_workflow = {
        label: float(records[label]["request_latency_ms"])
        for label in WORKFLOWS
    }
    g_by_workflow = {
        label: int(records[label]["g"]) for label in WORKFLOWS
    }
    return {
        "condition": run["condition"],
        "aggregate_ttft_ms": sum(ttft_by_workflow.values()),
        "aggregate_latency_ms": sum(latency_by_workflow.values()),
        "aggregate_g": sum(g_by_workflow.values()),
        "ttft_ms_by_workflow": ttft_by_workflow,
        "latency_ms_by_workflow": latency_by_workflow,
        "g_by_workflow": g_by_workflow,
        "input_token_digests": {
            label: records[label].get("input_token_digest")
            for label in WORKFLOWS
        },
        "candidate_positions": positions,
        "outcomes": {
            label: {
                name: records[label].get(name)
                for name in (
                    "h",
                    "e",
                    "g",
                    "ttft_ms",
                    "request_latency_ms",
                    "cached_tokens",
                    "server_prompt_tokens",
                )
            }
            for label in WORKFLOWS
        },
    }


def audit_paired_runtime_semantics(
    lm_run: Mapping[str, object],
    flowstate_run: Mapping[str, object],
) -> dict[str, object]:
    """验证 selected set、可执行前沿和公平条件的配对语义。"""
    lm_records = _records_by_workflow(lm_run)
    flow_records = _records_by_workflow(flowstate_run)
    lm_positions = _positions_by_workflow(lm_run)
    flow_positions = _positions_by_workflow(flowstate_run)
    same_positions = lm_positions == flow_positions
    same_inputs = all(
        lm_records[label].get("input_token_digest")
        == flow_records[label].get("input_token_digest")
        for label in WORKFLOWS
    )
    all_g_exact = all(
        int(record["g"]) == int(record["h"]) - int(record["e"])
        for record in (*lm_records.values(), *flow_records.values())
    )
    a_semantic = bool(
        int(lm_records["A"]["e"]) < lm_positions["A"]
        and int(lm_records["A"]["g"]) > 0
        and int(flow_records["A"]["e"]) >= flow_positions["A"]
    )
    b_semantic = bool(
        int(lm_records["B"]["e"]) < lm_positions["B"]
        and int(lm_records["B"]["g"]) > 0
        and int(flow_records["B"]["e"]) < flow_positions["B"]
        and int(flow_records["B"]["g"]) > 0
    )
    c_semantic = bool(
        int(lm_records["C"]["e"]) >= lm_positions["C"]
        and int(flow_records["C"]["e"]) >= flow_positions["C"]
    )
    d_semantic = bool(
        int(lm_records["D"]["e"]) >= lm_positions["D"]
        and int(flow_records["D"]["e"]) < flow_positions["D"]
        and int(flow_records["D"]["g"]) > 0
    )
    lm_mapping = lm_run.get("mapping_invariants")
    flowstate_mapping = flowstate_run.get("mapping_invariants")
    selected_residency_exact = bool(
        isinstance(lm_mapping, Mapping)
        and lm_mapping.get("selected_residency_exact") is True
        and isinstance(flowstate_mapping, Mapping)
        and flowstate_mapping.get("selected_residency_exact") is True
    )
    native_eviction = bool(
        lm_run.get("native_mamba_capacity_eviction")
        or flowstate_run.get("native_mamba_capacity_eviction")
    )
    fa_cascade = bool(
        lm_run.get("fa_kv_cascade")
        or flowstate_run.get("fa_kv_cascade")
    )
    future_information_used = bool(
        lm_run.get("future_information_used")
        or flowstate_run.get("future_information_used")
    )
    policy_specific = all(
        (a_semantic, b_semantic, c_semantic, d_semantic)
    )
    passed = bool(
        lm_run.get("status") == "PASS"
        and flowstate_run.get("status") == "PASS"
        and same_positions
        and same_inputs
        and all_g_exact
        and policy_specific
        and selected_residency_exact
        and not native_eviction
        and not fa_cascade
        and not future_information_used
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "candidate_positions_equal": same_positions,
        "request_input_ids_equal": same_inputs,
        "all_g_equals_h_minus_e": all_g_exact,
        "policy_specific_heg_reproduced": policy_specific,
        "workflow_semantics": {
            "A": a_semantic,
            "B": b_semantic,
            "C": c_semantic,
            "D": d_semantic,
        },
        "selected_residency_exact": selected_residency_exact,
        "native_mamba_capacity_eviction": native_eviction,
        "fa_kv_cascade": fa_cascade,
        "future_information_used": future_information_used,
    }


def build_paired_trial(
    trial_index: int,
    lm_run: Mapping[str, object],
    flowstate_run: Mapping[str, object],
) -> dict[str, object]:
    """计算单个配对 trial 的聚合 TTFT 和逐 workflow 差值。"""
    lm = summarize_policy_run(lm_run)
    flowstate = summarize_policy_run(flowstate_run)
    semantics = audit_paired_runtime_semantics(lm_run, flowstate_run)
    workflow_deltas = {
        label: (
            float(flowstate["ttft_ms_by_workflow"][label])
            - float(lm["ttft_ms_by_workflow"][label])
        )
        for label in WORKFLOWS
    }
    paired_delta = (
        float(flowstate["aggregate_ttft_ms"])
        - float(lm["aggregate_ttft_ms"])
    )
    passed = bool(semantics["status"] == "PASS")
    return {
        "trial": trial_index,
        "status": "PASS" if passed else "FAIL",
        "engine_lifecycles": [
            lm_run.get("engine_ordinal"),
            flowstate_run.get("engine_ordinal"),
        ],
        "lm": lm,
        "flowstate": flowstate,
        "paired_delta_ms": paired_delta,
        "flowstate_faster": paired_delta < 0,
        "workflow_paired_delta_ms": workflow_deltas,
        "runtime_correctness": semantics,
        "statistical_significance_claim": False,
    }


def _sample_statistics(values: Sequence[float]) -> dict[str, object]:
    """计算不包含显著性检验的描述统计。"""
    if not values:
        return {
            "values": [],
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "sample_std": None,
        }
    numeric = [float(value) for value in values]
    return {
        "values": numeric,
        "mean": mean(numeric),
        "median": median(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "sample_std": stdev(numeric) if len(numeric) >= 2 else None,
    }


def build_summary(
    *,
    artifact: Path,
    trials: Sequence[Mapping[str, object]],
    boundary_audit: Sequence[Mapping[str, object]],
    environment: Mapping[str, object] | None,
) -> dict[str, object]:
    """汇总五个配对 trial 的 TTFT 与 runtime correctness。"""
    valid_trials = [
        trial for trial in trials if trial.get("status") == "PASS"
    ]
    lm_values = [
        float(trial["lm"]["aggregate_ttft_ms"])
        for trial in valid_trials
    ]
    flowstate_values = [
        float(trial["flowstate"]["aggregate_ttft_ms"])
        for trial in valid_trials
    ]
    paired_values = [
        float(trial["paired_delta_ms"]) for trial in valid_trials
    ]
    lm_statistics = _sample_statistics(lm_values)
    flowstate_statistics = _sample_statistics(flowstate_values)
    paired_statistics = _sample_statistics(paired_values)
    input_signatures = [
        tuple(
            trial["lm"]["input_token_digests"][label]
            for label in WORKFLOWS
        )
        for trial in valid_trials
    ]
    position_signatures = [
        tuple(
            trial["lm"]["candidate_positions"][label]
            for label in WORKFLOWS
        )
        for trial in valid_trials
    ]
    same_inputs_across_trials = bool(
        len(valid_trials) == TRIAL_COUNT
        and len(set(input_signatures)) == 1
    )
    same_positions_across_trials = bool(
        len(valid_trials) == TRIAL_COUNT
        and len(set(position_signatures)) == 1
    )
    lm_mean = lm_statistics["mean"]
    flowstate_mean = flowstate_statistics["mean"]
    relative_mean_difference = (
        None
        if lm_mean in (None, 0)
        else (float(flowstate_mean) - float(lm_mean))
        / float(lm_mean)
        * 100.0
    )
    future_information_used = any(
        item.get("r_plus_2_message_consumed") is not False
        or item.get("r_plus_2_request_materialized") is not False
        or item.get("pending_assistant_output_read") is not False
        for item in boundary_audit
    ) or any(
        trial.get("runtime_correctness", {}).get(
            "future_information_used"
        )
        is True
        for trial in trials
    )
    all_g_exact = bool(
        len(valid_trials) == TRIAL_COUNT
        and all(
            trial["runtime_correctness"]["all_g_equals_h_minus_e"]
            is True
            for trial in valid_trials
        )
    )
    policy_specific = bool(
        len(valid_trials) == TRIAL_COUNT
        and all(
            trial["runtime_correctness"]["policy_specific_heg_reproduced"]
            is True
            for trial in valid_trials
        )
    )
    native_eviction = any(
        trial.get("runtime_correctness", {}).get(
            "native_mamba_capacity_eviction"
        )
        is True
        for trial in trials
    )
    fa_cascade = any(
        trial.get("runtime_correctness", {}).get("fa_kv_cascade") is True
        for trial in trials
    )
    complete = bool(
        len(trials) == TRIAL_COUNT
        and len(valid_trials) == TRIAL_COUNT
    )
    passed = bool(
        complete
        and all_g_exact
        and policy_specific
        and same_inputs_across_trials
        and same_positions_across_trials
        and not native_eviction
        and not fa_cascade
        and not future_information_used
    )
    return {
        "schema_version": "flowstate.openhands_policy_paired_ttft.v1",
        "status": "PASS" if passed else "FAIL",
        "verdict": "READY" if passed else "PARTIAL",
        "artifact": _display_path(artifact),
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_PAIRED_TTFT,
        "trial_count_requested": TRIAL_COUNT,
        "trial_count_completed": len(trials),
        "valid_pair_count": len(valid_trials),
        "engine_lifecycle_count_expected": ENGINE_LIFECYCLE_COUNT,
        "engine_lifecycle_count_observed": sum(
            len(trial.get("engine_lifecycles", ())) for trial in trials
        ),
        "fresh_engine_per_policy_per_trial": True,
        "policy_order_per_trial": list(POLICY_ORDER),
        "logical_k": 2,
        "budget_bytes": BUDGET_BYTES,
        "trials": list(trials),
        "paired_summary": {
            "flowstate_faster_count": sum(
                bool(trial["flowstate_faster"]) for trial in valid_trials
            ),
            "lm_aggregate_ttft_ms": lm_statistics,
            "flowstate_aggregate_ttft_ms": flowstate_statistics,
            "paired_delta_ms": paired_statistics,
            "relative_mean_difference_percent": relative_mean_difference,
            "relative_mean_improvement_percent": (
                None
                if relative_mean_difference is None
                else -relative_mean_difference
            ),
            "paired_delta_definition": (
                "FlowState 聚合 TTFT 减去 LM 聚合 TTFT"
            ),
            "negative_delta_meaning": "FlowState 更快",
        },
        "runtime_correctness": {
            "all_g_equals_h_minus_e": all_g_exact,
            "policy_specific_heg_reproduced": policy_specific,
            "same_request_inputs_across_trials": same_inputs_across_trials,
            "same_candidate_positions_across_trials": (
                same_positions_across_trials
            ),
            "native_mamba_capacity_eviction": native_eviction,
            "fa_kv_cascade": fa_cascade,
            "future_information_used": future_information_used,
        },
        "future_leakage": future_information_used,
        "online_information_boundary": list(boundary_audit),
        "statistical_significance_test_performed": False,
        "statistical_significance_claim": False,
        "environment": dict(environment) if environment is not None else None,
    }


def _incomplete_trial(
    trial_index: int,
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """记录因底层条件失败而未形成有效配对的 trial。"""
    return {
        "trial": trial_index,
        "status": "FAIL",
        "engine_lifecycles": [
            run.get("engine_ordinal") for run in runs
        ],
        "conditions_completed": [run.get("condition") for run in runs],
        "fatal_errors": [run.get("fatal_error") for run in runs],
        "runtime_correctness": {
            "all_g_equals_h_minus_e": False,
            "policy_specific_heg_reproduced": False,
            "native_mamba_capacity_eviction": any(
                run.get("native_mamba_capacity_eviction") for run in runs
            ),
            "fa_kv_cascade": any(
                run.get("fa_kv_cascade") for run in runs
            ),
            "future_information_used": any(
                run.get("future_information_used") for run in runs
            ),
        },
    }


def _run(artifact: Path) -> dict[str, object]:
    """按 LM、FlowState 交替顺序执行五个独立配对 trial。"""
    trials_path = artifact / "trials.jsonl"
    trials_path.write_text("", encoding="utf-8")
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
            "trial_count": TRIAL_COUNT,
            "policy_order_per_trial": list(POLICY_ORDER),
            "engine_lifecycle_count": ENGINE_LIFECYCLE_COUNT,
            "fresh_engine_per_policy_per_trial": True,
            "selected_sets": {
                key: list(value) for key, value in SELECTED_SETS.items()
            },
            "expected_evictions": {
                key: list(value) for key, value in EXPECTED_EVICTIONS.items()
            },
            "engine_configuration": ENGINE_CONFIGURATION_PAIRED_TTFT,
            "sampling_parameters": SAMPLING_PARAMETERS,
            "dataset_path": str(DATASET_PATH),
            "tokenizer_path": str(TOKENIZER_PATH),
            "setup_schedule": [
                f"{label}{turn}" for label, turn in SCHEDULE
            ],
            "outcome_schedule": [
                f"{label}{turn}" for label, turn in PENDING_SCHEDULE
            ],
            "formal_measurement_requests": [
                f"{label}{PENDING_TURN}" for label in WORKFLOWS
            ],
            "ttft_definition": "客户端边界首 token 时延",
            "paired_delta_definition": (
                "FlowState 聚合 TTFT 减去 LM 聚合 TTFT"
            ),
            "logical_k": 2,
            "budget_bytes": BUDGET_BYTES,
            "online_information_boundary": boundary_audit,
            "future_turn_materialized": False,
            "statistical_significance_test_performed": False,
            "environment": environment,
        },
    )
    trials = []
    stop = False
    for trial_index in range(1, TRIAL_COUNT + 1):
        paths = _trial_paths(
            artifact / f"trial_{trial_index:02d}"
        )
        runs = []
        for policy_offset, condition in enumerate(POLICY_ORDER):
            engine_ordinal = (
                (trial_index - 1) * len(POLICY_ORDER)
                + policy_offset
                + 1
            )
            result = run_condition(
                condition=condition,
                engine_ordinal=engine_ordinal,
                requests=requests,
                boundary_audit=boundary_audit,
                paths=paths,
            )
            runs.append(result)
            if result["status"] != "PASS":
                stop = True
                break
        if (
            len(runs) == len(POLICY_ORDER)
            and all(run.get("status") == "PASS" for run in runs)
        ):
            trial = build_paired_trial(
                trial_index,
                runs[0],
                runs[1],
            )
        else:
            trial = _incomplete_trial(trial_index, runs)
        trials.append(trial)
        _append_jsonl(trials_path, trial)
        if trial["status"] != "PASS":
            stop = True
        if stop:
            break
    summary = build_summary(
        artifact=artifact,
        trials=trials,
        boundary_audit=boundary_audit,
        environment=environment,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行唯一一次五配对 TTFT 门禁。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
