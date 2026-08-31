#!/usr/bin/env python3
"""执行 OpenHands 循环状态恢复的配对因果时延实验。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import statistics
import traceback
from typing import Mapping, Sequence

from transformers import AutoTokenizer

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
from evaluation.openhands_recurrent_only_causal_gate import (
    _census_unexpected_mamba_change,
    exact_lcp,
    locate_latest_checkpoint,
    validate_eviction_response,
    verify_final_checkpoint_state,
)
from evaluation.openhands_single_workflow_baseline10 import (
    ArtifactLogCapture,
    _append_jsonl,
    _write_json,
)
from evaluation.openhands_single_workflow_smoke import (
    DATASET_PATH,
    TOKENIZER_PATH,
    build_request_inputs,
    load_workflow_messages,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
WORKFLOW_LABEL = "B"
WORKFLOW_ID = WORKFLOWS[WORKFLOW_LABEL]
TARGET_TURNS = (1, 2, 3)
CONDITIONS = ("CONTROL", "EVICT")
REPETITIONS = 5
TRIAL_ORDER = tuple(
    (condition, repetition)
    for repetition in range(1, REPETITIONS + 1)
    for condition in CONDITIONS
)
ENGINE_CONFIGURATION_PAIRED_LATENCY = {
    **ENGINE_CONFIGURATION_128K,
    "max_mamba_cache_size": 28,
}


def _artifact_directory() -> Path:
    """创建不会覆盖既有结果的时间戳目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_recurrent_recovery_paired_latency_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """优先返回仓库内相对路径。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def prepare_requests(
    tokenizer: object,
) -> tuple[dict[int, dict[str, object]], int]:
    """只构造 Workflow B 的前三轮 recorded-history 请求。"""
    messages, n_turns = load_workflow_messages(
        dataset_path=DATASET_PATH,
        workflow_id=WORKFLOW_ID,
    )
    requests = build_request_inputs(
        tokenizer,
        messages,
        target_turns=TARGET_TURNS,
    )
    by_turn: dict[int, dict[str, object]] = {}
    previous_ids: list[int] | None = None
    for request in requests:
        turn = int(request["turn"])
        input_ids = request["input_ids"]
        if not isinstance(input_ids, list):
            raise TypeError("input_ids 必须是列表")
        by_turn[turn] = {
            "workflow_label": WORKFLOW_LABEL,
            "workflow_id": WORKFLOW_ID,
            "turn": turn,
            "input_ids": input_ids,
            "exact_adjacent_lcp": (
                None if previous_ids is None else exact_lcp(previous_ids, input_ids)
            ),
        }
        previous_ids = input_ids
    if set(by_turn) != set(TARGET_TURNS):
        raise RuntimeError("Workflow B 未构造出完整的前三轮请求")
    return by_turn, n_turns


def summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    """计算有限样本的描述统计，标准差使用样本定义。"""
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "count": 0,
            "values": [],
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "sample_std": None,
        }
    return {
        "count": len(numeric),
        "values": numeric,
        "mean": statistics.mean(numeric),
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "sample_std": (
            statistics.stdev(numeric) if len(numeric) >= 2 else None
        ),
    }


def _trial_request(
    request: Mapping[str, object],
    condition: str,
    repetition: int,
) -> dict[str, object]:
    """为独立 trial 生成稳定且唯一的请求标识。"""
    return {
        **request,
        "rid": (
            f"openhands-paired-{condition.lower()}-{repetition:02d}-"
            f"b{int(request['turn'])}"
        ),
    }


def _execute_trial_request(
    engine: object,
    client: object,
    request: Mapping[str, object],
    ordinal: int,
) -> dict[str, object]:
    """执行 trial 内请求并补充相邻 recorded-history LCP。"""
    record = execute_request(engine, client, request, ordinal)
    record["exact_adjacent_lcp"] = request["exact_adjacent_lcp"]
    return record


def _compact_eviction(
    response: Mapping[str, object],
    checkpoint_position: int,
    node_id: int,
) -> dict[str, object]:
    """保存正式 actuator 的关键证明和完整原始响应。"""
    before = response["before"]
    after = response["after"]
    proof = response["proof"]
    return {
        "checkpoint_position": checkpoint_position,
        "node_id": node_id,
        "recurrent_removed": bool(
            before["path"]["target_mamba_present"]
            and not after["path"]["target_mamba_present"]
        ),
        "fa_kv_preserved": bool(proof["fa_unchanged"]),
        "fa_allocator_before": before["accounting"]["full_allocator"],
        "fa_allocator_after": after["accounting"]["full_allocator"],
        "tree_identity_preserved": bool(proof["tree_unchanged"]),
        "path_identity_preserved": bool(proof["path_unchanged"]),
        "same_node": bool(proof["same_node"]),
        "cascade_called": bool(proof["cascade_called"]),
        "only_target_mamba_changed": bool(
            proof["only_target_mamba_changed"]
        ),
        "correctness_pass": validate_eviction_response(response),
        "formal_response": response,
    }


def evaluate_trial(
    trial: Mapping[str, object],
) -> dict[str, object]:
    """根据安全门和 B3 语义计算 trial 有效性。"""
    condition = str(trial["condition"])
    measured = trial.get("measured_request")
    semantic_pass = False
    if isinstance(measured, Mapping):
        h_value = measured.get("h")
        e_value = measured.get("e")
        g_value = measured.get("g")
        if all(value is not None for value in (h_value, e_value, g_value)):
            if condition == "CONTROL":
                semantic_pass = int(h_value) == int(e_value) and int(g_value) == 0
            else:
                semantic_pass = int(h_value) > int(e_value) and int(g_value) > 0
    eviction_pass = condition == "CONTROL" or bool(
        (trial.get("eviction") or {}).get("correctness_pass")
    )
    request_pass = bool(
        isinstance(measured, Mapping)
        and measured.get("request_completed") is True
        and measured.get("token_count_exact") is True
        and measured.get("runtime_metrics_valid") is True
    )
    valid = bool(
        trial.get("error") is None
        and trial.get("engine_shutdown_error") is None
        and trial.get("setup_requests_valid") is True
        and request_pass
        and semantic_pass
        and eviction_pass
        and trial.get("native_mamba_capacity_eviction") is False
        and trial.get("fa_kv_cascade") is False
        and trial.get("oom") is False
        and trial.get("truncation_or_clipping") is False
    )
    return {
        **trial,
        "semantic_pass": semantic_pass,
        "recurrent_only_correctness_pass": eviction_pass,
        "valid": valid,
        "status": "VALID" if valid else "INVALID",
    }


def run_trial(
    *,
    condition: str,
    repetition: int,
    prepared_requests: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """在独立 Engine 生命周期内执行一个 CONTROL 或 EVICT trial。"""
    trial_id = f"{condition}-{repetition}"
    engine = None
    client = None
    setup_records: list[dict[str, object]] = []
    measured_record = None
    eviction = None
    error_text = None
    shutdown_error = None
    native_eviction = False
    fa_cascade = False
    census_snapshots: list[dict[str, object]] = []
    try:
        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        print(f"[配对时延] 开始 {trial_id}", flush=True)
        engine = FormalEndToEndGateEngine(
            **ENGINE_CONFIGURATION_PAIRED_LATENCY
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        baseline = compact_census(
            client.census(f"openhands_paired:{trial_id}:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        census_snapshots.append(baseline)
        if baseline["mamba_node_count"] != 0:
            raise RuntimeError("独立 Engine 初始 Mamba census 非空")
        previous = baseline
        requests_by_turn = {
            turn: _trial_request(request, condition, repetition)
            for turn, request in prepared_requests.items()
        }

        for turn in (1, 2):
            request = requests_by_turn[turn]
            try:
                record = _execute_trial_request(
                    engine,
                    client,
                    request,
                    turn,
                )
            except Exception as request_error:
                record = _failure_record(request, turn, request_error)
                record["exact_adjacent_lcp"] = request["exact_adjacent_lcp"]
                setup_records.append(record)
                raise
            setup_records.append(record)
            census = compact_census(
                client.census(f"openhands_paired:{trial_id}:after:B{turn}"),
                ordinal=turn,
                request=request,
                previous=previous,
            )
            census_snapshots.append(census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{trial_id} 的 B{turn} token 数不一致")
            if census["native_mamba_capacity_eviction_inferred"]:
                native_eviction = True
                raise RuntimeError(f"{trial_id} 建态时发生原生 Mamba 驱逐")
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
                raise RuntimeError(f"{trial_id} 建态时发生 FA-KV 节点删除")

        if condition == "EVICT":
            handle, checkpoint_info = locate_latest_checkpoint(
                client,
                requests_by_turn[2],
                previous,
            )
            pre_eviction_census = previous
            adapter = SchedulerRuntimeAdapter(client)
            adapter.evict_mamba_only(handle)
            response = adapter.eviction_responses[-1]
            eviction = _compact_eviction(
                response,
                len(handle.token_ids),
                int(handle.expected_node_id),
            )
            post_eviction_census = compact_census(
                client.census(
                    f"openhands_paired:{trial_id}:after:evict:B2"
                ),
                ordinal=2,
                request=requests_by_turn[2],
                previous=previous,
            )
            census_snapshots.append(post_eviction_census)
            if _census_unexpected_mamba_change(
                post_eviction_census,
                int(handle.expected_node_id),
            ):
                native_eviction = True
            if post_eviction_census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
            if (
                pre_eviction_census["full_allocator"]
                != post_eviction_census["full_allocator"]
                or pre_eviction_census["full_device_node_ids"]
                != post_eviction_census["full_device_node_ids"]
                or pre_eviction_census["structure_node_ids"]
                != post_eviction_census["structure_node_ids"]
            ):
                fa_cascade = True
            final_state = verify_final_checkpoint_state(
                client,
                handle,
                expected_mamba_present=False,
            )
            eviction["checkpoint_info"] = checkpoint_info
            eviction["post_intervention_state"] = final_state
            eviction["correctness_pass"] = bool(
                eviction["correctness_pass"]
                and final_state["valid"]
                and not native_eviction
                and not fa_cascade
            )
            previous = post_eviction_census
            if not eviction["correctness_pass"]:
                raise RuntimeError(f"{trial_id} recurrent-only 正确性检查失败")

        request = requests_by_turn[3]
        try:
            measured_record = _execute_trial_request(
                engine,
                client,
                request,
                3,
            )
        except Exception as request_error:
            measured_record = _failure_record(request, 3, request_error)
            measured_record["exact_adjacent_lcp"] = request[
                "exact_adjacent_lcp"
            ]
            raise
        final_census = compact_census(
            client.census(f"openhands_paired:{trial_id}:after:B3"),
            ordinal=3,
            request=request,
            previous=previous,
        )
        census_snapshots.append(final_census)
        if final_census["native_mamba_capacity_eviction_inferred"]:
            native_eviction = True
        if final_census["fa_kv_cascade_eviction_inferred"]:
            fa_cascade = True
    except Exception as error:
        error_text = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                shutdown_error = repr(error)

    all_records = setup_records + (
        [] if measured_record is None else [measured_record]
    )
    lowered_errors = f"{error_text or ''} {shutdown_error or ''}".lower()
    trial = {
        "trial_id": trial_id,
        "condition": condition,
        "repetition": repetition,
        "engine_lifecycle": "independent",
        "setup_requests": setup_records,
        "measured_request": measured_record,
        "eviction": eviction,
        "setup_requests_valid": bool(
            len(setup_records) == 2
            and all(
                item.get("request_completed") is True
                and item.get("token_count_exact") is True
                for item in setup_records
            )
        ),
        "native_mamba_capacity_eviction": native_eviction,
        "fa_kv_cascade": fa_cascade,
        "oom": bool(
            any(item.get("oom") is True for item in all_records)
            or "out of memory" in lowered_errors
            or "oom" in lowered_errors
        ),
        "truncation_or_clipping": any(
            item.get("truncation_or_clipping") is True
            for item in all_records
        ),
        "census_summary": [
            {
                "request_ordinal": item["request_ordinal"],
                "mamba_available_slots": item["mamba_available_slots"],
                "mamba_node_count": item["mamba_node_count"],
                "removed_mamba_node_ids": item["removed_mamba_node_ids"],
                "changed_existing_mamba_node_ids": item[
                    "changed_existing_mamba_node_ids"
                ],
                "removed_full_device_node_ids": item[
                    "removed_full_device_node_ids"
                ],
                "removed_structure_node_ids": item[
                    "removed_structure_node_ids"
                ],
            }
            for item in census_snapshots
        ],
        "error": error_text,
        "engine_shutdown_error": shutdown_error,
    }
    evaluated = evaluate_trial(trial)
    print(
        f"[配对时延] 完成 {trial_id}，状态={evaluated['status']}",
        flush=True,
    )
    return evaluated


def build_summary(
    *,
    trials: Sequence[Mapping[str, object]],
    artifact: Path,
    recorded_turns: int | None,
    environment: Mapping[str, object] | None,
    fatal_error: str | None,
) -> dict[str, object]:
    """汇总有效 trial、配对差值和语义门。"""
    by_key = {
        (str(item["condition"]), int(item["repetition"])): item
        for item in trials
    }
    valid_control = [
        item
        for item in trials
        if item["condition"] == "CONTROL" and item["valid"]
    ]
    valid_evict = [
        item
        for item in trials
        if item["condition"] == "EVICT" and item["valid"]
    ]
    control_ttft = [
        float(item["measured_request"]["ttft_ms"]) for item in valid_control
    ]
    evict_ttft = [
        float(item["measured_request"]["ttft_ms"]) for item in valid_evict
    ]
    control_g = [
        int(item["measured_request"]["g"])
        for item in trials
        if item["condition"] == "CONTROL"
        and isinstance(item.get("measured_request"), Mapping)
        and item["measured_request"].get("g") is not None
    ]
    evict_g = [
        int(item["measured_request"]["g"])
        for item in trials
        if item["condition"] == "EVICT"
        and isinstance(item.get("measured_request"), Mapping)
        and item["measured_request"].get("g") is not None
    ]
    pairs = []
    for repetition in range(1, REPETITIONS + 1):
        control = by_key.get(("CONTROL", repetition))
        evict = by_key.get(("EVICT", repetition))
        valid = bool(control and evict and control["valid"] and evict["valid"])
        control_value = (
            None
            if not control or not isinstance(control.get("measured_request"), Mapping)
            else control["measured_request"].get("ttft_ms")
        )
        evict_value = (
            None
            if not evict or not isinstance(evict.get("measured_request"), Mapping)
            else evict["measured_request"].get("ttft_ms")
        )
        delta = (
            float(evict_value) - float(control_value)
            if valid and control_value is not None and evict_value is not None
            else None
        )
        pairs.append(
            {
                "repetition": repetition,
                "valid": valid,
                "control_ttft_ms": control_value,
                "evict_ttft_ms": evict_value,
                "delta_ms": delta,
            }
        )
    valid_deltas = [
        float(item["delta_ms"])
        for item in pairs
        if item["valid"] and item["delta_ms"] is not None
    ]
    control_consistency = sum(value == 0 for value in control_g)
    evict_consistency = sum(value > 0 for value in evict_g)
    eviction_correctness_all = bool(
        len([item for item in trials if item["condition"] == "EVICT"])
        == REPETITIONS
        and all(
            (item.get("eviction") or {}).get("correctness_pass") is True
            for item in trials
            if item["condition"] == "EVICT"
        )
    )
    valid_pair_count = sum(item["valid"] for item in pairs)
    native_eviction = any(
        item["native_mamba_capacity_eviction"] for item in trials
    )
    fa_cascade = any(item["fa_kv_cascade"] for item in trials)
    passed = bool(
        fatal_error is None
        and len(trials) == len(TRIAL_ORDER)
        and valid_pair_count >= 4
        and len(control_g) == REPETITIONS
        and control_consistency == REPETITIONS
        and len(evict_g) == REPETITIONS
        and evict_consistency == REPETITIONS
        and eviction_correctness_all
        and not native_eviction
        and not fa_cascade
    )
    return {
        "schema_version": "flowstate.openhands_recurrent_recovery_paired_latency.v1",
        "status": "PASS" if passed else "FAIL",
        "workflow": WORKFLOW_LABEL,
        "workflow_id": WORKFLOW_ID,
        "measured_request": "B3",
        "trial_order": [
            f"{condition}-{repetition}"
            for condition, repetition in TRIAL_ORDER
        ],
        "independent_engine_per_trial": True,
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_PAIRED_LATENCY,
        "sampling_parameters": SAMPLING_PARAMETERS,
        "recorded_n_turns": recorded_turns,
        "dataset_path": str(DATASET_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "artifact": _display_path(artifact),
        "trial_count": len(trials),
        "trials": list(trials),
        "control_semantic_consistency": {
            "g_values": control_g,
            "g_zero_count": control_consistency,
            "expected_count": REPETITIONS,
        },
        "evict_semantic_consistency": {
            "g_values": evict_g,
            "g_positive_count": evict_consistency,
            "expected_count": REPETITIONS,
        },
        "recurrent_only_correctness_all": eviction_correctness_all,
        "control_ttft_ms": summarize_values(control_ttft),
        "evict_ttft_ms": summarize_values(evict_ttft),
        "paired_results": pairs,
        "paired_delta_ms": summarize_values(valid_deltas),
        "positive_pair_count": sum(value > 0 for value in valid_deltas),
        "valid_pair_count": valid_pair_count,
        "native_mamba_capacity_eviction_observed": native_eviction,
        "fa_kv_cascade_eviction_observed": fa_cascade,
        "oom_observed": any(item["oom"] for item in trials),
        "truncation_or_clipping_observed": any(
            item["truncation_or_clipping"] for item in trials
        ),
        "policy_executed": False,
        "statistical_significance_tested": False,
        "fatal_error": fatal_error,
        "environment": dict(environment) if environment is not None else None,
    }


def _run(artifact: Path) -> dict[str, object]:
    """按确定性交替顺序执行十个独立 Engine trial。"""
    trials_path = artifact / "trials.jsonl"
    trials_path.write_text("", encoding="utf-8")
    trials: list[dict[str, object]] = []
    fatal_error = None
    environment = None
    recorded_turns = None
    try:
        environment = _environment()
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )
        prepared_requests, recorded_turns = prepare_requests(tokenizer)
        _write_json(
            artifact / "config.json",
            {
                "workflow": WORKFLOW_LABEL,
                "workflow_id": WORKFLOW_ID,
                "measured_request": "B3",
                "trial_order": [
                    f"{condition}-{repetition}"
                    for condition, repetition in TRIAL_ORDER
                ],
                "independent_engine_per_trial": True,
                "recorded_n_turns": recorded_turns,
                "offline_requests": {
                    f"B{turn}": {
                        "input_tokens": len(request["input_ids"]),
                        "exact_adjacent_lcp": request["exact_adjacent_lcp"],
                    }
                    for turn, request in prepared_requests.items()
                },
                "engine_configuration": ENGINE_CONFIGURATION_PAIRED_LATENCY,
                "sampling_parameters": SAMPLING_PARAMETERS,
                "dataset_path": str(DATASET_PATH),
                "tokenizer_path": str(TOKENIZER_PATH),
                "environment": environment,
                "policy_executed": False,
                "significance_test": False,
            },
        )
        for condition, repetition in TRIAL_ORDER:
            trial = run_trial(
                condition=condition,
                repetition=repetition,
                prepared_requests=prepared_requests,
            )
            trials.append(trial)
            _append_jsonl(trials_path, trial)
    except Exception as error:
        fatal_error = repr(error)
        traceback.print_exc()

    summary = build_summary(
        trials=trials,
        artifact=artifact,
        recorded_turns=recorded_turns,
        environment=environment,
        fatal_error=fatal_error,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行唯一一次配对时延实验。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
