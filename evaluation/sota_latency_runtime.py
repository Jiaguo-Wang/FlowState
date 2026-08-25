#!/usr/bin/env python3
"""按冻结 Step 9A 协议执行正式 SOTA latency benchmark。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import time
import traceback
from typing import Callable, Mapping, Sequence

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    RuntimeRepresentationMismatch,
    build_runtime_handles,
    build_runtime_workflows,
    inspect_after_allocation,
    query_runtime_metrics,
    wait_for_transport,
)
from evaluation.controlled_multiworkflow_v1.snapshot_runtime import (
    SnapshotSchedulerRuntimeAdapter,
    allocation_safety_snapshot,
    validate_clean_cache,
)
from evaluation.sota_latency_benchmark import (
    MAX_ESTIMATED_GPU_HOURS,
    MEASURED_REPETITIONS,
    POLICY_ORDER_SEED,
    REPRESENTATIVE_POINTS,
    REQUIRED_SAFETY_FLAGS,
    WARMUP_REPETITIONS,
    LatencyBenchmarkCase,
    LatencyCorrectnessError,
    aggregate_latency_records,
    build_benchmark_cases,
    build_dry_run_report,
    validate_latency_measurement,
    weighted_mean,
    weighted_quantile,
)
from evaluation.sota_runtime_correctness import (
    STEP8E_ENGINE_CONFIGURATION,
    EvaluationPolicyOptimizerAdapter,
    RuntimeCorrectnessCase,
    _pending_tokens,
    _scenario_for_case,
    build_flowstate_oracle_report,
    build_runtime_scenario_view,
)
from flowstate.controller import StateController
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_ROOT = _REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
_FORMAL_PRIMITIVE = (
    "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
)
MAX_DETERMINISTIC_RETRIES = 1


class CaseAttemptError(RuntimeError):
    """保存一次 case 尝试的失败阶段、分类和部分证据。"""

    def __init__(
        self,
        stage: str,
        cause: Exception,
        *,
        correctness_failure: bool,
        partial_record: Mapping[str, object],
    ) -> None:
        super().__init__(f"{stage}: {cause}")
        self.stage = stage
        self.cause = cause
        self.correctness_failure = correctness_failure
        self.partial_record = dict(partial_record)


@dataclass
class LatencyArtifactWriter:
    """增量保存原始样本、运行 metadata 和最终统计。"""

    directory: Path

    @classmethod
    def create(cls) -> "LatencyArtifactWriter":
        """创建不会覆盖既有证据的时间戳目录。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = _ARTIFACT_ROOT / f"sota_latency_{timestamp}"
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory)

    @property
    def raw_samples_path(self) -> Path:
        """返回增量原始样本路径。"""
        return self.directory / "raw_samples.jsonl"

    def append_raw_sample(self, record: Mapping[str, object]) -> None:
        """立即持久化一个 warmup、measured 或失败样本。"""
        with self.raw_samples_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    def write_metadata(self, metadata: Mapping[str, object]) -> None:
        """写出可在运行结束时更新的环境和协议信息。"""
        _write_json(self.directory / "run_metadata.json", metadata)

    def write_summary(
        self,
        summary: Mapping[str, object],
        gap_groups: Sequence[Mapping[str, object]],
    ) -> None:
        """写出 JSON、策略 CSV 与 recovery-gap CSV。"""
        _write_json(self.directory / "summary.json", summary)
        _write_policy_summary_csv(
            self.directory / "summary.csv",
            summary.get("policy_summaries", ()),
        )
        _write_gap_summary_csv(
            self.directory / "gap_group_summary.csv",
            gap_groups,
        )


def measure_streaming_request(
    engine: object,
    *,
    request_id: str,
    token_ids: Sequence[int],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """用首个流式 token 和请求结束时点测量 TTFT 与 latency。"""
    start_ns = clock_ns()
    stream = engine.generate(
        input_ids=list(token_ids),
        sampling_params=SAMPLING_PARAMETERS,
        rid=request_id,
        stream=True,
    )
    first_token_ns = None
    last_chunk = None
    for chunk in stream:
        if not isinstance(chunk, dict):
            raise RuntimeError(f"请求 {request_id} 返回了非对象流块")
        output_ids = chunk.get("output_ids") or []
        if output_ids and first_token_ns is None:
            first_token_ns = clock_ns()
        last_chunk = chunk
    end_ns = clock_ns()
    if first_token_ns is None or last_chunk is None:
        raise RuntimeError(f"请求 {request_id} 未产生首 token")
    output_ids = last_chunk.get("output_ids") or []
    metadata = last_chunk.get("meta_info") or {}
    if len(output_ids) != 1:
        raise RuntimeError(f"请求 {request_id} 的输出令牌数量异常")
    if int(metadata.get("completion_tokens", 1)) != 1:
        raise RuntimeError(f"请求 {request_id} 的完成长度异常")
    if int(metadata.get("num_retractions", 0) or 0) != 0:
        raise RuntimeError(f"请求 {request_id} 发生意外回撤")
    ttft_ms = (first_token_ns - start_ns) / 1_000_000.0
    request_latency_ms = (end_ns - start_ns) / 1_000_000.0
    if ttft_ms < 0.0 or request_latency_ms < ttft_ms:
        raise RuntimeError("流式请求计时边界无效")
    return {
        "output_token_id": int(output_ids[0]),
        "ttft_ms": ttft_ms,
        "request_latency_ms": request_latency_ms,
        "server_metadata": metadata,
    }


def execute_runtime_case_once(
    case: LatencyBenchmarkCase,
    *,
    engine: object,
    client: object,
    recovery_cost_model: RecoveryCostModel,
    retry_count: int,
) -> dict[str, object]:
    """在一个 fresh snapshot 中执行一次冻结 latency case。"""
    item = case.equivalence_class
    frozen_case = _runtime_correctness_case(case)
    scenario = _scenario_for_case(frozen_case)
    runtime_scenario = build_runtime_scenario_view(scenario)
    namespace = (
        f"flowstate_step9b_{_identifier(item.scenario_name)}_"
        f"k{item.budget_checkpoints}"
    )
    record: dict[str, object] = {
        "case_id": case.case_id,
        "scenario": item.scenario_name,
        "K": item.budget_checkpoints,
        "policy": item.policy_name,
        "continuation_id": item.representative_continuation_id,
        "equivalence_class": item.class_id,
        "equivalence_class_key": list(item.equivalence_key),
        "class_multiplicity": item.class_multiplicity,
        "repetition": case.repetition,
        "execution_order_position": case.execution_order_position,
        "is_warmup": case.is_warmup,
        "planning_H": item.planning_target,
        "planning_E": item.planning_executable_frontier,
        "planning_G": item.planning_gap_tokens,
        "selected_checkpoint_ids": list(item.selected_checkpoint_ids),
        "retry_count": retry_count,
        "correctness_pass": False,
        "status": "FAIL",
    }
    stage = "刷新并验证空缓存"
    try:
        started = time.perf_counter_ns()
        engine.flush_cache()
        census = client.census(
            f"{namespace}:{case.case_id}:retry{retry_count}:after_flush"
        )
        try:
            record["clean_cache"] = validate_clean_cache(census)
        except RuntimeError as error:
            raise LatencyCorrectnessError(str(error)) from error
        record["flush_ms"] = _elapsed_ms(started)

        stage = "重建并验证全部候选 checkpoint"
        started = time.perf_counter_ns()
        runtime_workflows, candidate_tokens = build_runtime_workflows(
            engine,
            runtime_scenario,
            request_namespace=(
                f"{namespace}_{case.case_id}_retry{retry_count}_build"
            ),
        )
        try:
            handles, before_paths, before_states = build_runtime_handles(
                client,
                runtime_scenario,
                candidate_tokens,
            )
        except RuntimeRepresentationMismatch as error:
            raise LatencyCorrectnessError(str(error)) from error
        record["snapshot_build_ms"] = _elapsed_ms(started)
        record["checkpoints_before_allocation"] = before_states

        stage = "执行冻结策略与 controller reconcile"
        runtime_adapter = SnapshotSchedulerRuntimeAdapter(
            client,
            f"{case.case_id}:retry{retry_count}",
            namespace,
        )
        optimizer = (
            GlobalOptimizer(recovery_cost_model)
            if item.policy_name == "FlowState"
            else EvaluationPolicyOptimizerAdapter(
                item.policy_name,
                scenario,
                recovery_cost_model,
            )
        )
        started = time.perf_counter_ns()
        allocation = StateController(
            optimizer,
            runtime_adapter,
        ).reconcile(
            scenario.continuations,
            scenario.candidates,
            handles,
            scenario.budget_bytes,
        )
        record["reconcile_ms"] = _elapsed_ms(started)
        selected_ids = tuple(
            candidate.checkpoint_id
            for candidate in allocation.selected
        )
        evicted_ids = tuple(runtime_adapter.evicted_checkpoint_ids)
        expected_evicted_ids = tuple(
            sorted(
                candidate.checkpoint_id
                for candidate in scenario.candidates
                if candidate.recurrent_resident
                and candidate.checkpoint_id not in selected_ids
            )
        )
        record["actual_selected_checkpoint_ids"] = list(selected_ids)
        record["evicted_checkpoint_ids"] = list(evicted_ids)
        if selected_ids != item.selected_checkpoint_ids:
            raise LatencyCorrectnessError(
                "运行时策略选择偏离冻结 selection："
                f"{selected_ids} != {item.selected_checkpoint_ids}"
            )
        if evicted_ids != expected_evicted_ids:
            raise LatencyCorrectnessError(
                "controller 驱逐动作与冻结 selected set 不一致"
            )

        stage = "验证 allocation 后状态与安全条件"
        started = time.perf_counter_ns()
        after_paths, after_states = inspect_after_allocation(
            client,
            candidate_tokens,
        )
        safety, allocator = allocation_safety_snapshot(
            before_paths=before_paths,
            after_paths=after_paths,
            after_states=after_states,
            selected_ids=selected_ids,
            evicted_ids=evicted_ids,
            eviction_responses=runtime_adapter.eviction_responses,
        )
        record["safety_validation_ms"] = _elapsed_ms(started)
        record["checkpoints_after_allocation"] = after_states
        record["runtime_safety"] = safety
        record["fa_allocator"] = allocator
        record["formal_mutation_primitive"] = _FORMAL_PRIMITIVE
        safety_flags = _latency_safety_flags(safety)
        record["safety_flags"] = safety_flags
        record["safety_pass"] = all(safety_flags.values())
        if not record["safety_pass"]:
            raise LatencyCorrectnessError(
                f"组件级安全条件失败：{safety_flags}"
            )

        stage = "确认 scheduler 安全时点"
        if client.ping().get("ok") is not True:
            raise RuntimeError("scheduler transport 未就绪")
        record["runtime_ready"] = True

        stage = "发送并计时唯一 target continuation"
        continuation = next(
            continuation
            for continuation in scenario.continuations
            if continuation.continuation_id
            == item.representative_continuation_id
        )
        request_id = (
            f"{namespace}_{case.case_id}_retry{retry_count}_target"
        )
        timing = measure_streaming_request(
            engine,
            request_id=request_id,
            token_ids=_pending_tokens(
                continuation,
                scenario,
                runtime_workflows,
            ),
        )
        record["ttft_ms"] = timing["ttft_ms"]
        record["request_latency_ms"] = timing["request_latency_ms"]
        record["server_metadata"] = timing["server_metadata"]

        stage = "读取 runtime H/E/G 并执行正确性门禁"
        metrics = query_runtime_metrics(client, request_id)
        runtime_h = int(metrics["physical_fa_hit"])
        runtime_e = int(metrics["executable_prefix"])
        runtime_g = int(metrics["replay_gap"])
        record.update(
            {
                "runtime_H": runtime_h,
                "runtime_E": runtime_e,
                "runtime_G": runtime_g,
                "H_match": runtime_h == item.planning_target,
                "E_match": (
                    runtime_e == item.planning_executable_frontier
                ),
                "G_match": runtime_g == item.planning_gap_tokens,
            }
        )
        measurement = validate_latency_measurement(
            case,
            runtime_metrics=metrics,
            safety=safety_flags,
            ttft_ms=float(timing["ttft_ms"]),
            request_latency_ms=float(timing["request_latency_ms"]),
            snapshot_build_ms=float(record["snapshot_build_ms"]),
            reconcile_ms=float(record["reconcile_ms"]),
        )
        record.update(measurement)
        record["planning_H"] = item.planning_target
        record["planning_E"] = item.planning_executable_frontier
        record["planning_G"] = item.planning_gap_tokens
        record["runtime_H"] = runtime_h
        record["runtime_E"] = runtime_e
        record["runtime_G"] = runtime_g
        record["correctness_pass"] = True
        record["status"] = "PASS"
        record["failure_stage"] = None
        return record
    except Exception as error:
        correctness_failure = isinstance(
            error,
            LatencyCorrectnessError,
        )
        record["failure_stage"] = stage
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
        record["correctness_failure"] = correctness_failure
        raise CaseAttemptError(
            stage,
            error,
            correctness_failure=correctness_failure,
            partial_record=record,
        ) from error


def execute_with_deterministic_retry(
    case: LatencyBenchmarkCase,
    execute_once: Callable[[LatencyBenchmarkCase, int], Mapping[str, object]],
) -> dict[str, object]:
    """非 correctness 失败最多重试一次，correctness 失败立即返回。"""
    attempt_errors = []
    for retry_count in range(MAX_DETERMINISTIC_RETRIES + 1):
        try:
            record = dict(execute_once(case, retry_count))
            record["retry_count"] = retry_count
            record["attempt_errors"] = attempt_errors
            return record
        except CaseAttemptError as error:
            attempt_errors.append(
                {
                    "retry_count": retry_count,
                    "failure_stage": error.stage,
                    "error": repr(error.cause),
                    "correctness_failure": error.correctness_failure,
                }
            )
            if error.correctness_failure or retry_count >= (
                MAX_DETERMINISTIC_RETRIES
            ):
                record = dict(error.partial_record)
                record["retry_count"] = retry_count
                record["attempt_errors"] = attempt_errors
                record["status"] = "FAIL"
                record["correctness_pass"] = False
                return record
    raise AssertionError("deterministic retry 控制流不可达")


def build_gap_group_summary(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """按 planning gap 汇总成功 measured 样本的描述性 TTFT。"""
    groups: dict[int, list[Mapping[str, object]]] = {}
    for record in records:
        if record.get("is_warmup") is not False:
            continue
        if record.get("correctness_pass") is not True:
            continue
        gap = int(record["planning_gap"])
        groups.setdefault(gap, []).append(record)
    result = []
    for gap in sorted(groups):
        samples = groups[gap]
        values = tuple(float(item["ttft_ms"]) for item in samples)
        weights = tuple(
            float(item["class_multiplicity"]) for item in samples
        )
        result.append(
            {
                "gap_tokens": gap,
                "sample_count": len(samples),
                "weighted_request_count": sum(weights),
                "ttft_weighted_mean_ms": weighted_mean(values, weights),
                "ttft_weighted_median_ms": weighted_quantile(
                    values,
                    weights,
                    0.5,
                ),
                "ttft_weighted_p95_ms": weighted_quantile(
                    values,
                    weights,
                    0.95,
                ),
            }
        )
    return tuple(result)


def build_runtime_summary(
    records: Sequence[Mapping[str, object]],
    *,
    total_runtime_ms: float,
    artifact_directory: Path,
) -> dict[str, object]:
    """构造最终完成度、正确性、加权 latency 与对比汇总。"""
    policy_summaries = tuple(aggregate_latency_records(records))
    warmup_records = tuple(
        record for record in records if record.get("is_warmup") is True
    )
    measured_records = tuple(
        record for record in records if record.get("is_warmup") is False
    )
    failed = tuple(
        record for record in records if record.get("status") != "PASS"
    )
    expected_warmup = 69 * WARMUP_REPETITIONS
    expected_measured = 69 * MEASURED_REPETITIONS
    warmup_completed = sum(
        record.get("status") == "PASS" for record in warmup_records
    )
    measured_completed = sum(
        record.get("status") == "PASS" for record in measured_records
    )
    correctness = {
        "H_mismatch": sum(
            record.get("H_match") is False for record in records
        ),
        "E_mismatch": sum(
            record.get("E_match") is False for record in records
        ),
        "G_mismatch": sum(
            record.get("G_match") is False for record in records
        ),
        "safety_failures": sum(
            record.get("safety_pass") is False for record in records
        ),
    }
    status = (
        "PASS"
        if warmup_completed == expected_warmup
        and measured_completed == expected_measured
        and not failed
        and not any(correctness.values())
        else "FAIL"
    )
    return {
        "schema_version": "flowstate.sota_latency_runtime.v1",
        "status": status,
        "warmup_expected": expected_warmup,
        "warmup_completed": warmup_completed,
        "measured_expected": expected_measured,
        "measured_completed": measured_completed,
        "failed": len(failed),
        "retried": sum(int(record.get("retry_count", 0)) > 0 for record in records),
        "correctness": correctness,
        "policy_summaries": policy_summaries,
        "policy_comparisons": _build_policy_comparisons(policy_summaries),
        "gap_group_summary": build_gap_group_summary(records),
        "oracle_reference": build_flowstate_oracle_report(),
        "total_runtime_ms": total_runtime_ms,
        "artifact_directory": str(artifact_directory),
        "formal_mutation_primitive": _FORMAL_PRIMITIVE,
    }


def _build_policy_comparisons(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """计算 FlowState 相对三个 runtime baseline 的加权降幅。"""
    by_key = {
        (str(row["scenario"]), int(row["K"]), str(row["policy"])): row
        for row in rows
    }
    comparisons = []
    for scenario_name, budget in REPRESENTATIVE_POINTS:
        flowstate = by_key.get((scenario_name, budget, "FlowState"))
        if flowstate is None:
            continue
        for baseline_name in (
            "Global-LRU",
            "KVFlow-style",
            "Marconi-style",
        ):
            baseline = by_key.get(
                (scenario_name, budget, baseline_name)
            )
            if baseline is None:
                continue
            comparisons.append(
                {
                    "scenario": scenario_name,
                    "K": budget,
                    "baseline": baseline_name,
                    "ttft_reduction": _metric_reduction(
                        flowstate["ttft_ms"],
                        baseline["ttft_ms"],
                    ),
                    "request_latency_reduction": _metric_reduction(
                        flowstate["request_latency_ms"],
                        baseline["request_latency_ms"],
                    ),
                }
            )
    return tuple(comparisons)


def _metric_reduction(
    current: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, float | None]:
    """计算 weighted mean、median 和 P95 的相对降幅。"""
    result = {}
    for statistic in (
        "weighted_mean",
        "weighted_median",
        "weighted_p95",
    ):
        current_value = float(current[statistic])
        baseline_value = float(baseline[statistic])
        result[statistic] = (
            (baseline_value - current_value) / baseline_value
            if baseline_value > 0.0
            else None
        )
    return result


def _runtime_correctness_case(
    case: LatencyBenchmarkCase,
) -> RuntimeCorrectnessCase:
    """把冻结 latency class 转换成 Step 8E runtime 只读视图。"""
    item = case.equivalence_class
    return RuntimeCorrectnessCase(
        scenario_name=item.scenario_name,
        budget_checkpoints=item.budget_checkpoints,
        policy_name=item.policy_name,
        continuation_id=item.representative_continuation_id,
        workflow_id=item.workflow_id,
        selected_checkpoint_ids=item.selected_checkpoint_ids,
        planning_target=item.planning_target,
        planning_executable_frontier=(
            item.planning_executable_frontier
        ),
        planning_gap_tokens=item.planning_gap_tokens,
    )


def _latency_safety_flags(
    safety: Mapping[str, bool],
) -> dict[str, bool]:
    """把正式 runtime proof 映射成 Step 9A 固定安全门禁。"""
    flags = {
        "fa_safety": bool(
            safety.get("fa_preserved")
            and safety.get("fa_identity_invariant")
        ),
        "mamba_safety": bool(
            safety.get("selected_mamba_resident")
            and safety.get("unselected_mamba_evicted")
            and safety.get("only_expected_mamba_changed")
        ),
        "allocator_safety": bool(safety.get("allocator_invariant")),
        "tree_safety": bool(
            safety.get("tree_invariant")
            and safety.get("path_invariant")
        ),
        "sanity_check": bool(safety.get("sanity_check")),
    }
    if set(flags) != set(REQUIRED_SAFETY_FLAGS):
        raise AssertionError("Step 9A 安全字段映射不完整")
    if safety.get("cascade_not_called") is not True:
        flags["tree_safety"] = False
    if safety.get("formal_primitive") is not True:
        flags["mamba_safety"] = False
    return flags


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """以 UTF-8 和稳定键序写出 JSON。"""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _write_policy_summary_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """写出十六个代表点策略的加权 latency 表。"""
    fieldnames = (
        "scenario",
        "K",
        "policy",
        "measured_case_count",
        "weighted_request_count",
        "ttft_weighted_mean_ms",
        "ttft_weighted_median_ms",
        "ttft_weighted_p95_ms",
        "latency_weighted_mean_ms",
        "latency_weighted_median_ms",
        "latency_weighted_p95_ms",
        "ttft_mean_reduction_vs_global_lru",
        "latency_mean_reduction_vs_global_lru",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            relative = row.get("relative_to_global_lru") or {}
            ttft_relative = relative.get("ttft_ms") or {}
            latency_relative = relative.get("request_latency_ms") or {}
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "K": row["K"],
                    "policy": row["policy"],
                    "measured_case_count": row[
                        "measured_case_count"
                    ],
                    "weighted_request_count": row[
                        "weighted_request_count"
                    ],
                    "ttft_weighted_mean_ms": row["ttft_ms"][
                        "weighted_mean"
                    ],
                    "ttft_weighted_median_ms": row["ttft_ms"][
                        "weighted_median"
                    ],
                    "ttft_weighted_p95_ms": row["ttft_ms"][
                        "weighted_p95"
                    ],
                    "latency_weighted_mean_ms": row[
                        "request_latency_ms"
                    ]["weighted_mean"],
                    "latency_weighted_median_ms": row[
                        "request_latency_ms"
                    ]["weighted_median"],
                    "latency_weighted_p95_ms": row[
                        "request_latency_ms"
                    ]["weighted_p95"],
                    "ttft_mean_reduction_vs_global_lru": (
                        ttft_relative.get("weighted_mean")
                    ),
                    "latency_mean_reduction_vs_global_lru": (
                        latency_relative.get("weighted_mean")
                    ),
                }
            )


def _write_gap_summary_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """写出按 planning gap 聚合的描述性 TTFT 表。"""
    fieldnames = (
        "gap_tokens",
        "sample_count",
        "weighted_request_count",
        "ttft_weighted_mean_ms",
        "ttft_weighted_median_ms",
        "ttft_weighted_p95_ms",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def _build_run_metadata(
    *,
    schedule: Sequence[LatencyBenchmarkCase],
    started_at: str,
    gpu_name: str,
) -> dict[str, object]:
    """构造正式运行所需的冻结环境与协议信息。"""
    return {
        "schema_version": "flowstate.sota_latency_metadata.v1",
        "environment": "冻结 SGLang Docker runtime",
        "model": "Qwen3.5-9B",
        "sglang_version": importlib.metadata.version("sglang"),
        "gpu": gpu_name,
        "tp": 1,
        "seed": POLICY_ORDER_SEED,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "representative_points": [
            {"scenario": scenario, "K": budget}
            for scenario, budget in REPRESENTATIVE_POINTS
        ],
        "policies": [
            "Global-LRU",
            "KVFlow-style",
            "Marconi-style",
            "FlowState",
        ],
        "equivalence_classes": 69,
        "multiplicity_total": 800,
        "warmup_cases_expected": sum(
            item.is_warmup for item in schedule
        ),
        "measured_cases_expected": sum(
            not item.is_warmup for item in schedule
        ),
        "total_expected_cases": len(schedule),
        "maximum_deterministic_retries": MAX_DETERMINISTIC_RETRIES,
        "start_time": started_at,
        "end_time": None,
        "status": "RUNNING",
        "engine_configuration": STEP8E_ENGINE_CONFIGURATION,
        "formal_mutation_primitive": _FORMAL_PRIMITIVE,
    }


def _elapsed_ms(started_ns: int) -> float:
    """返回从指定单调时点开始经过的毫秒数。"""
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _identifier(value: str) -> str:
    """把名称转换成 runtime namespace 可用的稳定片段。"""
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    ).strip("_")


def main() -> int:
    """执行唯一一次正式 GPU latency benchmark。"""
    dry_run = build_dry_run_report()
    estimate = dry_run["estimated_gpu_runtime"]
    if float(estimate["estimated_hours"]) > MAX_ESTIMATED_GPU_HOURS:
        print(
            "正式运行估计超过六小时，按冻结协议停止且不减少 repetition",
            flush=True,
        )
        return 2
    schedule = build_benchmark_cases()
    if len(schedule) != 828:
        raise RuntimeError(f"冻结 schedule 数量异常：{len(schedule)}")

    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import (
        FormalEndToEndGateEngine,
        requested_control_port,
    )
    writer = LatencyArtifactWriter.create()
    started_at = datetime.now().astimezone().isoformat()
    metadata = _build_run_metadata(
        schedule=schedule,
        started_at=started_at,
        gpu_name="NVIDIA H100 PCIe 80 GiB",
    )
    writer.write_metadata(metadata)
    records: list[dict[str, object]] = []
    engine = None
    fatal_error = None
    started = time.perf_counter_ns()
    model = RecoveryCostModel()
    try:
        engine = FormalEndToEndGateEngine(
            **STEP8E_ENGINE_CONFIGURATION
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        for index, case in enumerate(schedule, start=1):
            phase = "WARMUP" if case.is_warmup else "MEASURED"
            print(
                f"[STEP9B] {index}/{len(schedule)} {phase} "
                f"{case.scenario_name} K={case.budget_checkpoints} "
                f"{case.policy_name} "
                f"{case.equivalence_class.representative_continuation_id}",
                flush=True,
            )
            record = execute_with_deterministic_retry(
                case,
                lambda active_case, retry_count: (
                    execute_runtime_case_once(
                        active_case,
                        engine=engine,
                        client=client,
                        recovery_cost_model=model,
                        retry_count=retry_count,
                    )
                ),
            )
            records.append(record)
            writer.append_raw_sample(record)
            if record.get("correctness_failure") is True:
                fatal_error = (
                    "检测到 correctness failure，按协议立即停止："
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
    summary = build_runtime_summary(
        records,
        total_runtime_ms=total_runtime_ms,
        artifact_directory=writer.directory,
    )
    if fatal_error is not None:
        summary["fatal_error"] = fatal_error
        summary["status"] = "FAIL"
    gap_groups = tuple(summary["gap_group_summary"])
    writer.write_summary(summary, gap_groups)
    metadata.update(
        {
            "end_time": datetime.now().astimezone().isoformat(),
            "status": summary["status"],
            "total_runtime_ms": total_runtime_ms,
            "raw_records": len(records),
            "fatal_error": fatal_error,
        }
    )
    writer.write_metadata(metadata)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "warmup_completed": summary["warmup_completed"],
                "warmup_expected": summary["warmup_expected"],
                "measured_completed": summary["measured_completed"],
                "measured_expected": summary["measured_expected"],
                "failed": summary["failed"],
                "retried": summary["retried"],
                "artifacts": str(writer.directory),
                "fatal_error": fatal_error,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
