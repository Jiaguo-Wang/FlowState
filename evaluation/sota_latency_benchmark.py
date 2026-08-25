#!/usr/bin/env python3
"""构建论文级 SOTA latency benchmark 的冻结计划与统计工具。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from typing import Mapping, Sequence

from evaluation.sota_runtime_correctness import (
    EXPECTED_E2E_EQUIVALENCE_CASES,
    EXPECTED_TOTAL_CASES,
    GPU_POLICY_NAMES,
    RuntimeCorrectnessCase,
    build_flowstate_oracle_report,
    build_representative_cases,
    build_snapshot_audit_plans,
    validate_exact_runtime_observation,
)


WARMUP_REPETITIONS = 2
MEASURED_REPETITIONS = 10
POLICY_ORDER_SEED = 20_260_825
MAX_ESTIMATED_GPU_HOURS = 6.0
REFERENCE_STEP8E_RUNTIME_SECONDS = 1_550.8840360920876
REFERENCE_STEP8E_SNAPSHOT_COUNT = 85
RUNTIME_ESTIMATE_SAFETY_FACTOR = 1.25
REPRESENTATIVE_POINTS = (
    ("scalable_multiworkflow_v2_n16", 4),
    ("scalable_multiworkflow_v2_n16", 12),
    ("sota_signal_stress_v1", 4),
    ("sota_signal_stress_v1", 8),
)
REQUIRED_SAFETY_FLAGS = (
    "fa_safety",
    "mamba_safety",
    "allocator_safety",
    "tree_safety",
    "sanity_check",
)


class LatencyCorrectnessError(RuntimeError):
    """表示 latency case 未通过状态正确性门禁。"""


@dataclass(frozen=True)
class LatencyEquivalenceClass:
    """描述一个冻结恢复行为等价类及其 workload 权重。"""

    scenario_name: str
    budget_checkpoints: int
    policy_name: str
    representative_continuation_id: str
    workflow_id: str
    selected_checkpoint_ids: tuple[str, ...]
    planning_target: int
    planning_executable_frontier: int
    planning_gap_tokens: int
    class_multiplicity: int

    @property
    def equivalence_key(self) -> tuple[int, int, int]:
        """返回 Step 8E 冻结的恢复行为分组键。"""
        return (
            self.planning_target,
            self.planning_executable_frontier,
            self.planning_gap_tokens,
        )

    @property
    def class_id(self) -> str:
        """返回跨运行稳定的等价类标识。"""
        target, frontier, gap = self.equivalence_key
        return (
            f"{self.scenario_name}:K={self.budget_checkpoints}:"
            f"{self.policy_name}:T={target}:E={frontier}:G={gap}"
        )


@dataclass(frozen=True)
class LatencyBenchmarkCase:
    """描述一次 fresh-snapshot latency 请求执行。"""

    case_id: str
    equivalence_class: LatencyEquivalenceClass
    repetition: int
    is_warmup: bool
    execution_order_position: int

    @property
    def scenario_name(self) -> str:
        """返回 case 所属场景。"""
        return self.equivalence_class.scenario_name

    @property
    def budget_checkpoints(self) -> int:
        """返回 case 的 checkpoint 预算。"""
        return self.equivalence_class.budget_checkpoints

    @property
    def policy_name(self) -> str:
        """返回 case 使用的冻结策略。"""
        return self.equivalence_class.policy_name


def build_latency_equivalence_classes(
    cases: Sequence[RuntimeCorrectnessCase] | None = None,
) -> tuple[LatencyEquivalenceClass, ...]:
    """从 Step 8E 全量逻辑 case 恢复代表类和 multiplicity。"""
    active_cases = tuple(
        build_representative_cases() if cases is None else cases
    )
    plans = build_snapshot_audit_plans(active_cases)
    result = []
    for plan in plans:
        groups: dict[
            tuple[int, int, int],
            list[RuntimeCorrectnessCase],
        ] = defaultdict(list)
        for case in plan.logical_cases:
            groups[
                (
                    case.planning_target,
                    case.planning_executable_frontier,
                    case.planning_gap_tokens,
                )
            ].append(case)
        for key in sorted(groups):
            members = groups[key]
            representative = min(
                members,
                key=lambda item: item.continuation_id,
            )
            result.append(
                LatencyEquivalenceClass(
                    scenario_name=representative.scenario_name,
                    budget_checkpoints=(
                        representative.budget_checkpoints
                    ),
                    policy_name=representative.policy_name,
                    representative_continuation_id=(
                        representative.continuation_id
                    ),
                    workflow_id=representative.workflow_id,
                    selected_checkpoint_ids=(
                        representative.selected_checkpoint_ids
                    ),
                    planning_target=key[0],
                    planning_executable_frontier=key[1],
                    planning_gap_tokens=key[2],
                    class_multiplicity=len(members),
                )
            )
    classes = tuple(result)
    if len(classes) != EXPECTED_E2E_EQUIVALENCE_CASES:
        raise RuntimeError(f"冻结等价类数量异常：{len(classes)}")
    if sum(item.class_multiplicity for item in classes) != (
        EXPECTED_TOTAL_CASES
    ):
        raise RuntimeError("等价类 multiplicity 未覆盖全部逻辑请求")
    if any(item.class_multiplicity <= 0 for item in classes):
        raise RuntimeError("等价类 multiplicity 必须为正")
    return classes


def balanced_policy_order(
    point_index: int,
    repetition: int,
    *,
    seed: int = POLICY_ORDER_SEED,
) -> tuple[str, ...]:
    """用固定 seed 和循环轮换生成确定性 policy block 顺序。"""
    if point_index < 0 or repetition < 0:
        raise ValueError("代表点序号和 repetition 必须非负")
    policies = tuple(GPU_POLICY_NAMES)
    offset = (seed + point_index + repetition) % len(policies)
    return policies[offset:] + policies[:offset]


def build_benchmark_cases(
    equivalence_classes: Sequence[LatencyEquivalenceClass] | None = None,
    *,
    warmup_repetitions: int = WARMUP_REPETITIONS,
    measured_repetitions: int = MEASURED_REPETITIONS,
    seed: int = POLICY_ORDER_SEED,
) -> tuple[LatencyBenchmarkCase, ...]:
    """构建 warmup 与 measured 的完整 fresh-snapshot 执行计划。"""
    if warmup_repetitions < 0 or measured_repetitions <= 0:
        raise ValueError("repetition 配置无效")
    classes = tuple(
        build_latency_equivalence_classes()
        if equivalence_classes is None
        else equivalence_classes
    )
    grouped: dict[
        tuple[str, int, str],
        list[LatencyEquivalenceClass],
    ] = defaultdict(list)
    for item in classes:
        grouped[
            (
                item.scenario_name,
                item.budget_checkpoints,
                item.policy_name,
            )
        ].append(item)

    expected_groups = {
        (scenario_name, budget, policy_name)
        for scenario_name, budget in REPRESENTATIVE_POINTS
        for policy_name in GPU_POLICY_NAMES
    }
    if set(grouped) != expected_groups:
        raise RuntimeError("latency 计划偏离四个冻结代表点")

    schedule = []
    sequence = 0
    phases = (
        (True, warmup_repetitions, "warmup"),
        (False, measured_repetitions, "measured"),
    )
    for is_warmup, repetitions, phase_name in phases:
        for repetition in range(repetitions):
            for point_index, point in enumerate(REPRESENTATIVE_POINTS):
                policy_order = balanced_policy_order(
                    point_index,
                    repetition,
                    seed=seed,
                )
                for position, policy_name in enumerate(policy_order):
                    point_classes = sorted(
                        grouped[(point[0], point[1], policy_name)],
                        key=lambda item: (
                            item.equivalence_key,
                            item.representative_continuation_id,
                        ),
                    )
                    for item in point_classes:
                        sequence += 1
                        schedule.append(
                            LatencyBenchmarkCase(
                                case_id=(
                                    f"{phase_name}_{sequence:04d}_"
                                    f"{item.scenario_name}_k"
                                    f"{item.budget_checkpoints}_"
                                    f"{_identifier(policy_name)}_"
                                    f"r{repetition:02d}_"
                                    f"{_identifier(item.representative_continuation_id)}"
                                ),
                                equivalence_class=item,
                                repetition=repetition,
                                is_warmup=is_warmup,
                                execution_order_position=position,
                            )
                        )

    result = tuple(schedule)
    expected_warmup = len(classes) * warmup_repetitions
    expected_measured = len(classes) * measured_repetitions
    if sum(item.is_warmup for item in result) != expected_warmup:
        raise RuntimeError("warmup case 数量异常")
    if sum(not item.is_warmup for item in result) != expected_measured:
        raise RuntimeError("measured case 数量异常")
    if len({item.case_id for item in result}) != len(result):
        raise RuntimeError("latency case_id 不唯一")
    return result


def policy_position_distribution(
    cases: Sequence[LatencyBenchmarkCase],
    *,
    warmup: bool,
) -> dict[str, tuple[int, ...]]:
    """按 policy block 统计各执行位置出现次数。"""
    blocks = {
        (
            case.scenario_name,
            case.budget_checkpoints,
            case.repetition,
            case.policy_name,
            case.execution_order_position,
        )
        for case in cases
        if case.is_warmup is warmup
    }
    result = {}
    for policy_name in GPU_POLICY_NAMES:
        result[policy_name] = tuple(
            sum(
                policy == policy_name and position == target_position
                for _, _, _, policy, position in blocks
            )
            for target_position in range(len(GPU_POLICY_NAMES))
        )
    return result


def validate_latency_measurement(
    case: LatencyBenchmarkCase,
    *,
    runtime_metrics: Mapping[str, object],
    safety: Mapping[str, bool],
    ttft_ms: float,
    request_latency_ms: float,
    snapshot_build_ms: float,
    reconcile_ms: float,
) -> dict[str, object]:
    """验证一次请求的严格状态语义、安全条件与计时字段。"""
    missing_flags = tuple(
        flag for flag in REQUIRED_SAFETY_FLAGS if flag not in safety
    )
    failed_flags = tuple(
        flag for flag in REQUIRED_SAFETY_FLAGS if safety.get(flag) is not True
    )
    if missing_flags:
        raise LatencyCorrectnessError(
            f"缺少安全条件：{missing_flags}"
        )
    if failed_flags:
        raise LatencyCorrectnessError(
            f"安全条件失败：{failed_flags}"
        )

    frozen_case = RuntimeCorrectnessCase(
        scenario_name=case.scenario_name,
        budget_checkpoints=case.budget_checkpoints,
        policy_name=case.policy_name,
        continuation_id=(
            case.equivalence_class.representative_continuation_id
        ),
        workflow_id=case.equivalence_class.workflow_id,
        selected_checkpoint_ids=(
            case.equivalence_class.selected_checkpoint_ids
        ),
        planning_target=case.equivalence_class.planning_target,
        planning_executable_frontier=(
            case.equivalence_class.planning_executable_frontier
        ),
        planning_gap_tokens=(
            case.equivalence_class.planning_gap_tokens
        ),
    )
    try:
        observation = validate_exact_runtime_observation(
            frozen_case,
            runtime_metrics,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise LatencyCorrectnessError(
            f"planning/runtime 正确性门禁失败：{error}"
        ) from error

    timings = {
        "ttft_ms": float(ttft_ms),
        "request_latency_ms": float(request_latency_ms),
        "snapshot_build_ms": float(snapshot_build_ms),
        "reconcile_ms": float(reconcile_ms),
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in timings.values()
    ):
        raise ValueError("计时字段必须为有限非负数")
    if timings["ttft_ms"] > timings["request_latency_ms"]:
        raise ValueError("TTFT 不得超过完整请求 latency")

    item = case.equivalence_class
    return {
        "case_id": case.case_id,
        "scenario": item.scenario_name,
        "K": item.budget_checkpoints,
        "policy": item.policy_name,
        "continuation_id": item.representative_continuation_id,
        "equivalence_class": item.class_id,
        "class_multiplicity": item.class_multiplicity,
        "repetition": case.repetition,
        "execution_order_position": case.execution_order_position,
        "is_warmup": case.is_warmup,
        "planning_target": item.planning_target,
        "planning_frontier": item.planning_executable_frontier,
        "planning_gap": item.planning_gap_tokens,
        "runtime_fa_frontier": observation["runtime_fa_frontier"],
        "runtime_frontier": observation[
            "runtime_executable_frontier"
        ],
        "runtime_gap": observation["runtime_gap_tokens"],
        **timings,
        "correctness_pass": True,
    }


def weighted_mean(
    values: Sequence[float],
    weights: Sequence[float],
) -> float:
    """计算正权重样本的加权平均值。"""
    pairs = _validated_weighted_pairs(values, weights)
    total_weight = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total_weight


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    """按加权经验分布的逆函数计算分位数。"""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile 必须位于 [0, 1]")
    pairs = sorted(_validated_weighted_pairs(values, weights))
    threshold = quantile * sum(weight for _, weight in pairs)
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def aggregate_latency_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """忽略 warmup 和失败 case，按 multiplicity 汇总 latency。"""
    valid_records = tuple(
        record
        for record in records
        if record.get("is_warmup") is False
        and record.get("correctness_pass") is True
    )
    groups: dict[tuple[str, int, str], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for record in valid_records:
        groups[
            (
                str(record["scenario"]),
                int(record["K"]),
                str(record["policy"]),
            )
        ].append(record)

    rows = []
    for key in sorted(groups):
        samples = groups[key]
        weights = tuple(
            float(record["class_multiplicity"])
            for record in samples
        )
        ttft = tuple(float(record["ttft_ms"]) for record in samples)
        latency = tuple(
            float(record["request_latency_ms"]) for record in samples
        )
        per_class: dict[str, dict[str, object]] = defaultdict(
            lambda: {
                "multiplicity": 0,
                "ttft_ms": [],
                "request_latency_ms": [],
            }
        )
        for record in samples:
            entry = per_class[str(record["equivalence_class"])]
            entry["multiplicity"] = int(record["class_multiplicity"])
            entry["ttft_ms"].append(float(record["ttft_ms"]))
            entry["request_latency_ms"].append(
                float(record["request_latency_ms"])
            )
        rows.append(
            {
                "scenario": key[0],
                "K": key[1],
                "policy": key[2],
                "measured_case_count": len(samples),
                "weighted_request_count": sum(weights),
                "ttft_ms": _metric_summary(ttft, weights),
                "request_latency_ms": _metric_summary(
                    latency,
                    weights,
                ),
                "per_class_raw_samples": dict(per_class),
            }
        )

    by_key = {
        (row["scenario"], row["K"], row["policy"]): row
        for row in rows
    }
    for row in rows:
        baseline = by_key.get(
            (row["scenario"], row["K"], "Global-LRU")
        )
        row["relative_to_global_lru"] = (
            _relative_reductions(row, baseline)
            if baseline is not None
            else None
        )
    return tuple(rows)


def estimate_gpu_runtime(
    total_snapshot_cases: int,
) -> dict[str, float | bool]:
    """用 Step 8E 实测吞吐和保守系数估计正式运行时长。"""
    if total_snapshot_cases < 0:
        raise ValueError("snapshot case 数量不能为负")
    seconds_per_snapshot = (
        REFERENCE_STEP8E_RUNTIME_SECONDS
        / REFERENCE_STEP8E_SNAPSHOT_COUNT
    )
    estimated_seconds = (
        total_snapshot_cases
        * seconds_per_snapshot
        * RUNTIME_ESTIMATE_SAFETY_FACTOR
    )
    estimated_hours = estimated_seconds / 3_600.0
    return {
        "reference_seconds_per_snapshot": seconds_per_snapshot,
        "safety_factor": RUNTIME_ESTIMATE_SAFETY_FACTOR,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_hours,
        "within_six_hour_limit": (
            estimated_hours <= MAX_ESTIMATED_GPU_HOURS
        ),
    }


def build_dry_run_report() -> dict[str, object]:
    """生成不启动图形处理器的完整协议 dry-run 报告。"""
    classes = build_latency_equivalence_classes()
    schedule = build_benchmark_cases(classes)
    warmup_cases = sum(item.is_warmup for item in schedule)
    measured_cases = sum(not item.is_warmup for item in schedule)
    estimate = estimate_gpu_runtime(len(schedule))
    return {
        "schema_version": "flowstate.sota_latency_benchmark.v1",
        "mode": "仅协议 dry-run，不启动图形处理器",
        "representative_points": [
            {"scenario": scenario, "K": budget}
            for scenario, budget in REPRESENTATIVE_POINTS
        ],
        "policies": list(GPU_POLICY_NAMES),
        "equivalence_class_count": len(classes),
        "equivalence_classes": [
            {
                "class_id": item.class_id,
                "representative_continuation_id": (
                    item.representative_continuation_id
                ),
                "class_multiplicity": item.class_multiplicity,
            }
            for item in classes
        ],
        "multiplicity_sum": sum(
            item.class_multiplicity for item in classes
        ),
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "warmup_cases": warmup_cases,
        "measured_cases": measured_cases,
        "policy_order_seed": POLICY_ORDER_SEED,
        "warmup_policy_block_positions": (
            policy_position_distribution(schedule, warmup=True)
        ),
        "measured_policy_block_positions": (
            policy_position_distribution(schedule, warmup=False)
        ),
        "estimated_gpu_runtime": estimate,
        "correctness_gate": {
            "exact_planning_runtime_gap": True,
            "fa_mamba_tree_sanity_required": True,
            "failed_cases_enter_statistics": False,
        },
        "oracle": build_flowstate_oracle_report(),
    }


def _validated_weighted_pairs(
    values: Sequence[float],
    weights: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """校验并返回有限值和正权重二元组。"""
    if len(values) != len(weights) or not values:
        raise ValueError("values 与 weights 必须等长且非空")
    pairs = tuple(
        (float(value), float(weight))
        for value, weight in zip(values, weights, strict=True)
    )
    if any(
        not math.isfinite(value)
        or not math.isfinite(weight)
        or weight <= 0.0
        for value, weight in pairs
    ):
        raise ValueError("样本必须有限且权重必须为正")
    return pairs


def _metric_summary(
    values: Sequence[float],
    weights: Sequence[float],
) -> dict[str, float]:
    """生成论文表格使用的三个加权统计量。"""
    return {
        "weighted_mean": weighted_mean(values, weights),
        "weighted_median": weighted_quantile(values, weights, 0.5),
        "weighted_p95": weighted_quantile(values, weights, 0.95),
    }


def _relative_reductions(
    row: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, dict[str, float | None]]:
    """计算相对同一点 Global-LRU 的 latency 降幅。"""
    result = {}
    for metric_name in ("ttft_ms", "request_latency_ms"):
        metric = row[metric_name]
        reference = baseline[metric_name]
        reductions = {}
        for statistic in (
            "weighted_mean",
            "weighted_median",
            "weighted_p95",
        ):
            current = float(metric[statistic])
            base = float(reference[statistic])
            reductions[statistic] = (
                (base - current) / base if base > 0.0 else None
            )
        result[metric_name] = reductions
    return result


def _identifier(value: str) -> str:
    """把稳定名称转换成便于保存的 case_id 片段。"""
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    ).strip("_")


def main() -> int:
    """打印纯 CPU dry-run，不启动任何真实运行时。"""
    report = build_dry_run_report()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
