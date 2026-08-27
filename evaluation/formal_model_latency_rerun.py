#!/usr/bin/env python3
"""只重跑正式恢复模型改变选择的两个 H100 latency 代表点。"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import time
import traceback
from typing import Mapping, Sequence

from evaluation.scalable_multiworkflow_v2.scenario import (
    ScalableScenario,
    build_scenario as build_scalable_scenario,
)
from evaluation.sota_latency_benchmark import (
    MEASURED_REPETITIONS,
    POLICY_ORDER_SEED,
    REPRESENTATIVE_POINTS,
    WARMUP_REPETITIONS,
    LatencyBenchmarkCase,
    LatencyEquivalenceClass,
    aggregate_latency_records,
    balanced_policy_order,
    weighted_mean,
    weighted_quantile,
)
from evaluation.sota_latency_runtime import (
    MAX_DETERMINISTIC_RETRIES,
    execute_runtime_case_once,
    execute_with_deterministic_retry,
)
from evaluation.sota_runtime_correctness import (
    GPU_POLICY_NAMES,
    SCALABLE_SCENARIO_NAME,
    SIGNAL_SCENARIO_NAME,
    STEP8E_ENGINE_CONFIGURATION,
    select_gpu_policy_ids,
)
from evaluation.sota_signal_stress_v1.scenario import (
    SignalScenario,
    build_scenario as build_signal_scenario,
)
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import (
    FORMAL_RECOVERY_MODEL_METADATA,
    RecoveryCostModel,
)
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "evaluation" / "runtime_artifacts"
HISTORICAL_ARTIFACT = (
    ARTIFACT_ROOT / "sota_latency_20260825_113526_839592"
)
FORMAL_SELECTION_AUDIT = (
    ROOT
    / "evaluation"
    / "formal_model_regression"
    / "h100_selection_audit.json"
)
TARGET_POINTS = (
    (SCALABLE_SCENARIO_NAME, 4),
    (SIGNAL_SCENARIO_NAME, 8),
)
TARGET_POINT_LABELS = {
    (SCALABLE_SCENARIO_NAME, 4): "Scalable N16 K4",
    (SIGNAL_SCENARIO_NAME, 8): "SOTA-signal K8",
}
BASELINE_POLICIES = (
    "Global-LRU",
    "KVFlow-style",
    "Marconi-style",
)
EXPECTED_EQUIVALENCE_CLASSES = 33
EXPECTED_MULTIPLICITY_PER_REPETITION = 400
EXPECTED_WARMUP_CASES = 66
EXPECTED_MEASURED_CASES = 330
HISTORICAL_FLOWSTATE_MEAN_MS = {
    (SCALABLE_SCENARIO_NAME, 4): 251.195800505,
    (SIGNAL_SCENARIO_NAME, 8): 219.78845433000004,
}
PROTECTED_PATHS = (
    ROOT / "flowstate" / "recovery_model.py",
    ROOT / "flowstate" / "optimizer.py",
    ROOT / "evaluation" / "sota_policies.py",
    ROOT
    / "evaluation"
    / "controlled_multiworkflow_v1"
    / "policies.py",
    ROOT
    / "evaluation"
    / "public_agent_trace"
    / "tracelab_final_protocol.json",
)
HISTORICAL_SOURCE_PATHS = (
    HISTORICAL_ARTIFACT / "raw_samples.jsonl",
    HISTORICAL_ARTIFACT / "run_metadata.json",
    HISTORICAL_ARTIFACT / "summary.json",
    HISTORICAL_ARTIFACT / "summary.csv",
    FORMAL_SELECTION_AUDIT,
)


@dataclass(frozen=True)
class SelectionMetric:
    """记录一个选择集合在正式模型下的统一规划指标。"""

    total_recovery_cost_ms: float
    mean_recovery_cost_ms: float
    total_gap_tokens: int
    executable_hit_ratio: float
    gap_histogram: Mapping[int, int]


@dataclass(frozen=True)
class SelectionManifestEntry:
    """记录一个代表点与策略的冻结选择及正式规划指标。"""

    scenario: str
    budget_checkpoints: int
    policy: str
    selected_checkpoint_ids: tuple[str, ...]
    selection_source: str
    metric: SelectionMetric


class FormalRerunArtifactWriter:
    """增量保存定向重跑证据，不覆盖任何历史制品。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @classmethod
    def create(cls) -> "FormalRerunArtifactWriter":
        """创建带微秒时间戳的独立制品目录。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = ARTIFACT_ROOT / f"formal_model_latency_rerun_{timestamp}"
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory)

    @property
    def raw_jsonl_path(self) -> Path:
        """返回增量原始记录文件。"""
        return self.directory / "raw_trials.jsonl"

    def append_record(self, record: Mapping[str, object]) -> None:
        """立即保存一个 warmup、measured 或失败 trial。"""
        with self.raw_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    def write_json(self, name: str, payload: object) -> None:
        """以稳定格式写出 JSON。"""
        with (self.directory / name).open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    def write_text(self, name: str, content: str) -> None:
        """以 UTF-8 写出说明文字。"""
        (self.directory / name).write_text(content, encoding="utf-8")


def build_target_equivalence_classes(
    historical_artifact: Path = HISTORICAL_ARTIFACT,
    formal_audit_path: Path = FORMAL_SELECTION_AUDIT,
) -> tuple[LatencyEquivalenceClass, ...]:
    """合并冻结 baseline 类与正式新 FlowState 类。"""
    baseline_classes = _load_historical_baseline_classes(
        historical_artifact / "raw_samples.jsonl"
    )
    formal_expected = _load_formal_expected_selections(formal_audit_path)
    formal_model = RecoveryCostModel()
    result = list(baseline_classes)
    for point in TARGET_POINTS:
        scenario = _scenario_for_point(point)
        allocation = GlobalOptimizer(formal_model).select(
            scenario.continuations,
            scenario.candidates,
            scenario.budget_bytes,
        )
        selected_ids = tuple(
            candidate.checkpoint_id for candidate in allocation.selected
        )
        if selected_ids != formal_expected[point]:
            raise RuntimeError(
                f"{TARGET_POINT_LABELS[point]} 正式选择偏离 Step 10D.4："
                f"{selected_ids} != {formal_expected[point]}"
            )
        result.extend(
            _classes_from_selection(
                point,
                "FlowState",
                scenario.continuations,
                scenario.candidates,
                selected_ids,
            )
        )
    classes = tuple(result)
    _validate_target_classes(classes)
    return classes


def build_selection_manifest(
    classes: Sequence[LatencyEquivalenceClass] | None = None,
) -> tuple[SelectionManifestEntry, ...]:
    """验证所有选择来源并计算正式位置感知成本。"""
    active_classes = tuple(
        build_target_equivalence_classes() if classes is None else classes
    )
    formal_expected = _load_formal_expected_selections(
        FORMAL_SELECTION_AUDIT
    )
    model = RecoveryCostModel()
    entries = []
    for point in TARGET_POINTS:
        scenario = _scenario_for_point(point)
        for policy in GPU_POLICY_NAMES:
            rows = tuple(
                item
                for item in active_classes
                if (item.scenario_name, item.budget_checkpoints) == point
                and item.policy_name == policy
            )
            if not rows:
                raise RuntimeError(
                    f"{TARGET_POINT_LABELS[point]} 缺少策略 {policy}"
                )
            frozen_selection = rows[0].selected_checkpoint_ids
            if any(
                item.selected_checkpoint_ids != frozen_selection
                for item in rows
            ):
                raise RuntimeError("同一策略的等价类包含冲突选择")
            recomputed = select_gpu_policy_ids(policy, scenario, model)
            expected = (
                formal_expected[point]
                if policy == "FlowState"
                else frozen_selection
            )
            if recomputed != expected or frozen_selection != expected:
                source = (
                    "Step 10D.4 正式审计"
                    if policy == "FlowState"
                    else "Step 9B 冻结 baseline"
                )
                raise RuntimeError(
                    f"{TARGET_POINT_LABELS[point]} {policy} 偏离{source}："
                    f"recomputed={recomputed} frozen={frozen_selection} "
                    f"expected={expected}"
                )
            entries.append(
                SelectionManifestEntry(
                    scenario=point[0],
                    budget_checkpoints=point[1],
                    policy=policy,
                    selected_checkpoint_ids=frozen_selection,
                    selection_source=(
                        "Step 10D.4 正式位置感知优化器"
                        if policy == "FlowState"
                        else "Step 9B 历史制品"
                    ),
                    metric=_selection_metric(
                        scenario.continuations,
                        scenario.candidates,
                        frozen_selection,
                        model,
                    ),
                )
            )
    return tuple(entries)


def build_target_schedule(
    classes: Sequence[LatencyEquivalenceClass] | None = None,
    *,
    warmup_repetitions: int = WARMUP_REPETITIONS,
    measured_repetitions: int = MEASURED_REPETITIONS,
    seed: int = POLICY_ORDER_SEED,
) -> tuple[LatencyBenchmarkCase, ...]:
    """沿用 Step 9B 原代表点索引和循环顺序构造定向计划。"""
    if warmup_repetitions < 0 or measured_repetitions <= 0:
        raise ValueError("repetition 配置无效")
    active_classes = tuple(
        build_target_equivalence_classes() if classes is None else classes
    )
    _validate_target_classes(active_classes)
    grouped: dict[
        tuple[str, int, str], list[LatencyEquivalenceClass]
    ] = defaultdict(list)
    for item in active_classes:
        grouped[
            (item.scenario_name, item.budget_checkpoints, item.policy_name)
        ].append(item)

    schedule = []
    sequence = 0
    for is_warmup, repetitions, phase in (
        (True, warmup_repetitions, "warmup"),
        (False, measured_repetitions, "measured"),
    ):
        for repetition in range(repetitions):
            for point in TARGET_POINTS:
                original_point_index = REPRESENTATIVE_POINTS.index(point)
                policy_order = balanced_policy_order(
                    original_point_index,
                    repetition,
                    seed=seed,
                )
                for position, policy in enumerate(policy_order):
                    point_classes = sorted(
                        grouped[(point[0], point[1], policy)],
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
                                    f"formal_{phase}_{sequence:04d}_"
                                    f"{_identifier(item.scenario_name)}_"
                                    f"k{item.budget_checkpoints}_"
                                    f"{_identifier(policy)}_r{repetition:02d}_"
                                    f"{_identifier(item.representative_continuation_id)}"
                                ),
                                equivalence_class=item,
                                repetition=repetition,
                                is_warmup=is_warmup,
                                execution_order_position=position,
                            )
                        )
    result = tuple(schedule)
    expected_warmup = len(active_classes) * warmup_repetitions
    expected_measured = len(active_classes) * measured_repetitions
    if sum(item.is_warmup for item in result) != expected_warmup:
        raise RuntimeError("定向 warmup case 数量异常")
    if sum(not item.is_warmup for item in result) != expected_measured:
        raise RuntimeError("定向 measured case 数量异常")
    if len({item.case_id for item in result}) != len(result):
        raise RuntimeError("定向 latency case_id 不唯一")
    return result


def build_weighted_summary(
    records: Sequence[Mapping[str, object]],
    manifest: Sequence[SelectionManifestEntry],
) -> tuple[dict[str, object], ...]:
    """沿用 Step 9B 加权语义并补充原始 trial 与正式目标指标。"""
    modeled = {
        (row.scenario, row.budget_checkpoints, row.policy): row.metric
        for row in manifest
    }
    base_rows = aggregate_latency_records(records)
    result = []
    for row in base_rows:
        key = (str(row["scenario"]), int(row["K"]), str(row["policy"]))
        samples = tuple(
            item
            for item in records
            if item.get("is_warmup") is False
            and item.get("correctness_pass") is True
            and (
                str(item["scenario"]),
                int(item["K"]),
                str(item["policy"]),
            )
            == key
        )
        metric = modeled[key]
        ttft = tuple(float(item["ttft_ms"]) for item in samples)
        latency = tuple(
            float(item["request_latency_ms"]) for item in samples
        )
        result.append(
            {
                **row,
                "ttft_unweighted": _unweighted_summary(ttft),
                "request_latency_unweighted": _unweighted_summary(latency),
                "modeled_total_recovery_cost_ms": (
                    metric.total_recovery_cost_ms
                ),
                "modeled_mean_recovery_cost_ms": (
                    metric.mean_recovery_cost_ms
                ),
                "total_gap_tokens": metric.total_gap_tokens,
                "executable_hit_ratio": metric.executable_hit_ratio,
                "gap_histogram": {
                    str(gap): count
                    for gap, count in metric.gap_histogram.items()
                },
            }
        )
    return tuple(result)


def build_gap_group_summary(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """按代表点、策略和恢复间隔汇总真实 TTFT。"""
    groups: dict[
        tuple[str, int, str, int], list[Mapping[str, object]]
    ] = defaultdict(list)
    for record in records:
        if record.get("is_warmup") is not False:
            continue
        if record.get("correctness_pass") is not True:
            continue
        groups[
            (
                str(record["scenario"]),
                int(record["K"]),
                str(record["policy"]),
                int(record["planning_gap"]),
            )
        ].append(record)
    rows = []
    for key in sorted(groups):
        samples = groups[key]
        values = tuple(float(item["ttft_ms"]) for item in samples)
        weights = tuple(
            float(item["class_multiplicity"]) for item in samples
        )
        rows.append(
            {
                "scenario": key[0],
                "K": key[1],
                "policy": key[2],
                "gap_tokens": key[3],
                "sample_count": len(samples),
                "weighted_request_count": sum(weights),
                "ttft_weighted_mean_ms": weighted_mean(values, weights),
                "ttft_weighted_median_ms": weighted_quantile(
                    values, weights, 0.5
                ),
                "ttft_weighted_p95_ms": weighted_quantile(
                    values, weights, 0.95
                ),
            }
        )
    return tuple(rows)


def build_runtime_correctness(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """汇总定向运行完成度、H/E/G 与组件安全门禁。"""
    warmup = tuple(
        row for row in records if row.get("is_warmup") is True
    )
    measured = tuple(
        row for row in records if row.get("is_warmup") is False
    )
    failures = tuple(row for row in records if row.get("status") != "PASS")
    result = {
        "warmup_expected": EXPECTED_WARMUP_CASES,
        "warmup_success": sum(row.get("status") == "PASS" for row in warmup),
        "measured_expected": EXPECTED_MEASURED_CASES,
        "measured_success": sum(
            row.get("status") == "PASS" for row in measured
        ),
        "runtime_mismatch_count": sum(
            row.get("H_match") is False
            or row.get("E_match") is False
            or row.get("G_match") is False
            for row in records
        ),
        "H_mismatch": sum(row.get("H_match") is False for row in records),
        "E_mismatch": sum(row.get("E_match") is False for row in records),
        "G_mismatch": sum(row.get("G_match") is False for row in records),
        "safety_failures": sum(
            row.get("safety_pass") is False for row in records
        ),
        "failed_trials": len(failures),
        "retried_trials": sum(
            int(row.get("retry_count", 0)) > 0 for row in records
        ),
    }
    points = []
    for point in TARGET_POINTS:
        rows = tuple(
            row
            for row in records
            if (str(row.get("scenario")), int(row.get("K", -1))) == point
        )
        points.append(
            {
                "point": TARGET_POINT_LABELS[point],
                "trials": len(rows),
                "status": (
                    "PASS"
                    if rows
                    and all(row.get("status") == "PASS" for row in rows)
                    else "FAIL"
                ),
                "runtime_mismatch_count": sum(
                    row.get("H_match") is False
                    or row.get("E_match") is False
                    or row.get("G_match") is False
                    for row in rows
                ),
                "safety_failures": sum(
                    row.get("safety_pass") is False for row in rows
                ),
            }
        )
    result["points"] = points
    result["status"] = (
        "PASS"
        if result["warmup_success"] == EXPECTED_WARMUP_CASES
        and result["measured_success"] == EXPECTED_MEASURED_CASES
        and result["runtime_mismatch_count"] == 0
        and result["safety_failures"] == 0
        and not failures
        else "FAIL"
    )
    return result


def compare_with_historical(
    weighted_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """把本次四策略同源重跑与旧 FlowState 历史值分层记录。"""
    old_summary = json.loads(
        (HISTORICAL_ARTIFACT / "summary.json").read_text(encoding="utf-8")
    )
    old_rows = {
        (str(row["scenario"]), int(row["K"]), str(row["policy"])): row
        for row in old_summary["policy_summaries"]
    }
    current_rows = {
        (str(row["scenario"]), int(row["K"]), str(row["policy"])): row
        for row in weighted_rows
    }
    points = []
    for point in TARGET_POINTS:
        flowstate = current_rows[(point[0], point[1], "FlowState")]
        comparisons = {}
        for baseline_policy in BASELINE_POLICIES:
            baseline = current_rows[(point[0], point[1], baseline_policy)]
            comparisons[baseline_policy] = {
                statistic: _reduction_percent(
                    float(baseline["ttft_ms"][statistic]),
                    float(flowstate["ttft_ms"][statistic]),
                )
                for statistic in (
                    "weighted_mean",
                    "weighted_median",
                    "weighted_p95",
                )
            }
        old_flowstate = old_rows[(point[0], point[1], "FlowState")]
        points.append(
            {
                "point": TARGET_POINT_LABELS[point],
                "current_same_run_comparisons": comparisons,
                "historical_flowstate_ttft_ms": old_flowstate["ttft_ms"],
                "current_flowstate_ttft_ms": flowstate["ttft_ms"],
                "historical_flowstate_mean_expected_ms": (
                    HISTORICAL_FLOWSTATE_MEAN_MS[point]
                ),
                "selection_changed": True,
                "historical_interpretation": "仅作为历史参考",
            }
        )
    return {
        "source": str(HISTORICAL_ARTIFACT),
        "baseline_comparison_source": "本次同源重跑",
        "old_flowstate_use": "仅作为历史参考",
        "points": points,
    }


def protected_source_hashes() -> dict[str, str]:
    """计算正式模型、策略和 TraceLab 协议的只读指纹。"""
    return {_relative(path): _sha256(path) for path in PROTECTED_PATHS}


def historical_source_hashes() -> dict[str, str]:
    """计算 Step 9B 与 Step 10D.4 输入制品的只读指纹。"""
    return {
        _relative(path): _sha256(path) for path in HISTORICAL_SOURCE_PATHS
    }


def _load_historical_baseline_classes(
    raw_samples_path: Path,
) -> tuple[LatencyEquivalenceClass, ...]:
    """从 Step 9B 成功样本恢复三个 baseline 的冻结类。"""
    groups: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    with raw_samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            point = (str(row["scenario"]), int(row["K"]))
            policy = str(row["policy"])
            if point not in TARGET_POINTS or policy not in BASELINE_POLICIES:
                continue
            if row.get("correctness_pass") is not True:
                raise RuntimeError("Step 9B baseline 来源包含失败样本")
            groups[
                (point[0], point[1], policy, str(row["equivalence_class"]))
            ].append(row)
    result = []
    for key in sorted(groups):
        rows = groups[key]
        first = rows[0]
        identity = (
            str(first["continuation_id"]),
            int(first["class_multiplicity"]),
            int(first["planning_H"]),
            int(first["planning_E"]),
            int(first["planning_G"]),
            tuple(first["selected_checkpoint_ids"]),
        )
        if any(
            (
                str(row["continuation_id"]),
                int(row["class_multiplicity"]),
                int(row["planning_H"]),
                int(row["planning_E"]),
                int(row["planning_G"]),
                tuple(row["selected_checkpoint_ids"]),
            )
            != identity
            for row in rows
        ):
            raise RuntimeError(f"Step 9B 等价类记录不一致：{key}")
        scenario = _scenario_for_point((key[0], key[1]))
        workflow_id = next(
            continuation.workflow_id
            for continuation in scenario.continuations
            if continuation.continuation_id == identity[0]
        )
        result.append(
            LatencyEquivalenceClass(
                scenario_name=key[0],
                budget_checkpoints=key[1],
                policy_name=key[2],
                representative_continuation_id=identity[0],
                workflow_id=workflow_id,
                selected_checkpoint_ids=identity[5],
                planning_target=identity[2],
                planning_executable_frontier=identity[3],
                planning_gap_tokens=identity[4],
                class_multiplicity=identity[1],
            )
        )
    return tuple(result)


def _load_formal_expected_selections(
    audit_path: Path,
) -> dict[tuple[str, int], tuple[str, ...]]:
    """只读取 Step 10D.4 中标记为 CHANGED 的正式选择。"""
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    result = {}
    for row in payload["points"]:
        point = (str(row["scenario"]), int(row["budget_checkpoints"]))
        if point not in TARGET_POINTS:
            continue
        if row["classification"] != "CHANGED":
            raise RuntimeError(f"目标点未被 Step 10D.4 标记为 CHANGED：{point}")
        result[point] = tuple(row["new_formal_selection"])
    if set(result) != set(TARGET_POINTS):
        raise RuntimeError("Step 10D.4 正式选择审计未覆盖两个目标点")
    return result


def _classes_from_selection(
    point: tuple[str, int],
    policy: str,
    continuations: Sequence[PendingContinuation],
    candidates: Sequence[CheckpointCandidate],
    selected_ids: tuple[str, ...],
) -> tuple[LatencyEquivalenceClass, ...]:
    """沿用 Step 9B 的 T/E/G 键和字典序代表规则构造类。"""
    candidates_by_id = {
        candidate.checkpoint_id: candidate for candidate in candidates
    }
    selected = tuple(candidates_by_id[item] for item in selected_ids)
    groups: dict[
        tuple[int, int, int], list[PendingContinuation]
    ] = defaultdict(list)
    for continuation in continuations:
        frontier = executable_frontier(continuation, selected)
        gap = recovery_gap(continuation, selected)
        groups[(continuation.planning_target, frontier, gap)].append(
            continuation
        )
    rows = []
    for key in sorted(groups):
        members = groups[key]
        representative = min(
            members, key=lambda item: item.continuation_id
        )
        rows.append(
            LatencyEquivalenceClass(
                scenario_name=point[0],
                budget_checkpoints=point[1],
                policy_name=policy,
                representative_continuation_id=(
                    representative.continuation_id
                ),
                workflow_id=representative.workflow_id,
                selected_checkpoint_ids=selected_ids,
                planning_target=key[0],
                planning_executable_frontier=key[1],
                planning_gap_tokens=key[2],
                class_multiplicity=len(members),
            )
        )
    return tuple(rows)


def _validate_target_classes(
    classes: Sequence[LatencyEquivalenceClass],
) -> None:
    """拒绝额外代表点、策略缺失和 multiplicity 漂移。"""
    if len(classes) != EXPECTED_EQUIVALENCE_CLASSES:
        raise RuntimeError(f"定向等价类数量异常：{len(classes)}")
    points = {
        (item.scenario_name, item.budget_checkpoints) for item in classes
    }
    if points != set(TARGET_POINTS):
        raise RuntimeError(f"定向运行包含非预注册代表点：{points}")
    groups = {
        (item.scenario_name, item.budget_checkpoints, item.policy_name)
        for item in classes
    }
    expected_groups = {
        (point[0], point[1], policy)
        for point in TARGET_POINTS
        for policy in GPU_POLICY_NAMES
    }
    if groups != expected_groups:
        raise RuntimeError("定向运行策略覆盖不完整")
    multiplicity = sum(item.class_multiplicity for item in classes)
    if multiplicity != EXPECTED_MULTIPLICITY_PER_REPETITION:
        raise RuntimeError(f"定向 multiplicity 总和异常：{multiplicity}")


def _selection_metric(
    continuations: Sequence[PendingContinuation],
    candidates: Sequence[CheckpointCandidate],
    selected_ids: Sequence[str],
    model: RecoveryCostModel,
) -> SelectionMetric:
    """使用正式 Phi(G,T) 计算统一规划成本、gap 和 EHR。"""
    candidates_by_id = {
        candidate.checkpoint_id: candidate for candidate in candidates
    }
    selected = tuple(candidates_by_id[item] for item in selected_ids)
    total_target = 0
    total_frontier = 0
    total_gap = 0
    total_cost = 0.0
    histogram: dict[int, int] = defaultdict(int)
    for continuation in continuations:
        target = continuation.planning_target
        frontier = executable_frontier(continuation, selected)
        gap = recovery_gap(continuation, selected)
        total_target += target
        total_frontier += frontier
        total_gap += gap
        total_cost += model.estimate(gap, target)
        histogram[gap] += 1
    count = len(continuations)
    return SelectionMetric(
        total_recovery_cost_ms=total_cost,
        mean_recovery_cost_ms=total_cost / count if count else 0.0,
        total_gap_tokens=total_gap,
        executable_hit_ratio=(
            total_frontier / total_target if total_target else 0.0
        ),
        gap_histogram=dict(sorted(histogram.items())),
    )


def _scenario_for_point(
    point: tuple[str, int],
) -> ScalableScenario | SignalScenario:
    """只重建两个预注册、未修改的受控场景。"""
    if point == (SCALABLE_SCENARIO_NAME, 4):
        return build_scalable_scenario(16, 4)
    if point == (SIGNAL_SCENARIO_NAME, 8):
        return build_signal_scenario(8)
    raise ValueError(f"未预注册的定向 H100 代表点：{point}")


def _unweighted_summary(values: Sequence[float]) -> dict[str, float]:
    """生成不使用 workload multiplicity 的原始 trial 摘要。"""
    if not values:
        raise ValueError("原始 trial 摘要不能为空")
    weights = (1.0,) * len(values)
    return {
        "mean": statistics.fmean(values),
        "median": weighted_quantile(values, weights, 0.5),
        "p95": weighted_quantile(values, weights, 0.95),
    }


def _reduction_percent(baseline: float, current: float) -> float:
    """计算 FlowState 相对同次 baseline 的百分比降幅。"""
    if baseline <= 0.0 or not math.isfinite(baseline + current):
        raise ValueError("相对降幅输入必须为有限正 baseline")
    return (baseline - current) / baseline * 100.0


def _manifest_payload(
    rows: Sequence[SelectionManifestEntry],
) -> list[dict[str, object]]:
    """把不可变 manifest 转换为 JSON 友好的结构。"""
    payload = []
    for row in rows:
        metric = asdict(row.metric)
        metric["gap_histogram"] = {
            str(gap): count
            for gap, count in row.metric.gap_histogram.items()
        }
        payload.append(
            {
                "scenario": row.scenario,
                "K": row.budget_checkpoints,
                "policy": row.policy,
                "selected_checkpoint_ids": list(
                    row.selected_checkpoint_ids
                ),
                "selection_source": row.selection_source,
                "formal_metric": metric,
            }
        )
    return payload


def _schedule_payload(
    rows: Sequence[LatencyBenchmarkCase],
) -> list[dict[str, object]]:
    """保存完整确定性执行次序。"""
    return [
        {
            "case_id": row.case_id,
            "scenario": row.scenario_name,
            "K": row.budget_checkpoints,
            "policy": row.policy_name,
            "continuation_id": (
                row.equivalence_class.representative_continuation_id
            ),
            "class_multiplicity": (
                row.equivalence_class.class_multiplicity
            ),
            "repetition": row.repetition,
            "is_warmup": row.is_warmup,
            "execution_order_position": row.execution_order_position,
        }
        for row in rows
    ]


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """把嵌套字段稳定序列化后写入 CSV。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row.get(field)) for field in fieldnames}
            )


def _write_raw_trials_csv(
    path: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    """写出可直接审计的 trial 级平面表。"""
    fields = (
        "case_id",
        "scenario",
        "K",
        "policy",
        "continuation_id",
        "equivalence_class",
        "class_multiplicity",
        "repetition",
        "execution_order_position",
        "is_warmup",
        "planning_H",
        "planning_E",
        "planning_G",
        "runtime_H",
        "runtime_E",
        "runtime_G",
        "ttft_ms",
        "request_latency_ms",
        "snapshot_build_ms",
        "reconcile_ms",
        "correctness_pass",
        "safety_pass",
        "retry_count",
        "status",
        "failure_stage",
        "error",
        "selected_checkpoint_ids",
        "actual_selected_checkpoint_ids",
        "evicted_checkpoint_ids",
        "safety_flags",
    )
    _write_csv(path, records, fields)


def _write_weighted_summary_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """写出论文比较使用的同源加权结果。"""
    flat = []
    for row in rows:
        flat.append(
            {
                "scenario": row["scenario"],
                "K": row["K"],
                "policy": row["policy"],
                "measured_case_count": row["measured_case_count"],
                "weighted_request_count": row["weighted_request_count"],
                "ttft_weighted_mean_ms": row["ttft_ms"]["weighted_mean"],
                "ttft_weighted_median_ms": row["ttft_ms"]["weighted_median"],
                "ttft_weighted_p95_ms": row["ttft_ms"]["weighted_p95"],
                "ttft_unweighted_mean_ms": row["ttft_unweighted"]["mean"],
                "ttft_unweighted_median_ms": row["ttft_unweighted"]["median"],
                "ttft_unweighted_p95_ms": row["ttft_unweighted"]["p95"],
                "latency_weighted_mean_ms": row["request_latency_ms"]["weighted_mean"],
                "latency_weighted_median_ms": row["request_latency_ms"]["weighted_median"],
                "latency_weighted_p95_ms": row["request_latency_ms"]["weighted_p95"],
                "modeled_total_recovery_cost_ms": row["modeled_total_recovery_cost_ms"],
                "modeled_mean_recovery_cost_ms": row["modeled_mean_recovery_cost_ms"],
                "total_gap_tokens": row["total_gap_tokens"],
                "executable_hit_ratio": row["executable_hit_ratio"],
                "gap_histogram": row["gap_histogram"],
            }
        )
    _write_csv(path, flat, tuple(flat[0]) if flat else ())


def _csv_value(value: object) -> object:
    """把容器转换成稳定 JSON，其余值保持原样。"""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _environment_text() -> str:
    """记录本次嵌入式 runtime 可验证的环境信息。"""
    try:
        import torch

        gpu = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "不可见"
        )
    except Exception as error:
        gpu = f"读取失败：{error!r}"
    return "\n".join(
        (
            f"记录时间：{datetime.now().astimezone().isoformat()}",
            f"操作系统：{platform.platform()}",
            f"Python：{platform.python_version()}",
            f"SGLang：{importlib.metadata.version('sglang')}",
            "模型：Qwen3.5-9B（/model）",
            f"图形处理器：{gpu}",
            "张量并行：1",
            "运行形态：冻结 SGLang Docker 内嵌 Engine",
            "",
        )
    )


def _server_command_text() -> str:
    """记录本次容器内入口与嵌入式引擎配置。"""
    return (
        "容器内入口：\n"
        "python3 -m evaluation.formal_model_latency_rerun\n\n"
        "SGLang 使用 FormalEndToEndGateEngine 内嵌启动。\n"
        "引擎参数：\n"
        + json.dumps(
            STEP8E_ENGINE_CONFIGURATION,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _readme_text() -> str:
    """生成制品的数据来源、口径与解释边界。"""
    return """# 正式位置感知模型定向 H100 latency 重跑

本目录只包含 Scalable N16 K4 与 SOTA-signal K8 两个代表点。
三个 baseline 的选择来自 Step 9B 冻结制品；FlowState 的选择来自
Step 10D.4 正式位置感知恢复模型审计。所有策略在本次运行中使用同一
runtime、同一 fresh-snapshot 协议和同一计时边界。

`class_multiplicity` 用于恢复逻辑 workload 权重；warmup 不进入统计。
checkpoint rebuild、flush 与 reconcile 均不计入目标请求 TTFT。
旧 FlowState latency 只作为历史参考，正式 baseline 对比必须使用本次
同源重跑数据。真实 latency 不由正式恢复模型预测值替代。
"""


def _config_payload(
    classes: Sequence[LatencyEquivalenceClass],
    schedule: Sequence[LatencyBenchmarkCase],
    protected_hashes: Mapping[str, str],
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    """构造不可在运行中调整的预注册配置。"""
    return {
        "schema_version": "flowstate.formal_model_latency_rerun.v1",
        "target_points": [
            {"scenario": point[0], "K": point[1]}
            for point in TARGET_POINTS
        ],
        "policies": list(GPU_POLICY_NAMES),
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "policy_order_seed": POLICY_ORDER_SEED,
        "maximum_deterministic_retries": MAX_DETERMINISTIC_RETRIES,
        "equivalence_classes": len(classes),
        "multiplicity_per_repetition": sum(
            item.class_multiplicity for item in classes
        ),
        "warmup_cases": sum(item.is_warmup for item in schedule),
        "measured_cases": sum(not item.is_warmup for item in schedule),
        "engine_configuration": STEP8E_ENGINE_CONFIGURATION,
        "formal_model": asdict(FORMAL_RECOVERY_MODEL_METADATA),
        "historical_artifact": str(HISTORICAL_ARTIFACT),
        "formal_selection_audit": str(FORMAL_SELECTION_AUDIT),
        "protected_source_hashes_before": dict(protected_hashes),
        "input_artifact_hashes_before": dict(source_hashes),
    }


def _sha256(path: Path) -> str:
    """计算只读输入文件的 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    """优先返回相对仓库的稳定路径。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _identifier(value: str) -> str:
    """把稳定名称转换成 runtime namespace 片段。"""
    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    ).strip("_")


def main() -> int:
    """在一个 GPU 进程内完成两个定向代表点的正式重跑。"""
    protected_before = protected_source_hashes()
    source_before = historical_source_hashes()
    classes = build_target_equivalence_classes()
    manifest = build_selection_manifest(classes)
    schedule = build_target_schedule(classes)
    if len(schedule) != EXPECTED_WARMUP_CASES + EXPECTED_MEASURED_CASES:
        raise RuntimeError(f"定向 schedule 数量异常：{len(schedule)}")

    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import (
        FormalEndToEndGateEngine,
        requested_control_port,
    )
    from evaluation.controlled_multiworkflow_v1.runtime_gate import (
        wait_for_transport,
    )

    writer = FormalRerunArtifactWriter.create()
    writer.write_text("README.md", _readme_text())
    writer.write_text("server_command.txt", _server_command_text())
    writer.write_json("selection_manifest.json", _manifest_payload(manifest))
    writer.write_json("execution_order.json", _schedule_payload(schedule))
    writer.write_json(
        "config.json",
        _config_payload(
            classes,
            schedule,
            protected_before,
            source_before,
        ),
    )
    writer.write_text("environment.txt", _environment_text())

    engine = None
    records: list[dict[str, object]] = []
    fatal_error = None
    started_ns = time.perf_counter_ns()
    model = RecoveryCostModel()
    try:
        engine = FormalEndToEndGateEngine(**STEP8E_ENGINE_CONFIGURATION)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        for index, case in enumerate(schedule, start=1):
            phase = "预热" if case.is_warmup else "测量"
            print(
                f"[STEP10D5] {index}/{len(schedule)} {phase} "
                f"{case.scenario_name} K={case.budget_checkpoints} "
                f"{case.policy_name} "
                f"{case.equivalence_class.representative_continuation_id}",
                flush=True,
            )
            record = execute_with_deterministic_retry(
                case,
                lambda active_case, retry_count: execute_runtime_case_once(
                    active_case,
                    engine=engine,
                    client=client,
                    recovery_cost_model=model,
                    retry_count=retry_count,
                ),
            )
            records.append(record)
            writer.append_record(record)
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

    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    correctness = build_runtime_correctness(records)
    if fatal_error is not None:
        correctness["status"] = "FAIL"
        correctness["fatal_error"] = fatal_error
    correctness["total_runtime_ms"] = elapsed_ms
    protected_after = protected_source_hashes()
    source_after = historical_source_hashes()
    correctness["protected_sources_unchanged"] = (
        protected_before == protected_after
    )
    correctness["input_artifacts_unchanged"] = source_before == source_after
    if (
        not correctness["protected_sources_unchanged"]
        or not correctness["input_artifacts_unchanged"]
    ):
        correctness["status"] = "FAIL"

    weighted_rows = build_weighted_summary(records, manifest)
    gap_rows = build_gap_group_summary(records)
    writer.write_json("runtime_correctness.json", correctness)
    writer.write_json("weighted_summary.json", weighted_rows)
    writer.write_json("gap_group_summary.json", gap_rows)
    _write_raw_trials_csv(writer.directory / "raw_trials.csv", records)
    _write_weighted_summary_csv(
        writer.directory / "weighted_summary.csv", weighted_rows
    )
    _write_csv(
        writer.directory / "gap_group_summary.csv",
        gap_rows,
        tuple(gap_rows[0]) if gap_rows else (),
    )
    if correctness["status"] == "PASS":
        comparison = compare_with_historical(weighted_rows)
    else:
        comparison = {
            "status": "未完成",
            "reason": "runtime correctness 未通过",
            "source": str(HISTORICAL_ARTIFACT),
        }
    writer.write_json("comparison_to_historical.json", comparison)

    result = {
        "status": correctness["status"],
        "warmup": (
            f"{correctness['warmup_success']} / "
            f"{correctness['warmup_expected']}"
        ),
        "measured": (
            f"{correctness['measured_success']} / "
            f"{correctness['measured_expected']}"
        ),
        "runtime_mismatch_count": correctness["runtime_mismatch_count"],
        "safety_failures": correctness["safety_failures"],
        "artifacts": str(writer.directory),
        "fatal_error": fatal_error,
    }
    print(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        flush=True,
    )
    return 0 if correctness["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
