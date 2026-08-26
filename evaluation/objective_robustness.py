#!/usr/bin/env python3
"""精确分析正式 FlowState 分配在冻结成本不确定性下的目标后悔值。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
from typing import Iterator, Mapping, Sequence, Union

from evaluation.cost_model_sensitivity import (
    CostModel,
    ExactGapPerturbedCostModel,
    LOCAL_KNOT_GAPS,
    LOCAL_SCALE_FACTORS,
    REPRESENTATIVE_POINTS,
    RepresentativePoint,
    build_point_scenario,
    load_profiler_v2_model,
    scenario_fingerprint,
    select_flowstate,
    select_marconi,
)
from evaluation.scalable_multiworkflow_v2.scenario import ScalableScenario
from evaluation.sota_signal_stress_v1.scenario import SignalScenario
from flowstate.executable_state import recovery_gap
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_JSON_PATH = _REPOSITORY_ROOT / "evaluation/objective_robustness.json"
_OUTPUT_MARKDOWN_PATH = (
    _REPOSITORY_ROOT / "evaluation/OBJECTIVE_ROBUSTNESS.md"
)
_FLOAT_TOLERANCE_MS = 1e-9
EXPECTED_UNCERTAINTY_MODEL_COUNT = 18


@dataclass(frozen=True)
class UncertaintyModel:
    """绑定一个冻结成本模型及其可审计来源。"""

    model_name: str
    model_kind: str
    perturbed_gap: int | None
    perturbation_fraction: float | None
    estimator: CostModel


@dataclass(frozen=True)
class LocalSubsetOption:
    """记录一个 workflow 内局部选择及其 gap histogram。"""

    selected_mask: int
    selected_count: int
    gap_counts: tuple[int, ...]


@dataclass(frozen=True)
class FeasibleSubset:
    """记录一个全局可行集合及其预计算 gap histogram。"""

    selected_mask: int
    selected_count: int
    gap_counts: tuple[int, ...]


@dataclass(frozen=True)
class FeasibleSubsetSpace:
    """保存一次精确枚举所需的候选索引与局部 histogram。"""

    candidate_ids: tuple[str, ...]
    gap_values: tuple[int, ...]
    capacity: int
    workflow_options: tuple[tuple[LocalSubsetOption, ...], ...]

    def iter_subsets(self) -> Iterator[FeasibleSubset]:
        """只遍历一次全部满足全局预算的可行集合。"""
        counts = [0] * len(self.gap_values)

        def visit(
            workflow_index: int,
            used: int,
            selected_mask: int,
        ) -> Iterator[FeasibleSubset]:
            if workflow_index == len(self.workflow_options):
                yield FeasibleSubset(
                    selected_mask=selected_mask,
                    selected_count=used,
                    gap_counts=tuple(counts),
                )
                return
            for option in self.workflow_options[workflow_index]:
                next_used = used + option.selected_count
                if next_used > self.capacity:
                    continue
                for index, value in enumerate(option.gap_counts):
                    counts[index] += value
                yield from visit(
                    workflow_index + 1,
                    next_used,
                    selected_mask | option.selected_mask,
                )
                for index, value in enumerate(option.gap_counts):
                    counts[index] -= value

        yield from visit(0, 0, 0)


@dataclass(frozen=True)
class GapRisk:
    """描述一个选择产生的 recovery-gap 风险分布。"""

    total_gap_tokens: int
    mean_gap_tokens: float
    max_gap_tokens: int
    p95_gap_tokens: int


@dataclass(frozen=True)
class ModelRegret:
    """记录一个不确定性模型下的 exact objective regret。"""

    model_name: str
    model_kind: str
    perturbed_gap: int | None
    perturbation_fraction: float | None
    formal_checkpoint_ids: tuple[str, ...]
    exact_best_checkpoint_ids: tuple[str, ...]
    exact_selection_changed: bool
    formal_cost_ms: float
    optimal_cost_ms: float
    absolute_regret_ms: float
    relative_regret: float | None
    optimal_cost_is_zero: bool
    formal_gap_risk: GapRisk
    optimal_gap_risk: GapRisk


@dataclass(frozen=True)
class PointRobustness:
    """汇总一个冻结代表点的全部 exact regret 指标。"""

    point: RepresentativePoint
    candidate_count: int
    continuation_count: int
    capacity: int
    feasible_subset_count: int
    distinct_histogram_count: int
    uncertainty_model_count: int
    formal_checkpoint_ids: tuple[str, ...]
    model_results: tuple[ModelRegret, ...]
    exact_selection_change_count: int
    mean_relative_regret: float
    median_relative_regret: float
    max_relative_regret: float
    profiler_v2_relative_regret: float
    worst_case_model_name: str


@dataclass(frozen=True)
class SotaK8ProfilerComparison:
    """记录 SOTA-signal K8 在独立 v2 模型下的重点比较。"""

    formal_checkpoint_ids: tuple[str, ...]
    profiler_v2_exact_best_checkpoint_ids: tuple[str, ...]
    marconi_checkpoint_ids: tuple[str, ...]
    profiler_v2_best_equals_marconi: bool
    formal_cost_under_v2_ms: float
    optimal_cost_under_v2_ms: float
    absolute_regret_ms: float
    relative_regret: float | None


@dataclass(frozen=True)
class ObjectiveRobustnessResult:
    """保存四个冻结代表点的完整目标鲁棒性结果。"""

    schema_version: str
    uncertainty_model_count: int
    points: tuple[PointRobustness, ...]
    sota_k8_profiler_v2: SotaK8ProfilerComparison
    exact_selection_instability_present: bool
    objective_regret_observed: bool
    data_isolation: Mapping[str, object]


Scenario = Union[ScalableScenario, SignalScenario]


def build_uncertainty_models(
    formal_model: CostModel | None = None,
    profiler_v2_model: CostModel | None = None,
) -> tuple[UncertaintyModel, ...]:
    """构造 Step 9E 已冻结且不含整体缩放的十八个模型。"""
    active_formal = formal_model or RecoveryCostModel()
    active_profiler = profiler_v2_model or load_profiler_v2_model()
    models = [
        UncertaintyModel(
            model_name="Old Phi",
            model_kind="formal",
            perturbed_gap=None,
            perturbation_fraction=None,
            estimator=active_formal,
        ),
        UncertaintyModel(
            model_name="Profiler-v2 piecewise",
            model_kind="profiler_v2",
            perturbed_gap=None,
            perturbation_fraction=None,
            estimator=active_profiler,
        ),
    ]
    for gap in LOCAL_KNOT_GAPS:
        for scale in LOCAL_SCALE_FACTORS:
            models.append(
                UncertaintyModel(
                    model_name=(
                        f"Old Phi G={gap} {scale - 1.0:+.0%}"
                    ),
                    model_kind="local_knot_perturbation",
                    perturbed_gap=gap,
                    perturbation_fraction=scale - 1.0,
                    estimator=ExactGapPerturbedCostModel(
                        active_formal,
                        gap,
                        scale,
                    ),
                )
            )
    if len(models) != EXPECTED_UNCERTAINTY_MODEL_COUNT:
        raise RuntimeError("冻结 uncertainty model 数量异常")
    return tuple(models)


def build_feasible_subset_space(scenario: Scenario) -> FeasibleSubsetSpace:
    """按 workflow 预计算局部 histogram，并保留全局精确预算枚举。"""
    eligible = tuple(
        sorted(
            (
                candidate
                for candidate in scenario.candidates
                if candidate.recurrent_resident
            ),
            key=lambda candidate: candidate.checkpoint_id,
        )
    )
    if not eligible:
        gaps = tuple(
            recovery_gap(continuation, ())
            for continuation in scenario.continuations
        )
        gap_values = tuple(sorted(set(gaps)))
        counts = tuple(gaps.count(gap) for gap in gap_values)
        return FeasibleSubsetSpace(
            candidate_ids=(),
            gap_values=gap_values,
            capacity=0,
            workflow_options=((LocalSubsetOption(0, 0, counts),),),
        )

    checkpoint_size = eligible[0].memory_bytes
    if any(
        candidate.memory_bytes != checkpoint_size for candidate in eligible
    ):
        raise ValueError("objective robustness 只支持冻结的等大小 checkpoint")
    capacity = min(
        scenario.budget_bytes // checkpoint_size,
        len(eligible),
    )
    candidate_ids = tuple(
        candidate.checkpoint_id for candidate in eligible
    )
    bit_by_id = {
        checkpoint_id: 1 << index
        for index, checkpoint_id in enumerate(candidate_ids)
    }
    candidates_by_workflow: dict[str, list[CheckpointCandidate]] = {}
    for candidate in eligible:
        candidates_by_workflow.setdefault(candidate.workflow_id, []).append(
            candidate
        )
    continuations_by_workflow: dict[str, list[PendingContinuation]] = {}
    for continuation in scenario.continuations:
        continuations_by_workflow.setdefault(
            continuation.workflow_id,
            [],
        ).append(continuation)
    workflow_ids = tuple(
        sorted(set(candidates_by_workflow) | set(continuations_by_workflow))
    )

    raw_options: list[list[tuple[int, int, dict[int, int]]]] = []
    all_gaps: set[int] = set()
    for workflow_id in workflow_ids:
        local_candidates = tuple(
            sorted(
                candidates_by_workflow.get(workflow_id, ()),
                key=lambda candidate: candidate.checkpoint_id,
            )
        )
        local_continuations = tuple(
            continuations_by_workflow.get(workflow_id, ())
        )
        workflow_rows = []
        for subset_size in range(len(local_candidates) + 1):
            for subset in combinations(local_candidates, subset_size):
                histogram: dict[int, int] = {}
                for continuation in local_continuations:
                    gap = recovery_gap(continuation, subset)
                    histogram[gap] = histogram.get(gap, 0) + 1
                    all_gaps.add(gap)
                selected_mask = 0
                for candidate in subset:
                    selected_mask |= bit_by_id[candidate.checkpoint_id]
                workflow_rows.append(
                    (selected_mask, subset_size, histogram)
                )
        raw_options.append(workflow_rows)

    gap_values = tuple(sorted(all_gaps))
    workflow_options = tuple(
        tuple(
            LocalSubsetOption(
                selected_mask=selected_mask,
                selected_count=selected_count,
                gap_counts=tuple(
                    histogram.get(gap, 0) for gap in gap_values
                ),
            )
            for selected_mask, selected_count, histogram in rows
        )
        for rows in raw_options
    )
    return FeasibleSubsetSpace(
        candidate_ids=candidate_ids,
        gap_values=gap_values,
        capacity=capacity,
        workflow_options=workflow_options,
    )


def selected_ids_from_mask(
    space: FeasibleSubsetSpace,
    selected_mask: int,
) -> tuple[str, ...]:
    """把稳定候选位图转换成按 ID 字典序排列的集合。"""
    return tuple(
        checkpoint_id
        for index, checkpoint_id in enumerate(space.candidate_ids)
        if selected_mask & (1 << index)
    )


def selected_mask_from_ids(
    space: FeasibleSubsetSpace,
    selected_ids: Sequence[str],
) -> int:
    """验证并把 checkpoint IDs 转成稳定候选位图。"""
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected checkpoint IDs 不能重复")
    index_by_id = {
        checkpoint_id: index
        for index, checkpoint_id in enumerate(space.candidate_ids)
    }
    selected_mask = 0
    for checkpoint_id in selected_ids:
        if checkpoint_id not in index_by_id:
            raise ValueError(f"checkpoint 不属于可行候选：{checkpoint_id}")
        selected_mask |= 1 << index_by_id[checkpoint_id]
    if bin(selected_mask).count("1") > space.capacity:
        raise ValueError("selected set 超过冻结预算")
    return selected_mask


def selection_gap_histogram(
    scenario: Scenario,
    selected_ids: Sequence[str],
) -> dict[int, int]:
    """直接通过核心 recovery_gap 计算一个选择的 histogram。"""
    candidate_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    try:
        selected = tuple(candidate_by_id[item] for item in selected_ids)
    except KeyError as error:
        raise ValueError(f"未知 checkpoint ID：{error}") from error
    histogram: dict[int, int] = {}
    for continuation in scenario.continuations:
        gap = recovery_gap(continuation, selected)
        histogram[gap] = histogram.get(gap, 0) + 1
    return histogram


def objective_from_histogram(
    histogram: Mapping[int, int],
    model: CostModel,
) -> float:
    """只用 gap histogram 与成本模型计算统一 objective。"""
    return sum(
        count * model.estimate(gap)
        for gap, count in histogram.items()
    )


def gap_risk_from_histogram(histogram: Mapping[int, int]) -> GapRisk:
    """计算 histogram 的总量、均值、最大值与经验 P95。"""
    if any(gap < 0 or count < 0 for gap, count in histogram.items()):
        raise ValueError("gap 与 histogram count 必须大于等于零")
    count_total = sum(histogram.values())
    if count_total <= 0:
        return GapRisk(0, 0.0, 0, 0)
    total_gap = sum(gap * count for gap, count in histogram.items())
    rank = math.ceil(0.95 * count_total)
    cumulative = 0
    p95 = 0
    for gap in sorted(histogram):
        cumulative += histogram[gap]
        if cumulative >= rank:
            p95 = gap
            break
    return GapRisk(
        total_gap_tokens=total_gap,
        mean_gap_tokens=total_gap / count_total,
        max_gap_tokens=max(
            gap for gap, count in histogram.items() if count > 0
        ),
        p95_gap_tokens=p95,
    )


def analyze_point_objective_regret(
    point: RepresentativePoint,
    uncertainty_models: Sequence[UncertaintyModel],
) -> PointRobustness:
    """精确枚举一次可行集合，并同时评估全部成本模型。"""
    scenario = build_point_scenario(point)
    fingerprint_before = scenario_fingerprint(scenario)
    space = build_feasible_subset_space(scenario)
    formal_ids = tuple(
        sorted(select_flowstate(scenario, uncertainty_models[0].estimator))
    )
    formal_mask = selected_mask_from_ids(space, formal_ids)
    formal_histogram = selection_gap_histogram(scenario, formal_ids)
    formal_counts = tuple(
        formal_histogram.get(gap, 0) for gap in space.gap_values
    )
    costs_by_model = tuple(
        tuple(
            model.estimator.estimate(gap) for gap in space.gap_values
        )
        for model in uncertainty_models
    )
    best_costs = [math.inf] * len(uncertainty_models)
    best_masks: list[int | None] = [None] * len(uncertainty_models)
    best_gap_counts: list[tuple[int, ...] | None] = [
        None
    ] * len(uncertainty_models)
    histogram_cost_cache: dict[tuple[int, ...], tuple[float, ...]] = {}
    histogram_risk_cache: dict[tuple[int, ...], GapRisk] = {}
    feasible_count = 0

    def ids_for(mask: int) -> tuple[str, ...]:
        return selected_ids_from_mask(space, mask)

    for subset in space.iter_subsets():
        feasible_count += 1
        costs = histogram_cost_cache.get(subset.gap_counts)
        if costs is None:
            costs = tuple(
                sum(
                    count * gap_cost
                    for count, gap_cost in zip(
                        subset.gap_counts,
                        model_costs,
                    )
                )
                for model_costs in costs_by_model
            )
            histogram_cost_cache[subset.gap_counts] = costs
        for index, cost in enumerate(costs):
            current_mask = best_masks[index]
            if cost < best_costs[index] - _FLOAT_TOLERANCE_MS:
                best_costs[index] = cost
                best_masks[index] = subset.selected_mask
                best_gap_counts[index] = subset.gap_counts
            elif (
                abs(cost - best_costs[index]) <= _FLOAT_TOLERANCE_MS
                and current_mask is not None
                and ids_for(subset.selected_mask) < ids_for(current_mask)
            ):
                best_masks[index] = subset.selected_mask
                best_gap_counts[index] = subset.gap_counts

    expected_count = sum(
        math.comb(len(space.candidate_ids), size)
        for size in range(space.capacity + 1)
    )
    if feasible_count != expected_count:
        raise RuntimeError(
            "可行集合枚举数量异常："
            f"{feasible_count} != {expected_count}"
        )
    formal_costs = tuple(
        sum(
            count * gap_cost
            for count, gap_cost in zip(formal_counts, model_costs)
        )
        for model_costs in costs_by_model
    )
    formal_risk = gap_risk_from_histogram(formal_histogram)
    model_results = []
    for index, uncertainty in enumerate(uncertainty_models):
        best_mask = best_masks[index]
        optimal_counts = best_gap_counts[index]
        if best_mask is None or optimal_counts is None:
            raise RuntimeError("精确枚举没有产生最优选择")
        optimal_cost = best_costs[index]
        formal_cost = formal_costs[index]
        regret = formal_cost - optimal_cost
        if regret < -_FLOAT_TOLERANCE_MS:
            raise RuntimeError("exact optimum 不得劣于正式选择")
        regret = max(0.0, regret)
        optimal_risk = histogram_risk_cache.get(optimal_counts)
        if optimal_risk is None:
            optimal_histogram = {
                gap: count
                for gap, count in zip(space.gap_values, optimal_counts)
                if count
            }
            optimal_risk = gap_risk_from_histogram(optimal_histogram)
            histogram_risk_cache[optimal_counts] = optimal_risk
        relative = None if optimal_cost == 0.0 else regret / optimal_cost
        model_results.append(
            ModelRegret(
                model_name=uncertainty.model_name,
                model_kind=uncertainty.model_kind,
                perturbed_gap=uncertainty.perturbed_gap,
                perturbation_fraction=uncertainty.perturbation_fraction,
                formal_checkpoint_ids=formal_ids,
                exact_best_checkpoint_ids=ids_for(best_mask),
                exact_selection_changed=(best_mask != formal_mask),
                formal_cost_ms=formal_cost,
                optimal_cost_ms=optimal_cost,
                absolute_regret_ms=regret,
                relative_regret=relative,
                optimal_cost_is_zero=(optimal_cost == 0.0),
                formal_gap_risk=formal_risk,
                optimal_gap_risk=optimal_risk,
            )
        )

    old_phi = model_results[0]
    if old_phi.absolute_regret_ms > _FLOAT_TOLERANCE_MS:
        raise RuntimeError("Old Phi 下正式 FlowState selection regret 不为零")
    relative_values = tuple(
        row.relative_regret
        for row in model_results
        if row.relative_regret is not None
    )
    profiler_row = next(
        row for row in model_results if row.model_kind == "profiler_v2"
    )
    worst_row = max(
        model_results,
        key=lambda row: (
            -1.0 if row.relative_regret is None else row.relative_regret
        ),
    )
    if fingerprint_before != scenario_fingerprint(scenario):
        raise RuntimeError("objective robustness 分析修改了冻结 workload")
    return PointRobustness(
        point=point,
        candidate_count=len(scenario.candidates),
        continuation_count=len(scenario.continuations),
        capacity=space.capacity,
        feasible_subset_count=feasible_count,
        distinct_histogram_count=len(histogram_cost_cache),
        uncertainty_model_count=len(uncertainty_models),
        formal_checkpoint_ids=formal_ids,
        model_results=tuple(model_results),
        exact_selection_change_count=sum(
            row.exact_selection_changed for row in model_results
        ),
        mean_relative_regret=statistics.fmean(relative_values),
        median_relative_regret=statistics.median(relative_values),
        max_relative_regret=max(relative_values),
        profiler_v2_relative_regret=(
            0.0
            if profiler_row.relative_regret is None
            else profiler_row.relative_regret
        ),
        worst_case_model_name=worst_row.model_name,
    )


def run_objective_robustness_analysis() -> ObjectiveRobustnessResult:
    """运行四个冻结点，且不写回正式模型或策略。"""
    formal_model = RecoveryCostModel()
    formal_before = tuple(
        formal_model.estimate(gap) for gap in (0,) + LOCAL_KNOT_GAPS
    )
    models = build_uncertainty_models(
        formal_model,
        load_profiler_v2_model(),
    )
    points = tuple(
        analyze_point_objective_regret(point, models)
        for point in REPRESENTATIVE_POINTS
    )
    formal_after = tuple(
        formal_model.estimate(gap) for gap in (0,) + LOCAL_KNOT_GAPS
    )
    if formal_before != formal_after:
        raise RuntimeError("objective robustness 分析修改了正式 Phi")

    signal_scenario = build_point_scenario(REPRESENTATIVE_POINTS[-1])
    signal_point = points[-1]
    profiler_row = next(
        row
        for row in signal_point.model_results
        if row.model_kind == "profiler_v2"
    )
    marconi_ids = tuple(sorted(select_marconi(signal_scenario)))
    comparison = SotaK8ProfilerComparison(
        formal_checkpoint_ids=signal_point.formal_checkpoint_ids,
        profiler_v2_exact_best_checkpoint_ids=(
            profiler_row.exact_best_checkpoint_ids
        ),
        marconi_checkpoint_ids=marconi_ids,
        profiler_v2_best_equals_marconi=(
            set(profiler_row.exact_best_checkpoint_ids) == set(marconi_ids)
        ),
        formal_cost_under_v2_ms=profiler_row.formal_cost_ms,
        optimal_cost_under_v2_ms=profiler_row.optimal_cost_ms,
        absolute_regret_ms=profiler_row.absolute_regret_ms,
        relative_regret=profiler_row.relative_regret,
    )
    return ObjectiveRobustnessResult(
        schema_version="flowstate.objective_robustness.v1",
        uncertainty_model_count=len(models),
        points=points,
        sota_k8_profiler_v2=comparison,
        exact_selection_instability_present=any(
            point.exact_selection_change_count > 0 for point in points
        ),
        objective_regret_observed=any(
            point.max_relative_regret > 0.0 for point in points
        ),
        data_isolation={
            "step9b_data_used_for_optimization": False,
            "formal_phi_changed": False,
            "policy_changed": False,
            "workload_changed": False,
            "core_changed": False,
            "motivation_changed": False,
            "enumeration": "每个场景与预算只枚举一次全部可行集合",
            "objective_source": "预计算 recovery-gap histogram",
        },
    )


def write_artifacts(result: ObjectiveRobustnessResult) -> None:
    """写出结构化 JSON 与中文审计说明。"""
    _OUTPUT_JSON_PATH.write_text(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _OUTPUT_MARKDOWN_PATH.write_text(
        build_markdown(result),
        encoding="utf-8",
    )


def build_markdown(result: ObjectiveRobustnessResult) -> str:
    """把 exact regret 结果渲染为可复核的中文报告。"""
    lines = [
        "# FlowState objective robustness 审计",
        "",
        "## 结论",
        "",
        "本报告区分 checkpoint ID 是否改变的 selection instability，与正式选择在真实成本模型下产生多少额外目标成本的 objective instability。选择改变但 regret 很小，表示多个集合在目标上近似等价，不能据此断言系统性能不稳。",
        "",
        "| 代表点 | 模型数 | Exact selection changes | Mean regret | Median regret | Max regret | Profiler-v2 regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for point in result.points:
        lines.append(
            f"| {point.point.point_name} | {point.uncertainty_model_count} | "
            f"{point.exact_selection_change_count} | "
            f"{point.mean_relative_regret:.6%} | "
            f"{point.median_relative_regret:.6%} | "
            f"{point.max_relative_regret:.6%} | "
            f"{point.profiler_v2_relative_regret:.6%} |"
        )
    lines.extend(
        [
            "",
            "## 固定分析方法",
            "",
            "- 不确定性集合固定为 Old Phi、Step 9D 独立 Profiler-v2，以及四个 gap knots 各自的 ±5%/±10% 局部扰动，共十八个模型。",
            "- 不包含 Step 9E 的整体比例缩放，因为整体正比例不会改变 exact objective 排名，本步骤按用户冻结定义只保留局部 knot 扰动。",
            "- 每个场景与预算只枚举一次全部可行 checkpoint 子集；每个子集先计算 recovery-gap histogram，再同时评估十八个模型。",
            "- Exact best 使用目标值优先、checkpoint IDs 字典序次优先的确定性规则，不使用 greedy approximation。",
            "- Step 9B measured TTFT 不参与成本模型、exact best 或 regret 计算。",
            "",
            "## 全部模型结果",
            "",
            "| 代表点 | 模型 | Formal cost（ms） | Optimal cost（ms） | Absolute regret（ms） | Relative regret | Selection changed | Formal gap total/mean/max/P95 | Optimal gap total/mean/max/P95 | Exact best |",
            "|---|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for point in result.points:
        for row in point.model_results:
            relative = (
                "N/A"
                if row.relative_regret is None
                else f"{row.relative_regret:.6%}"
            )
            formal_risk = row.formal_gap_risk
            optimal_risk = row.optimal_gap_risk
            lines.append(
                f"| {point.point.point_name} | {row.model_name} | "
                f"{row.formal_cost_ms:.6f} | {row.optimal_cost_ms:.6f} | "
                f"{row.absolute_regret_ms:.6f} | {relative} | "
                f"{row.exact_selection_changed} | "
                f"{formal_risk.total_gap_tokens}/{formal_risk.mean_gap_tokens:.3f}/{formal_risk.max_gap_tokens}/{formal_risk.p95_gap_tokens} | "
                f"{optimal_risk.total_gap_tokens}/{optimal_risk.mean_gap_tokens:.3f}/{optimal_risk.max_gap_tokens}/{optimal_risk.p95_gap_tokens} | "
                f"{';'.join(row.exact_best_checkpoint_ids)} |"
            )
    signal = result.sota_k8_profiler_v2
    relative = (
        "N/A"
        if signal.relative_regret is None
        else f"{signal.relative_regret:.6%}"
    )
    lines.extend(
        [
            "",
            "## SOTA-signal K8 Profiler-v2",
            "",
            f"- Old-Phi formal selection：{';'.join(signal.formal_checkpoint_ids)}。",
            f"- Profiler-v2 exact best：{';'.join(signal.profiler_v2_exact_best_checkpoint_ids)}。",
            f"- Marconi selection：{';'.join(signal.marconi_checkpoint_ids)}。",
            f"- Profiler-v2 exact best 与 Marconi 相同：{signal.profiler_v2_best_equals_marconi}。",
            f"- Formal/optimal v2 cost：{signal.formal_cost_under_v2_ms:.6f} / {signal.optimal_cost_under_v2_ms:.6f} ms。",
            f"- Absolute/relative regret：{signal.absolute_regret_ms:.6f} ms / {relative}。",
            "",
            "## 数据隔离",
            "",
            "- Step 9B 评估数据用于优化：False。",
            "- 正式 Phi、policy、workload、metadata、flowstate 核心实现与 motivation 均未修改。",
            "- 本结果是预定义 uncertainty models 下的离线反事实审计，不是 post-hoc tuning。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """执行精确离线分析并保存 artifacts。"""
    result = run_objective_robustness_analysis()
    write_artifacts(result)
    print(
        json.dumps(
            {
                "points": len(result.points),
                "uncertainty_models": result.uncertainty_model_count,
                "output_json": str(_OUTPUT_JSON_PATH),
                "output_markdown": str(_OUTPUT_MARKDOWN_PATH),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
