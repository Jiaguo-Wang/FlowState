#!/usr/bin/env python3
"""分析四个冻结代表点对恢复成本估计误差的选择敏感性。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from evaluation.recovery_profiler_v2.analyze import (
    CALIBRATION_GAPS,
    MonotonePiecewiseRecoveryModel,
)
from evaluation.scalable_multiworkflow_v2.scenario import (
    ScalableScenario,
    build_scenario as build_scalable_scenario,
)
from evaluation.sota_metadata import (
    build_marconi_flop_saved,
    build_marconi_recency,
)
from evaluation.sota_policies import MarconiStylePolicy
from evaluation.sota_signal_stress_v1.scenario import (
    SignalScenario,
    build_scenario as build_signal_scenario,
)
from flowstate.executable_state import recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PROFILER_V2_MODEL_PATH = (
    _REPOSITORY_ROOT
    / "evaluation"
    / "recovery_profiler_v2"
    / "model_comparison.json"
)
_OUTPUT_JSON_PATH = (
    _REPOSITORY_ROOT / "evaluation" / "cost_model_sensitivity.json"
)
_OUTPUT_MARKDOWN_PATH = (
    _REPOSITORY_ROOT / "evaluation" / "COST_MODEL_SENSITIVITY.md"
)
_FLOAT_TOLERANCE_MS = 1e-9

GLOBAL_SCALE_FACTORS = (0.90, 0.95, 1.05, 1.10)
LOCAL_KNOT_GAPS = (4_096, 8_192, 16_384, 32_768)
LOCAL_SCALE_FACTORS = GLOBAL_SCALE_FACTORS
EXPECTED_PERTURBATION_COUNT = 21


class CostModel(Protocol):
    """描述敏感性分析所需的最小恢复成本接口。"""

    def estimate(self, replay_tokens: int) -> float:
        """返回指定恢复间隔的估计成本。"""


@dataclass(frozen=True)
class RepresentativePoint:
    """标识一个冻结场景与预算点。"""

    point_name: str
    scenario_name: str
    budget_checkpoints: int


@dataclass(frozen=True)
class SelectionEvaluation:
    """记录一个模型下的 FlowState 分配与统一目标。"""

    model_name: str
    selected_checkpoint_ids: tuple[str, ...]
    objective_ms: float
    total_gap_tokens: int
    same_as_formal_selection: bool


@dataclass(frozen=True)
class RankingMargin:
    """记录正式选择到最佳替代集合的目标距离。"""

    best_checkpoint_ids: tuple[str, ...]
    best_objective_ms: float
    second_best_checkpoint_ids: tuple[str, ...]
    second_best_objective_ms: float
    margin_ms: float
    relative_margin: float | None


@dataclass(frozen=True)
class PointSensitivity:
    """汇总一个代表点的全部固定扰动结果。"""

    point: RepresentativePoint
    candidate_count: int
    continuation_count: int
    budget_bytes: int
    formal: SelectionEvaluation
    ranking_margin: RankingMargin
    perturbations: tuple[SelectionEvaluation, ...]
    selection_change_count: int
    selection_stability_rate: float
    classification: str
    local_change_count_by_gap: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SotaK8Counterfactual:
    """记录 SOTA-signal K8 三个选择在 v2 模型下的成本。"""

    old_phi_checkpoint_ids: tuple[str, ...]
    old_phi_objective_under_v2_ms: float
    profiler_v2_checkpoint_ids: tuple[str, ...]
    profiler_v2_objective_ms: float
    marconi_checkpoint_ids: tuple[str, ...]
    marconi_objective_under_v2_ms: float


@dataclass(frozen=True)
class SensitivityAnalysisResult:
    """保存全部代表点与关键反事实比较。"""

    schema_version: str
    perturbation_count_per_point: int
    points: tuple[PointSensitivity, ...]
    sota_k8_counterfactual: SotaK8Counterfactual
    most_sensitive_gaps: tuple[int, ...]
    flowstate_generally_robust: bool
    near_tie_sensitivity_present: bool
    data_isolation: Mapping[str, object]


@dataclass(frozen=True)
class ScaledCostModel:
    """对正式 Phi 的全部非零成本执行固定比例缩放。"""

    base_model: CostModel
    scale: float

    def estimate(self, replay_tokens: int) -> float:
        """保持零点不变，并缩放全部非零恢复成本。"""
        value = self.base_model.estimate(replay_tokens)
        return 0.0 if replay_tokens == 0 else value * self.scale


@dataclass(frozen=True)
class ExactGapPerturbedCostModel:
    """只扰动一个指定恢复间隔的正式 Phi 成本。"""

    base_model: CostModel
    target_gap: int
    scale: float

    def estimate(self, replay_tokens: int) -> float:
        """只在 replay_tokens 精确命中目标 knot 时应用比例。"""
        value = self.base_model.estimate(replay_tokens)
        if replay_tokens == self.target_gap:
            return value * self.scale
        return value


REPRESENTATIVE_POINTS = (
    RepresentativePoint("Scalable N16 K4", "scalable_n16", 4),
    RepresentativePoint("Scalable N16 K12", "scalable_n16", 12),
    RepresentativePoint("SOTA-signal K4", "sota_signal", 4),
    RepresentativePoint("SOTA-signal K8", "sota_signal", 8),
)


def build_point_scenario(
    point: RepresentativePoint,
) -> ScalableScenario | SignalScenario:
    """通过冻结 builder 构造代表点，不复制 workload 定义。"""
    if point.scenario_name == "scalable_n16":
        return build_scalable_scenario(16, point.budget_checkpoints)
    if point.scenario_name == "sota_signal":
        return build_signal_scenario(point.budget_checkpoints)
    raise ValueError(f"未知代表场景：{point.scenario_name}")


def load_profiler_v2_model(
    path: Path = _PROFILER_V2_MODEL_PATH,
) -> MonotonePiecewiseRecoveryModel:
    """只从 Step 9D 独立 calibration artifact 读取 v2 knots。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("step9b_data_used_for_fitting") is not False:
        raise ValueError("Profiler v2 artifact 未证明与 Step 9B 拟合隔离")
    fit_split = payload.get("fit_split")
    if not isinstance(fit_split, dict):
        raise ValueError("Profiler v2 artifact 缺少数据切分说明")
    if fit_split.get("validation_used_for_fitting") is not False:
        raise ValueError("Profiler v2 held-out validation 不能参与拟合")
    if tuple(fit_split.get("calibration_gaps", ())) != CALIBRATION_GAPS:
        raise ValueError("Profiler v2 calibration gaps 与冻结定义不一致")

    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("Profiler v2 artifact 缺少模型参数")
    piecewise = models.get("Monotone piecewise v2")
    if not isinstance(piecewise, dict):
        raise ValueError("Profiler v2 artifact 缺少单调分段模型")
    raw_knots = piecewise.get("knots")
    if not isinstance(raw_knots, list):
        raise ValueError("Profiler v2 artifact 缺少 calibration knots")
    knots = tuple(
        (int(item[0]), float(item[1]))
        for item in raw_knots
    )
    model = MonotonePiecewiseRecoveryModel(knots=knots)
    if model.estimate(0) != 0.0:
        raise ValueError("Profiler v2 模型必须满足 Phi(0)=0")
    return model


def select_flowstate(
    scenario: ScalableScenario | SignalScenario,
    model: CostModel,
) -> tuple[str, ...]:
    """使用现有 GlobalOptimizer 计算一个反事实分配。"""
    allocation = GlobalOptimizer(model).select(
        scenario.continuations,
        scenario.candidates,
        scenario.budget_bytes,
    )
    return tuple(
        candidate.checkpoint_id for candidate in allocation.selected
    )


def evaluate_selection(
    scenario: ScalableScenario | SignalScenario,
    selected_ids: Sequence[str],
    model: CostModel,
    *,
    model_name: str,
    formal_selected_ids: Sequence[str],
) -> SelectionEvaluation:
    """使用核心 recovery_gap 计算选择的成本与总间隔。"""
    candidate_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected checkpoint IDs 不能重复")
    try:
        selected = tuple(candidate_by_id[item] for item in selected_ids)
    except KeyError as error:
        raise ValueError(f"selected ID 不属于冻结 candidate set：{error}") from error
    if any(not candidate.recurrent_resident for candidate in selected):
        raise ValueError("不能选择非 recurrent-resident candidate")
    if sum(candidate.memory_bytes for candidate in selected) > (
        scenario.budget_bytes
    ):
        raise ValueError("selected set 超过冻结 budget")
    gaps = tuple(
        recovery_gap(continuation, selected)
        for continuation in scenario.continuations
    )
    return SelectionEvaluation(
        model_name=model_name,
        selected_checkpoint_ids=tuple(selected_ids),
        objective_ms=sum(model.estimate(gap) for gap in gaps),
        total_gap_tokens=sum(gaps),
        same_as_formal_selection=(
            set(selected_ids) == set(formal_selected_ids)
        ),
    )


def find_ranking_margin(
    scenario: ScalableScenario | SignalScenario,
    formal_selected_ids: Sequence[str],
    model: CostModel,
) -> RankingMargin:
    """精确寻找正式选择之外目标最小的可行 checkpoint 集合。"""
    formal_ids = tuple(sorted(formal_selected_ids))
    formal = evaluate_selection(
        scenario,
        formal_ids,
        model,
        model_name="Old Phi",
        formal_selected_ids=formal_ids,
    )
    ranked = _two_best_feasible_selections(scenario, model)
    if not ranked:
        raise RuntimeError("没有可行的 checkpoint selection")
    best_cost = ranked[0][0]
    if formal.objective_ms > best_cost + _FLOAT_TOLERANCE_MS:
        raise RuntimeError(
            "正式 FlowState selection 不是当前 objective 的最优解："
            f"{formal.objective_ms} > {best_cost}"
        )
    alternate = next(
        (item for item in ranked if item[1] != formal_ids),
        None,
    )
    if alternate is None:
        raise RuntimeError("无法找到与正式选择不同的可行第二名")
    margin = alternate[0] - formal.objective_ms
    if margin < -_FLOAT_TOLERANCE_MS:
        raise RuntimeError("第二名目标不能优于正式最优选择")
    margin = max(0.0, margin)
    relative = (
        margin / formal.objective_ms
        if formal.objective_ms > 0.0
        else None
    )
    return RankingMargin(
        best_checkpoint_ids=tuple(formal_selected_ids),
        best_objective_ms=formal.objective_ms,
        second_best_checkpoint_ids=alternate[1],
        second_best_objective_ms=alternate[0],
        margin_ms=margin,
        relative_margin=relative,
    )


def analyze_point(
    point: RepresentativePoint,
    formal_model: CostModel,
    profiler_v2_model: CostModel,
) -> PointSensitivity:
    """执行一个点的正式选择、margin 和固定扰动分析。"""
    scenario = build_point_scenario(point)
    before_fingerprint = scenario_fingerprint(scenario)
    formal_ids = select_flowstate(scenario, formal_model)
    formal = evaluate_selection(
        scenario,
        formal_ids,
        formal_model,
        model_name="Old Phi",
        formal_selected_ids=formal_ids,
    )
    margin = find_ranking_margin(scenario, formal_ids, formal_model)

    perturbations = []
    profiler_ids = select_flowstate(scenario, profiler_v2_model)
    perturbations.append(
        evaluate_selection(
            scenario,
            profiler_ids,
            profiler_v2_model,
            model_name="Profiler-v2 piecewise",
            formal_selected_ids=formal_ids,
        )
    )
    for scale in GLOBAL_SCALE_FACTORS:
        model = ScaledCostModel(formal_model, scale)
        selected_ids = select_flowstate(scenario, model)
        perturbations.append(
            evaluate_selection(
                scenario,
                selected_ids,
                model,
                model_name=f"Old Phi global {scale - 1.0:+.0%}",
                formal_selected_ids=formal_ids,
            )
        )
    local_change_counts = []
    for gap in LOCAL_KNOT_GAPS:
        changed = 0
        for scale in LOCAL_SCALE_FACTORS:
            model = ExactGapPerturbedCostModel(formal_model, gap, scale)
            selected_ids = select_flowstate(scenario, model)
            result = evaluate_selection(
                scenario,
                selected_ids,
                model,
                model_name=f"Old Phi G={gap} {scale - 1.0:+.0%}",
                formal_selected_ids=formal_ids,
            )
            changed += not result.same_as_formal_selection
            perturbations.append(result)
        local_change_counts.append((gap, changed))
    if len(perturbations) != EXPECTED_PERTURBATION_COUNT:
        raise RuntimeError("预定义 perturbation 数量异常")

    change_count = sum(
        not result.same_as_formal_selection for result in perturbations
    )
    stability_rate = (len(perturbations) - change_count) / len(
        perturbations
    )
    if before_fingerprint != scenario_fingerprint(scenario):
        raise RuntimeError("敏感性分析修改了冻结 scenario")
    return PointSensitivity(
        point=point,
        candidate_count=len(scenario.candidates),
        continuation_count=len(scenario.continuations),
        budget_bytes=scenario.budget_bytes,
        formal=formal,
        ranking_margin=margin,
        perturbations=tuple(perturbations),
        selection_change_count=change_count,
        selection_stability_rate=stability_rate,
        classification=classify_stability(stability_rate),
        local_change_count_by_gap=tuple(local_change_counts),
    )


def run_sensitivity_analysis() -> SensitivityAnalysisResult:
    """运行四个冻结点并生成不写回正式模型的离线结果。"""
    formal_model = RecoveryCostModel()
    profiler_v2_model = load_profiler_v2_model()
    formal_before = tuple(
        formal_model.estimate(gap)
        for gap in (0,) + LOCAL_KNOT_GAPS
    )
    points = tuple(
        analyze_point(point, formal_model, profiler_v2_model)
        for point in REPRESENTATIVE_POINTS
    )
    if formal_before != tuple(
        formal_model.estimate(gap)
        for gap in (0,) + LOCAL_KNOT_GAPS
    ):
        raise RuntimeError("敏感性分析修改了正式 Phi")

    signal_point = REPRESENTATIVE_POINTS[-1]
    signal_scenario = build_point_scenario(signal_point)
    old_ids = points[-1].formal.selected_checkpoint_ids
    profiler_result = points[-1].perturbations[0]
    marconi_ids = select_marconi(signal_scenario)
    old_under_v2 = evaluate_selection(
        signal_scenario,
        old_ids,
        profiler_v2_model,
        model_name="Old-Phi selection under v2",
        formal_selected_ids=old_ids,
    )
    marconi_under_v2 = evaluate_selection(
        signal_scenario,
        marconi_ids,
        profiler_v2_model,
        model_name="Marconi selection under v2",
        formal_selected_ids=old_ids,
    )

    aggregate_local_changes = {
        gap: sum(
            dict(point.local_change_count_by_gap)[gap]
            for point in points
        )
        for gap in LOCAL_KNOT_GAPS
    }
    maximum_changes = max(aggregate_local_changes.values())
    most_sensitive = tuple(
        gap
        for gap in LOCAL_KNOT_GAPS
        if aggregate_local_changes[gap] == maximum_changes
    )
    return SensitivityAnalysisResult(
        schema_version="flowstate.cost_model_sensitivity.v1",
        perturbation_count_per_point=EXPECTED_PERTURBATION_COUNT,
        points=points,
        sota_k8_counterfactual=SotaK8Counterfactual(
            old_phi_checkpoint_ids=old_ids,
            old_phi_objective_under_v2_ms=old_under_v2.objective_ms,
            profiler_v2_checkpoint_ids=(
                profiler_result.selected_checkpoint_ids
            ),
            profiler_v2_objective_ms=profiler_result.objective_ms,
            marconi_checkpoint_ids=marconi_ids,
            marconi_objective_under_v2_ms=marconi_under_v2.objective_ms,
        ),
        most_sensitive_gaps=most_sensitive,
        flowstate_generally_robust=all(
            point.selection_stability_rate >= 0.90 for point in points
        ),
        near_tie_sensitivity_present=any(
            point.ranking_margin.relative_margin is not None
            and point.ranking_margin.relative_margin < 0.01
            for point in points
        ),
        data_isolation={
            "step9b_data_used_for_optimization": False,
            "profiler_v2_source": str(_PROFILER_V2_MODEL_PATH),
            "profiler_v2_is_independent_step9d_data": True,
            "formal_phi_changed": False,
            "policy_changed": False,
            "workload_changed": False,
        },
    )


def select_marconi(
    scenario: ScalableScenario | SignalScenario,
) -> tuple[str, ...]:
    """用冻结 metadata 计算 Marconi-style 选择。"""
    recency = build_marconi_recency(
        scenario.candidates,
        scenario.metadata.checkpoint_recency,
    )
    alpha = float(getattr(scenario.metadata, "marconi_alpha", 1.0))
    selection = MarconiStylePolicy().select(
        scenario.candidates,
        scenario.metadata.budget_checkpoints,
        recency,
        build_marconi_flop_saved(scenario.candidates),
        alpha,
    )
    return selection.selected_checkpoint_ids


def classify_stability(rate: float) -> str:
    """按冻结阈值把 selection stability 分成三档。"""
    if not 0.0 <= rate <= 1.0:
        raise ValueError("stability rate 必须位于 [0, 1]")
    if rate >= 0.90:
        return "Stable"
    if rate >= 0.50:
        return "Moderately sensitive"
    return "Sensitive"


def scenario_fingerprint(
    scenario: ScalableScenario | SignalScenario,
) -> tuple[object, ...]:
    """生成用于证明 workload 与 budget 未改变的完整逻辑指纹。"""
    return (
        scenario.budget_bytes,
        tuple(
            (
                continuation.continuation_id,
                continuation.workflow_id,
                continuation.lineage_path,
                continuation.anchor_pos,
                continuation.resident_fa_frontier,
            )
            for continuation in scenario.continuations
        ),
        tuple(
            (
                candidate.checkpoint_id,
                candidate.workflow_id,
                candidate.lineage_path,
                candidate.token_pos,
                candidate.memory_bytes,
                candidate.recurrent_resident,
                candidate.fa_resident,
            )
            for candidate in scenario.candidates
        ),
    )


def _two_best_feasible_selections(
    scenario: ScalableScenario | SignalScenario,
    model: CostModel,
) -> tuple[tuple[float, tuple[str, ...]], ...]:
    """按 workflow 可加性精确求全部可行集合中的前两名。"""
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
        cost = sum(
            model.estimate(recovery_gap(continuation, ()))
            for continuation in scenario.continuations
        )
        return ((cost, ()),)
    checkpoint_size = eligible[0].memory_bytes
    if any(
        candidate.memory_bytes != checkpoint_size for candidate in eligible
    ):
        raise ValueError("敏感性枚举只支持冻结的等大小 checkpoints")
    capacity = min(scenario.budget_bytes // checkpoint_size, len(eligible))
    candidates_by_workflow: dict[str, list[CheckpointCandidate]] = {}
    for candidate in eligible:
        candidates_by_workflow.setdefault(candidate.workflow_id, []).append(
            candidate
        )
    continuations_by_workflow: dict[str, list[PendingContinuation]] = {}
    for continuation in scenario.continuations:
        continuations_by_workflow.setdefault(
            continuation.workflow_id, []
        ).append(continuation)
    workflow_ids = tuple(
        sorted(set(candidates_by_workflow) | set(continuations_by_workflow))
    )

    states: dict[int, tuple[tuple[float, tuple[str, ...]], ...]] = {
        0: ((0.0, ()),)
    }
    for workflow_id in workflow_ids:
        local_candidates = tuple(candidates_by_workflow.get(workflow_id, ()))
        local_continuations = tuple(
            continuations_by_workflow.get(workflow_id, ())
        )
        local_options = []
        for subset_size in range(len(local_candidates) + 1):
            for subset in combinations(local_candidates, subset_size):
                selected_ids = tuple(
                    candidate.checkpoint_id for candidate in subset
                )
                cost = sum(
                    model.estimate(recovery_gap(continuation, subset))
                    for continuation in local_continuations
                )
                local_options.append((subset_size, cost, selected_ids))

        next_states: dict[int, list[tuple[float, tuple[str, ...]]]] = {}
        for prior_size, prior_rows in states.items():
            for prior_cost, prior_ids in prior_rows:
                for local_size, local_cost, local_ids in local_options:
                    total_size = prior_size + local_size
                    if total_size > capacity:
                        continue
                    next_states.setdefault(total_size, []).append(
                        (
                            prior_cost + local_cost,
                            tuple(sorted(prior_ids + local_ids)),
                        )
                    )
        states = {
            size: _rank_unique(rows, limit=2)
            for size, rows in next_states.items()
        }

    feasible = tuple(
        row for rows in states.values() for row in rows
    )
    return _rank_unique(feasible, limit=2)


def _rank_unique(
    rows: Sequence[tuple[float, tuple[str, ...]]],
    *,
    limit: int,
) -> tuple[tuple[float, tuple[str, ...]], ...]:
    """按目标和字典序排序、去重并保留固定数量。"""
    by_ids: dict[tuple[str, ...], float] = {}
    for cost, selected_ids in rows:
        previous = by_ids.get(selected_ids)
        if previous is None or cost < previous:
            by_ids[selected_ids] = cost
    ordered = sorted(
        ((cost, selected_ids) for selected_ids, cost in by_ids.items()),
        key=lambda item: (item[0], item[1]),
    )
    return tuple(ordered[:limit])


def write_artifacts(result: SensitivityAnalysisResult) -> None:
    """写出结构化结果与中文审计报告。"""
    with _OUTPUT_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            asdict(result),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    _OUTPUT_MARKDOWN_PATH.write_text(
        _build_markdown(result),
        encoding="utf-8",
    )


def _build_markdown(result: SensitivityAnalysisResult) -> str:
    """把完整敏感性结果渲染成可审阅的中文 Markdown。"""
    lines = [
        "# Recovery cost model 敏感性分析",
        "",
        "## 结论",
        "",
        "本分析仅使用冻结 workload、正式 Old Phi 与 Step 9D 独立 Profiler v2。Step 9B latency samples 不参与模型构造、选择或目标计算。所有结论均为离线反事实，不写回正式 FlowState。",
        "",
        "| 代表点 | 正式目标（ms） | Margin（ms） | Relative margin | 稳定率 | 分类 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for point in result.points:
        relative = point.ranking_margin.relative_margin
        relative_text = "N/A" if relative is None else f"{relative:.6%}"
        lines.append(
            f"| {point.point.point_name} | {point.formal.objective_ms:.6f} | "
            f"{point.ranking_margin.margin_ms:.6f} | {relative_text} | "
            f"{point.selection_stability_rate:.2%} | {point.classification} |"
        )
    lines.extend(
        [
            "",
            "## 固定方法",
            "",
            "- 反事实集合固定为 Profiler-v2、四个整体比例扰动和十六个单 knot 扰动，共二十一个模型。",
            "- Selection 一致性按 checkpoint ID 集合判断，optimizer 的选择顺序另行保留。",
            "- Ranking margin 使用所有满足预算的 recurrent-resident checkpoint 集合做精确比较；存在等目标替代集合时 margin 为零。",
            "- Profiler-v2 只读取 Step 9D 的 monotone piecewise calibration knots。",
            "",
            "## 全部扰动结果",
            "",
            "| 代表点 | 模型 | Selected | Objective（ms） | Total gap | 与正式选择相同 |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for point in result.points:
        for row in point.perturbations:
            lines.append(
                f"| {point.point.point_name} | {row.model_name} | "
                f"{';'.join(row.selected_checkpoint_ids)} | "
                f"{row.objective_ms:.6f} | {row.total_gap_tokens} | "
                f"{row.same_as_formal_selection} |"
            )
    counterfactual = result.sota_k8_counterfactual
    lines.extend(
        [
            "",
            "## SOTA-signal K8 Profiler-v2 反事实",
            "",
            "| 选择来源 | Selected | v2 objective（ms） |",
            "|---|---|---:|",
            f"| Old Phi | {';'.join(counterfactual.old_phi_checkpoint_ids)} | {counterfactual.old_phi_objective_under_v2_ms:.6f} |",
            f"| Profiler-v2 | {';'.join(counterfactual.profiler_v2_checkpoint_ids)} | {counterfactual.profiler_v2_objective_ms:.6f} |",
            f"| Marconi-style | {';'.join(counterfactual.marconi_checkpoint_ids)} | {counterfactual.marconi_objective_under_v2_ms:.6f} |",
            "",
            "## 数据隔离",
            "",
            "- Step 9B evaluation data used for optimization：False。",
            "- 正式 Phi、policy、workload、flowstate 核心实现均未修改。",
            "- 本报告不得解释为对 Step 9B 的 post-hoc 调参。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """执行离线分析并保存可复核 artifacts。"""
    result = run_sensitivity_analysis()
    write_artifacts(result)
    print(
        json.dumps(
            {
                "points": len(result.points),
                "perturbations_per_point": (
                    result.perturbation_count_per_point
                ),
                "flowstate_generally_robust": (
                    result.flowstate_generally_robust
                ),
                "near_tie_sensitivity_present": (
                    result.near_tie_sensitivity_present
                ),
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
