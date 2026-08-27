#!/usr/bin/env python3
"""预注册恢复模型选择与独立留出验证。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback
from typing import Callable, Mapping, Sequence

from evaluation.recovery_position_audit import (
    ARTIFACT_ROOT,
    MATRIX_SUFFIX_SEED,
    ORDER_SEED,
    RAW_FIELDS,
    REPOSITORY_ROOT,
    _collect_environment,
    _csv_value,
    _environment_text,
    _protected_hashes,
    _write_csv,
)
from evaluation.recovery_profiler_128k import (
    ENGINE_CONFIGURATION_128K,
    FORMAL_MUTATION_PRIMITIVE,
    MODEL_PATH,
    ProfileCase,
    _load_context_configuration,
    execute_profile_case,
    validate_context_capabilities,
)
from evaluation.recovery_profiler_v2.analyze import (
    MEASURED_REPETITIONS,
    WARMUP_REPETITIONS,
)


STEP10D1_DIRECTORY = (
    ARTIFACT_ROOT / "recovery_profiler_128k_20260826_133712_596813"
)
STEP10D2_DIRECTORY = (
    ARTIFACT_ROOT / "recovery_position_audit_20260826_144654_852303"
)
CALIBRATION_POSITION_TARGETS = (32_768, 65_536, 98_304, 131_072)
CALIBRATION_POSITION_GAPS = (4_096, 8_192, 16_384, 32_768)
CALIBRATION_LONG_GAPS = (49_152, 65_536, 98_304, 131_072)
HELDOUT_TARGETS = (49_152, 81_920, 114_688)
HELDOUT_GAPS_BY_TARGET = {
    49_152: (0, 6_144, 12_288, 24_576, 36_864),
    81_920: (0, 10_240, 20_480, 40_960, 61_440),
    114_688: (0, 14_336, 28_672, 57_344, 86_016),
}
HELDOUT_FRACTIONS = (0.125, 0.25, 0.5, 0.75)
STRUCTURAL_TARGETS = (32_768, 49_152, 65_536, 81_920, 98_304, 114_688, 131_072)
STRUCTURAL_GAP_STEP = 4_096
MODEL_ORDER = ("M0", "M1", "M2")
EXPECTED_CALIBRATION_POINTS = 20
EXPECTED_HELDOUT_CONFIGURATIONS = 15
EXPECTED_HELDOUT_TRIALS = EXPECTED_HELDOUT_CONFIGURATIONS * (
    WARMUP_REPETITIONS + MEASURED_REPETITIONS
)
HELDOUT_SOURCE_LABEL = "Step 10D.3 独立留出测量"


@dataclass(frozen=True)
class CalibrationPoint:
    """描述一个冻结的非零恢复成本校准点。"""

    source: str
    subset: str
    target_tokens: int
    gap_tokens: int
    measured_phi_ms: float

    @property
    def target_ki(self) -> float:
        """返回以 Ki token 表示的目标位置。"""
        return self.target_tokens / 1024.0

    @property
    def gap_ki(self) -> float:
        """返回以 Ki token 表示的恢复缺口。"""
        return self.gap_tokens / 1024.0


@dataclass(frozen=True)
class HeldoutCase:
    """描述一个冻结的独立留出 trial。"""

    case_id: str
    target_position: int
    target_frontier: int
    target_gap: int
    recovery_fraction: float
    repetition: int
    is_warmup: bool
    pair_order_position: int
    execution_order_position: int


@dataclass
class FreezeArtifactWriter:
    """增量保存模型选择与独立验证证据。"""

    directory: Path

    @classmethod
    def create(
        cls,
        root: Path = ARTIFACT_ROOT,
        timestamp: str | None = None,
    ) -> "FreezeArtifactWriter":
        """创建不会覆盖已有证据的时间戳目录。"""
        resolved = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = root / f"recovery_model_freeze_{resolved}"
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory)

    def append_raw(self, record: Mapping[str, object]) -> None:
        """在每个 trial 后立即追加原始记录。"""
        path = self.directory / "heldout_raw.csv"
        is_new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(
                {field: _csv_value(record.get(field)) for field in RAW_FIELDS}
            )

    def write_json(self, name: str, payload: object) -> None:
        """稳定写出 JSON。"""
        (self.directory / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def write_text(self, name: str, text: str) -> None:
        """写出 UTF-8 文本。"""
        (self.directory / name).write_text(text, encoding="utf-8")

    def ensure_required_files(self) -> None:
        """确保失败运行也保留全部约定文件。"""
        for name in (
            "calibration_points.csv",
            "calibration_diagnostics.csv",
            "heldout_raw.csv",
            "heldout_summary.csv",
            "heldout_predictions.csv",
        ):
            path = self.directory / name
            if not path.exists():
                path.write_text("\n", encoding="utf-8")
        for name in (
            "candidate_models.json",
            "model_selection.json",
            "structural_validation.json",
            "execution_order.json",
            "config.json",
        ):
            path = self.directory / name
            if not path.exists():
                self.write_json(name, {"status": "运行未完成"})
        for name in (
            "README.md",
            "theory_compatibility.md",
            "environment.txt",
            "server_command.txt",
        ):
            path = self.directory / name
            if not path.exists():
                path.write_text("运行未完成。\n", encoding="utf-8")


def load_calibration_points(
    position_directory: Path = STEP10D2_DIRECTORY,
    long_gap_directory: Path = STEP10D1_DIRECTORY,
) -> tuple[CalibrationPoint, ...]:
    """只从两个冻结 calibration artifact 读取 20 个非零点。"""
    points: list[CalibrationPoint] = []
    position_path = position_directory / "position_matrix_summary.csv"
    with position_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target = int(row["target_position"])
            gap = int(row["gap_tokens"])
            if (
                target in CALIBRATION_POSITION_TARGETS
                and gap in CALIBRATION_POSITION_GAPS
            ):
                points.append(
                    CalibrationPoint(
                        source=str(position_path.relative_to(REPOSITORY_ROOT)),
                        subset="position_matrix",
                        target_tokens=target,
                        gap_tokens=gap,
                        measured_phi_ms=float(row["measured_phi_ms"]),
                    )
                )
    long_path = long_gap_directory / "summary.csv"
    with long_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gap = int(row["gap_tokens"])
            if gap in CALIBRATION_LONG_GAPS:
                points.append(
                    CalibrationPoint(
                        source=str(long_path.relative_to(REPOSITORY_ROOT)),
                        subset="long_gap",
                        target_tokens=131_072,
                        gap_tokens=gap,
                        measured_phi_ms=float(row["measured_phi_ms"]),
                    )
                )
    result = tuple(
        sorted(points, key=lambda point: (point.target_tokens, point.gap_tokens))
    )
    validate_calibration_points(result)
    return result


def validate_calibration_points(points: Sequence[CalibrationPoint]) -> None:
    """验证 calibration 集合严格等于预注册的 20 个点。"""
    expected = {
        (target, gap)
        for target in CALIBRATION_POSITION_TARGETS
        for gap in CALIBRATION_POSITION_GAPS
    }
    expected.update((131_072, gap) for gap in CALIBRATION_LONG_GAPS)
    actual = {(point.target_tokens, point.gap_tokens) for point in points}
    if len(points) != EXPECTED_CALIBRATION_POINTS or actual != expected:
        raise ValueError("calibration 点不等于预注册的 20 个配置")
    if len(actual) != len(points):
        raise ValueError("calibration 点存在重复计权")
    if sum(point.subset == "position_matrix" for point in points) != 16:
        raise ValueError("position matrix calibration 必须恰好包含 16 点")
    if sum(point.subset == "long_gap" for point in points) != 4:
        raise ValueError("long-gap calibration 必须恰好包含 4 点")
    if any(
        point.gap_tokens <= 0
        or point.gap_tokens > point.target_tokens
        or point.measured_phi_ms <= 0.0
        for point in points
    ):
        raise ValueError("calibration 点违反 G、T 或 cost 的定义域")


def build_heldout_pairs() -> tuple[tuple[int, int], ...]:
    """返回预注册的 3 个 baseline 与 12 个非零留出配置。"""
    pairs = tuple(
        (target, gap)
        for target in HELDOUT_TARGETS
        for gap in HELDOUT_GAPS_BY_TARGET[target]
    )
    if len(pairs) != EXPECTED_HELDOUT_CONFIGURATIONS:
        raise RuntimeError("留出配置数量不正确")
    if any(gap > target for target, gap in pairs):
        raise RuntimeError("留出配置违反 G<=T")
    return pairs


def build_heldout_schedule(seed: int = ORDER_SEED) -> tuple[HeldoutCase, ...]:
    """构建固定种子的平衡循环留出执行计划。"""
    pairs = list(build_heldout_pairs())
    random.Random(seed).shuffle(pairs)
    if pairs == sorted(pairs):
        raise RuntimeError("留出执行顺序不能退化为递增顺序")
    result: list[HeldoutCase] = []
    execution_position = 0
    cycle_index = 0
    for is_warmup, repetitions in (
        (True, WARMUP_REPETITIONS),
        (False, MEASURED_REPETITIONS),
    ):
        for repetition in range(repetitions):
            offset = cycle_index % len(pairs)
            order = pairs[offset:] + pairs[:offset]
            for pair_position, (target, gap) in enumerate(order):
                phase = "warmup" if is_warmup else "measured"
                result.append(
                    HeldoutCase(
                        case_id=(
                            f"heldout_{phase}_r{repetition:02d}_"
                            f"p{pair_position:02d}_t{target}_g{gap}"
                        ),
                        target_position=target,
                        target_frontier=target - gap,
                        target_gap=gap,
                        recovery_fraction=gap / target,
                        repetition=repetition,
                        is_warmup=is_warmup,
                        pair_order_position=pair_position,
                        execution_order_position=execution_position,
                    )
                )
                execution_position += 1
            cycle_index += 1
    return tuple(result)


def fit_candidate_models(
    points: Sequence[CalibrationPoint],
) -> dict[str, object]:
    """只用 calibration 点拟合三个预注册候选模型。"""
    validate_calibration_points(points)
    grouped: dict[int, list[float]] = {}
    for point in points:
        grouped.setdefault(point.gap_tokens, []).append(point.measured_phi_ms)
    knots = {0: 0.0}
    knots.update(
        {
            gap: statistics.fmean(values)
            for gap, values in sorted(grouped.items())
        }
    )
    x1 = [[point.gap_ki, point.gap_ki * point.target_ki] for point in points]
    x2 = [
        [
            point.gap_ki,
            point.gap_ki * point.target_ki,
            point.gap_ki * point.gap_ki,
        ]
        for point in points
    ]
    outcomes = [point.measured_phi_ms for point in points]
    m1 = _least_squares(x1, outcomes)
    m2 = _least_squares(x2, outcomes)
    return {
        "calibration_point_count": len(points),
        "fit_units": "Ki token",
        "heldout_used_for_fit": False,
        "M0": {
            "formula": "固定 gap knots 上的 gap-only 线性插值 f(g)",
            "parameter_count": len(knots) - 1,
            "parameters": {
                "knots": [
                    {"gap_ki": gap / 1024.0, "cost_ms": value}
                    for gap, value in sorted(knots.items())
                ]
            },
        },
        "M1": {
            "formula": "a*g + b*g*t",
            "parameter_count": 2,
            "parameters": {"a": m1[0], "b": m1[1]},
        },
        "M2": {
            "formula": "a*g + b*g*t + c*g^2",
            "parameter_count": 3,
            "parameters": {"a": m2[0], "b": m2[1], "c": m2[2]},
        },
    }


def predict_model(
    model_name: str,
    model: Mapping[str, object],
    target_tokens: int,
    gap_tokens: int,
) -> float:
    """按冻结参数预测 token 输入对应的恢复成本。"""
    if gap_tokens < 0 or target_tokens < 0 or gap_tokens > target_tokens:
        raise ValueError("预测输入必须满足 0<=G<=T")
    if gap_tokens == 0:
        return 0.0
    parameters = model["parameters"]
    g = gap_tokens / 1024.0
    t = target_tokens / 1024.0
    if model_name == "M0":
        knots = {
            int(round(float(row["gap_ki"]) * 1024)): float(row["cost_ms"])
            for row in parameters["knots"]
        }
        return _piecewise_interpolate(gap_tokens, knots)
    if model_name == "M1":
        return float(parameters["a"]) * g + float(parameters["b"]) * g * t
    if model_name == "M2":
        return (
            float(parameters["a"]) * g
            + float(parameters["b"]) * g * t
            + float(parameters["c"]) * g * g
        )
    raise ValueError(f"未知候选模型：{model_name}")


def calibration_diagnostics(
    points: Sequence[CalibrationPoint],
    models: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """分别计算 16 点、4 点和合并 20 点 calibration 误差。"""
    subsets = (
        ("position_matrix", tuple(p for p in points if p.subset == "position_matrix")),
        ("long_gap", tuple(p for p in points if p.subset == "long_gap")),
        ("combined", tuple(points)),
    )
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        for subset_name, subset_points in subsets:
            actual = [point.measured_phi_ms for point in subset_points]
            predicted = [
                predict_model(
                    model_name,
                    models[model_name],
                    point.target_tokens,
                    point.gap_tokens,
                )
                for point in subset_points
            ]
            rows.append(
                {
                    "model": model_name,
                    "subset": subset_name,
                    "point_count": len(subset_points),
                    **prediction_metrics(actual, predicted),
                }
            )
    return tuple(rows)


def structural_validation(models: Mapping[str, object]) -> dict[str, object]:
    """在完整预注册网格验证零点、非负性与固定 T 单调性。"""
    result: dict[str, object] = {
        "targets": list(STRUCTURAL_TARGETS),
        "gap_step": STRUCTURAL_GAP_STEP,
        "models": {},
    }
    for model_name in MODEL_ORDER:
        zero_failures = []
        negative_failures = []
        monotonic_failures = []
        checked_points = 0
        for target in STRUCTURAL_TARGETS:
            values = []
            gaps = range(0, target + 1, STRUCTURAL_GAP_STEP)
            for gap in gaps:
                value = predict_model(model_name, models[model_name], target, gap)
                values.append((gap, value))
                checked_points += 1
                if gap == 0 and value != 0.0:
                    zero_failures.append({"T": target, "value": value})
                if value < 0.0:
                    negative_failures.append({"T": target, "G": gap, "value": value})
            for previous, current in zip(values, values[1:]):
                if current[1] + 1e-9 < previous[1]:
                    monotonic_failures.append(
                        {
                            "T": target,
                            "previous_G": previous[0],
                            "previous_value": previous[1],
                            "current_G": current[0],
                            "current_value": current[1],
                        }
                    )
        passed = not zero_failures and not negative_failures and not monotonic_failures
        result["models"][model_name] = {
            "status": "PASS" if passed else "FAIL",
            "checked_points": checked_points,
            "phi_zero_failures": zero_failures,
            "negative_failures": negative_failures,
            "monotonic_failures": monotonic_failures,
        }
    return result


def summarize_heldout(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """按每个留出 T 自己的 G=0 基线计算 MeasuredPhi。"""
    grouped: dict[tuple[int, int], list[float]] = {}
    for record in records:
        if (
            bool(record.get("is_warmup"))
            or record.get("status") != "PASS"
            or not bool(record.get("correctness_pass"))
        ):
            continue
        key = (int(record["target_H"]), int(record["target_gap"]))
        grouped.setdefault(key, []).append(float(record["ttft_ms"]))
    rows: list[dict[str, object]] = []
    for target, gap in build_heldout_pairs():
        baseline_values = grouped.get((target, 0), [])
        values = grouped.get((target, gap), [])
        if len(baseline_values) != MEASURED_REPETITIONS:
            raise ValueError(f"T={target} 的 baseline measured 数量不完整")
        if len(values) != MEASURED_REPETITIONS:
            raise ValueError(f"T={target},G={gap} 的 measured 数量不完整")
        baseline = statistics.fmean(baseline_values)
        mean_value = statistics.fmean(values)
        measured_phi = 0.0 if gap == 0 else mean_value - baseline
        rows.append(
            {
                "target_tokens": target,
                "target_ki": target / 1024.0,
                "executable_frontier_tokens": target - gap,
                "gap_tokens": gap,
                "gap_ki": gap / 1024.0,
                "recovery_fraction": gap / target,
                "warmup_count": _record_count(records, target, gap, True),
                "measured_count": len(values),
                "baseline_mean_ms": baseline,
                "mean_latency_ms": mean_value,
                "median_latency_ms": statistics.median(values),
                "std_latency_ms": statistics.stdev(values),
                "measured_phi_ms": measured_phi,
            }
        )
    return tuple(rows)


def heldout_predictions(
    summary_rows: Sequence[Mapping[str, object]],
    models: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """使用 GPU 测量前冻结的模型参数预测 12 个非零留出点。"""
    rows: list[dict[str, object]] = []
    for summary in summary_rows:
        gap = int(summary["gap_tokens"])
        if gap == 0:
            continue
        target = int(summary["target_tokens"])
        actual = float(summary["measured_phi_ms"])
        row: dict[str, object] = {
            "target_tokens": target,
            "target_ki": target / 1024.0,
            "gap_tokens": gap,
            "gap_ki": gap / 1024.0,
            "recovery_fraction": float(summary["recovery_fraction"]),
            "measured_phi_ms": actual,
        }
        for model_name in MODEL_ORDER:
            prediction = predict_model(
                model_name,
                models[model_name],
                target,
                gap,
            )
            absolute = abs(prediction - actual)
            row[f"{model_name}_prediction_ms"] = prediction
            row[f"{model_name}_absolute_error_ms"] = absolute
            row[f"{model_name}_relative_error"] = absolute / abs(actual)
        rows.append(row)
    if len(rows) != 12:
        raise ValueError("独立留出预测必须恰好包含 12 个非零点")
    return tuple(rows)


def heldout_metrics(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """计算全局、按 T、按 recovery fraction 的留出误差。"""
    result: dict[str, object] = {}
    for model_name in MODEL_ORDER:
        overall = _metrics_for_prediction_rows(prediction_rows, model_name)
        by_target = {
            str(target): _metrics_for_prediction_rows(
                tuple(row for row in prediction_rows if int(row["target_tokens"]) == target),
                model_name,
            )
            for target in HELDOUT_TARGETS
        }
        by_fraction = {
            str(fraction): _metrics_for_prediction_rows(
                tuple(
                    row
                    for row in prediction_rows
                    if math.isclose(float(row["recovery_fraction"]), fraction)
                ),
                model_name,
            )
            for fraction in HELDOUT_FRACTIONS
        }
        result[model_name] = {
            "overall": overall,
            "by_target": by_target,
            "by_recovery_fraction": by_fraction,
        }
    return result


def grade_model(
    structural_status: str,
    metrics: Mapping[str, float],
) -> str:
    """按预注册阈值评定候选模型。"""
    if structural_status != "PASS":
        return "FAIL"
    mape = float(metrics["mape_percent"])
    maximum = float(metrics["max_relative_error_percent"])
    if mape > 10.0 or maximum > 20.0:
        return "FAIL"
    if mape <= 5.0 and maximum <= 10.0:
        return "PASS"
    return "WEAK"


def select_model(statuses: Mapping[str, str]) -> tuple[str, str]:
    """只在 MODEL PASS 中按 M0、M1、M2 的预注册复杂度选择。"""
    passed = [name for name in MODEL_ORDER if statuses.get(name) == "PASS"]
    if not passed:
        return "NONE", "没有候选模型通过预注册 MODEL PASS 门禁"
    selected = passed[0]
    if len(passed) == 1:
        return selected, f"只有 {selected} 通过 MODEL PASS 门禁"
    return selected, f"多个模型通过门禁，按预注册复杂度选择 {selected}"


def submodular_structure_preserved(
    selected_model: str,
    models: Mapping[str, object],
    structural: Mapping[str, object],
) -> bool:
    """检查固定 T 下单调 cost 是否保持 max-coverage 结构。"""
    if selected_model == "NONE":
        return False
    if structural["models"][selected_model]["status"] != "PASS":
        return False
    for target in STRUCTURAL_TARGETS:
        previous_benefit = -math.inf
        full_cost = predict_model(selected_model, models[selected_model], target, target)
        for executable in range(0, target + 1, STRUCTURAL_GAP_STEP):
            gap = target - executable
            benefit = full_cost - predict_model(
                selected_model,
                models[selected_model],
                target,
                gap,
            )
            if benefit + 1e-9 < previous_benefit:
                return False
            previous_benefit = benefit
    return True


def prediction_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float]:
    """计算预注册的绝对、相对与拟合优度指标。"""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual 与 predicted 必须等长且非空")
    absolute = [abs(p - a) for a, p in zip(actual, predicted)]
    relative = [error / abs(a) for error, a in zip(absolute, actual)]
    mean_actual = statistics.fmean(actual)
    residual = sum((p - a) ** 2 for a, p in zip(actual, predicted))
    total = sum((a - mean_actual) ** 2 for a in actual)
    return {
        "mae_ms": statistics.fmean(absolute),
        "mape_percent": statistics.fmean(relative) * 100.0,
        "max_absolute_error_ms": max(absolute),
        "max_relative_error_percent": max(relative) * 100.0,
        "r_squared": 1.0 - residual / total if total > 0.0 else float("nan"),
    }


def _metrics_for_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    model_name: str,
) -> dict[str, float]:
    """从留出预测行提取指定模型的指标。"""
    return prediction_metrics(
        [float(row["measured_phi_ms"]) for row in rows],
        [float(row[f"{model_name}_prediction_ms"]) for row in rows],
    )


def _least_squares(
    features: Sequence[Sequence[float]],
    outcomes: Sequence[float],
) -> tuple[float, ...]:
    """通过正规方程求解无截距的小规模最小二乘。"""
    if not features or len(features) != len(outcomes):
        raise ValueError("最小二乘输入必须非空且等长")
    width = len(features[0])
    if width == 0 or any(len(row) != width for row in features):
        raise ValueError("特征矩阵宽度不一致")
    matrix = [
        [
            sum(float(row[left]) * float(row[right]) for row in features)
            for right in range(width)
        ]
        for left in range(width)
    ]
    vector = [
        sum(float(row[index]) * float(value) for row, value in zip(features, outcomes))
        for index in range(width)
    ]
    return _solve_linear_system(matrix, vector)


def _solve_linear_system(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, ...]:
    """使用带主元选择的高斯消元求解小型线性方程。"""
    size = len(vector)
    augmented = [
        [float(value) for value in matrix[index]] + [float(vector[index])]
        for index in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise ValueError("候选模型正规方程不可逆")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            augmented[row] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return tuple(augmented[row][-1] for row in range(size))


def _piecewise_interpolate(gap_tokens: int, knots: Mapping[int, float]) -> float:
    """在固定 knots 间做线性插值，不执行 clamp。"""
    ordered = sorted((int(gap), float(value)) for gap, value in knots.items())
    if gap_tokens < ordered[0][0] or gap_tokens > ordered[-1][0]:
        raise ValueError("gap 超出 M0 冻结 knot 范围")
    for left, right in zip(ordered, ordered[1:]):
        if gap_tokens == left[0]:
            return left[1]
        if left[0] < gap_tokens <= right[0]:
            fraction = (gap_tokens - left[0]) / (right[0] - left[0])
            return left[1] + fraction * (right[1] - left[1])
    return ordered[-1][1]


def _record_count(
    records: Sequence[Mapping[str, object]],
    target: int,
    gap: int,
    is_warmup: bool,
) -> int:
    """统计指定 T/G/阶段的原始 trial 数量。"""
    return sum(
        int(record["target_H"]) == target
        and int(record["target_gap"]) == gap
        and bool(record["is_warmup"]) is is_warmup
        for record in records
    )


def _calibration_rows(points: Sequence[CalibrationPoint]) -> tuple[dict[str, object], ...]:
    """把 calibration 点转换为稳定 CSV 行。"""
    return tuple(
        {
            "source": point.source,
            "subset": point.subset,
            "target_tokens": point.target_tokens,
            "target_ki": point.target_ki,
            "gap_tokens": point.gap_tokens,
            "gap_ki": point.gap_ki,
            "measured_phi_ms": point.measured_phi_ms,
        }
        for point in points
    )


def _file_sha256(path: Path) -> str:
    """返回单个文件的 SHA256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execute_heldout_phase(
    writer: FreezeArtifactWriter,
    schedule: Sequence[HeldoutCase],
) -> tuple[list[dict[str, object]], str | None]:
    """用正式 runtime recovery 路径执行全部独立留出 trial。"""
    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import FormalEndToEndGateEngine, requested_control_port
    from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport

    records: list[dict[str, object]] = []
    failure = None
    engine = None
    try:
        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION_128K)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        for index, case in enumerate(schedule, start=1):
            phase = "WARMUP" if case.is_warmup else "MEASURED"
            print(
                f"[STEP10D3-HELDOUT] {index}/{len(schedule)} {phase} "
                f"T={case.target_position} G={case.target_gap}",
                flush=True,
            )
            profile_case = ProfileCase(
                case_id=case.case_id,
                target_gap=case.target_gap,
                target_frontier=case.target_frontier,
                repetition=case.repetition,
                is_warmup=case.is_warmup,
                gap_order_position=case.pair_order_position,
                execution_order_position=case.execution_order_position,
            )
            record = execute_profile_case(
                profile_case,
                engine=engine,
                client=client,
                target_position=case.target_position,
                namespace_prefix="flowstate_step10d3",
                suffix_seed=MATRIX_SUFFIX_SEED,
            )
            record = {"source": HELDOUT_SOURCE_LABEL, **record}
            records.append(record)
            writer.append_raw(record)
            if record.get("status") != "PASS":
                failure = (
                    f"held-out case {case.case_id} 失败："
                    f"{record.get('failure_stage')} {record.get('error')}"
                )
                break
    except Exception as error:
        failure = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                failure = failure or f"关闭 held-out runtime 失败：{error!r}"
    return records, failure


def _render_theory(selected: str, preserved: bool) -> str:
    """生成固定 target 下理论结构兼容性说明。"""
    return "\n".join(
        (
            "# 理论结构兼容性",
            "",
            f"推荐候选：{selected}。",
            "",
            "在单次 allocation snapshot 中，每个 pending continuation 的目标位置 T_p 固定。兼容 checkpoint 集合决定最深 executable frontier E_p(S)，因此 G_p(S)=T_p-E_p(S)。",
            "",
            "只要 Phi(G,T_p) 对固定 T_p 随 G 单调不减，选择更深 compatible checkpoint 就不会增加恢复成本。相对空集合的恢复收益可写为各 compatible checkpoint 单点收益的最大值，因此仍保持 max-coverage 的单调次模结构。",
            "",
            f"SUBMODULAR_STRUCTURE_PRESERVED：{'YES' if preserved else 'NO'}。",
            "",
            "本检查仅验证数学兼容性，没有修改 optimizer。",
            "",
        )
    )


def _render_readme(config: Mapping[str, object]) -> str:
    """生成 artifact 的实验边界说明。"""
    return "\n".join(
        (
            "# 预注册恢复模型选择与独立留出验证",
            "",
            "Calibration 参数只来自 Step 10D.2 的 16 个 position-matrix 点与 Step 10D.1 的 4 个 long-gap 点。独立留出 GPU 数据不参与拟合，也不会改变候选公式。",
            "",
            "候选模型仅为 M0 gap-only 分段线性、M1 position-aware bilinear、M2 position-aware quadratic。正式 Phi、policy 和 TraceLab protocol 均未修改。",
            "",
            f"运行状态：{config.get('status', '未知')}。",
            "",
        )
    )


def _server_command_text() -> str:
    """记录 held-out 阶段的进程内引擎配置。"""
    parameters = " ".join(
        f"{key}={value!r}"
        for key, value in sorted(ENGINE_CONFIGURATION_128K.items())
    )
    return (
        "进程入口：python -m evaluation.recovery_model_freeze\n"
        f"Held-out 引擎参数：{parameters}\n"
        f"正式 mutation primitive：{FORMAL_MUTATION_PRIMITIVE}\n"
    )


def main() -> int:
    """冻结候选参数，执行独立 GPU 留出验证并按规则选择模型。"""
    writer = FreezeArtifactWriter.create()
    protected_before = _protected_hashes()
    calibration_points = load_calibration_points()
    models = fit_candidate_models(calibration_points)
    calibration_rows = _calibration_rows(calibration_points)
    diagnostics = calibration_diagnostics(calibration_points, models)
    structural = structural_validation(models)
    schedule = build_heldout_schedule()
    _write_csv(
        writer.directory / "calibration_points.csv",
        calibration_rows,
        (
            "source",
            "subset",
            "target_tokens",
            "target_ki",
            "gap_tokens",
            "gap_ki",
            "measured_phi_ms",
        ),
    )
    _write_csv(
        writer.directory / "calibration_diagnostics.csv",
        diagnostics,
        (
            "model",
            "subset",
            "point_count",
            "mae_ms",
            "mape_percent",
            "max_absolute_error_ms",
            "max_relative_error_percent",
            "r_squared",
        ),
    )
    writer.write_json("candidate_models.json", models)
    writer.write_json("structural_validation.json", structural)
    candidate_hash_before = _file_sha256(writer.directory / "candidate_models.json")
    writer.write_json(
        "execution_order.json",
        {
            "seed": ORDER_SEED,
            "cases": [case.__dict__ for case in schedule],
        },
    )
    writer.write_text("server_command.txt", _server_command_text())
    config: dict[str, object] = {
        "schema_version": "flowstate.recovery_model_freeze.v1",
        "status": "RUNNING",
        "calibration_point_count": len(calibration_points),
        "calibration_sources": {
            "position_matrix": str(STEP10D2_DIRECTORY.relative_to(REPOSITORY_ROOT)),
            "long_gap": str(STEP10D1_DIRECTORY.relative_to(REPOSITORY_ROOT)),
        },
        "calibration_source_hashes": {
            "position_matrix_summary.csv": _file_sha256(
                STEP10D2_DIRECTORY / "position_matrix_summary.csv"
            ),
            "long_gap_summary.csv": _file_sha256(STEP10D1_DIRECTORY / "summary.csv"),
        },
        "heldout_targets": list(HELDOUT_TARGETS),
        "heldout_gaps_by_target": {
            str(target): list(gaps)
            for target, gaps in HELDOUT_GAPS_BY_TARGET.items()
        },
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "seed": ORDER_SEED,
        "engine_configuration": ENGINE_CONFIGURATION_128K,
        "heldout_used_for_fit": False,
        "policy_comparison_executed": False,
        "start_time": datetime.now().astimezone().isoformat(),
    }
    writer.write_json("config.json", config)
    failure = None
    failure_stage = "环境检查"
    heldout_records: list[dict[str, object]] = []
    started = time.perf_counter_ns()
    try:
        environment = _collect_environment()
        model_config, tokenizer_config = _load_context_configuration(MODEL_PATH)
        context = validate_context_capabilities(model_config, tokenizer_config)
        config["environment"] = environment
        config["context_capabilities"] = context
        writer.write_text("environment.txt", _environment_text(environment))
        writer.write_json("config.json", config)

        failure_stage = "独立 held-out GPU 测量"
        heldout_records, failure = _execute_heldout_phase(writer, schedule)
        if failure is not None:
            raise RuntimeError(failure)

        failure_stage = "held-out 汇总与预注册模型选择"
        candidate_hash_after = _file_sha256(writer.directory / "candidate_models.json")
        if candidate_hash_before != candidate_hash_after:
            raise RuntimeError("held-out 测量期间候选模型参数发生变化")
        summary_rows = summarize_heldout(heldout_records)
        prediction_rows = heldout_predictions(summary_rows, models)
        metrics = heldout_metrics(prediction_rows)
        statuses = {
            model_name: grade_model(
                str(structural["models"][model_name]["status"]),
                metrics[model_name]["overall"],
            )
            for model_name in MODEL_ORDER
        }
        selected, selection_reason = select_model(statuses)
        preserved = submodular_structure_preserved(selected, models, structural)
        ready = selected != "NONE" and preserved
        _write_csv(
            writer.directory / "heldout_summary.csv",
            summary_rows,
            (
                "target_tokens",
                "target_ki",
                "executable_frontier_tokens",
                "gap_tokens",
                "gap_ki",
                "recovery_fraction",
                "warmup_count",
                "measured_count",
                "baseline_mean_ms",
                "mean_latency_ms",
                "median_latency_ms",
                "std_latency_ms",
                "measured_phi_ms",
            ),
        )
        _write_csv(
            writer.directory / "heldout_predictions.csv",
            prediction_rows,
            (
                "target_tokens",
                "target_ki",
                "gap_tokens",
                "gap_ki",
                "recovery_fraction",
                "measured_phi_ms",
                "M0_prediction_ms",
                "M0_absolute_error_ms",
                "M0_relative_error",
                "M1_prediction_ms",
                "M1_absolute_error_ms",
                "M1_relative_error",
                "M2_prediction_ms",
                "M2_absolute_error_ms",
                "M2_relative_error",
            ),
        )
        selection = {
            "model_statuses": statuses,
            "heldout_metrics": metrics,
            "selected_model": selected,
            "selection_reason": selection_reason,
            "submodular_structure_preserved": preserved,
            "ready_to_freeze_formal_recovery_model": ready,
            "candidate_models_hash_before_heldout": candidate_hash_before,
            "candidate_models_hash_after_heldout": candidate_hash_after,
            "heldout_used_for_fit": False,
        }
        writer.write_json("model_selection.json", selection)
        writer.write_text(
            "theory_compatibility.md",
            _render_theory(selected, preserved),
        )
        config["selection"] = {
            "statuses": statuses,
            "selected_model": selected,
            "selection_reason": selection_reason,
            "submodular_structure_preserved": preserved,
            "ready_to_freeze_formal_recovery_model": ready,
        }
    except Exception as error:
        failure = repr(error)
        traceback.print_exc()
    finally:
        protected_after = _protected_hashes()
        semantic_correctness = (
            failure is None
            and len(heldout_records) == EXPECTED_HELDOUT_TRIALS
            and all(record.get("status") == "PASS" for record in heldout_records)
            and all(bool(record.get("correctness_pass")) for record in heldout_records)
        )
        config.update(
            {
                "status": "PASS" if semantic_correctness else "FAIL",
                "end_time": datetime.now().astimezone().isoformat(),
                "total_runtime_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                "heldout_completed_trials": len(heldout_records),
                "failure_stage": None if semantic_correctness else failure_stage,
                "fatal_error": failure,
                "protected_hashes_before": protected_before,
                "protected_hashes_after": protected_after,
                "gates": {
                    "calibration_correctness": len(calibration_points)
                    == EXPECTED_CALIBRATION_POINTS,
                    "heldout_semantic_correctness": semantic_correctness,
                    "formal_recovery_model_unchanged": (
                        protected_before.get("flowstate/recovery_model.py")
                        == protected_after.get("flowstate/recovery_model.py")
                    ),
                    "protected_files_unchanged": protected_before == protected_after,
                },
            }
        )
        writer.write_json("config.json", config)
        writer.write_text("README.md", _render_readme(config))
        writer.ensure_required_files()
    print(
        json.dumps(
            {
                "status": config["status"],
                "artifact_directory": str(writer.directory),
                "heldout_trials": len(heldout_records),
                "failure_stage": config["failure_stage"],
                "fatal_error": config["fatal_error"],
                "selection": config.get("selection"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if config["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
