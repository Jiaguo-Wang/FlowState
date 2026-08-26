#!/usr/bin/env python3
"""分析独立 Recovery Profiler v2 样本并比较候选成本模型。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Callable, Mapping, Sequence

from flowstate.recovery_model import RecoveryCostModel


CALIBRATION_GAPS = (0, 4096, 8192, 16384, 32768)
VALIDATION_GAPS = (2048, 6144, 12288, 24576)
ALL_GAPS = tuple(sorted(CALIBRATION_GAPS + VALIDATION_GAPS))
WARMUP_REPETITIONS = 2
MEASURED_REPETITIONS = 12
STEP9C_REFERENCE_INCREMENTAL_MS = {
    4096: 131.588,
    8192: 290.719,
    16384: 625.997,
    32768: 1505.932,
}


@dataclass(frozen=True)
class LinearRecoveryModel:
    """表示通过原点且单调非减的线性恢复成本模型。"""

    slope_ms_per_token: float

    def estimate(self, gap_tokens: int) -> float:
        """估计指定 gap 的增量 TTFT。"""
        if gap_tokens < 0:
            raise ValueError("gap_tokens 必须大于等于零")
        return self.slope_ms_per_token * gap_tokens


@dataclass(frozen=True)
class MonotonePiecewiseRecoveryModel:
    """表示 calibration knots 上的单调分段线性模型。"""

    knots: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not self.knots or self.knots[0] != (0, 0.0):
            raise ValueError("分段模型必须从 (0, 0) 开始")
        for previous, current in zip(self.knots, self.knots[1:]):
            if current[0] <= previous[0]:
                raise ValueError("分段模型的 gap knots 必须严格递增")
            if current[1] < previous[1]:
                raise ValueError("分段模型的成本必须单调非减")

    def estimate(self, gap_tokens: int) -> float:
        """在相邻 calibration knots 间做线性插值。"""
        if gap_tokens < 0:
            raise ValueError("gap_tokens 必须大于等于零")
        if gap_tokens > self.knots[-1][0]:
            raise ValueError("分段模型不在最大 calibration knot 外推")
        for lower, upper in zip(self.knots, self.knots[1:]):
            if gap_tokens <= upper[0]:
                position = (gap_tokens - lower[0]) / (
                    upper[0] - lower[0]
                )
                return lower[1] + position * (upper[1] - lower[1])
        return self.knots[-1][1]


def load_raw_samples(path: Path) -> tuple[dict[str, object], ...]:
    """读取 profiler 原始 JSONL 样本。"""
    with path.open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def summarize_measurements(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """排除 warmup，并按 gap 汇总 TTFT 分布。"""
    _validate_records(records)
    measured = tuple(
        record for record in records if record.get("is_warmup") is False
    )
    grouped = {
        gap: tuple(
            float(record["ttft_ms"])
            for record in measured
            if int(record["target_gap"]) == gap
        )
        for gap in ALL_GAPS
    }
    baseline_values = grouped[0]
    baseline_mean = statistics.fmean(baseline_values)
    result = []
    for gap in ALL_GAPS:
        values = grouped[gap]
        if len(values) != MEASURED_REPETITIONS:
            raise ValueError(
                f"gap={gap} 的 measured 样本数异常：{len(values)}"
            )
        mean_ttft = statistics.fmean(values)
        measured_phi = 0.0 if gap == 0 else mean_ttft - baseline_mean
        if measured_phi < 0.0:
            raise ValueError(f"gap={gap} 的 measured Phi 为负")
        result.append(
            {
                "gap_tokens": gap,
                "split": (
                    "calibration"
                    if gap in CALIBRATION_GAPS
                    else "held_out_validation"
                ),
                "n": len(values),
                "ttft_mean_ms": mean_ttft,
                "ttft_median_ms": statistics.median(values),
                "ttft_p95_ms": _empirical_quantile(values, 0.95),
                "ttft_std_ms": statistics.stdev(values),
                "measured_phi_ms": measured_phi,
            }
        )
    return tuple(result)


def fit_linear_model(
    calibration: Mapping[int, float],
) -> LinearRecoveryModel:
    """只用 calibration points 拟合通过原点的线性模型。"""
    _validate_calibration_mapping(calibration)
    nonzero = tuple(gap for gap in CALIBRATION_GAPS if gap > 0)
    denominator = sum(float(gap * gap) for gap in nonzero)
    slope = sum(gap * float(calibration[gap]) for gap in nonzero) / denominator
    if slope < 0.0:
        raise ValueError("线性模型斜率不能为负")
    return LinearRecoveryModel(slope_ms_per_token=slope)


def fit_piecewise_model(
    calibration: Mapping[int, float],
) -> MonotonePiecewiseRecoveryModel:
    """只用 calibration knots 构造单调分段线性模型。"""
    _validate_calibration_mapping(calibration)
    knots = tuple(
        (gap, 0.0 if gap == 0 else float(calibration[gap]))
        for gap in CALIBRATION_GAPS
    )
    return MonotonePiecewiseRecoveryModel(knots=knots)


def compare_models(
    measurement_rows: Sequence[Mapping[str, object]],
    old_model: RecoveryCostModel | None = None,
) -> dict[str, object]:
    """在 held-out gaps 上比较旧 Phi、线性和分段模型。"""
    by_gap = {
        int(row["gap_tokens"]): float(row["measured_phi_ms"])
        for row in measurement_rows
    }
    if set(by_gap) != set(ALL_GAPS):
        raise ValueError("measurement rows 未覆盖全部 profiling gaps")
    calibration = {gap: by_gap[gap] for gap in CALIBRATION_GAPS}
    validation = {gap: by_gap[gap] for gap in VALIDATION_GAPS}
    linear = fit_linear_model(calibration)
    piecewise = fit_piecewise_model(calibration)
    active_old_model = old_model or RecoveryCostModel()

    estimators: dict[str, Callable[[int], float]] = {
        "Old Phi": active_old_model.estimate,
        "Linear v2": linear.estimate,
        "Monotone piecewise v2": piecewise.estimate,
    }
    comparison = {}
    for name, estimate in estimators.items():
        predictions = {
            gap: float(estimate(gap)) for gap in VALIDATION_GAPS
        }
        errors = {
            gap: abs(predictions[gap] - validation[gap])
            for gap in VALIDATION_GAPS
        }
        relative_errors = {
            gap: errors[gap] / validation[gap] * 100.0
            for gap in VALIDATION_GAPS
        }
        comparison[name] = {
            "predictions_ms": predictions,
            "absolute_errors_ms": errors,
            "mae_ms": statistics.fmean(errors.values()),
            "mape_percent": statistics.fmean(relative_errors.values()),
            "max_absolute_error_ms": max(errors.values()),
        }
    best_model = min(
        comparison,
        key=lambda name: (comparison[name]["mae_ms"], name),
    )
    return {
        "fit_split": {
            "calibration_gaps": list(CALIBRATION_GAPS),
            "held_out_validation_gaps": list(VALIDATION_GAPS),
            "validation_used_for_fitting": False,
            "step9b_data_used_for_fitting": False,
        },
        "models": {
            "Linear v2": {
                "slope_ms_per_token": linear.slope_ms_per_token,
                "phi_zero_ms": linear.estimate(0),
            },
            "Monotone piecewise v2": {
                "knots": [list(knot) for knot in piecewise.knots],
                "phi_zero_ms": piecewise.estimate(0),
            },
            "Old Phi": {
                "phi_zero_ms": active_old_model.estimate(0),
            },
        },
        "held_out_actual_ms": validation,
        "comparison": comparison,
        "best_held_out_model": best_model,
    }


def analyze_records(
    records: Sequence[Mapping[str, object]],
    old_model: RecoveryCostModel | None = None,
) -> dict[str, object]:
    """从独立 raw samples 生成全部统计和模型比较。"""
    measurements = summarize_measurements(records)
    comparison = compare_models(measurements, old_model)
    by_gap = {int(row["gap_tokens"]): row for row in measurements}
    step9b_comparison = {
        gap: {
            "profiler_v2_incremental_ms": float(
                by_gap[gap]["measured_phi_ms"]
            ),
            "step9c_step9b_incremental_ms": reference,
            "difference_ms": (
                float(by_gap[gap]["measured_phi_ms"]) - reference
            ),
        }
        for gap, reference in STEP9C_REFERENCE_INCREMENTAL_MS.items()
    }
    return {
        "schema_version": "flowstate.recovery_profiler_v2.analysis.v1",
        "data_source": "Recovery Profiler v2 独立 profiling samples",
        "measurements": measurements,
        "model_comparison": comparison,
        "step9b_comparison": step9b_comparison,
        "step9b_data_used_for_fitting": False,
    }


def write_analysis_artifacts(
    analysis: Mapping[str, object],
    output_directory: Path,
) -> None:
    """写出 calibration、validation 和模型比较 artifacts。"""
    measurements = tuple(analysis["measurements"])
    calibration_rows = tuple(
        row for row in measurements if row["split"] == "calibration"
    )
    validation_rows = tuple(
        row
        for row in measurements
        if row["split"] == "held_out_validation"
    )
    _write_measurement_csv(
        output_directory / "calibration_summary.csv",
        calibration_rows,
    )
    model_comparison = analysis["model_comparison"]
    comparison_by_name = model_comparison["comparison"]
    validation_fieldnames = (
        "gap_tokens",
        "n",
        "ttft_mean_ms",
        "ttft_median_ms",
        "ttft_p95_ms",
        "ttft_std_ms",
        "measured_phi_ms",
        "old_phi_prediction_ms",
        "linear_v2_prediction_ms",
        "piecewise_v2_prediction_ms",
    )
    with (output_directory / "validation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=validation_fieldnames)
        writer.writeheader()
        for row in validation_rows:
            gap = int(row["gap_tokens"])
            writer.writerow(
                {
                    **{field: row[field] for field in validation_fieldnames[:7]},
                    "old_phi_prediction_ms": comparison_by_name[
                        "Old Phi"
                    ]["predictions_ms"][gap],
                    "linear_v2_prediction_ms": comparison_by_name[
                        "Linear v2"
                    ]["predictions_ms"][gap],
                    "piecewise_v2_prediction_ms": comparison_by_name[
                        "Monotone piecewise v2"
                    ]["predictions_ms"][gap],
                }
            )
    payload = {
        **model_comparison,
        "step9b_comparison": analysis["step9b_comparison"],
        "step9b_data_used_for_fitting": False,
    }
    _write_json(output_directory / "model_comparison.json", payload)


def _validate_records(records: Sequence[Mapping[str, object]]) -> None:
    """核验 case 数量、分组、正确性和 warmup 标记。"""
    expected = len(ALL_GAPS) * (
        WARMUP_REPETITIONS + MEASURED_REPETITIONS
    )
    if len(records) != expected:
        raise ValueError(f"raw sample 数量异常：{len(records)}，预期 {expected}")
    case_ids = {str(record["case_id"]) for record in records}
    if len(case_ids) != expected:
        raise ValueError("raw samples 存在重复 case_id")
    for record in records:
        gap = int(record["target_gap"])
        if gap not in ALL_GAPS:
            raise ValueError(f"发现未冻结 gap：{gap}")
        if record.get("status") != "PASS":
            raise ValueError(f"case 未通过：{record['case_id']}")
        if record.get("gap_match") is not True:
            raise ValueError(f"case gap 不匹配：{record['case_id']}")
        ttft_ms = float(record["ttft_ms"])
        if not math.isfinite(ttft_ms) or ttft_ms <= 0.0:
            raise ValueError(f"case TTFT 无效：{record['case_id']}")
    for gap in ALL_GAPS:
        warmup_count = sum(
            int(record["target_gap"]) == gap
            and record.get("is_warmup") is True
            for record in records
        )
        measured_count = sum(
            int(record["target_gap"]) == gap
            and record.get("is_warmup") is False
            for record in records
        )
        if warmup_count != WARMUP_REPETITIONS:
            raise ValueError(f"gap={gap} 的 warmup 数量异常：{warmup_count}")
        if measured_count != MEASURED_REPETITIONS:
            raise ValueError(
                f"gap={gap} 的 measured 数量异常：{measured_count}"
            )


def _validate_calibration_mapping(calibration: Mapping[int, float]) -> None:
    """核验拟合输入只包含冻结 calibration gaps。"""
    if set(calibration) != set(CALIBRATION_GAPS):
        raise ValueError("拟合输入必须且只能包含 calibration gaps")
    if float(calibration[0]) != 0.0:
        raise ValueError("calibration 必须满足 Phi(0)=0")
    values = tuple(float(calibration[gap]) for gap in CALIBRATION_GAPS)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("calibration cost 必须是有限非负数")
    if any(current < previous for previous, current in zip(values, values[1:])):
        raise ValueError("calibration cost 必须单调非减")


def _empirical_quantile(values: Sequence[float], quantile: float) -> float:
    """按经验分布逆函数计算未加权分位数。"""
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("分位数输入无效")
    ordered = sorted(float(value) for value in values)
    threshold = quantile * len(ordered)
    index = max(0, math.ceil(threshold) - 1)
    return ordered[index]


def _write_measurement_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """写出一个数据 split 的测量汇总。"""
    fieldnames = (
        "gap_tokens",
        "split",
        "n",
        "ttft_mean_ms",
        "ttft_median_ms",
        "ttft_p95_ms",
        "ttft_std_ms",
        "measured_phi_ms",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """稳定写出 UTF-8 JSON。"""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    """分析当前目录中的独立 profiler raw samples。"""
    directory = Path(__file__).resolve().parent
    records = load_raw_samples(directory / "raw_samples.jsonl")
    analysis = analyze_records(records)
    write_analysis_artifacts(analysis, directory)
    print(
        json.dumps(
            {
                "best_held_out_model": analysis["model_comparison"][
                    "best_held_out_model"
                ],
                "step9b_data_used_for_fitting": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
