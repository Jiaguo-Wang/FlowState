#!/usr/bin/env python3
"""审计恢复成本对绝对上下文位置的依赖。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback
from typing import Mapping, Sequence

from evaluation.recovery_profiler_128k import (
    ENGINE_CONFIGURATION_128K,
    FORMAL_MUTATION_PRIMITIVE,
    MODEL_PATH,
    ProfileCase as PositionProfileCase,
    _collect_environment,
    _environment_text,
    _load_context_configuration,
    build_position_scenario,
    execute_profile_case as execute_position_case,
    validate_context_capabilities,
)
from evaluation.recovery_profiler_v2.analyze import (
    CALIBRATION_GAPS,
    MEASURED_REPETITIONS,
    WARMUP_REPETITIONS,
)
from evaluation.recovery_profiler_v2.profile_runner import (
    ANCHOR_POS as LEGACY_TARGET_POSITION,
    build_profile_schedule as build_legacy_full_schedule,
    execute_profile_case as execute_legacy_case,
)
from evaluation.sota_runtime_correctness import STEP8E_ENGINE_CONFIGURATION


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
STEP9D_DIRECTORY = REPOSITORY_ROOT / "evaluation" / "recovery_profiler_v2"
STEP10D1_DIRECTORY = (
    ARTIFACT_ROOT / "recovery_profiler_128k_20260826_133712_596813"
)
TARGET_POSITIONS = (32_768, 65_536, 98_304, 131_072)
POSITION_GAPS = (0, 4_096, 8_192, 16_384, 32_768)
ORDER_SEED = 20_260_826
MATRIX_SUFFIX_SEED = 741_019
POSITION_RATIO_PASS_LIMIT = 0.05
POSITION_RATIO_WEAK_LIMIT = 0.10
SIGNIFICANT_TREND_LIMIT = 0.05
EXPECTED_LEGACY_TRIALS = len(POSITION_GAPS) * (
    WARMUP_REPETITIONS + MEASURED_REPETITIONS
)
EXPECTED_MATRIX_TRIALS = len(TARGET_POSITIONS) * len(POSITION_GAPS) * (
    WARMUP_REPETITIONS + MEASURED_REPETITIONS
)
RAW_FIELDS = (
    "source",
    "case_id",
    "target_H",
    "target_E",
    "target_gap",
    "repetition",
    "is_warmup",
    "gap_order_position",
    "execution_order_position",
    "status",
    "correctness_pass",
    "gap_match",
    "runtime_H",
    "runtime_E",
    "runtime_G",
    "ttft_ms",
    "request_latency_ms",
    "request_input_tokens",
    "failure_stage",
    "error",
    "fa_preserved",
    "safety_pass",
    "selected_checkpoint_ids",
    "evicted_checkpoint_ids",
    "fa_allocator",
    "safety",
    "server_metadata",
)
PROTECTED_PATHS = (
    REPOSITORY_ROOT / "flowstate" / "recovery_model.py",
    REPOSITORY_ROOT / "flowstate" / "optimizer.py",
    REPOSITORY_ROOT / "evaluation" / "controlled_multiworkflow_v1" / "policies.py",
    REPOSITORY_ROOT / "evaluation" / "sota_policies.py",
    REPOSITORY_ROOT / "evaluation" / "sota_metadata.py",
    REPOSITORY_ROOT
    / "evaluation"
    / "public_agent_trace"
    / "tracelab_nontrivial_protocol.py",
    REPOSITORY_ROOT / "motivation" / "README.md",
)


@dataclass(frozen=True)
class MatrixCase:
    """描述一个固定 T、E、G 的位置矩阵 trial。"""

    case_id: str
    target_position: int
    target_frontier: int
    target_gap: int
    repetition: int
    is_warmup: bool
    pair_order_position: int
    execution_order_position: int


@dataclass
class AuditArtifactWriter:
    """增量保存位置审计的全部证据。"""

    directory: Path

    @classmethod
    def create(
        cls,
        root: Path = ARTIFACT_ROOT,
        timestamp: str | None = None,
    ) -> "AuditArtifactWriter":
        """创建不会覆盖既有证据的时间戳目录。"""
        resolved = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = root / f"recovery_position_audit_{resolved}"
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory)

    def append_record(self, name: str, record: Mapping[str, object]) -> None:
        """把单个 trial 立即追加到指定 CSV。"""
        path = self.directory / name
        is_new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(
                {
                    field: _csv_value(record.get(field))
                    for field in RAW_FIELDS
                }
            )

    def write_json(self, name: str, value: object) -> None:
        """稳定写出 UTF-8 JSON。"""
        (self.directory / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def write_text(self, name: str, value: str) -> None:
        """写出 UTF-8 文本。"""
        (self.directory / name).write_text(value, encoding="utf-8")

    def ensure_required_files(self) -> None:
        """确保失败运行也保留约定的全部文件。"""
        for name in (
            "legacy_reproduction_raw.csv",
            "legacy_reproduction.csv",
            "position_matrix_raw.csv",
            "position_matrix_summary.csv",
        ):
            path = self.directory / name
            if not path.exists():
                path.write_text("\n", encoding="utf-8")
        for name in ("position_dependence.json", "model_diagnostics.json"):
            path = self.directory / name
            if not path.exists():
                self.write_json(name, {"status": "测量不完整，无法分析"})


def build_legacy_schedule() -> tuple[object, ...]:
    """从 Step 9D 原始计划中过滤出五个复现 gap。"""
    return tuple(
        case
        for case in build_legacy_full_schedule(ORDER_SEED)
        if case.target_gap in POSITION_GAPS
    )


def build_matrix_schedule(seed: int = ORDER_SEED) -> tuple[MatrixCase, ...]:
    """构建跨 T 与 G 平衡交错的确定性位置矩阵计划。"""
    pairs = [(target, gap) for target in TARGET_POSITIONS for gap in POSITION_GAPS]
    random.Random(seed).shuffle(pairs)
    if pairs == sorted(pairs):
        raise RuntimeError("位置矩阵顺序不能退化为递增顺序")
    result = []
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
                    MatrixCase(
                        case_id=(
                            f"position_{phase}_r{repetition:02d}_"
                            f"p{pair_position:02d}_t{target}_g{gap}"
                        ),
                        target_position=target,
                        target_frontier=target - gap,
                        target_gap=gap,
                        repetition=repetition,
                        is_warmup=is_warmup,
                        pair_order_position=pair_position,
                        execution_order_position=execution_position,
                    )
                )
                execution_position += 1
            cycle_index += 1
    return tuple(result)


def summarize_legacy(
    records: Sequence[Mapping[str, object]],
    historical: Mapping[int, float],
) -> dict[str, object]:
    """用旧协议自己的 G=0 基线评估历史复现误差。"""
    grouped = _valid_measured_by_key(records, lambda row: int(row["target_gap"]))
    baseline = statistics.fmean(grouped[0])
    rows = []
    for gap in POSITION_GAPS:
        values = grouped[gap]
        stats = _latency_statistics(values)
        measured = 0.0 if gap == 0 else float(stats["mean_ms"]) - baseline
        reference = float(historical[gap])
        absolute = abs(measured - reference)
        relative = None if gap == 0 else absolute / abs(reference)
        rows.append(
            {
                "gap_tokens": gap,
                **stats,
                "legacy_measured_phi_ms": measured,
                "historical_phi_ms": reference,
                "absolute_error_ms": absolute,
                "relative_error": relative,
            }
        )
    grade = grade_legacy_reproduction(
        tuple(
            float(row["relative_error"])
            for row in rows
            if row["relative_error"] is not None
        )
    )
    return {"rows": rows, "grade": grade, "baseline_mean_ms": baseline}


def grade_legacy_reproduction(relative_errors: Sequence[float]) -> str:
    """按预注册 5% 与 10% 阈值判定旧协议复现等级。"""
    if not relative_errors:
        raise ValueError("缺少非零 gap 的复现误差")
    maximum = max(float(value) for value in relative_errors)
    if maximum <= 0.05:
        return "PASS"
    if maximum <= 0.10:
        return "WEAK"
    return "FAIL"


def summarize_position_matrix(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """按每个 T 自己的 G=0 基线计算位置特定恢复成本。"""
    grouped = _valid_measured_by_key(
        records,
        lambda row: (int(row["target_H"]), int(row["target_gap"])),
    )
    rows = []
    for target in TARGET_POSITIONS:
        baseline = statistics.fmean(grouped[(target, 0)])
        for gap in POSITION_GAPS:
            values = grouped[(target, gap)]
            stats = _latency_statistics(values)
            measured = (
                0.0 if gap == 0 else float(stats["mean_ms"]) - baseline
            )
            rows.append(
                {
                    "target_position": target,
                    "executable_frontier": target - gap,
                    "gap_tokens": gap,
                    "warmup_count": _count_records(records, target, gap, True),
                    "valid_measured_count": len(values),
                    "failure_count": _failure_count(records, target, gap),
                    "mismatch_count": _mismatch_count(records, target, gap),
                    **stats,
                    "position_baseline_mean_ms": baseline,
                    "measured_phi_ms": measured,
                }
            )
    return tuple(rows)


def analyze_position_dependence(
    summary_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """计算 same-gap 跨位置范围、趋势和预注册 gate。"""
    by_key = {
        (int(row["target_position"]), int(row["gap_tokens"])): float(
            row["measured_phi_ms"]
        )
        for row in summary_rows
    }
    gaps = []
    for gap in POSITION_GAPS[1:]:
        values = [by_key[(target, gap)] for target in TARGET_POSITIONS]
        mean_value = statistics.fmean(values)
        if mean_value == 0.0:
            raise ValueError("非零 gap 的平均恢复成本不能为零")
        ratio = (max(values) - min(values)) / abs(mean_value)
        increments = [
            current - previous
            for previous, current in zip(values, values[1:])
        ]
        endpoint_change = (values[-1] - values[0]) / abs(mean_value)
        increasing = all(value >= 0.0 for value in increments)
        decreasing = all(value <= 0.0 for value in increments)
        significant = (
            (increasing or decreasing)
            and abs(endpoint_change) > SIGNIFICANT_TREND_LIMIT
        )
        trend = (
            "单调增长"
            if significant and increasing
            else "单调下降"
            if significant and decreasing
            else "无显著单调趋势"
        )
        gaps.append(
            {
                "gap_tokens": gap,
                "phi_by_target": {
                    str(target): value
                    for target, value in zip(TARGET_POSITIONS, values)
                },
                "position_range_ratio": ratio,
                "increments_ms": increments,
                "endpoint_relative_change": endpoint_change,
                "significant_monotonic_trend": significant,
                "trend": trend,
            }
        )
    grade = grade_gap_only_assumption(gaps)
    increasing_count = sum(
        row["trend"] == "单调增长" for row in gaps
    )
    consistency = (
        "YES"
        if increasing_count >= 2 and grade == "FAIL"
        else "WEAK"
        if increasing_count >= 1
        else "NO"
    )
    return {
        "gap_rows": gaps,
        "gap_only_assumption": grade,
        "step10d1_sublinear_consistency": consistency,
        "thresholds": {
            "pass_ratio": POSITION_RATIO_PASS_LIMIT,
            "weak_ratio": POSITION_RATIO_WEAK_LIMIT,
            "significant_trend_endpoint_change": SIGNIFICANT_TREND_LIMIT,
            "multiple_trends_for_fail": 2,
        },
    }


def grade_gap_only_assumption(
    gap_rows: Sequence[Mapping[str, object]],
) -> str:
    """按冻结范围比例和趋势规则判定 gap-only 假设。"""
    ratios = [float(row["position_range_ratio"]) for row in gap_rows]
    trends = sum(bool(row["significant_monotonic_trend"]) for row in gap_rows)
    if any(value > POSITION_RATIO_WEAK_LIMIT for value in ratios) or trends >= 2:
        return "FAIL"
    if any(value > POSITION_RATIO_PASS_LIMIT for value in ratios) or trends:
        return "WEAK"
    return "PASS"


def diagnostic_models(
    summary_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """比较通过原点的 gap-only 与最小双线性位置模型。"""
    observations = tuple(
        (
            int(row["target_position"]),
            int(row["gap_tokens"]),
            float(row["measured_phi_ms"]),
        )
        for row in summary_rows
        if int(row["gap_tokens"]) > 0
    )
    gap_only = _fit_gap_only(observations)
    position_aware = _fit_position_aware(observations)
    gap_predictions = {
        (target, gap): _predict_gap_only(gap_only, gap)
        for target, gap, _ in observations
    }
    position_predictions = {
        (target, gap): _predict_position_aware(
            position_aware,
            target,
            gap,
        )
        for target, gap, _ in observations
    }
    return {
        "status": "仅作维度诊断，未写回正式 Phi",
        "gap_only": {
            "formula": "cost = beta_gap * G",
            "coefficients": gap_only,
            **_model_metrics(observations, gap_predictions),
            "leave_one_position_out": _leave_one_position_out(
                observations,
                "gap_only",
            ),
        },
        "position_aware": {
            "formula": "cost = beta_gap * G + beta_position * G * (T / 32768)",
            "coefficients": position_aware,
            **_model_metrics(observations, position_predictions),
            "leave_one_position_out": _leave_one_position_out(
                observations,
                "position_aware",
            ),
        },
    }


def leave_one_position_out_splits() -> tuple[dict[str, object], ...]:
    """返回四个互斥且完整的留一位置划分。"""
    return tuple(
        {
            "held_out_target": held_out,
            "training_targets": tuple(
                target for target in TARGET_POSITIONS if target != held_out
            ),
        }
        for held_out in TARGET_POSITIONS
    )


def classify_discrepancy(
    legacy_grade: str,
    gap_only_grade: str,
) -> str:
    """按预注册组合规则分类历史差异来源。"""
    if legacy_grade == "PASS" and gap_only_grade == "FAIL":
        return "POSITION_DEPENDENCE"
    if legacy_grade != "PASS" and gap_only_grade in {"WEAK", "FAIL"}:
        return "MIXED"
    if legacy_grade == "FAIL" and gap_only_grade == "PASS":
        return "ENVIRONMENT_OR_PROTOCOL_DRIFT"
    return "UNRESOLVED"


def build_protocol_diff() -> dict[str, object]:
    """从源码与冻结 artifacts 构建逐项协议差异。"""
    step9d = json.loads(
        (STEP9D_DIRECTORY / "run_metadata.json").read_text(encoding="utf-8")
    )
    step10d1 = json.loads(
        (STEP10D1_DIRECTORY / "config.json").read_text(encoding="utf-8")
    )
    legacy_positions = {
        str(gap): {
            "T": LEGACY_TARGET_POSITION,
            "E": LEGACY_TARGET_POSITION - gap,
            "G": gap,
        }
        for gap in POSITION_GAPS
    }
    extended_positions = {
        str(gap): {"T": 131_072, "E": 131_072 - gap, "G": gap}
        for gap in POSITION_GAPS
    }
    return {
        "step9d": {
            "model": step9d["model"],
            "model_path": step9d["engine_configuration"]["model_path"],
            "model_revision": "artifact 中不可用",
            "sglang_version": step9d["sglang_version"],
            "container_image": "artifact 中不可用",
            "gpu": step9d["gpu"],
            "tp": step9d["tp"],
            "engine_configuration": step9d["engine_configuration"],
            "output_tokens": 1,
            "request_endpoint": "进程内 Engine.generate(stream=True)",
            "measurement_boundary": step9d["ttft_boundary"],
            "warmup": step9d["warmup_repetitions"],
            "measured": step9d["measured_repetitions"],
            "prompt_construction": (
                "make_tokens 的固定算术序列；checkpoint 基种子 51001，"
                "target suffix 种子 541019，suffix 长度 63"
            ),
            "positions": legacy_positions,
            "checkpoint_construction": "按候选 token_pos 递增构建同 lineage 节点",
            "fa_residency": "构建后保留全部 FA-KV",
            "recurrent_eviction": FORMAL_MUTATION_PRIMITIVE,
            "service_reset": "每个 trial 前 flush_cache 并验证空缓存",
        },
        "step10d1": {
            "model": step10d1["environment"]["model"],
            "model_path": step10d1["engine_configuration"]["model_path"],
            "model_revision": "artifact 中不可用",
            "sglang_version": step10d1["environment"]["sglang_version"],
            "container_image": "artifact 中不可用",
            "gpu": step10d1["environment"]["gpu"],
            "tp": step10d1["environment"]["tp"],
            "engine_configuration": step10d1["engine_configuration"],
            "output_tokens": 1,
            "request_endpoint": "进程内 Engine.generate(stream=True)",
            "measurement_boundary": "复用 Step 9D 的首个流式 token client-side 计时",
            "warmup": step10d1["warmup_repetitions"],
            "measured": step10d1["measured_repetitions"],
            "prompt_construction": (
                "make_tokens 的固定算术序列；checkpoint 基种子 51001，"
                "target suffix 种子 741019，suffix 长度 63"
            ),
            "positions": extended_positions,
            "checkpoint_construction": (
                "按候选 token_pos 递增构建，并显式清理 chunk-boundary 循环状态"
            ),
            "fa_residency": "构建后保留全部 FA-KV",
            "recurrent_eviction": FORMAL_MUTATION_PRIMITIVE,
            "service_reset": "每个 trial 前 flush_cache 并验证空缓存",
        },
        "key_differences": (
            "T 从固定 32768 改为固定 131072",
            "context_length 从 45056 提高到 131200",
            "target suffix 的确定性种子从 541019 改为 741019",
            "长上下文构造显式清理 chunk-boundary 循环状态",
        ),
    }


def render_protocol_diff(diff: Mapping[str, object]) -> str:
    """把协议差异写成可审阅的 Markdown。"""
    left = diff["step9d"]
    right = diff["step10d1"]
    fields = (
        ("模型", "model"),
        ("模型路径", "model_path"),
        ("模型 revision", "model_revision"),
        ("SGLang", "sglang_version"),
        ("容器镜像", "container_image"),
        ("GPU", "gpu"),
        ("TP", "tp"),
        ("输出 token", "output_tokens"),
        ("请求入口", "request_endpoint"),
        ("计时边界", "measurement_boundary"),
        ("warmup", "warmup"),
        ("measured", "measured"),
        ("prompt/token 构造", "prompt_construction"),
        ("checkpoint 构造", "checkpoint_construction"),
        ("FA-KV 构造", "fa_residency"),
        ("循环状态驱逐", "recurrent_eviction"),
        ("服务重置", "service_reset"),
    )
    lines = [
        "# Step 9D 与 Step 10D.1 协议差异",
        "",
        "| 项目 | Step 9D | Step 10D.1 |",
        "|---|---|---|",
    ]
    for label, key in fields:
        lines.append(f"| {label} | {left[key]} | {right[key]} |")
    lines.extend(("", "## 引擎参数", ""))
    keys = sorted(set(left["engine_configuration"]) | set(right["engine_configuration"]))
    lines.extend(("| 参数 | Step 9D | Step 10D.1 |", "|---|---:|---:|"))
    for key in keys:
        lines.append(
            f"| `{key}` | {left['engine_configuration'].get(key)} | "
            f"{right['engine_configuration'].get(key)} |"
        )
    lines.extend(("", "## T / E / G", "", "| G | Step 9D T/E | Step 10D.1 T/E |", "|---:|---:|---:|"))
    for gap in POSITION_GAPS:
        old = left["positions"][str(gap)]
        new = right["positions"][str(gap)]
        lines.append(f"| {gap} | {old['T']} / {old['E']} | {new['T']} / {new['E']} |")
    lines.extend(("", "## 关键差异", ""))
    lines.extend(f"- {item}" for item in diff["key_differences"])
    lines.append("")
    return "\n".join(lines)


def _historical_phi() -> dict[int, float]:
    """读取 Step 9D 冻结 calibration 汇总。"""
    with (STEP9D_DIRECTORY / "calibration_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = tuple(csv.DictReader(handle))
    values = {
        int(row["gap_tokens"]): float(row["measured_phi_ms"])
        for row in rows
    }
    if set(values) != set(POSITION_GAPS):
        raise RuntimeError("Step 9D 历史 calibration gap 不完整")
    return values


def _valid_measured_by_key(
    records: Sequence[Mapping[str, object]],
    key_function: object,
) -> dict[object, tuple[float, ...]]:
    """严格筛选通过所有门禁的 measured 样本。"""
    grouped: dict[object, list[float]] = {}
    for record in records:
        if record.get("is_warmup") is True:
            continue
        if not (
            record.get("status") == "PASS"
            and record.get("correctness_pass") is True
            and record.get("gap_match") is True
        ):
            raise ValueError("存在失败或 mismatch 的 measured trial")
        key = key_function(record)
        grouped.setdefault(key, []).append(float(record["ttft_ms"]))
    for key, values in grouped.items():
        if len(values) != MEASURED_REPETITIONS:
            raise ValueError(f"{key!r} 的 measured 数量异常：{len(values)}")
    return {key: tuple(values) for key, values in grouped.items()}


def _latency_statistics(values: Sequence[float]) -> dict[str, float]:
    """计算固定样本的均值、中位数、标准差和经验 P95。"""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("延迟样本不能为空")
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "std_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _count_records(
    records: Sequence[Mapping[str, object]],
    target: int,
    gap: int,
    warmup: bool,
) -> int:
    """统计指定矩阵单元的 warmup 或 measured 数量。"""
    return sum(
        int(record["target_H"]) == target
        and int(record["target_gap"]) == gap
        and record.get("is_warmup") is warmup
        for record in records
    )


def _failure_count(
    records: Sequence[Mapping[str, object]], target: int, gap: int
) -> int:
    """统计指定矩阵单元的失败 trial。"""
    return sum(
        int(record["target_H"]) == target
        and int(record["target_gap"]) == gap
        and record.get("status") != "PASS"
        for record in records
    )


def _mismatch_count(
    records: Sequence[Mapping[str, object]], target: int, gap: int
) -> int:
    """统计指定矩阵单元的 H/E/G mismatch。"""
    return sum(
        int(record["target_H"]) == target
        and int(record["target_gap"]) == gap
        and record.get("gap_match") is not True
        for record in records
    )


def _fit_gap_only(
    observations: Sequence[tuple[int, int, float]],
) -> dict[str, float]:
    """拟合通过原点的单变量 gap 线性模型。"""
    denominator = sum(float(gap * gap) for _, gap, _ in observations)
    if denominator == 0.0:
        raise ValueError("gap-only 拟合缺少非零 gap")
    beta = sum(gap * value for _, gap, value in observations) / denominator
    return {"beta_gap_ms_per_token": beta}


def _fit_position_aware(
    observations: Sequence[tuple[int, int, float]],
) -> dict[str, float]:
    """拟合 gap 与 gap-position 双线性诊断模型。"""
    rows = [
        (float(gap), float(gap) * target / 32_768.0, float(value))
        for target, gap, value in observations
    ]
    a11 = sum(x1 * x1 for x1, _, _ in rows)
    a12 = sum(x1 * x2 for x1, x2, _ in rows)
    a22 = sum(x2 * x2 for _, x2, _ in rows)
    b1 = sum(x1 * value for x1, _, value in rows)
    b2 = sum(x2 * value for _, x2, value in rows)
    determinant = a11 * a22 - a12 * a12
    if determinant == 0.0:
        raise ValueError("position-aware 设计矩阵奇异")
    beta_gap = (b1 * a22 - b2 * a12) / determinant
    beta_position = (a11 * b2 - a12 * b1) / determinant
    return {
        "beta_gap_ms_per_token": beta_gap,
        "beta_gap_position_ms_per_token": beta_position,
        "position_scale_tokens": 32_768.0,
    }


def _predict_gap_only(coefficients: Mapping[str, float], gap: int) -> float:
    """计算 gap-only 诊断预测。"""
    return float(coefficients["beta_gap_ms_per_token"]) * gap


def _predict_position_aware(
    coefficients: Mapping[str, float], target: int, gap: int
) -> float:
    """计算 position-aware 双线性诊断预测。"""
    return gap * (
        float(coefficients["beta_gap_ms_per_token"])
        + float(coefficients["beta_gap_position_ms_per_token"])
        * target
        / float(coefficients["position_scale_tokens"])
    )


def _model_metrics(
    observations: Sequence[tuple[int, int, float]],
    predictions: Mapping[tuple[int, int], float],
) -> dict[str, float]:
    """计算诊断模型的 MAE、MAPE、最大误差和 R²。"""
    actual = [value for _, _, value in observations]
    predicted = [predictions[(target, gap)] for target, gap, _ in observations]
    errors = [abs(left - right) for left, right in zip(actual, predicted)]
    mean_actual = statistics.fmean(actual)
    total = sum((value - mean_actual) ** 2 for value in actual)
    residual = sum((left - right) ** 2 for left, right in zip(actual, predicted))
    return {
        "mae_ms": statistics.fmean(errors),
        "mape_percent": statistics.fmean(
            error / abs(value) * 100.0 for error, value in zip(errors, actual)
        ),
        "max_absolute_error_ms": max(errors),
        "r_squared": 1.0 - residual / total,
    }


def _leave_one_position_out(
    observations: Sequence[tuple[int, int, float]],
    model_name: str,
) -> dict[str, object]:
    """用三个 T 训练并依次预测第四个 T。"""
    predictions: dict[tuple[int, int], float] = {}
    folds = []
    for split in leave_one_position_out_splits():
        held_out = int(split["held_out_target"])
        training = tuple(row for row in observations if row[0] != held_out)
        testing = tuple(row for row in observations if row[0] == held_out)
        if model_name == "gap_only":
            coefficients = _fit_gap_only(training)
            fold_predictions = {
                (target, gap): _predict_gap_only(coefficients, gap)
                for target, gap, _ in testing
            }
        elif model_name == "position_aware":
            coefficients = _fit_position_aware(training)
            fold_predictions = {
                (target, gap): _predict_position_aware(
                    coefficients, target, gap
                )
                for target, gap, _ in testing
            }
        else:
            raise ValueError(f"未知诊断模型：{model_name}")
        predictions.update(fold_predictions)
        folds.append(
            {
                **split,
                "coefficients": coefficients,
                **_model_metrics(testing, fold_predictions),
            }
        )
    return {"folds": folds, **_model_metrics(observations, predictions)}


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """按固定列顺序写出 CSV。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _csv_value(value: object) -> object:
    """把嵌套值稳定编码为 CSV 单元格。"""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _protected_hashes() -> dict[str, str]:
    """记录本步骤不得修改的冻结文件摘要。"""
    return {
        str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in PROTECTED_PATHS
    }


def _server_command_text() -> str:
    """记录两个阶段的进程内引擎配置。"""
    legacy = " ".join(
        f"{key}={value!r}"
        for key, value in sorted(STEP8E_ENGINE_CONFIGURATION.items())
    )
    matrix = " ".join(
        f"{key}={value!r}"
        for key, value in sorted(ENGINE_CONFIGURATION_128K.items())
    )
    return (
        "进程入口：python -m evaluation.recovery_position_audit\n"
        f"Legacy 引擎参数：{legacy}\n"
        f"Position matrix 引擎参数：{matrix}\n"
        "两个阶段均使用正式 Mamba-only eviction primitive。\n"
    )


def _render_readme(config: Mapping[str, object]) -> str:
    """生成说明实验边界与证据用途的中文 README。"""
    return "\n".join(
        (
            "# Recovery Cost Context-Position Audit",
            "",
            "本目录审计 recovery cost 是否只由 gap G 决定，或还依赖绝对目标位置 T。",
            "",
            "Legacy 阶段直接调用 Step 9D 原 profiler implementation；position matrix 阶段复用 Step 10D.1 的正式 runtime recovery 路径。每个 trial 均验证真实 H/E/G、FA-KV、循环状态和树结构。",
            "",
            "每个 T 使用自己的 G=0 baseline。所有诊断模型只用于维度审计，不会写回正式 Phi，也不读取任何 policy selection 或 policy performance。",
            "",
            f"运行状态：{config.get('status', '未知')}。",
            "",
        )
    )


def _execute_legacy_phase(
    writer: AuditArtifactWriter,
    schedule: Sequence[object],
) -> tuple[list[dict[str, object]], str | None]:
    """用 Step 9D 原实现执行旧协议复现阶段。"""
    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import FormalEndToEndGateEngine, requested_control_port
    from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport

    records: list[dict[str, object]] = []
    failure = None
    engine = None
    try:
        engine = FormalEndToEndGateEngine(**STEP8E_ENGINE_CONFIGURATION)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        for index, case in enumerate(schedule, start=1):
            phase = "WARMUP" if case.is_warmup else "MEASURED"
            print(
                f"[STEP10D2-LEGACY] {index}/{len(schedule)} {phase} "
                f"gap={case.target_gap}",
                flush=True,
            )
            record = execute_legacy_case(case, engine=engine, client=client)
            record = {"source": "Step 9D 原实现的当前环境复现", **record}
            records.append(record)
            writer.append_record("legacy_reproduction_raw.csv", record)
            if record["status"] != "PASS":
                failure = (
                    f"legacy case {case.case_id} 失败："
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
                failure = failure or f"关闭 legacy runtime 失败：{error!r}"
    return records, failure


def _execute_matrix_phase(
    writer: AuditArtifactWriter,
    schedule: Sequence[MatrixCase],
) -> tuple[list[dict[str, object]], str | None]:
    """在统一 128K 配置中执行完整 T/G 位置矩阵。"""
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
                f"[STEP10D2-MATRIX] {index}/{len(schedule)} {phase} "
                f"T={case.target_position} G={case.target_gap}",
                flush=True,
            )
            profile_case = PositionProfileCase(
                case_id=case.case_id,
                target_gap=case.target_gap,
                target_frontier=case.target_frontier,
                repetition=case.repetition,
                is_warmup=case.is_warmup,
                gap_order_position=case.pair_order_position,
                execution_order_position=case.execution_order_position,
            )
            record = execute_position_case(
                profile_case,
                engine=engine,
                client=client,
                target_position=case.target_position,
                namespace_prefix="flowstate_step10d2",
                suffix_seed=MATRIX_SUFFIX_SEED,
            )
            record = {"source": "Step 10D.2 新测量", **record}
            records.append(record)
            writer.append_record("position_matrix_raw.csv", record)
            if record["status"] != "PASS":
                failure = (
                    f"matrix case {case.case_id} 失败："
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
                failure = failure or f"关闭 matrix runtime 失败：{error!r}"
    return records, failure


def main() -> int:
    """依次完成协议审计、legacy 复现和完整位置矩阵。"""
    writer = AuditArtifactWriter.create()
    legacy_schedule = build_legacy_schedule()
    matrix_schedule = build_matrix_schedule()
    protected_before = _protected_hashes()
    protocol_diff = build_protocol_diff()
    config: dict[str, object] = {
        "schema_version": "flowstate.recovery_position_audit.v1",
        "status": "RUNNING",
        "target_positions": list(TARGET_POSITIONS),
        "gap_points": list(POSITION_GAPS),
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "seed": ORDER_SEED,
        "legacy_engine_configuration": STEP8E_ENGINE_CONFIGURATION,
        "matrix_engine_configuration": ENGINE_CONFIGURATION_128K,
        "position_gate_thresholds": {
            "pass": POSITION_RATIO_PASS_LIMIT,
            "weak": POSITION_RATIO_WEAK_LIMIT,
            "trend": SIGNIFICANT_TREND_LIMIT,
        },
        "step10d1_raw_reused": False,
        "policy_comparison_executed": False,
        "policy_result_used": False,
        "start_time": datetime.now().astimezone().isoformat(),
    }
    writer.write_text("protocol_diff.md", render_protocol_diff(protocol_diff))
    writer.write_text("server_command.txt", _server_command_text())
    writer.write_json(
        "execution_order.json",
        {
            "seed": ORDER_SEED,
            "legacy_cases": [case.__dict__ for case in legacy_schedule],
            "matrix_cases": [case.__dict__ for case in matrix_schedule],
        },
    )
    failure = None
    failure_stage = "环境检查"
    legacy_records: list[dict[str, object]] = []
    matrix_records: list[dict[str, object]] = []
    started = time.perf_counter_ns()
    try:
        environment = _collect_environment()
        model_config, tokenizer_config = _load_context_configuration(MODEL_PATH)
        context = validate_context_capabilities(model_config, tokenizer_config)
        config["environment"] = environment
        config["context_capabilities"] = context
        config["sglang_version"] = importlib.metadata.version("sglang")
        writer.write_text("environment.txt", _environment_text(environment))
        writer.write_json("config.json", config)

        failure_stage = "Legacy reproduction"
        legacy_records, failure = _execute_legacy_phase(writer, legacy_schedule)
        if failure is not None:
            raise RuntimeError(failure)

        failure_stage = "Position matrix"
        matrix_records, failure = _execute_matrix_phase(writer, matrix_schedule)
        if failure is not None:
            raise RuntimeError(failure)

        failure_stage = "离线汇总与维度诊断"
        legacy = summarize_legacy(legacy_records, _historical_phi())
        matrix_summary = summarize_position_matrix(matrix_records)
        dependence = analyze_position_dependence(matrix_summary)
        models = diagnostic_models(matrix_summary)
        classification = classify_discrepancy(
            str(legacy["grade"]),
            str(dependence["gap_only_assumption"]),
        )
        ready = classification != "UNRESOLVED"
        _write_csv(
            writer.directory / "legacy_reproduction.csv",
            tuple(legacy["rows"]),
            (
                "gap_tokens",
                "mean_ms",
                "median_ms",
                "std_ms",
                "p95_ms",
                "min_ms",
                "max_ms",
                "legacy_measured_phi_ms",
                "historical_phi_ms",
                "absolute_error_ms",
                "relative_error",
            ),
        )
        _write_csv(
            writer.directory / "position_matrix_summary.csv",
            matrix_summary,
            (
                "target_position",
                "executable_frontier",
                "gap_tokens",
                "warmup_count",
                "valid_measured_count",
                "failure_count",
                "mismatch_count",
                "mean_ms",
                "median_ms",
                "std_ms",
                "p95_ms",
                "min_ms",
                "max_ms",
                "position_baseline_mean_ms",
                "measured_phi_ms",
            ),
        )
        dependence = {
            **dependence,
            "legacy_reproduction": legacy["grade"],
            "discrepancy_classification": classification,
            "ready_to_freeze_recovery_model": ready,
        }
        writer.write_json("position_dependence.json", dependence)
        writer.write_json("model_diagnostics.json", models)
        config["analysis"] = {
            "legacy_reproduction": legacy["grade"],
            "gap_only_assumption": dependence["gap_only_assumption"],
            "discrepancy_classification": classification,
            "ready_to_freeze_recovery_model": ready,
        }
    except Exception as error:
        failure = repr(error)
        traceback.print_exc()
    finally:
        protected_after = _protected_hashes()
        correctness = (
            failure is None
            and len(legacy_records) == EXPECTED_LEGACY_TRIALS
            and len(matrix_records) == EXPECTED_MATRIX_TRIALS
            and all(record.get("status") == "PASS" for record in legacy_records)
            and all(record.get("status") == "PASS" for record in matrix_records)
        )
        config.update(
            {
                "status": "PASS" if correctness else "FAIL",
                "end_time": datetime.now().astimezone().isoformat(),
                "total_runtime_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                "legacy_completed_trials": len(legacy_records),
                "matrix_completed_trials": len(matrix_records),
                "failure_stage": None if correctness else failure_stage,
                "fatal_error": failure,
                "protected_hashes_before": protected_before,
                "protected_hashes_after": protected_after,
                "gates": {
                    "recovery_semantic_correctness": correctness,
                    "position_matrix_correctness": correctness,
                    "formal_phi_unchanged": (
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
                "legacy_trials": len(legacy_records),
                "matrix_trials": len(matrix_records),
                "failure_stage": config["failure_stage"],
                "fatal_error": config["fatal_error"],
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
