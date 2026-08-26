#!/usr/bin/env python3
"""把已验证的独立恢复测量路径扩展到 128K。"""

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
from typing import Callable, Mapping, Sequence

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    ENGINE_CONFIGURATION,
    PENDING_SUFFIX_LENGTH,
    build_runtime_handles,
    build_runtime_workflows,
    compact_state,
    inspect_after_allocation,
    inspect_checkpoint,
    make_tokens,
    path_state,
    query_runtime_metrics,
    wait_for_transport,
)
from evaluation.controlled_multiworkflow_v1.scenario import (
    CHECKPOINT_SIZE_BYTES,
    CheckpointRecency,
    ControlledScenario,
    WorkflowSpec,
    WorkloadMetadata,
)
from evaluation.controlled_multiworkflow_v1.snapshot_runtime import (
    SnapshotSchedulerRuntimeAdapter,
    allocation_safety_snapshot,
    validate_clean_cache,
)
from evaluation.recovery_profiler_v2.profile_runner import (
    FixedSelectionOptimizer,
    _no_mutation_safety,
)
from evaluation.sota_latency_runtime import measure_streaming_request
from flowstate.controller import StateController
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
MODEL_PATH = Path("/model")
FIXED_GAP_POINTS = (
    0,
    4_096,
    8_192,
    16_384,
    32_768,
    49_152,
    65_536,
    98_304,
    131_072,
)
SHORT_RANGE_GAPS = FIXED_GAP_POINTS[:5]
LONG_RANGE_GAPS = FIXED_GAP_POINTS[5:]
LONG_SHAPE_GAPS = (32_768, 49_152, 65_536, 98_304, 131_072)
PIECEWISE_DIAGNOSTIC_KNOTS = (0, 32_768, 65_536, 131_072)
WARMUP_REPETITIONS = 2
MEASURED_REPETITIONS = 12
ORDER_SEED = 20_260_826
ANCHOR_POS = 131_072
TARGET_REQUEST_INPUT_TOKENS = ANCHOR_POS + 1 + PENDING_SUFFIX_LENGTH
EFFECTIVE_CONTEXT_LENGTH = 131_200
EXPECTED_SGLANG_VERSION = "0.5.17"
PROFILE_WORKFLOW_ID = "RECOVERY_PROFILE_128K"
PROFILE_ROOT_LINEAGE = "PROFILE_ROOT"
PROFILE_CONTINUATION_ID = "RECOVERY_PROFILE_128K-B"
DEEP_CHECKPOINT_ID = "PROFILE_DEEP"
SHALLOW_CHECKPOINT_ID = "PROFILE_SHALLOW"
RUNTIME_BUILD_CHUNK_TOKENS = int(
    ENGINE_CONFIGURATION["chunked_prefill_size"]
)
FORMAL_MUTATION_PRIMITIVE = (
    "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
)
ENGINE_CONFIGURATION_128K = {
    **ENGINE_CONFIGURATION,
    "context_length": EFFECTIVE_CONTEXT_LENGTH,
    "chunked_prefill_size": ENGINE_CONFIGURATION["chunked_prefill_size"],
    "max_mamba_cache_size": 24,
}
RAW_TRIAL_FIELDS = (
    "case_id",
    "target_gap",
    "target_H",
    "target_E",
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
    REPOSITORY_ROOT / "evaluation" / "controlled_multiworkflow_v1" / "scenario.py",
    REPOSITORY_ROOT / "evaluation" / "controlled_multiworkflow_v1" / "policies.py",
    REPOSITORY_ROOT / "evaluation" / "scalable_multiworkflow_v2" / "scenario.py",
    REPOSITORY_ROOT / "evaluation" / "sota_policies.py",
    REPOSITORY_ROOT / "evaluation" / "sota_metadata.py",
    REPOSITORY_ROOT / "motivation" / "README.md",
)


@dataclass(frozen=True)
class ProfileCase:
    """描述一个独立快照恢复测量 case。"""

    case_id: str
    target_gap: int
    target_frontier: int
    repetition: int
    is_warmup: bool
    gap_order_position: int
    execution_order_position: int


@dataclass
class ProfilerArtifactWriter:
    """增量保存 128K profiler 的完整证据。"""

    directory: Path

    @classmethod
    def create(
        cls,
        root: Path = ARTIFACT_ROOT,
        timestamp: str | None = None,
    ) -> "ProfilerArtifactWriter":
        """创建不会覆盖既有实验的时间戳目录。"""
        resolved_timestamp = timestamp or datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        directory = root / f"recovery_profiler_128k_{resolved_timestamp}"
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory=directory)

    @property
    def raw_trials_path(self) -> Path:
        """返回逐 trial CSV 路径。"""
        return self.directory / "raw_trials.csv"

    def append_trial(self, record: Mapping[str, object]) -> None:
        """在每个 trial 结束后立即写入扁平化记录。"""
        is_new = not self.raw_trials_path.exists()
        with self.raw_trials_path.open(
            "a", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_TRIAL_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(
                {
                    field: _csv_value(record.get(field))
                    for field in RAW_TRIAL_FIELDS
                }
            )

    def write_json(self, name: str, payload: object) -> None:
        """稳定写出 UTF-8 JSON。"""
        path = self.directory / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def write_text(self, name: str, text: str) -> None:
        """写出 UTF-8 文本文件。"""
        (self.directory / name).write_text(text, encoding="utf-8")

    def ensure_required_files(self) -> None:
        """确保失败运行也保留全部约定文件。"""
        if not self.raw_trials_path.exists():
            with self.raw_trials_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                csv.DictWriter(handle, fieldnames=RAW_TRIAL_FIELDS).writeheader()
        for name in ("summary.csv", "gap_audit.csv"):
            path = self.directory / name
            if not path.exists():
                path.write_text("\n", encoding="utf-8")
        if not (self.directory / "fit_diagnostics.json").exists():
            self.write_json("fit_diagnostics.json", {"status": "不可用"})


def build_profile_schedule(seed: int = ORDER_SEED) -> tuple[ProfileCase, ...]:
    """构建固定种子的平衡循环 gap 顺序。"""
    base_order = list(FIXED_GAP_POINTS)
    random.Random(seed).shuffle(base_order)
    if tuple(base_order) == FIXED_GAP_POINTS:
        raise RuntimeError("gap 基序列不能退化为递增顺序")
    cases = []
    execution_position = 0
    cycle_index = 0
    for is_warmup, repetitions in (
        (True, WARMUP_REPETITIONS),
        (False, MEASURED_REPETITIONS),
    ):
        for repetition in range(repetitions):
            offset = cycle_index % len(base_order)
            order = base_order[offset:] + base_order[:offset]
            for gap_position, gap in enumerate(order):
                phase = "warmup" if is_warmup else "measured"
                cases.append(
                    ProfileCase(
                        case_id=(
                            f"profile128k_{phase}_r{repetition:02d}_"
                            f"p{gap_position:02d}_g{gap}"
                        ),
                        target_gap=gap,
                        target_frontier=ANCHOR_POS - gap,
                        repetition=repetition,
                        is_warmup=is_warmup,
                        gap_order_position=gap_position,
                        execution_order_position=execution_position,
                    )
                )
                execution_position += 1
            cycle_index += 1
    return tuple(cases)


def build_profile_scenario(
    target_gap: int,
) -> tuple[ControlledScenario, tuple[str, ...]]:
    """固定 H=128K，并仅通过循环状态驻留改变 E。"""
    if target_gap not in FIXED_GAP_POINTS:
        raise ValueError(f"未冻结的 profiling gap：{target_gap}")
    return build_position_scenario(ANCHOR_POS, target_gap)


def build_position_scenario(
    target_position: int,
    target_gap: int,
) -> tuple[ControlledScenario, tuple[str, ...]]:
    """在指定绝对目标位置构造仅改变循环状态前沿的快照。"""
    if target_position <= 0:
        raise ValueError("target_position 必须大于零")
    if target_gap < 0 or target_gap > target_position:
        raise ValueError("target_gap 必须位于零和 target_position 之间")
    target_frontier = target_position - target_gap
    workflow = WorkflowSpec(
        workflow_id=PROFILE_WORKFLOW_ID,
        root_lineage=PROFILE_ROOT_LINEAGE,
        anchor_pos=target_position,
        pending_branches=("B",),
    )
    continuation = PendingContinuation(
        continuation_id=PROFILE_CONTINUATION_ID,
        workflow_id=PROFILE_WORKFLOW_ID,
        lineage_path=(PROFILE_ROOT_LINEAGE, "B"),
        anchor_pos=target_position,
        resident_fa_frontier=target_position,
    )
    candidate_specs = []
    if 0 < target_frontier < target_position:
        candidate_specs.append((SHALLOW_CHECKPOINT_ID, target_frontier))
    cleanup_pos = target_frontier + RUNTIME_BUILD_CHUNK_TOKENS
    while cleanup_pos < target_position:
        candidate_specs.append(
            (f"PROFILE_CLEANUP_{cleanup_pos}", cleanup_pos)
        )
        cleanup_pos += RUNTIME_BUILD_CHUNK_TOKENS
    candidate_specs.append((DEEP_CHECKPOINT_ID, target_position))
    candidates = tuple(
        CheckpointCandidate(
            checkpoint_id=checkpoint_id,
            workflow_id=PROFILE_WORKFLOW_ID,
            lineage_path=(PROFILE_ROOT_LINEAGE,),
            token_pos=token_pos,
            memory_bytes=CHECKPOINT_SIZE_BYTES,
        )
        for checkpoint_id, token_pos in candidate_specs
    )
    recency = tuple(
        CheckpointRecency(
            checkpoint_id=checkpoint_id,
            creation_order=index,
            last_access_order=index,
        )
        for index, (checkpoint_id, _) in enumerate(
            candidate_specs,
            start=1,
        )
    )
    if target_gap == 0:
        selected_ids = (DEEP_CHECKPOINT_ID,)
    elif target_frontier > 0:
        selected_ids = (SHALLOW_CHECKPOINT_ID,)
    else:
        selected_ids = ()
    budget_checkpoints = len(selected_ids)
    metadata = WorkloadMetadata(
        workflows=(workflow,),
        checkpoint_recency=recency,
        workflow_order=(PROFILE_WORKFLOW_ID,),
        checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
        budget_checkpoints=budget_checkpoints,
    )
    return (
        ControlledScenario(
            continuations=(continuation,),
            candidates=candidates,
            budget_bytes=budget_checkpoints * CHECKPOINT_SIZE_BYTES,
            metadata=metadata,
        ),
        selected_ids,
    )


def validate_runtime_gap(
    target_gap: int,
    metrics: Mapping[str, object],
    target_position: int = ANCHOR_POS,
) -> dict[str, int]:
    """严格验证真实 H、E、G 与冻结目标逐 token 相等。"""
    runtime_h = int(metrics["physical_fa_hit"])
    runtime_e = int(metrics["executable_prefix"])
    runtime_g = int(metrics["replay_gap"])
    expected_e = target_position - target_gap
    if runtime_h - runtime_e != runtime_g:
        raise RuntimeError("runtime recovery gap 不等于 H-E")
    if runtime_h != target_position:
        raise RuntimeError(
            f"runtime H 不匹配：{runtime_h} != {target_position}"
        )
    if runtime_e != expected_e:
        raise RuntimeError(f"runtime E 不匹配：{runtime_e} != {expected_e}")
    if runtime_g != target_gap:
        raise RuntimeError(f"runtime G 不匹配：{runtime_g} != {target_gap}")
    return {
        "runtime_H": runtime_h,
        "runtime_E": runtime_e,
        "runtime_G": runtime_g,
    }


def validate_context_capabilities(
    model_config: Mapping[str, object],
    tokenizer_config: Mapping[str, object],
    effective_context_length: int = EFFECTIVE_CONTEXT_LENGTH,
) -> dict[str, object]:
    """验证模型、tokenizer 与服务 admission 均覆盖正式请求。"""
    text_config = model_config.get("text_config")
    if not isinstance(text_config, Mapping):
        raise ValueError("模型配置缺少 text_config")
    model_max = int(text_config.get("max_position_embeddings", 0))
    tokenizer_max = int(tokenizer_config.get("model_max_length", 0))
    required = TARGET_REQUEST_INPUT_TOKENS + 1
    if model_max < required:
        raise ValueError(f"模型原生上下文不足：{model_max} < {required}")
    if tokenizer_max < required:
        raise ValueError(
            f"tokenizer 原生上下文不足：{tokenizer_max} < {required}"
        )
    if effective_context_length < required:
        raise ValueError(
            "SGLang 有效上下文不足："
            f"{effective_context_length} < {required}"
        )
    return {
        "model_advertised_max_context": model_max,
        "tokenizer_model_max_length": tokenizer_max,
        "sglang_effective_max_context": effective_context_length,
        "formal_anchor_tokens": ANCHOR_POS,
        "formal_request_input_tokens": TARGET_REQUEST_INPUT_TOKENS,
        "formal_request_with_output_tokens": required,
        "admission_limit_changed_from_step9d": (
            effective_context_length
            != int(ENGINE_CONFIGURATION["context_length"])
        ),
        "model_semantics_changed": False,
        "cache_policy_changed": False,
    }


def validate_feasibility_response(
    metadata: Mapping[str, object],
    requested_input_tokens: int = ANCHOR_POS,
) -> dict[str, object]:
    """拒绝截断、静默裁剪、回撤或异常输出的 128K 请求。"""
    prompt_tokens = int(metadata.get("prompt_tokens", -1))
    completion_tokens = int(metadata.get("completion_tokens", -1))
    retractions = int(metadata.get("num_retractions", 0) or 0)
    truncated = prompt_tokens != requested_input_tokens
    if truncated:
        raise RuntimeError(
            "128K feasibility request 发生截断或静默裁剪："
            f"prompt_tokens={prompt_tokens}"
        )
    if completion_tokens != 1:
        raise RuntimeError("128K feasibility request 输出长度异常")
    if retractions != 0:
        raise RuntimeError("128K feasibility request 发生意外回撤")
    return {
        "request_completed": True,
        "requested_input_tokens": requested_input_tokens,
        "reported_prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "oom": False,
        "truncation": False,
        "silent_clipping": False,
        "output_normal": True,
    }


def summarize_trials(
    records: Sequence[Mapping[str, object]],
    old_model: RecoveryCostModel | None = None,
) -> dict[str, object]:
    """汇总有效 measured trial，并生成误差与形状审计。"""
    grouped = {gap: [] for gap in FIXED_GAP_POINTS}
    counts = {
        gap: {
            "warmup_count": 0,
            "valid_measured_count": 0,
            "failure_count": 0,
            "mismatch_count": 0,
        }
        for gap in FIXED_GAP_POINTS
    }
    for record in records:
        gap = int(record["target_gap"])
        if gap not in grouped:
            raise ValueError(f"出现未冻结 gap：{gap}")
        if record.get("is_warmup") is True:
            counts[gap]["warmup_count"] += 1
        if record.get("status") != "PASS":
            counts[gap]["failure_count"] += 1
        if record.get("gap_match") is False:
            counts[gap]["mismatch_count"] += 1
        if (
            record.get("is_warmup") is False
            and record.get("status") == "PASS"
            and record.get("correctness_pass") is True
            and record.get("gap_match") is True
        ):
            value = float(record["ttft_ms"])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"gap={gap} 存在无效 TTFT")
            grouped[gap].append(value)
            counts[gap]["valid_measured_count"] += 1

    complete = all(
        counts[gap]["warmup_count"] == WARMUP_REPETITIONS
        and counts[gap]["valid_measured_count"] == MEASURED_REPETITIONS
        and counts[gap]["failure_count"] == 0
        and counts[gap]["mismatch_count"] == 0
        for gap in FIXED_GAP_POINTS
    )
    baseline_mean = (
        statistics.fmean(grouped[0])
        if len(grouped[0]) == MEASURED_REPETITIONS
        else None
    )
    rows = []
    for gap in FIXED_GAP_POINTS:
        values = tuple(grouped[gap])
        stats = _latency_statistics(values)
        measured_phi = None
        if baseline_mean is not None and stats["mean_ms"] is not None:
            measured_phi = (
                0.0 if gap == 0 else float(stats["mean_ms"]) - baseline_mean
            )
        rows.append(
            {
                "gap_tokens": gap,
                **counts[gap],
                **stats,
                "measured_phi_ms": measured_phi,
            }
        )
    result: dict[str, object] = {
        "gap_rows": rows,
        "all_fixed_points_measured": complete,
        "total_records": len(records),
        "expected_records": len(FIXED_GAP_POINTS)
        * (WARMUP_REPETITIONS + MEASURED_REPETITIONS),
        "gap_mismatch_count": sum(
            row["mismatch_count"] for row in counts.values()
        ),
    }
    if complete:
        by_gap = {
            int(row["gap_tokens"]): float(row["measured_phi_ms"])
            for row in rows
        }
        model = old_model or RecoveryCostModel()
        result["old_phi_audit"] = old_phi_audit(by_gap, model)
        result["long_gap_shape"] = audit_long_gap_shape(by_gap)
        result["fit_diagnostics"] = diagnostic_fits(by_gap)
    else:
        result["old_phi_audit"] = None
        result["long_gap_shape"] = None
        result["fit_diagnostics"] = {"status": "测量不完整，未执行诊断拟合"}
    return result


def old_phi_audit(
    measured_phi_by_gap: Mapping[int, float],
    old_model: object,
) -> dict[str, object]:
    """分别审计旧 Phi 校准区间与 32K 以上外推。"""
    point_rows = []
    for gap in FIXED_GAP_POINTS:
        measured = float(measured_phi_by_gap[gap])
        predicted = float(old_model.estimate(gap))
        absolute_error = abs(predicted - measured)
        relative_error = (
            None if gap == 0 or measured == 0.0
            else absolute_error / abs(measured) * 100.0
        )
        point_rows.append(
            {
                "gap_tokens": gap,
                "range": (
                    "既有校准区间" if gap <= 32_768 else "旧模型外推区间"
                ),
                "measured_phi_ms": measured,
                "old_phi_ms": predicted,
                "absolute_error_ms": absolute_error,
                "relative_error_percent": relative_error,
                "validated_prediction": gap <= 32_768,
            }
        )
    return {
        "points": point_rows,
        "within_32k": _error_summary(
            tuple(row for row in point_rows if 0 < row["gap_tokens"] <= 32_768)
        ),
        "above_32k_extrapolation": _error_summary(
            tuple(row for row in point_rows if row["gap_tokens"] > 32_768)
        ),
    }


def audit_long_gap_shape(
    measured_phi_by_gap: Mapping[int, float],
) -> dict[str, object]:
    """计算长 gap 相邻区间斜率并按预注册阈值描述形状。"""
    values = [float(measured_phi_by_gap[gap]) for gap in LONG_SHAPE_GAPS]
    monotonic = all(
        current >= previous for previous, current in zip(values, values[1:])
    )
    slopes = []
    for lower, upper in zip(LONG_SHAPE_GAPS, LONG_SHAPE_GAPS[1:]):
        slope = (
            float(measured_phi_by_gap[upper])
            - float(measured_phi_by_gap[lower])
        ) / ((upper - lower) / 1_024)
        slopes.append(
            {
                "lower_gap_tokens": lower,
                "upper_gap_tokens": upper,
                "ms_per_ki_token": slope,
            }
        )
    slope_values = [row["ms_per_ki_token"] for row in slopes]
    slope_mean = statistics.fmean(slope_values)
    slope_cv = (
        statistics.pstdev(slope_values) / abs(slope_mean)
        if slope_mean != 0.0
        else math.inf
    )
    if monotonic and slope_cv <= 0.10:
        approximately_linear = "YES"
    elif monotonic and slope_cv <= 0.25:
        approximately_linear = "WEAK"
    else:
        approximately_linear = "NO"
    if approximately_linear == "YES":
        shape = "approximately-linear"
    elif slope_values[-1] > slope_values[0] * 1.10:
        shape = "super-linear tendency"
    elif slope_values[-1] < slope_values[0] * 0.90:
        shape = "sub-linear tendency"
    else:
        shape = "irregular"
    return {
        "slopes": slopes,
        "monotonic": monotonic,
        "slope_mean_ms_per_ki_token": slope_mean,
        "slope_coefficient_of_variation": slope_cv,
        "approximately_linear": approximately_linear,
        "shape": shape,
        "classification_rule": (
            "斜率变异系数不超过 10% 为 YES，不超过 25% 为 WEAK，否则为 NO"
        ),
    }


def diagnostic_fits(
    measured_phi_by_gap: Mapping[int, float],
) -> dict[str, object]:
    """计算不会写回正式 Phi 的线性与固定 knots 分段诊断。"""
    denominator = sum(float(gap * gap) for gap in FIXED_GAP_POINTS if gap)
    slope = sum(
        gap * float(measured_phi_by_gap[gap])
        for gap in FIXED_GAP_POINTS
        if gap
    ) / denominator
    linear_predictions = {
        gap: slope * gap for gap in FIXED_GAP_POINTS
    }
    piecewise_predictions = {
        gap: _piecewise_estimate(
            gap,
            {
                knot: float(measured_phi_by_gap[knot])
                for knot in PIECEWISE_DIAGNOSTIC_KNOTS
            },
        )
        for gap in FIXED_GAP_POINTS
    }
    actual = {gap: float(measured_phi_by_gap[gap]) for gap in FIXED_GAP_POINTS}
    return {
        "status": "仅作诊断，未写回正式 Phi",
        "linear": {
            "slope_ms_per_token": slope,
            "within_32k": _prediction_metrics(actual, linear_predictions, SHORT_RANGE_GAPS),
            "full_128k": _prediction_metrics(actual, linear_predictions, FIXED_GAP_POINTS),
        },
        "piecewise_linear": {
            "fixed_knots": list(PIECEWISE_DIAGNOSTIC_KNOTS),
            "within_32k": _prediction_metrics(actual, piecewise_predictions, SHORT_RANGE_GAPS),
            "full_128k": _prediction_metrics(actual, piecewise_predictions, FIXED_GAP_POINTS),
        },
    }


def execute_feasibility_gate(engine: object, client: object) -> dict[str, object]:
    """在正式 trial 前执行一次精确 128K 请求门禁。"""
    engine.flush_cache()
    clean = validate_clean_cache(client.census("step10d1:feasibility:after_flush"))
    token_ids = make_tokens(701_003, ANCHOR_POS)
    timing = measure_streaming_request(
        engine,
        request_id="flowstate_step10d1_feasibility_128k",
        token_ids=token_ids,
    )
    validation = validate_feasibility_response(timing["server_metadata"])
    engine.flush_cache()
    validate_clean_cache(client.census("step10d1:feasibility:after_cleanup"))
    return {
        **validation,
        "clean_cache_before": all(clean.values()),
        "ttft_ms_diagnostic_only": timing["ttft_ms"],
        "request_latency_ms_diagnostic_only": timing["request_latency_ms"],
    }


def execute_profile_case(
    case: ProfileCase,
    *,
    engine: object,
    client: object,
    target_position: int = ANCHOR_POS,
    namespace_prefix: str = "flowstate_step10d1",
    suffix_seed: int = 741_019,
) -> dict[str, object]:
    """复用 Step 9D 路径执行一个指定绝对位置的快照 trial。"""
    if case.target_frontier != target_position - case.target_gap:
        raise ValueError("case 的 E、G、T 关系不成立")
    scenario, expected_selected_ids = build_position_scenario(
        target_position,
        case.target_gap,
    )
    namespace = f"{namespace_prefix}_{case.case_id}"
    record: dict[str, object] = {
        "case_id": case.case_id,
        "target_gap": case.target_gap,
        "target_H": target_position,
        "target_E": case.target_frontier,
        "repetition": case.repetition,
        "is_warmup": case.is_warmup,
        "gap_order_position": case.gap_order_position,
        "execution_order_position": case.execution_order_position,
        "status": "FAIL",
        "correctness_pass": False,
        "gap_match": False,
        "fa_preserved": False,
        "safety_pass": False,
    }
    stage = "刷新并验证空缓存"
    try:
        engine.flush_cache()
        record["clean_cache"] = validate_clean_cache(
            client.census(f"{namespace}:after_flush")
        )

        stage = "构建与验证指定位置的循环检查点"
        runtime_workflows, candidate_tokens = build_runtime_workflows(
            engine,
            scenario,
            request_namespace=f"{namespace}_build",
        )
        handles, before_paths, before_states = build_runtime_handles(
            client,
            scenario,
            candidate_tokens,
        )
        before_census = client.census(f"{namespace}:before_allocation")
        if before_states[DEEP_CHECKPOINT_ID]["token_pos"] != target_position:
            raise RuntimeError("深层 checkpoint 的精确前缀长度不匹配")
        if not all(
            state["fa_resident"] and state["mamba_resident"]
            for state in before_states.values()
        ):
            raise RuntimeError("计时前候选驻留状态不完整")

        stage = "执行预注册循环状态保留与驱逐"
        runtime_adapter = SnapshotSchedulerRuntimeAdapter(
            client,
            case.case_id,
            namespace_prefix,
        )
        allocation = StateController(
            FixedSelectionOptimizer(expected_selected_ids),
            runtime_adapter,
        ).reconcile(
            scenario.continuations,
            scenario.candidates,
            handles,
            scenario.budget_bytes,
        )
        selected_ids = tuple(
            candidate.checkpoint_id for candidate in allocation.selected
        )
        evicted_ids = tuple(runtime_adapter.evicted_checkpoint_ids)
        if selected_ids != expected_selected_ids:
            raise RuntimeError("运行时保留集合偏离预注册 gap 状态")
        record["selected_checkpoint_ids"] = selected_ids
        record["evicted_checkpoint_ids"] = evicted_ids

        stage = "验证分配后状态与 FA 安全条件"
        after_paths, after_states = inspect_after_allocation(
            client,
            candidate_tokens,
        )
        after_census = client.census(f"{namespace}:after_allocation")
        if evicted_ids:
            safety, allocator = allocation_safety_snapshot(
                before_paths=before_paths,
                after_paths=after_paths,
                after_states=after_states,
                selected_ids=selected_ids,
                evicted_ids=evicted_ids,
                eviction_responses=runtime_adapter.eviction_responses,
            )
        else:
            safety, allocator = _no_mutation_safety(
                client=client,
                candidate_tokens=candidate_tokens,
                before_paths=before_paths,
                before_states=before_states,
                after_paths=after_paths,
                after_states=after_states,
                before_census=before_census,
                after_census=after_census,
                selected_ids=selected_ids,
            )
        record["fa_allocator"] = allocator
        record["safety"] = safety
        record["safety_pass"] = all(safety.values())
        record["fa_preserved"] = bool(safety["fa_preserved"])
        record["formal_mutation_primitive"] = (
            FORMAL_MUTATION_PRIMITIVE if evicted_ids else None
        )
        if not record["safety_pass"]:
            raise RuntimeError(f"profiler 安全条件失败：{safety}")

        stage = "确认 scheduler 已进入安全时点"
        if client.ping().get("ok") is not True:
            raise RuntimeError("scheduler transport 未就绪")

        stage = "发送唯一目标请求并测量 TTFT"
        runtime_workflow = runtime_workflows[0]
        request_tokens = (
            runtime_workflow.anchor_tokens
            + (runtime_workflow.anchor_output,)
            + make_tokens(suffix_seed, PENDING_SUFFIX_LENGTH)
        )
        expected_request_tokens = (
            target_position + 1 + PENDING_SUFFIX_LENGTH
        )
        if len(request_tokens) != expected_request_tokens:
            raise RuntimeError("目标请求输入长度不匹配")
        request_id = f"{namespace}_target"
        timing = measure_streaming_request(
            engine,
            request_id=request_id,
            token_ids=request_tokens,
        )
        metadata = timing["server_metadata"]
        if int(metadata.get("prompt_tokens", -1)) != len(request_tokens):
            raise RuntimeError("目标请求发生截断或静默裁剪")
        record["request_input_tokens"] = len(request_tokens)
        record["ttft_ms"] = timing["ttft_ms"]
        record["request_latency_ms"] = timing["request_latency_ms"]
        record["server_metadata"] = metadata

        stage = "读取并严格验证真实 H、E、G"
        raw_metrics = query_runtime_metrics(client, request_id)
        record.update(
            {
                "runtime_H": int(raw_metrics["physical_fa_hit"]),
                "runtime_E": int(raw_metrics["executable_prefix"]),
                "runtime_G": int(raw_metrics["replay_gap"]),
            }
        )
        runtime = validate_runtime_gap(
            case.target_gap,
            raw_metrics,
            target_position,
        )
        record.update(runtime)
        record["gap_match"] = runtime["runtime_G"] == case.target_gap
        record["correctness_pass"] = bool(
            record["gap_match"]
            and record["safety_pass"]
            and runtime["runtime_H"] == target_position
            and runtime["runtime_E"] == case.target_frontier
        )
        if not record["correctness_pass"]:
            raise RuntimeError("profiler trial 未通过正确性门禁")
        record["status"] = "PASS"
        record["failure_stage"] = None
        return record
    except Exception as error:
        record["failure_stage"] = stage
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
        return record


def write_analysis_artifacts(
    writer: ProfilerArtifactWriter,
    analysis: Mapping[str, object],
) -> None:
    """写出逐 gap 汇总、旧 Phi 审计与诊断拟合。"""
    gap_rows = tuple(analysis["gap_rows"])
    _write_csv(
        writer.directory / "summary.csv",
        gap_rows,
        (
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
            "measured_phi_ms",
        ),
    )
    old_audit = analysis.get("old_phi_audit")
    audit_rows = tuple(old_audit["points"]) if old_audit else ()
    _write_csv(
        writer.directory / "gap_audit.csv",
        audit_rows,
        (
            "gap_tokens",
            "range",
            "measured_phi_ms",
            "old_phi_ms",
            "absolute_error_ms",
            "relative_error_percent",
            "validated_prediction",
        ),
    )
    writer.write_json(
        "fit_diagnostics.json",
        analysis["fit_diagnostics"],
    )


def render_artifact_readme(config: Mapping[str, object]) -> str:
    """生成说明测量语义、独立性与门禁的中文 README。"""
    return "\n".join(
        (
            "# Recovery Profiler 128K",
            "",
            "本目录保存 Step 10D.1 的独立恢复测量。测量对象是服务端内部 recovery/TTFT 路径，不是纯 replay CUDA kernel latency。",
            "",
            "每个 trial 均复用 Step 9D 的 fresh/flush、checkpoint 构建、正式 Mamba-only 驱逐、FA 安全验证、流式首 token TTFT 与运行时 H/E/G instrumentation。所有 gap 使用相同服务配置、2 次 warmup 和 12 次正式测量。",
            "",
            "本 profiler 不读取 TraceLab policy selection、policy objective 或 policy performance。唯一使用的 TraceLab 协议信息是 recovery model 必须覆盖至 131,072 tokens。",
            "",
            "正式 Phi 未修改。`gap_audit.csv` 中 32K 以上的 OldPhi 仅标记为旧模型外推，不是已验证预测。线性与分段线性结果仅作形状诊断。",
            "",
            f"运行状态：{config.get('status', '未知')}。",
            "",
        )
    )


def _load_context_configuration(model_path: Path) -> tuple[dict, dict]:
    """读取模型和 tokenizer 的原生上下文配置。"""
    model_config_path = model_path / "config.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not model_config_path.is_file() or not tokenizer_config_path.is_file():
        raise FileNotFoundError("模型目录缺少上下文能力配置")
    return (
        json.loads(model_config_path.read_text(encoding="utf-8")),
        json.loads(tokenizer_config_path.read_text(encoding="utf-8")),
    )


def _collect_environment() -> dict[str, object]:
    """读取真实包版本、GPU 名称与显存容量。"""
    import torch

    sglang_version = importlib.metadata.version("sglang")
    if sglang_version != EXPECTED_SGLANG_VERSION:
        raise RuntimeError(
            "当前 FlowState 正式环境要求 SGLang 0.5.17，"
            f"实际为 {sglang_version}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"可见 GPU 数必须为 1，实际为 {torch.cuda.device_count()}"
        )
    properties = torch.cuda.get_device_properties(0)
    gpu_name = str(properties.name)
    total_gib = int(properties.total_memory) / (1024**3)
    if "H100" not in gpu_name or "PCIe" not in gpu_name:
        raise RuntimeError(f"GPU 与冻结环境不一致：{gpu_name}")
    if not 79.0 <= total_gib <= 81.0:
        raise RuntimeError(f"GPU 显存与 80 GiB 配置不一致：{total_gib:.3f}")
    return {
        "model": "Qwen3.5-9B",
        "sglang_version": sglang_version,
        "gpu": gpu_name,
        "gpu_memory_gib": total_gib,
        "visible_gpu_count": torch.cuda.device_count(),
        "tp": 1,
    }


def _environment_text(environment: Mapping[str, object]) -> str:
    """把冻结运行环境写成便于审阅的文本。"""
    return "\n".join(
        (
            f"模型：{environment['model']}",
            f"SGLang：{environment['sglang_version']}",
            f"GPU：{environment['gpu']}",
            f"GPU 显存：{float(environment['gpu_memory_gib']):.3f} GiB",
            f"可见 GPU 数：{environment['visible_gpu_count']}",
            f"TP：{environment['tp']}",
            "测量路径：服务端内部 recovery/TTFT",
            "",
        )
    )


def _server_command_text() -> str:
    """记录本次进程内 SGLang 引擎的等价启动配置。"""
    arguments = " ".join(
        f"{key}={value!r}"
        for key, value in sorted(ENGINE_CONFIGURATION_128K.items())
    )
    return (
        "进程入口：python -m evaluation.recovery_profiler_128k\n"
        f"SGLang Engine 参数：{arguments}\n"
        "说明：context_length 仅从 Step 9D 的 45056 提高到 131200；"
        "模型语义与 cache policy 不变。\n"
    )


def _latency_statistics(values: Sequence[float]) -> dict[str, float | None]:
    """计算单个 gap 的完整原始延迟统计。"""
    if not values:
        return {
            "mean_ms": None,
            "median_ms": None,
            "std_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "std_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "p95_ms": _empirical_quantile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _error_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """汇总绝对误差、相对误差和最大误差。"""
    errors = [float(row["absolute_error_ms"]) for row in rows]
    relative = [
        float(row["relative_error_percent"])
        for row in rows
        if row["relative_error_percent"] is not None
    ]
    return {
        "mae_ms": statistics.fmean(errors),
        "mape_percent": statistics.fmean(relative),
        "max_absolute_error_ms": max(errors),
    }


def _prediction_metrics(
    actual: Mapping[int, float],
    predicted: Mapping[int, float],
    gaps: Sequence[int],
) -> dict[str, float]:
    """计算固定 gap 集合上的诊断拟合误差与 R²。"""
    actual_values = [float(actual[gap]) for gap in gaps]
    predicted_values = [float(predicted[gap]) for gap in gaps]
    errors = [
        abs(prediction - observation)
        for prediction, observation in zip(predicted_values, actual_values)
    ]
    relative = [
        error / abs(observation) * 100.0
        for error, observation in zip(errors, actual_values)
        if observation != 0.0
    ]
    actual_mean = statistics.fmean(actual_values)
    total = sum((value - actual_mean) ** 2 for value in actual_values)
    residual = sum(
        (observation - prediction) ** 2
        for observation, prediction in zip(actual_values, predicted_values)
    )
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else (
        math.nan if total == 0.0 else 1.0 - residual / total
    )
    return {
        "mae_ms": statistics.fmean(errors),
        "mape_percent": statistics.fmean(relative),
        "max_absolute_error_ms": max(errors),
        "r_squared": r_squared,
    }


def _piecewise_estimate(gap: int, knots: Mapping[int, float]) -> float:
    """在预注册固定 knots 间执行线性插值。"""
    ordered = sorted((int(key), float(value)) for key, value in knots.items())
    if gap < ordered[0][0] or gap > ordered[-1][0]:
        raise ValueError("gap 超出分段诊断区间")
    for lower, upper in zip(ordered, ordered[1:]):
        if gap <= upper[0]:
            position = (gap - lower[0]) / (upper[0] - lower[0])
            return lower[1] + position * (upper[1] - lower[1])
    return ordered[-1][1]


def _empirical_quantile(values: Sequence[float], probability: float) -> float:
    """按经验分布逆函数计算未加权分位数。"""
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("分位数输入无效")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """按固定字段顺序写出 CSV。"""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _csv_value(value: object) -> object:
    """把嵌套结构稳定编码为单个 CSV 单元格。"""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _file_hashes(paths: Sequence[Path] = PROTECTED_PATHS) -> dict[str, str]:
    """记录不得被 profiler 修改的冻结文件摘要。"""
    return {
        str(path.relative_to(REPOSITORY_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }


def main() -> int:
    """在单个 GPU 进程中先做 feasibility，再完成全部正式测量。"""
    writer = ProfilerArtifactWriter.create()
    schedule = build_profile_schedule()
    protected_before = _file_hashes()
    config: dict[str, object] = {
        "schema_version": "flowstate.recovery_profiler_128k.v1",
        "status": "RUNNING",
        "measurement_semantic": "服务端内部 recovery/TTFT path",
        "pure_cuda_kernel_latency": False,
        "gap_points": list(FIXED_GAP_POINTS),
        "anchor_tokens": ANCHOR_POS,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "measured_repetitions": MEASURED_REPETITIONS,
        "seed": ORDER_SEED,
        "engine_configuration": ENGINE_CONFIGURATION_128K,
        "policy_comparison_executed": False,
        "policy_result_used": False,
        "trace_protocol_domain_only": REQUIRED_DOMAIN_DESCRIPTION,
        "start_time": datetime.now().astimezone().isoformat(),
    }
    writer.write_json(
        "execution_order.json",
        {
            "seed": ORDER_SEED,
            "balanced_cyclic": True,
            "cases": [case.__dict__ for case in schedule],
        },
    )
    writer.write_text("server_command.txt", _server_command_text())
    engine = None
    records = []
    fatal_error = None
    failure_stage = "环境检查"
    started = time.perf_counter_ns()
    try:
        environment = _collect_environment()
        model_config, tokenizer_config = _load_context_configuration(MODEL_PATH)
        context = validate_context_capabilities(model_config, tokenizer_config)
        config["environment"] = environment
        config["context_capabilities"] = context
        writer.write_text("environment.txt", _environment_text(environment))
        writer.write_json("config.json", config)

        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        failure_stage = "启动冻结 SGLang runtime"
        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION_128K)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)

        failure_stage = "128K feasibility gate"
        feasibility = execute_feasibility_gate(engine, client)
        config["runtime_feasibility"] = {"status": "PASS", **feasibility}
        writer.write_json("config.json", config)

        failure_stage = "正式 128K profiler"
        for index, case in enumerate(schedule, start=1):
            phase = "WARMUP" if case.is_warmup else "MEASURED"
            print(
                f"[STEP10D1] {index}/{len(schedule)} {phase} "
                f"gap={case.target_gap}",
                flush=True,
            )
            record = execute_profile_case(case, engine=engine, client=client)
            records.append(record)
            writer.append_trial(record)
            if record["status"] != "PASS":
                fatal_error = (
                    f"case {case.case_id} 失败："
                    f"{record.get('failure_stage')} {record.get('error')}"
                )
                break
    except Exception as error:
        fatal_error = repr(error)
        config.setdefault(
            "runtime_feasibility",
            {
                "status": "FAIL",
                "failure_stage": failure_stage,
                "error": repr(error),
            },
        )
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                if fatal_error is None:
                    fatal_error = f"关闭 runtime 失败：{error!r}"

    analysis = summarize_trials(records)
    write_analysis_artifacts(writer, analysis)
    protected_after = _file_hashes()
    formal_phi_unchanged = (
        protected_before["flowstate/recovery_model.py"]
        == protected_after["flowstate/recovery_model.py"]
    )
    all_protected_unchanged = protected_before == protected_after
    semantic_pass = bool(
        records
        and all(record.get("correctness_pass") is True for record in records)
        and analysis["gap_mismatch_count"] == 0
    )
    fa_preserved = bool(
        records and all(record.get("fa_preserved") is True for record in records)
    )
    feasibility_pass = (
        config.get("runtime_feasibility", {}).get("status") == "PASS"
    )
    status = (
        "PASS"
        if fatal_error is None
        and feasibility_pass
        and semantic_pass
        and fa_preserved
        and analysis["all_fixed_points_measured"]
        and formal_phi_unchanged
        and all_protected_unchanged
        else "FAIL"
    )
    config.update(
        {
            "status": status,
            "end_time": datetime.now().astimezone().isoformat(),
            "total_runtime_ms": (time.perf_counter_ns() - started) / 1_000_000,
            "completed_trials": sum(
                record.get("status") == "PASS" for record in records
            ),
            "expected_trials": len(schedule),
            "failure_stage": None if status == "PASS" else failure_stage,
            "fatal_error": fatal_error,
            "gates": {
                "128k_feasibility": feasibility_pass,
                "recovery_semantic_correctness": semantic_pass,
                "fa_kv_preserved": fa_preserved,
                "all_fixed_points_measured": analysis["all_fixed_points_measured"],
                "formal_phi_unchanged": formal_phi_unchanged,
                "all_protected_files_unchanged": all_protected_unchanged,
            },
            "protected_hashes_before": protected_before,
            "protected_hashes_after": protected_after,
        }
    )
    writer.write_json("config.json", config)
    writer.write_text("README.md", render_artifact_readme(config))
    writer.ensure_required_files()
    print(
        json.dumps(
            {
                "status": status,
                "artifact_directory": str(writer.directory),
                "completed_trials": config["completed_trials"],
                "expected_trials": config["expected_trials"],
                "failure_stage": config["failure_stage"],
                "fatal_error": fatal_error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if status == "PASS" else 1


REQUIRED_DOMAIN_DESCRIPTION = (
    "仅使用 TraceLab 冻结协议要求 recovery gap 覆盖至 131072 tokens；"
    "未读取任何策略选择或表现"
)


if __name__ == "__main__":
    raise SystemExit(main())
