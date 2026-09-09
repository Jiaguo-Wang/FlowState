#!/usr/bin/env python3
"""测量冻结分配快照上的 FlowState 控制面开销并汇总运行时记录。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence

from evaluation.rq3_agentx_formal_policy_evaluation import (
    build_agentx_formal_snapshots,
)
from evaluation.rq3_formal_policy_evaluation import (
    compute_budget_ks,
    create_budget_variant,
    load_eligible_snapshots,
)
from evaluation.rq3_frozen_snapshot_evaluator import (
    AllocationSnapshot,
    _select_policy,
    evaluate_objective,
)
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import is_compatible


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPENHANDS_ROOT = (
    REPOSITORY_ROOT
    / "evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017"
)
RQ4_ROOT = REPOSITORY_ROOT / "rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652"
DEFAULT_OUTPUT_PARENT = REPOSITORY_ROOT / "evaluation/rq6_overhead_output"
CPU_TIMING_REPETITIONS = 3
RUNTIME_REPETITIONS = 3
RUNTIME_SNAPSHOT_COUNT = 24


@dataclass(frozen=True)
class PreparedExecutableState:
    """保存只从冻结在线输入构造的执行态规划材料。"""

    continuations: tuple[Any, ...]
    candidates: tuple[Any, ...]
    planning_targets: tuple[tuple[str, int], ...]
    compatibility: tuple[tuple[str, str], ...]
    standalone_benefits_ms: tuple[tuple[str, str, float], ...]


@dataclass(frozen=True)
class TimedSelection:
    """保存分阶段计时及与冻结 selector 等价的选择结果。"""

    selected_checkpoint_ids: tuple[str, ...]
    construction_ns: int
    allocation_ns: int
    total_ns: int
    snapshot_digest_before: str
    snapshot_digest_after: str


@dataclass
class StageTimer:
    """用单调高分辨率时钟记录一个明确代码区间。"""

    enabled: bool = True
    elapsed_ns: int = 0
    _start_ns: int | None = None

    def __enter__(self) -> "StageTimer":
        """进入计时边界。"""
        if self.enabled:
            self._start_ns = perf_counter_ns()
        return self

    def __exit__(self, *_errors: object) -> None:
        """退出计时边界且不吞掉异常。"""
        if self.enabled and self._start_ns is not None:
            self.elapsed_ns = perf_counter_ns() - self._start_ns


def construct_executable_state(snapshot: AllocationSnapshot) -> PreparedExecutableState:
    """从冻结在线字段构造目标、兼容关系与空集合单点收益。"""
    if (
        snapshot.online_boundary.future_continuation_included
        or snapshot.online_boundary.future_request_included
        or snapshot.online_boundary.future_latency_included
    ):
        raise ValueError("执行态构造禁止读取未来信息")
    continuations = snapshot.core_continuations()
    candidates = snapshot.core_candidates()
    model = RecoveryCostModel()
    targets = tuple(
        (continuation.continuation_id, continuation.planning_target)
        for continuation in continuations
    )
    compatibility: list[tuple[str, str]] = []
    benefits: list[tuple[str, str, float]] = []
    for candidate in candidates:
        for continuation in continuations:
            if not is_compatible(candidate, continuation):
                continue
            compatibility.append(
                (candidate.checkpoint_id, continuation.continuation_id)
            )
            target = continuation.planning_target
            baseline = model.estimate(target, target)
            frontier = min(candidate.token_pos, target)
            gain = baseline - model.estimate(target - frontier, target)
            benefits.append(
                (
                    candidate.checkpoint_id,
                    continuation.continuation_id,
                    max(0.0, gain),
                )
            )
    return PreparedExecutableState(
        continuations=continuations,
        candidates=candidates,
        planning_targets=targets,
        compatibility=tuple(compatibility),
        standalone_benefits_ms=tuple(benefits),
    )


def timed_flowstate_selection(
    snapshot: AllocationSnapshot,
    *,
    instrumentation: bool = True,
) -> TimedSelection:
    """直接计时连续的执行态构造与冻结 GlobalOptimizer 调用。"""
    digest_before = snapshot.content_digest()
    total_timer = StageTimer(enabled=instrumentation)
    construction_timer = StageTimer(enabled=instrumentation)
    allocation_timer = StageTimer(enabled=instrumentation)
    with total_timer:
        with construction_timer:
            prepared = construct_executable_state(snapshot)
        with allocation_timer:
            result = GlobalOptimizer(RecoveryCostModel()).select(
                prepared.continuations,
                prepared.candidates,
                snapshot.budget_bytes,
            )
    digest_after = snapshot.content_digest()
    if digest_after != digest_before:
        raise RuntimeError("计时路径修改了冻结 snapshot")
    selected = tuple(item.checkpoint_id for item in result.selected)
    return TimedSelection(
        selected_checkpoint_ids=selected,
        construction_ns=construction_timer.elapsed_ns,
        allocation_ns=allocation_timer.elapsed_ns,
        total_ns=total_timer.elapsed_ns,
        snapshot_digest_before=digest_before,
        snapshot_digest_after=digest_after,
    )


def reference_flowstate_selection(snapshot: AllocationSnapshot) -> tuple[str, ...]:
    """调用 RQ3 冻结 dispatcher 得到无新增计时包装的参考选择。"""
    selected, _ = _select_policy("FlowState", snapshot, evaluate_objective)
    return tuple(selected)


def cpu_overhead_record(
    snapshot: AllocationSnapshot,
    *,
    population: str,
    budget_ratio: float,
    timing_repetition: int,
) -> dict[str, Any]:
    """生成单个冻结 epoch 的 CPU 控制面原始计时记录。"""
    reference = reference_flowstate_selection(snapshot)
    timed = timed_flowstate_selection(snapshot)
    repeated = timed_flowstate_selection(snapshot)
    if timed.selected_checkpoint_ids != reference:
        raise RuntimeError("计时开关改变了 FlowState selected set")
    if repeated.selected_checkpoint_ids != timed.selected_checkpoint_ids:
        raise RuntimeError("FlowState 计时选择不确定")
    return {
        "schema_version": "flowstate.rq6_per_epoch_overhead.v1",
        "record_kind": "cpu_allocation",
        "population": population,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_digest": timed.snapshot_digest_before,
        "budget_ratio": budget_ratio,
        "k": snapshot.logical_budget_k,
        "pending_count": len(snapshot.pending_continuations),
        "candidate_count": len(snapshot.eligible_candidates),
        "timing_repetition": timing_repetition,
        "timings_ns": {
            "introspection": None,
            "construction": timed.construction_ns,
            "allocation": timed.allocation_ns,
            "reconciliation": None,
            "total_control": timed.total_ns,
        },
        "selected_checkpoint_ids": list(timed.selected_checkpoint_ids),
        "reference_selected_checkpoint_ids": list(reference),
        "instrumentation_equivalent": timed.selected_checkpoint_ids == reference,
        "snapshot_immutable": timed.snapshot_digest_before == timed.snapshot_digest_after,
        "future_leakage": False,
        "status": "PASS",
    }


def collect_cpu_records(repetitions: int = CPU_TIMING_REPETITIONS) -> list[dict[str, Any]]:
    """遍历 168 个 OpenHands 与 23 个 AgentX 正式快照及去重预算。"""
    if repetitions < 1:
        raise ValueError("CPU timing repetitions 必须大于零")
    openhands = load_eligible_snapshots(OPENHANDS_ROOT)
    agentx_wrapped, _ = build_agentx_formal_snapshots()
    populations = {
        "OpenHands": openhands,
        "AgentX": [item.snapshot for item in agentx_wrapped],
    }
    if len(openhands) != 168 or len(agentx_wrapped) != 23:
        raise RuntimeError("RQ3 冻结 population 数量不正确")
    records: list[dict[str, Any]] = []
    for population, snapshots in populations.items():
        for frozen in snapshots:
            before = frozen.content_digest()
            for ratio, k in compute_budget_ks(len(frozen.eligible_candidates)):
                variant = create_budget_variant(frozen, k)
                for repetition in range(repetitions):
                    records.append(
                        cpu_overhead_record(
                            variant,
                            population=population,
                            budget_ratio=ratio,
                            timing_repetition=repetition,
                        )
                    )
            if frozen.content_digest() != before:
                raise RuntimeError(f"CPU 计时修改了冻结快照：{frozen.snapshot_id}")
    return records


def percentile(values: Sequence[float], percent: float) -> float:
    """按线性插值计算有限样本百分位数。"""
    if not values:
        raise ValueError("百分位输入不能为空")
    if not 0.0 <= percent <= 100.0:
        raise ValueError("百分位必须位于 0 到 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    """汇总计数、均值、中位数、P95 与最大值。"""
    if not values:
        raise ValueError("汇总输入不能为空")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "p95": percentile(numeric, 95.0),
        "max": max(numeric),
    }


def _rank(values: Sequence[float]) -> list[float]:
    """为 Spearman 相关计算带并列平均名次的秩。"""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        rank = (offset + end - 1) / 2.0 + 1.0
        for index in order[offset:end]:
            ranks[index] = rank
        offset = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """计算简单 Spearman 相关；常量或不足两点时返回空。"""
    if len(left) != len(right) or len(left) < 2:
        return None
    x = _rank(left)
    y = _rank(right)
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x)
        * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def validate_record_schema(record: Mapping[str, Any]) -> None:
    """拒绝缺少 stage、规模、边界或正确性字段的原始记录。"""
    required = {
        "schema_version",
        "record_kind",
        "population",
        "snapshot_id",
        "k",
        "pending_count",
        "candidate_count",
        "timings_ns",
        "instrumentation_equivalent",
        "future_leakage",
        "status",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"RQ6 原始记录缺少字段：{sorted(missing)}")
    timings = record["timings_ns"]
    if not isinstance(timings, Mapping):
        raise ValueError("timings_ns 必须是映射")
    for stage in (
        "introspection",
        "construction",
        "allocation",
        "reconciliation",
        "total_control",
    ):
        if stage not in timings:
            raise ValueError(f"timings_ns 缺少 stage：{stage}")


def overhead_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """分别汇总 CPU、runtime 以及 runtime 完整控制路径。"""
    for record in records:
        validate_record_schema(record)
    result: dict[str, Any] = {
        "record_count": len(records),
        "by_kind": {},
        "by_population": {},
    }
    for field, key_name in (("record_kind", "by_kind"), ("population", "by_population")):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record[field])].append(record)
        for key, rows in sorted(groups.items()):
            stages: dict[str, Any] = {}
            for stage in (
                "introspection",
                "construction",
                "allocation",
                "reconciliation",
                "total_control",
            ):
                milliseconds = [
                    float(row["timings_ns"][stage]) / 1_000_000.0
                    for row in rows
                    if row["timings_ns"].get(stage) is not None
                ]
                stages[stage] = summarize_values(milliseconds) if milliseconds else None
            result[key_name][key] = {"record_count": len(rows), "stages_ms": stages}
    return result


def scaling_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """按 P、C、K 分组并报告自然工作负载上的秩相关。"""
    passed = [row for row in records if row.get("status") == "PASS"]
    allocation_rows = [
        row for row in passed if row["timings_ns"].get("allocation") is not None
    ]
    runtime_rows = [row for row in passed if row["record_kind"] == "runtime_control"]

    def grouped(variable: str, rows: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
        values: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            timing = row["timings_ns"].get(stage)
            if timing is not None:
                values[int(row[variable])].append(float(timing) / 1_000_000.0)
        return {
            str(key): summarize_values(stage_values)
            for key, stage_values in sorted(values.items())
        }

    allocation_ms = [
        float(row["timings_ns"]["allocation"]) / 1_000_000.0
        for row in allocation_rows
    ]
    total_runtime_ms = [
        float(row["timings_ns"]["total_control"]) / 1_000_000.0
        for row in runtime_rows
    ]
    return {
        "source": "仅冻结自然 population；未派生 synthetic case",
        "observed_ranges": {
            "pending_count": [
                min(int(row["pending_count"]) for row in passed),
                max(int(row["pending_count"]) for row in passed),
            ],
            "candidate_count": [
                min(int(row["candidate_count"]) for row in passed),
                max(int(row["candidate_count"]) for row in passed),
            ],
            "k": [min(int(row["k"]) for row in passed), max(int(row["k"]) for row in passed)],
        },
        "allocation_by_pending_count_ms": grouped(
            "pending_count", allocation_rows, "allocation"
        ),
        "allocation_by_candidate_count_ms": grouped(
            "candidate_count", allocation_rows, "allocation"
        ),
        "allocation_by_k_ms": grouped("k", allocation_rows, "allocation"),
        "total_control_by_pending_count_ms": grouped(
            "pending_count", runtime_rows, "total_control"
        ),
        "total_control_by_candidate_count_ms": grouped(
            "candidate_count", runtime_rows, "total_control"
        ),
        "spearman": {
            "allocation_vs_pending_count": spearman(
                [float(row["pending_count"]) for row in allocation_rows], allocation_ms
            ),
            "allocation_vs_candidate_count": spearman(
                [float(row["candidate_count"]) for row in allocation_rows], allocation_ms
            ),
            "allocation_vs_k": spearman(
                [float(row["k"]) for row in allocation_rows], allocation_ms
            ),
            "runtime_total_vs_pending_count": spearman(
                [float(row["pending_count"]) for row in runtime_rows], total_runtime_ms
            ),
            "runtime_total_vs_candidate_count": spearman(
                [float(row["candidate_count"]) for row in runtime_rows], total_runtime_ms
            ),
        },
    }


def instrumentation_equivalence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """汇总 selected set、快照不可变性、未来边界与 H/E/G 等价证据。"""
    checks = {
        "selected_set_equivalent": all(
            row.get("instrumentation_equivalent") is True for row in records
        ),
        "snapshot_immutable": all(
            row.get("snapshot_immutable", True) is True for row in records
        ),
        "no_future_information": all(
            row.get("future_leakage") is False for row in records
        ),
        "heg_equivalent": all(
            row.get("heg_equivalent", True) is True for row in records
        ),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def runtime_correctness(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """对 24×3 个 runtime control run 执行 fail-closed 汇总门禁。"""
    runtime = [row for row in records if row.get("record_kind") == "runtime_control"]
    expected = RUNTIME_SNAPSHOT_COUNT * RUNTIME_REPETITIONS
    fields = (
        "handle_mapping_pass",
        "recurrent_residency_validation_pass",
        "fa_preserved",
        "native_recurrent_eviction_zero",
        "unexpected_rematerialization_zero",
        "fa_cascade_zero",
        "oom_zero",
        "truncation_zero",
        "future_leakage_zero",
        "fresh_engine_empty",
        "gpu_cleanup_pass",
    )
    gates = {
        "run_count": len(runtime) == expected,
        **{
            field: len(runtime) == expected
            and all(row.get("correctness", {}).get(field) is True for row in runtime)
            for field in fields
        },
        "cross_run_contamination_no": len(runtime) == expected
        and all(row.get("cross_run_contamination") is False for row in runtime),
    }
    invalid = [row.get("run_id") for row in runtime if row.get("status") != "PASS"]
    return {
        "status": "PASS" if all(gates.values()) and not invalid else "FAIL",
        "expected_runs": expected,
        "observed_runs": len(runtime),
        "invalid_run_ids": invalid,
        "gates": gates,
    }


def _sha256(path: Path) -> str:
    """流式计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """稳定写入 UTF-8 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """向新建 JSONL 追加原始 epoch 记录。"""
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            validate_record_schema(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_records(path: Path) -> list[dict[str, Any]]:
    """读取原始 epoch JSONL。"""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def initialize_artifact(output_root: Path) -> None:
    """创建完整且不会覆盖既有结果的 RQ6 artifact 目录。"""
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在：{output_root}")
    for relative in ("raw", "analysis", "correctness", "provenance", "logs", "runtime_runs"):
        (output_root / relative).mkdir(parents=True, exist_ok=False)
    (output_root / "raw/per_epoch_overhead.jsonl").touch(exist_ok=False)


def source_paths() -> tuple[Path, ...]:
    """返回 RQ6 与冻结决策路径的源码集合。"""
    return (
        REPOSITORY_ROOT / "evaluation/rq6_system_overhead.py",
        REPOSITORY_ROOT / "evaluation/rq6_runtime_overhead.py",
        REPOSITORY_ROOT / "evaluation/rq3_frozen_snapshot_evaluator.py",
        REPOSITORY_ROOT / "evaluation/rq3_formal_policy_evaluation.py",
        REPOSITORY_ROOT / "evaluation/rq4_unified_runtime_harness.py",
        REPOSITORY_ROOT / "flowstate/optimizer.py",
        REPOSITORY_ROOT / "flowstate/recovery_model.py",
        REPOSITORY_ROOT / "flowstate/executable_state.py",
        REPOSITORY_ROOT / "flowstate/controller.py",
    )


def source_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    """记录源码路径、大小和 SHA256。"""
    return {
        "files": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in paths
        }
    }


def input_manifest() -> dict[str, Any]:
    """锁定三个正式 population 及预算协议来源。"""
    input_paths = (
        OPENHANDS_ROOT / "FORMAL_COLLECTION.json",
        OPENHANDS_ROOT / "collection_summary.json",
        REPOSITORY_ROOT
        / "evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/EVALUATION_PROTOCOL.json",
        REPOSITORY_ROOT / "rq3_agentx_policy_eval_20260906_214956/manifest.json",
        RQ4_ROOT / "frozen_protocol.json",
        RQ4_ROOT / "manifest.json",
    )
    return {
        "openhands_snapshot_count": 168,
        "agentx_snapshot_count": 23,
        "runtime_snapshot_count": 24,
        "cpu_budget_ratios": [0.25, 0.50, 0.75],
        "runtime_budget_ratio": 0.25,
        "files": {
            str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in input_paths
        },
    }


def collect_cpu(output_root: Path, repetitions: int = CPU_TIMING_REPETITIONS) -> None:
    """采集 CPU 记录并立即写入新 artifact。"""
    records = collect_cpu_records(repetitions)
    append_records(output_root / "raw/per_epoch_overhead.jsonl", records)


def analyze(output_root: Path) -> dict[str, Any]:
    """汇总已有 CPU 与 runtime 记录并写出正确性和 scaling 结果。"""
    records = read_records(output_root / "raw/per_epoch_overhead.jsonl")
    summary = overhead_summary(records)
    scaling = scaling_summary(records)
    equivalence = instrumentation_equivalence(records)
    correctness = runtime_correctness(records)
    write_json(output_root / "analysis/overhead_summary.json", summary)
    write_json(output_root / "analysis/scaling_summary.json", scaling)
    write_json(
        output_root / "correctness/instrumentation_equivalence.json", equivalence
    )
    write_json(output_root / "correctness/runtime_correctness.json", correctness)
    return {
        "summary": summary,
        "scaling": scaling,
        "equivalence": equivalence,
        "correctness": correctness,
    }


def main(argv: Iterable[str] | None = None) -> int:
    """执行 artifact 初始化、CPU 采集或统一分析。"""
    parser = argparse.ArgumentParser(description="RQ6 控制面开销评估")
    parser.add_argument("command", choices=("init", "cpu", "analyze"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repetitions", type=int, default=CPU_TIMING_REPETITIONS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = args.output_root or (
        DEFAULT_OUTPUT_PARENT
        / f"rq6_overhead_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if args.command == "init":
        initialize_artifact(output)
        sources = source_manifest(source_paths())
        inputs = input_manifest()
        write_json(output / "provenance/source_manifest.json", sources)
        write_json(output / "provenance/input_manifest.json", inputs)
    elif args.command == "cpu":
        collect_cpu(output, args.repetitions)
    else:
        analyze(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
