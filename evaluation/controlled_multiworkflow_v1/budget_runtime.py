#!/usr/bin/env python3
"""在一个图形处理器进程内执行 K=1 和 K=4 固定快照比较。"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT))


from evaluation.controlled_multiworkflow_v1.scenario import (
    ControlledScenario,
    build_scenario,
)
from evaluation.controlled_multiworkflow_v1.snapshot_cases import (
    POLICY_NAMES,
    PolicyPlanningSummary,
    SnapshotEvaluationCase,
    build_planning_summaries,
    build_snapshot_cases,
)
from evaluation.controlled_multiworkflow_v1.snapshot_runtime import (
    CaseArtifactWriter,
    build_summary,
    run_snapshot_case,
    wait_for_transport,
)
from flowstate.recovery_model import RecoveryCostModel


_ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"
_RUNTIME_BUDGETS = (1, 4)
_AGGREGATE_BUDGETS = (1, 3, 4)
_CASES_PER_BUDGET = 28
_EXISTING_K3_ARTIFACT = (
    _ARTIFACT_ROOT / "snapshot_runtime_20260825_022400_327223"
)
_PERFORMANCE_INTERPRETATION = (
    "CORRECTNESS / MEMORY-RECOVERY TRADEOFF ONLY"
)


@dataclass(frozen=True)
class RuntimeBudgetPlan:
    """保存一个预算对应的逻辑场景、case 与规划汇总。"""

    budget_checkpoints: int
    scenario: ControlledScenario
    cases: tuple[SnapshotEvaluationCase, ...]
    planning_summaries: tuple[PolicyPlanningSummary, ...]


def build_budget_scenario(
    budget_checkpoints: int,
    scenario: ControlledScenario | None = None,
) -> ControlledScenario:
    """只替换预算，不改变冻结 workload 的请求、候选或元数据顺序。"""
    if budget_checkpoints <= 0:
        raise ValueError("预算检查点数量必须为正数")
    base = scenario or build_scenario()
    checkpoint_size = base.metadata.checkpoint_size_bytes
    metadata = replace(
        base.metadata,
        budget_checkpoints=budget_checkpoints,
    )
    return replace(
        base,
        budget_bytes=budget_checkpoints * checkpoint_size,
        metadata=metadata,
    )


def build_runtime_budget_plan(
    budget_checkpoints: int,
    recovery_cost_model: RecoveryCostModel | None = None,
) -> RuntimeBudgetPlan:
    """直接用现有 scenario、策略和 GlobalOptimizer 构造预算计划。"""
    model = recovery_cost_model or RecoveryCostModel()
    scenario = build_budget_scenario(budget_checkpoints)
    cases = build_snapshot_cases(scenario, model)
    planning_summaries = build_planning_summaries(scenario, model)
    if len(cases) != _CASES_PER_BUDGET:
        raise RuntimeError(f"隔离 case 数量异常：{len(cases)}")
    return RuntimeBudgetPlan(
        budget_checkpoints=budget_checkpoints,
        scenario=scenario,
        cases=cases,
        planning_summaries=planning_summaries,
    )


def load_case_records(artifact_directory: Path) -> tuple[dict, ...]:
    """读取一个已保存 artifact 的逐 case 记录。"""
    records = []
    with (artifact_directory / "cases.jsonl").open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"cases.jsonl 第 {line_number} 行不是有效 JSON"
                ) from error
    return tuple(records)


def validate_case_agreement(records: Sequence[dict]) -> None:
    """按 Step 7E 的一个令牌容差重新验证全部 case。"""
    if len(records) != _CASES_PER_BUDGET:
        raise ValueError(
            f"预算 artifact 必须包含 {_CASES_PER_BUDGET} 个 case"
        )
    for record in records:
        if record.get("status") != "PASS":
            raise ValueError(f"case 未通过：{record.get('case_id')}")
        planning_gap = int(record["planning_gap"])
        runtime_gap = int(record["runtime_gap"])
        agreement = (
            runtime_gap in (0, 1)
            if planning_gap == 0
            else abs(runtime_gap - planning_gap) <= 1
        )
        if not agreement:
            raise ValueError(
                "规划与运行时恢复间隔不一致："
                f"{record.get('case_id')}，planning={planning_gap}，"
                f"runtime={runtime_gap}"
            )


def load_runtime_summary(artifact_directory: Path) -> dict:
    """读取并返回一个固定快照运行时汇总。"""
    with (artifact_directory / "summary.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError("运行时 summary.json 顶层必须是对象")
    return summary


def validate_runtime_summary(summary: Mapping[str, object]) -> None:
    """验证单预算汇总的完整性与正式安全条件。"""
    if summary.get("status") != "PASS":
        raise ValueError("运行时预算汇总未通过")
    if int(summary.get("cases_passed", 0)) != _CASES_PER_BUDGET:
        raise ValueError("运行时预算汇总的通过 case 数量异常")
    if int(summary.get("cases_failed", -1)) != 0:
        raise ValueError("运行时预算汇总包含失败 case")
    if summary.get("planning_runtime_agreement") is not True:
        raise ValueError("运行时预算汇总未通过 planning/runtime 校验")

    safety = summary.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("运行时预算汇总缺少安全字段")
    required_true = (
        "fa_preserved",
        "allocator_invariant",
        "tree_path_invariant",
        "sanity_check",
    )
    if not all(safety.get(field) is True for field in required_true):
        raise ValueError(f"运行时预算安全条件失败：{safety}")
    if safety.get("cascade_called") is not False:
        raise ValueError("运行时预算检测到 cascade eviction")

    runtime_summary = summary.get("runtime_summary")
    if not isinstance(runtime_summary, dict):
        raise ValueError("运行时预算汇总缺少逐策略指标")
    if set(runtime_summary) != set(POLICY_NAMES):
        raise ValueError("运行时预算汇总的策略集合不完整")
    if not all(
        int(runtime_summary[name].get("n_cases", 0)) == 7
        for name in POLICY_NAMES
    ):
        raise ValueError("运行时预算汇总的逐策略 case 数量异常")


def build_combined_budget_rows(
    summaries_by_budget: Mapping[int, Mapping[str, object]],
    sources_by_budget: Mapping[int, str] | None = None,
) -> tuple[dict, ...]:
    """把 K=1、K=3、K=4 汇总为统一的 memory-recovery 表。"""
    if set(summaries_by_budget) != set(_AGGREGATE_BUDGETS):
        raise ValueError("聚合预算必须且只能包含 K=1、K=3、K=4")
    sources = sources_by_budget or {}
    rows = []
    for budget_checkpoints in _AGGREGATE_BUDGETS:
        summary = summaries_by_budget[budget_checkpoints]
        validate_runtime_summary(summary)
        runtime_summary = summary["runtime_summary"]
        for policy_name in POLICY_NAMES:
            policy = runtime_summary[policy_name]
            rows.append(
                {
                    "K": budget_checkpoints,
                    "policy": policy_name,
                    "runtime_total_gap": int(
                        policy["runtime_total_gap"]
                    ),
                    "runtime_EPR": float(
                        policy["executable_prefix_ratio"]
                    ),
                    "estimated_recovery_cost_ms": float(
                        policy["estimated_recovery_cost_ms"]
                    ),
                    "mean_gap_per_request": float(
                        policy["mean_gap_per_request"]
                    ),
                    "mean_request_e2e_ms": policy.get(
                        "mean_request_e2e_ms"
                    ),
                    "source_artifact": sources.get(
                        budget_checkpoints,
                        str(summary.get("artifact_directory", "")),
                    ),
                }
            )
    validate_flowstate_monotonicity(rows)
    return tuple(rows)


def validate_flowstate_monotonicity(rows: Sequence[Mapping[str, object]]) -> None:
    """确认 FlowState 的运行时恢复成本随预算单调不增。"""
    flowstate_rows = sorted(
        (
            row for row in rows
            if row.get("policy") == "FlowState"
        ),
        key=lambda row: int(row["K"]),
    )
    if tuple(int(row["K"]) for row in flowstate_rows) != _AGGREGATE_BUDGETS:
        raise ValueError("FlowState 单调性检查缺少 K=1、K=3、K=4")
    costs = tuple(
        float(row["estimated_recovery_cost_ms"])
        for row in flowstate_rows
    )
    if any(
        following > current + 1e-9
        for current, following in zip(costs, costs[1:])
    ):
        raise ValueError(f"FlowState 运行时恢复成本非单调：{costs}")


def write_combined_budget_csv(
    rows: Sequence[Mapping[str, object]],
    artifact_root: Path = _ARTIFACT_ROOT,
    timestamp: str | None = None,
) -> Path:
    """写出不会覆盖旧结果的三预算聚合 CSV。"""
    active_timestamp = timestamp or datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    path = artifact_root / f"budget_runtime_summary_{active_timestamp}.csv"
    if path.exists():
        raise FileExistsError(f"聚合 artifact 已存在：{path}")
    fieldnames = (
        "K",
        "policy",
        "runtime_total_gap",
        "runtime_EPR",
        "estimated_recovery_cost_ms",
        "mean_gap_per_request",
        "mean_request_e2e_ms",
        "source_artifact",
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return path


def _create_budget_writer(
    budget_checkpoints: int,
    timestamp: str,
    artifact_root: Path = _ARTIFACT_ROOT,
) -> CaseArtifactWriter:
    """为一个预算创建符合冻结命名规则的 artifact 目录。"""
    directory = (
        artifact_root
        / f"budget_runtime_k{budget_checkpoints}_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    writer = CaseArtifactWriter(directory)
    writer.cases_path.touch(exist_ok=False)
    return writer


def _execute_budget(
    *,
    plan: RuntimeBudgetPlan,
    writer: CaseArtifactWriter,
    engine: object,
    client: object,
    recovery_cost_model: RecoveryCostModel,
) -> tuple[dict, tuple[dict, ...]]:
    """在已启动的同一引擎内完成一个预算的二十八个隔离 case。"""
    records = []
    failure_stage = None
    error_text = None
    started = time.perf_counter()
    runtime_namespace = f"flowstate_step7g_k{plan.budget_checkpoints}"
    for index, case in enumerate(plan.cases, start=1):
        print(
            f"[STEP7G] K={plan.budget_checkpoints} "
            f"CASE {index}/{_CASES_PER_BUDGET} "
            f"{case.policy_name} {case.continuation_id}",
            flush=True,
        )
        record = run_snapshot_case(
            index=index,
            case=case,
            engine=engine,
            client=client,
            scenario=plan.scenario,
            recovery_cost_model=recovery_cost_model,
            runtime_namespace=runtime_namespace,
        )
        record["budget_checkpoints"] = plan.budget_checkpoints
        records.append(record)
        writer.append_case(record)
        if record["status"] != "PASS":
            failure_stage = str(record["failure_stage"])
            error_text = str(record.get("error"))
            break

    summary = build_summary(
        records=records,
        planning_summaries=plan.planning_summaries,
        artifact_directory=writer.directory,
        failure_stage=failure_stage,
        error=error_text,
        budget_checkpoints=plan.budget_checkpoints,
        expected_cases=_CASES_PER_BUDGET,
    )
    summary["schema_version"] = (
        "flowstate.controlled_multiworkflow.budget_runtime.v1"
    )
    summary["performance_interpretation"] = _PERFORMANCE_INTERPRETATION
    summary["wall_time_ms"] = (
        time.perf_counter() - started
    ) * 1_000.0
    writer.write_summary(summary)
    return summary, tuple(records)


def _write_unexecuted_summary(
    *,
    plan: RuntimeBudgetPlan,
    writer: CaseArtifactWriter,
    failure_stage: str,
    error_text: str,
) -> dict:
    """为初始化或前序失败后未执行的预算保存明确记录。"""
    summary = build_summary(
        records=(),
        planning_summaries=plan.planning_summaries,
        artifact_directory=writer.directory,
        failure_stage=failure_stage,
        error=error_text,
        budget_checkpoints=plan.budget_checkpoints,
        expected_cases=_CASES_PER_BUDGET,
    )
    summary["schema_version"] = (
        "flowstate.controlled_multiworkflow.budget_runtime.v1"
    )
    summary["performance_interpretation"] = _PERFORMANCE_INTERPRETATION
    writer.write_summary(summary)
    return summary


def main() -> int:
    """只启动一次引擎并依次完成 K=1、K=4 两个预算。"""
    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import (
        FormalEndToEndGateEngine,
        requested_control_port,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    recovery_cost_model = RecoveryCostModel()
    plans = {
        budget: build_runtime_budget_plan(
            budget,
            recovery_cost_model,
        )
        for budget in _RUNTIME_BUDGETS
    }
    writers = {
        budget: _create_budget_writer(budget, timestamp)
        for budget in _RUNTIME_BUDGETS
    }
    summaries: dict[int, dict] = {}
    engine = None
    overall_failure_stage = None
    error_text = None
    aggregate_path = None
    started = time.perf_counter()

    try:
        from evaluation.controlled_multiworkflow_v1.runtime_gate import (
            ENGINE_CONFIGURATION,
        )

        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)

        for budget in _RUNTIME_BUDGETS:
            summary, records = _execute_budget(
                plan=plans[budget],
                writer=writers[budget],
                engine=engine,
                client=client,
                recovery_cost_model=recovery_cost_model,
            )
            summaries[budget] = summary
            if summary["status"] != "PASS":
                overall_failure_stage = (
                    f"K={budget}：{summary['failure_stage']}"
                )
                error_text = str(summary.get("error"))
                break
    except Exception as error:
        overall_failure_stage = overall_failure_stage or "初始化预算扫描运行时"
        error_text = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as shutdown_error:
                if error_text is None:
                    overall_failure_stage = "关闭预算扫描运行时"
                    error_text = repr(shutdown_error)

    for budget in _RUNTIME_BUDGETS:
        if budget in summaries:
            continue
        summaries[budget] = _write_unexecuted_summary(
            plan=plans[budget],
            writer=writers[budget],
            failure_stage=overall_failure_stage or "前序预算失败后停止",
            error_text=error_text or "前序预算失败后未执行",
        )

    status = "PASS"
    try:
        if error_text is not None:
            raise RuntimeError(error_text)
        k3_summary = load_runtime_summary(_EXISTING_K3_ARTIFACT)
        all_summaries = {
            1: summaries[1],
            3: k3_summary,
            4: summaries[4],
        }
        artifact_directories = {
            1: writers[1].directory,
            3: _EXISTING_K3_ARTIFACT,
            4: writers[4].directory,
        }
        for budget, directory in artifact_directories.items():
            validate_case_agreement(load_case_records(directory))
            validate_runtime_summary(all_summaries[budget])
        rows = build_combined_budget_rows(
            all_summaries,
            {
                budget: str(directory)
                for budget, directory in artifact_directories.items()
            },
        )
        aggregate_path = write_combined_budget_csv(
            rows,
            timestamp=timestamp,
        )
    except Exception as error:
        status = "FAIL"
        overall_failure_stage = overall_failure_stage or "生成三预算聚合"
        error_text = error_text or repr(error)
        traceback.print_exc()

    result = {
        "status": status,
        "runtime_budgets": list(_RUNTIME_BUDGETS),
        "cases": {
            str(budget): {
                "passed": summaries[budget]["cases_passed"],
                "failed": summaries[budget]["cases_failed"],
                "expected": summaries[budget]["cases_expected"],
            }
            for budget in _RUNTIME_BUDGETS
        },
        "summaries": {
            str(budget): summaries[budget]
            for budget in _RUNTIME_BUDGETS
        },
        "k3_source": str(_EXISTING_K3_ARTIFACT),
        "k3_rerun": False,
        "aggregate_path": (
            str(aggregate_path) if aggregate_path is not None else None
        ),
        "performance_interpretation": _PERFORMANCE_INTERPRETATION,
        "failure_stage": overall_failure_stage,
        "error": error_text,
        "wall_time_ms": (time.perf_counter() - started) * 1_000.0,
    }
    print(
        "[STEP7G] RESULT="
        + json.dumps(result, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
