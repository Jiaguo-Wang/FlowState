#!/usr/bin/env python3
"""执行 Step 8E 的代表点固定快照运行时正确性验证。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Mapping, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_PROBE_DIRECTORY = (
    _REPOSITORY_ROOT
    / "motivation"
    / "artifacts"
    / "wp3b_gate_20260820"
)
_TEST_RUNTIME_DIRECTORY = _REPOSITORY_ROOT / "tests" / "runtime"
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_LEGACY_PROBE_DIRECTORY))
sys.path.insert(0, str(_TEST_RUNTIME_DIRECTORY))

from evaluation.controlled_multiworkflow_v1.policies import (
    select_global_lru,
)
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    ENGINE_CONFIGURATION,
    PENDING_SUFFIX_LENGTH,
    RuntimeWorkflow,
    build_runtime_handles,
    build_runtime_workflows,
    generate,
    inspect_after_allocation,
    make_tokens,
    optional_latency_ms,
    query_runtime_metrics,
    wait_for_transport,
)
from evaluation.controlled_multiworkflow_v1.snapshot_runtime import (
    SnapshotSchedulerRuntimeAdapter,
    allocation_safety_snapshot,
    validate_clean_cache,
)
from evaluation.scalable_multiworkflow_v2.offline_analysis import (
    load_offline_artifact as load_scalable_offline_artifact,
)
from evaluation.scalable_multiworkflow_v2.scenario import (
    ScalableScenario,
    build_scenario as build_scalable_scenario,
)
from evaluation.sota_metadata import (
    CONTROLLED_MARCONI_ALPHA,
    build_kvflow_steps,
    build_marconi_flop_saved,
    build_marconi_recency,
)
from evaluation.sota_policies import KVFlowStylePolicy, MarconiStylePolicy
from evaluation.sota_signal_stress_v1.offline_analysis import (
    load_offline_artifact as load_signal_offline_artifact,
)
from evaluation.sota_signal_stress_v1.scenario import (
    SignalScenario,
    build_scenario as build_signal_scenario,
)
from flowstate.controller import StateController
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import AllocationResult, GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


GPU_POLICY_NAMES = (
    "Global-LRU",
    "KVFlow-style",
    "Marconi-style",
    "FlowState",
)
SCALABLE_SCENARIO_NAME = "scalable_multiworkflow_v2_n16"
SIGNAL_SCENARIO_NAME = "sota_signal_stress_v1"
SCALABLE_BUDGETS = (4, 12)
SIGNAL_BUDGETS = (4, 8)
EXPECTED_SCALABLE_CASES = 480
EXPECTED_SIGNAL_CASES = 320
EXPECTED_TOTAL_CASES = 800
EXPECTED_E2E_EQUIVALENCE_CASES = 69
_FORMAL_PRIMITIVE = (
    "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
)
_RUNTIME_ARTIFACT_ROOT = (
    _REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
)

# N=16 可扩展场景包含二十个同时驻留的候选，因此测试运行时预留二十四个槽位。
STEP8E_ENGINE_CONFIGURATION = {
    **ENGINE_CONFIGURATION,
    "max_mamba_cache_size": 24,
}


@dataclass(frozen=True)
class RuntimeCorrectnessCase:
    """记录一个策略与单个待续请求的固定快照预期。"""

    scenario_name: str
    budget_checkpoints: int
    policy_name: str
    continuation_id: str
    workflow_id: str
    selected_checkpoint_ids: tuple[str, ...]
    planning_target: int
    planning_executable_frontier: int
    planning_gap_tokens: int


@dataclass(frozen=True)
class SnapshotAuditPlan:
    """记录一个场景、预算和策略对应的全量状态审计计划。"""

    scenario_name: str
    budget_checkpoints: int
    policy_name: str
    selected_checkpoint_ids: tuple[str, ...]
    logical_cases: tuple[RuntimeCorrectnessCase, ...]


@dataclass(frozen=True)
class RuntimeWorkflowSpecView:
    """把不同 evaluation workload 的字段统一为既有构建器接口。"""

    workflow_id: str
    anchor_pos: int
    pending_fanout: int


@dataclass(frozen=True)
class RuntimeMetadataView:
    """保存真实运行时构建器所需的最小工作流 metadata。"""

    workflows: tuple[RuntimeWorkflowSpecView, ...]


@dataclass(frozen=True)
class RuntimeScenarioView:
    """为既有 runtime 构建器提供不改变逻辑场景的只读视图。"""

    continuations: tuple[PendingContinuation, ...]
    candidates: tuple[CheckpointCandidate, ...]
    budget_bytes: int
    metadata: RuntimeMetadataView


class EvaluationPolicyOptimizerAdapter:
    """把冻结的 SOTA-style 选择转换为 StateController 接口。"""

    def __init__(
        self,
        policy_name: str,
        scenario: ScalableScenario | SignalScenario,
        recovery_cost_model: RecoveryCostModel,
    ) -> None:
        if policy_name not in GPU_POLICY_NAMES[:-1]:
            raise ValueError(f"不支持的运行时基线策略：{policy_name}")
        self._policy_name = policy_name
        self._scenario = scenario
        self._recovery_cost_model = recovery_cost_model

    def select(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        budget_bytes: int,
    ) -> AllocationResult:
        """调用冻结策略，并用统一恢复模型构造分配结果。"""
        selected_ids = select_gpu_policy_ids(
            self._policy_name,
            self._scenario,
            self._recovery_cost_model,
            continuations=continuations,
            candidates=candidates,
            budget_bytes=budget_bytes,
        )
        candidates_by_id = {
            candidate.checkpoint_id: candidate
            for candidate in candidates
        }
        selected = tuple(
            candidates_by_id[checkpoint_id]
            for checkpoint_id in selected_ids
        )
        cost_before = _total_recovery_cost(
            continuations,
            (),
            self._recovery_cost_model,
        )
        cost_after = _total_recovery_cost(
            continuations,
            selected,
            self._recovery_cost_model,
        )
        benefit = cost_before - cost_after
        if benefit < -1e-9:
            raise RuntimeError(f"冻结策略产生负恢复收益：{benefit} ms")
        return AllocationResult(
            selected=selected,
            total_benefit_ms=max(benefit, 0.0),
            recovery_cost_before_ms=cost_before,
            recovery_cost_after_ms=cost_after,
            used_bytes=sum(
                candidate.memory_bytes for candidate in selected
            ),
        )


@dataclass
class RuntimeArtifactWriter:
    """增量保存单个 workload 的 case 与汇总 artifact。"""

    directory: Path

    @classmethod
    def create(
        cls,
        root: Path,
        prefix: str,
        timestamp: str,
    ) -> "RuntimeArtifactWriter":
        """创建不会覆盖已有证据的时间戳目录。"""
        root.mkdir(parents=True, exist_ok=True)
        directory = root / f"{prefix}_{timestamp}"
        directory.mkdir(parents=False, exist_ok=False)
        return cls(directory=directory)

    @property
    def snapshot_cases_path(self) -> Path:
        """返回 snapshot 审计 JSONL 路径。"""
        return self.directory / "snapshot_cases.jsonl"

    @property
    def e2e_cases_path(self) -> Path:
        """返回真实请求等价类 JSONL 路径。"""
        return self.directory / "e2e_cases.jsonl"

    def append_snapshot_case(self, record: dict) -> None:
        """立即持久化一个 snapshot 审计记录。"""
        with self.snapshot_cases_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    def append_e2e_case(self, record: dict) -> None:
        """立即持久化一个真实请求等价类记录。"""
        with self.e2e_cases_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    def write_summary(self, summary: dict) -> None:
        """写出 JSON 汇总和按预算、策略展开的 CSV。"""
        json_path = self.directory / "summary.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        fieldnames = (
            "scenario",
            "budget_checkpoints",
            "policy",
            "total_cases",
            "gap_match_count",
            "safety_pass_count",
            "total_planning_gap",
            "total_runtime_gap",
            "planning_executable_prefix_ratio",
            "runtime_executable_prefix_ratio",
            "mean_request_e2e_ms",
            "selected_checkpoint_ids",
        )
        with (self.directory / "summary.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for group in summary["groups"]:
                row = dict(group)
                row["selected_checkpoint_ids"] = ";".join(
                    group["selected_checkpoint_ids"]
                )
                writer.writerow(
                    {field: row.get(field) for field in fieldnames}
                )


def select_gpu_policy_ids(
    policy_name: str,
    scenario: ScalableScenario | SignalScenario,
    recovery_cost_model: RecoveryCostModel,
    *,
    continuations: Sequence[PendingContinuation] | None = None,
    candidates: Sequence[CheckpointCandidate] | None = None,
    budget_bytes: int | None = None,
) -> tuple[str, ...]:
    """只用冻结 policy 与 metadata 计算真实运行时所需选择。"""
    active_continuations = tuple(
        scenario.continuations
        if continuations is None
        else continuations
    )
    active_candidates = tuple(
        scenario.candidates if candidates is None else candidates
    )
    active_budget = (
        scenario.budget_bytes if budget_bytes is None else budget_bytes
    )
    last_access = build_marconi_recency(
        active_candidates,
        scenario.metadata.checkpoint_recency,
    )

    if policy_name == "Global-LRU":
        return select_global_lru(
            active_candidates,
            scenario.metadata.checkpoint_recency,
            active_budget,
        )
    if policy_name == "KVFlow-style":
        steps = _kvflow_steps_for_scenario(
            scenario,
            active_continuations,
        )
        return KVFlowStylePolicy().select(
            active_continuations,
            active_candidates,
            scenario.metadata.budget_checkpoints,
            steps,
            last_access,
        ).selected_checkpoint_ids
    if policy_name == "Marconi-style":
        alpha = float(
            getattr(
                scenario.metadata,
                "marconi_alpha",
                CONTROLLED_MARCONI_ALPHA,
            )
        )
        return MarconiStylePolicy().select(
            active_candidates,
            scenario.metadata.budget_checkpoints,
            last_access,
            build_marconi_flop_saved(active_candidates),
            alpha,
        ).selected_checkpoint_ids
    if policy_name == "FlowState":
        allocation = GlobalOptimizer(recovery_cost_model).select(
            active_continuations,
            active_candidates,
            active_budget,
        )
        return tuple(
            candidate.checkpoint_id
            for candidate in allocation.selected
        )
    raise ValueError(f"不支持的图形处理器策略：{policy_name}")


def build_representative_cases(
    recovery_cost_model: RecoveryCostModel | None = None,
) -> tuple[RuntimeCorrectnessCase, ...]:
    """从冻结离线 artifact 构造四个代表点的全部隔离 case。"""
    model = recovery_cost_model or RecoveryCostModel()
    scalable_offline = load_scalable_offline_artifact()
    signal_offline = load_signal_offline_artifact()
    cases = []

    for budget in SCALABLE_BUDGETS:
        scenario = build_scalable_scenario(16, budget)
        rows = {
            row.policy_name: row
            for row in scalable_offline.rows
            if row.workflow_count == 16
            and row.budget_checkpoints == budget
        }
        cases.extend(
            _build_point_cases(
                SCALABLE_SCENARIO_NAME,
                scenario,
                rows,
                model,
            )
        )

    for budget in SIGNAL_BUDGETS:
        scenario = build_signal_scenario(budget)
        rows = {
            row.policy_name: row
            for row in signal_offline.rows
            if row.budget_checkpoints == budget
        }
        cases.extend(
            _build_point_cases(
                SIGNAL_SCENARIO_NAME,
                scenario,
                rows,
                model,
            )
        )

    result = tuple(cases)
    validate_case_plan(result)
    return result


def validate_case_plan(
    cases: Sequence[RuntimeCorrectnessCase],
) -> None:
    """验证 case 数量、唯一性和代表点覆盖完整。"""
    expected_by_scenario = {
        SCALABLE_SCENARIO_NAME: EXPECTED_SCALABLE_CASES,
        SIGNAL_SCENARIO_NAME: EXPECTED_SIGNAL_CASES,
    }
    observed_by_scenario = {
        scenario_name: sum(
            case.scenario_name == scenario_name for case in cases
        )
        for scenario_name in expected_by_scenario
    }
    if observed_by_scenario != expected_by_scenario:
        raise RuntimeError(
            f"代表点 case 数量异常：{observed_by_scenario}"
        )
    if len(cases) != EXPECTED_TOTAL_CASES:
        raise RuntimeError(f"总 case 数量异常：{len(cases)}")
    identities = {
        (
            case.scenario_name,
            case.budget_checkpoints,
            case.policy_name,
            case.continuation_id,
        )
        for case in cases
    }
    if len(identities) != len(cases):
        raise RuntimeError("代表点 case 包含重复身份")


def build_snapshot_audit_plans(
    cases: Sequence[RuntimeCorrectnessCase] | None = None,
) -> tuple[SnapshotAuditPlan, ...]:
    """把八百个逻辑 case 合并为十六个真实 snapshot 审计。"""
    active_cases = tuple(
        build_representative_cases() if cases is None else cases
    )
    groups: dict[
        tuple[str, int, str],
        list[RuntimeCorrectnessCase],
    ] = {}
    for case in active_cases:
        key = (
            case.scenario_name,
            case.budget_checkpoints,
            case.policy_name,
        )
        groups.setdefault(key, []).append(case)

    plans = tuple(
        SnapshotAuditPlan(
            scenario_name=key[0],
            budget_checkpoints=key[1],
            policy_name=key[2],
            selected_checkpoint_ids=tuple(rows[0].selected_checkpoint_ids),
            logical_cases=tuple(rows),
        )
        for key, rows in groups.items()
    )
    if len(plans) != 16:
        raise RuntimeError(f"snapshot 审计数量异常：{len(plans)}")
    if sum(len(plan.logical_cases) for plan in plans) != (
        EXPECTED_TOTAL_CASES
    ):
        raise RuntimeError("snapshot 审计未覆盖全部逻辑 continuation")
    return plans


def build_e2e_equivalence_cases(
    cases: Sequence[RuntimeCorrectnessCase] | None = None,
) -> tuple[RuntimeCorrectnessCase, ...]:
    """按规划恢复行为分组并确定性选择字典序最小代表。"""
    plans = build_snapshot_audit_plans(cases)
    representatives = []
    for plan in plans:
        classes: dict[
            tuple[int, int, int],
            list[RuntimeCorrectnessCase],
        ] = {}
        for case in plan.logical_cases:
            key = (
                case.planning_target,
                case.planning_executable_frontier,
                case.planning_gap_tokens,
            )
            classes.setdefault(key, []).append(case)
        for key in sorted(classes):
            representatives.append(
                min(
                    classes[key],
                    key=lambda item: item.continuation_id,
                )
            )
    return tuple(representatives)


def build_equivalence_class_report(
    cases: Sequence[RuntimeCorrectnessCase] | None = None,
) -> dict:
    """生成启动图形处理器前必须打印的等价类数量报告。"""
    plans = build_snapshot_audit_plans(cases)
    rows = []
    total = 0
    for plan in plans:
        class_count = len(
            {
                (
                    case.planning_target,
                    case.planning_executable_frontier,
                    case.planning_gap_tokens,
                )
                for case in plan.logical_cases
            }
        )
        total += class_count
        rows.append(
            {
                "scenario": plan.scenario_name,
                "budget_checkpoints": plan.budget_checkpoints,
                "policy": plan.policy_name,
                "equivalence_classes": class_count,
            }
        )
    return {
        "snapshot_runtime_runs": len(plans),
        "logical_cases_checked": sum(
            len(plan.logical_cases) for plan in plans
        ),
        "groups": rows,
        "total_e2e_cases": total,
        "gpu_allowed_by_case_count": (
            total == EXPECTED_E2E_EQUIVALENCE_CASES
        ),
        "expected_e2e_cases": EXPECTED_E2E_EQUIVALENCE_CASES,
    }


def validate_exact_runtime_observation(
    case: RuntimeCorrectnessCase,
    metrics: Mapping[str, object],
) -> dict[str, int]:
    """严格验证真实 H、E、G 与固定规划结果逐令牌一致。"""
    physical_frontier = int(metrics["physical_fa_hit"])
    executable = int(metrics["executable_prefix"])
    gap = int(metrics["replay_gap"])
    if min(physical_frontier, executable, gap) < 0:
        raise RuntimeError("真实运行时 H、E、G 不能为负")
    if gap != physical_frontier - executable:
        raise RuntimeError("真实运行时恢复间隔不满足 G=H-E")
    if physical_frontier != case.planning_target:
        raise RuntimeError(
            "真实 FA 前沿与规划目标不一致："
            f"{physical_frontier} != {case.planning_target}"
        )
    if executable != case.planning_executable_frontier:
        raise RuntimeError(
            "真实可执行前沿与规划前沿不一致："
            f"{executable} != {case.planning_executable_frontier}"
        )
    if gap != case.planning_gap_tokens:
        raise RuntimeError(
            "真实恢复间隔与规划间隔不一致："
            f"{gap} != {case.planning_gap_tokens}"
        )
    return {
        "runtime_fa_frontier": physical_frontier,
        "runtime_executable_frontier": executable,
        "runtime_gap_tokens": gap,
    }


def build_runtime_summary(
    records: Sequence[dict],
    *,
    scenario_name: str,
    expected_cases: int,
    artifact_directory: Path,
    failure_stage: str | None,
    error: str | None,
) -> dict:
    """按场景、预算和策略聚合严格正确性结果。"""
    groups = []
    budgets = (
        SCALABLE_BUDGETS
        if scenario_name == SCALABLE_SCENARIO_NAME
        else SIGNAL_BUDGETS
    )
    for budget in budgets:
        for policy_name in GPU_POLICY_NAMES:
            group_records = tuple(
                record
                for record in records
                if record.get("scenario") == scenario_name
                and record.get("budget_checkpoints") == budget
                and record.get("policy") == policy_name
            )
            if not group_records:
                continue
            planning_target_sum = sum(
                int(record["planning_target"])
                for record in group_records
            )
            planning_executable_sum = sum(
                int(record["planning_executable_frontier"])
                for record in group_records
            )
            runtime_physical_sum = sum(
                int(record.get("runtime_fa_frontier", 0))
                for record in group_records
            )
            runtime_executable_sum = sum(
                int(record.get("runtime_executable_frontier", 0))
                for record in group_records
            )
            timings = tuple(
                float(record["request_e2e_ms"])
                for record in group_records
                if record.get("request_e2e_ms") is not None
            )
            groups.append(
                {
                    "scenario": scenario_name,
                    "budget_checkpoints": budget,
                    "policy": policy_name,
                    "total_cases": len(group_records),
                    "gap_match_count": sum(
                        record.get("gap_match") is True
                        for record in group_records
                    ),
                    "safety_pass_count": sum(
                        record.get("safety_pass") is True
                        for record in group_records
                    ),
                    "total_planning_gap": sum(
                        int(record["planning_gap_tokens"])
                        for record in group_records
                    ),
                    "total_runtime_gap": sum(
                        int(record.get("runtime_gap_tokens", 0))
                        for record in group_records
                    ),
                    "planning_executable_prefix_ratio": (
                        planning_executable_sum / planning_target_sum
                        if planning_target_sum
                        else 0.0
                    ),
                    "runtime_executable_prefix_ratio": (
                        runtime_executable_sum / runtime_physical_sum
                        if runtime_physical_sum
                        else 0.0
                    ),
                    "mean_request_e2e_ms": (
                        sum(timings) / len(timings)
                        if timings
                        else None
                    ),
                    "selected_checkpoint_ids": list(
                        group_records[0]["selected_checkpoint_ids"]
                    ),
                }
            )

    passed = sum(record.get("status") == "PASS" for record in records)
    failed = sum(record.get("status") == "FAIL" for record in records)
    completed_safety = tuple(
        record["safety"]
        for record in records
        if record.get("status") == "PASS"
    )
    status = (
        "PASS"
        if passed == expected_cases and failed == 0
        else "FAIL"
    )
    return {
        "schema_version": "flowstate.sota_runtime_correctness.v1",
        "scenario": scenario_name,
        "status": status,
        "cases_expected": expected_cases,
        "cases_passed": passed,
        "cases_failed": failed,
        "groups": groups,
        "global_gap_match": (
            status == "PASS"
            and all(record.get("gap_match") is True for record in records)
        ),
        "safety": {
            "fa_preserved": bool(completed_safety) and all(
                item["fa_preserved"] for item in completed_safety
            ),
            "mamba_selection_respected": bool(completed_safety) and all(
                item["selected_mamba_resident"]
                and item["unselected_mamba_evicted"]
                for item in completed_safety
            ),
            "allocator_unchanged": bool(completed_safety) and all(
                item["allocator_invariant"] for item in completed_safety
            ),
            "tree_identity_preserved": bool(completed_safety) and all(
                item["tree_invariant"] and item["path_invariant"]
                for item in completed_safety
            ),
            "sanity_check": bool(completed_safety) and all(
                item["sanity_check"] for item in completed_safety
            ),
            "cascade_called": any(
                not item["cascade_not_called"]
                for item in completed_safety
            ),
        },
        "formal_mutation_primitive": _FORMAL_PRIMITIVE,
        "timing_interpretation": "仅用于正确性诊断",
        "failure_stage": failure_stage,
        "error": error,
        "artifact_directory": str(artifact_directory),
        "engine_configuration": STEP8E_ENGINE_CONFIGURATION,
    }


def run_runtime_case(
    *,
    index: int,
    case: RuntimeCorrectnessCase,
    engine: object,
    client: object,
    recovery_cost_model: RecoveryCostModel,
) -> dict:
    """在独立 flush/rebuild 生命周期中执行一个严格 case。"""
    case_id = _case_id(index, case)
    scenario = _scenario_for_case(case)
    runtime_scenario = build_runtime_scenario_view(scenario)
    namespace = f"flowstate_step8e_{case.scenario_name}"
    stage = "刷新并验证空缓存"
    record: dict[str, object] = {
        "case_id": case_id,
        "scenario": case.scenario_name,
        "budget_checkpoints": case.budget_checkpoints,
        "policy": case.policy_name,
        "continuation_id": case.continuation_id,
        "workflow_id": case.workflow_id,
        "selected_checkpoint_ids": list(case.selected_checkpoint_ids),
        "planning_target": case.planning_target,
        "planning_executable_frontier": (
            case.planning_executable_frontier
        ),
        "planning_gap_tokens": case.planning_gap_tokens,
        "gap_match": False,
        "safety_pass": False,
        "status": "FAIL",
    }
    try:
        engine.flush_cache()
        census = client.census(f"{namespace}:{case_id}:after_flush")
        record["clean_cache"] = validate_clean_cache(census)

        stage = "重建场景全部检查点"
        runtime_workflows, candidate_tokens = build_runtime_workflows(
            engine,
            runtime_scenario,
            request_namespace=f"{namespace}_{case_id}_build",
        )

        stage = "验证分配前全部检查点驻留"
        handles, before_paths, before_states = build_runtime_handles(
            client,
            runtime_scenario,
            candidate_tokens,
        )
        record["checkpoints_before_allocation"] = before_states

        stage = "执行冻结策略与 controller"
        runtime_adapter = SnapshotSchedulerRuntimeAdapter(
            client,
            case_id,
            namespace,
        )
        optimizer = (
            GlobalOptimizer(recovery_cost_model)
            if case.policy_name == "FlowState"
            else EvaluationPolicyOptimizerAdapter(
                case.policy_name,
                scenario,
                recovery_cost_model,
            )
        )
        allocation = StateController(
            optimizer,
            runtime_adapter,
        ).reconcile(
            scenario.continuations,
            scenario.candidates,
            handles,
            scenario.budget_bytes,
        )
        selected_ids = tuple(
            candidate.checkpoint_id
            for candidate in allocation.selected
        )
        if selected_ids != case.selected_checkpoint_ids:
            raise RuntimeError(
                "真实执行前的策略选择偏离冻结离线结果："
                f"{selected_ids} != {case.selected_checkpoint_ids}"
            )
        evicted_ids = tuple(runtime_adapter.evicted_checkpoint_ids)
        expected_evicted_ids = tuple(
            sorted(
                candidate.checkpoint_id
                for candidate in scenario.candidates
                if candidate.recurrent_resident
                and candidate.checkpoint_id not in selected_ids
            )
        )
        if evicted_ids != expected_evicted_ids:
            raise RuntimeError(
                "controller 驱逐动作与 selected set 不一致："
                f"{evicted_ids} != {expected_evicted_ids}"
            )
        record["selected_checkpoint_ids"] = list(selected_ids)
        record["evicted_checkpoint_ids"] = list(evicted_ids)

        stage = "验证组件级安全条件"
        after_paths, after_states = inspect_after_allocation(
            client,
            candidate_tokens,
        )
        safety, allocator = allocation_safety_snapshot(
            before_paths=before_paths,
            after_paths=after_paths,
            after_states=after_states,
            selected_ids=selected_ids,
            evicted_ids=evicted_ids,
            eviction_responses=runtime_adapter.eviction_responses,
        )
        record["checkpoints_after_allocation"] = after_states
        record["safety"] = safety
        record["fa_allocator"] = allocator
        record["safety_pass"] = all(safety.values())
        if not record["safety_pass"]:
            raise RuntimeError(f"组件级安全条件失败：{safety}")

        stage = "发送唯一目标待续请求"
        continuation = next(
            continuation
            for continuation in scenario.continuations
            if continuation.continuation_id == case.continuation_id
        )
        request_id = f"{namespace}_{case_id}_pending"
        _, metadata = generate(
            engine,
            request_id,
            _pending_tokens(
                continuation,
                scenario,
                runtime_workflows,
            ),
        )
        metrics = query_runtime_metrics(client, request_id)
        observation = validate_exact_runtime_observation(case, metrics)
        record.update(observation)
        record["gap_match"] = True
        record["frontier_match"] = True
        record["request_e2e_ms"] = optional_latency_ms(
            metadata,
            "e2e_latency",
        )
        record["status"] = "PASS"
        record["failure_stage"] = None
        return record
    except Exception as error:
        record["failure_stage"] = stage
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
        return record


def run_snapshot_audit(
    *,
    index: int,
    plan: SnapshotAuditPlan,
    engine: object,
    client: object,
    recovery_cost_model: RecoveryCostModel,
) -> dict:
    """执行一次真实 reconcile，并由 inspect 驻留状态审计全部逻辑请求。"""
    case_id = (
        f"snapshot_{index:02d}_{plan.scenario_name}_"
        f"k{plan.budget_checkpoints}_"
        f"{plan.policy_name.lower().replace('-', '_')}"
    )
    seed_case = plan.logical_cases[0]
    scenario = _scenario_for_case(seed_case)
    runtime_scenario = build_runtime_scenario_view(scenario)
    namespace = f"flowstate_step8e1_{plan.scenario_name}"
    stage = "刷新并验证空缓存"
    record: dict[str, object] = {
        "layer": "snapshot_audit",
        "case_id": case_id,
        "scenario": plan.scenario_name,
        "budget_checkpoints": plan.budget_checkpoints,
        "policy": plan.policy_name,
        "selected_checkpoint_ids": list(plan.selected_checkpoint_ids),
        "logical_cases_expected": len(plan.logical_cases),
        "status": "FAIL",
    }
    try:
        engine.flush_cache()
        census = client.census(f"{namespace}:{case_id}:after_flush")
        record["clean_cache"] = validate_clean_cache(census)

        stage = "重建场景全部检查点"
        _, candidate_tokens = build_runtime_workflows(
            engine,
            runtime_scenario,
            request_namespace=f"{namespace}_{case_id}_build",
        )
        handles, before_paths, before_states = build_runtime_handles(
            client,
            runtime_scenario,
            candidate_tokens,
        )
        record["checkpoints_before_allocation"] = before_states

        stage = "执行冻结策略与 controller"
        runtime_adapter = SnapshotSchedulerRuntimeAdapter(
            client,
            case_id,
            namespace,
        )
        optimizer = (
            GlobalOptimizer(recovery_cost_model)
            if plan.policy_name == "FlowState"
            else EvaluationPolicyOptimizerAdapter(
                plan.policy_name,
                scenario,
                recovery_cost_model,
            )
        )
        allocation = StateController(
            optimizer,
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
        if selected_ids != plan.selected_checkpoint_ids:
            raise RuntimeError(
                "snapshot 策略选择偏离冻结离线结果："
                f"{selected_ids} != {plan.selected_checkpoint_ids}"
            )
        evicted_ids = tuple(runtime_adapter.evicted_checkpoint_ids)
        expected_evicted_ids = tuple(
            sorted(
                candidate.checkpoint_id
                for candidate in scenario.candidates
                if candidate.recurrent_resident
                and candidate.checkpoint_id not in selected_ids
            )
        )
        if evicted_ids != expected_evicted_ids:
            raise RuntimeError("snapshot controller 驱逐动作与选择不一致")
        record["evicted_checkpoint_ids"] = list(evicted_ids)

        stage = "检查全部候选的真实驻留与组件安全"
        after_paths, after_states = inspect_after_allocation(
            client,
            candidate_tokens,
        )
        safety, allocator = allocation_safety_snapshot(
            before_paths=before_paths,
            after_paths=after_paths,
            after_states=after_states,
            selected_ids=selected_ids,
            evicted_ids=evicted_ids,
            eviction_responses=runtime_adapter.eviction_responses,
        )
        actual_resident_ids = tuple(
            sorted(
                checkpoint_id
                for checkpoint_id, state in after_states.items()
                if state["mamba_resident"]
            )
        )
        residency_match = set(actual_resident_ids) == set(selected_ids)
        safety["selection_residency_match"] = residency_match
        record["checkpoints_after_allocation"] = after_states
        record["actual_resident_checkpoint_ids"] = list(
            actual_resident_ids
        )
        record["candidates_inspected"] = len(after_states)
        record["safety"] = safety
        record["fa_allocator"] = allocator
        record["safety_pass"] = all(safety.values())
        if not record["safety_pass"]:
            raise RuntimeError(f"snapshot 安全条件失败：{safety}")

        stage = "由真实驻留状态审计全部 continuation"
        candidates_by_id = {
            candidate.checkpoint_id: candidate
            for candidate in scenario.candidates
        }
        actual_resident = tuple(
            candidates_by_id[checkpoint_id]
            for checkpoint_id in actual_resident_ids
        )
        continuations_by_id = {
            continuation.continuation_id: continuation
            for continuation in scenario.continuations
        }
        logical_results = []
        frontier_mismatches = 0
        gap_mismatches = 0
        gap_differences = []
        for logical_case in plan.logical_cases:
            continuation = continuations_by_id[
                logical_case.continuation_id
            ]
            runtime_frontier = executable_frontier(
                continuation,
                actual_resident,
            )
            runtime_gap = recovery_gap(
                continuation,
                actual_resident,
            )
            frontier_match = runtime_frontier == (
                logical_case.planning_executable_frontier
            )
            gap_match = runtime_gap == logical_case.planning_gap_tokens
            frontier_mismatches += not frontier_match
            gap_mismatches += not gap_match
            gap_differences.append(
                abs(runtime_gap - logical_case.planning_gap_tokens)
            )
            logical_results.append(
                {
                    "continuation_id": logical_case.continuation_id,
                    "workflow_id": logical_case.workflow_id,
                    "planning_target": logical_case.planning_target,
                    "planning_executable_frontier": (
                        logical_case.planning_executable_frontier
                    ),
                    "runtime_executable_frontier": runtime_frontier,
                    "planning_gap_tokens": (
                        logical_case.planning_gap_tokens
                    ),
                    "runtime_gap_tokens": runtime_gap,
                    "frontier_match": frontier_match,
                    "gap_match": gap_match,
                }
            )
        record["logical_results"] = logical_results
        record["logical_continuations_checked"] = len(logical_results)
        record["planning_runtime_frontier_mismatch"] = (
            frontier_mismatches
        )
        record["planning_runtime_gap_mismatch"] = gap_mismatches
        record["max_absolute_gap_difference"] = max(
            gap_differences,
            default=0,
        )
        if frontier_mismatches or gap_mismatches:
            raise RuntimeError(
                "真实驻留状态推导结果与规划不一致："
                f"frontier={frontier_mismatches}，gap={gap_mismatches}"
            )

        record["status"] = "PASS"
        record["failure_stage"] = None
        return record
    except Exception as error:
        record["failure_stage"] = stage
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
        return record


def build_runtime_scenario_view(
    scenario: ScalableScenario | SignalScenario,
) -> RuntimeScenarioView:
    """在不修改冻结 workload 的前提下统一 runtime 字段名称。"""
    workflows = []
    for workflow in scenario.metadata.workflows:
        anchor_pos = (
            workflow.anchor_pos
            if hasattr(workflow, "anchor_pos")
            else workflow.anchor_depth
        )
        fanout = (
            workflow.pending_fanout
            if hasattr(workflow, "pending_fanout")
            else workflow.fanout
        )
        workflows.append(
            RuntimeWorkflowSpecView(
                workflow_id=workflow.workflow_id,
                anchor_pos=int(anchor_pos),
                pending_fanout=int(fanout),
            )
        )
    return RuntimeScenarioView(
        continuations=scenario.continuations,
        candidates=scenario.candidates,
        budget_bytes=scenario.budget_bytes,
        metadata=RuntimeMetadataView(workflows=tuple(workflows)),
    )


def _build_point_cases(
    scenario_name: str,
    scenario: ScalableScenario | SignalScenario,
    offline_rows: Mapping[str, object],
    recovery_cost_model: RecoveryCostModel,
) -> tuple[RuntimeCorrectnessCase, ...]:
    """用核心 E/G 和冻结离线 selection 构造一个代表点。"""
    candidates_by_id = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    cases = []
    for policy_name in GPU_POLICY_NAMES:
        row = offline_rows.get(policy_name)
        if row is None:
            raise RuntimeError(
                f"{scenario_name} K={scenario.metadata.budget_checkpoints} "
                f"缺少冻结策略 {policy_name}"
            )
        selected_ids = tuple(row.selected_checkpoint_ids)
        recomputed_ids = select_gpu_policy_ids(
            policy_name,
            scenario,
            recovery_cost_model,
        )
        if recomputed_ids != selected_ids:
            raise RuntimeError(
                f"{policy_name} 当前选择偏离冻结离线 artifact："
                f"{recomputed_ids} != {selected_ids}"
            )
        selected = tuple(
            candidates_by_id[checkpoint_id]
            for checkpoint_id in selected_ids
        )
        for continuation in scenario.continuations:
            frontier = executable_frontier(continuation, selected)
            gap = recovery_gap(continuation, selected)
            cases.append(
                RuntimeCorrectnessCase(
                    scenario_name=scenario_name,
                    budget_checkpoints=(
                        scenario.metadata.budget_checkpoints
                    ),
                    policy_name=policy_name,
                    continuation_id=continuation.continuation_id,
                    workflow_id=continuation.workflow_id,
                    selected_checkpoint_ids=selected_ids,
                    planning_target=continuation.planning_target,
                    planning_executable_frontier=frontier,
                    planning_gap_tokens=gap,
                )
            )
    return tuple(cases)


def _kvflow_steps_for_scenario(
    scenario: ScalableScenario | SignalScenario,
    continuations: Sequence[PendingContinuation],
) -> Mapping[str, int]:
    """读取信号场景冻结距离，或构造直接下一步场景的固定距离。"""
    steps = getattr(
        scenario.metadata,
        "steps_to_execution_by_continuation",
        None,
    )
    if steps is not None:
        return steps
    return build_kvflow_steps(continuations)


def _total_recovery_cost(
    continuations: Sequence[PendingContinuation],
    selected: Sequence[CheckpointCandidate],
    recovery_cost_model: RecoveryCostModel,
) -> float:
    """使用核心恢复间隔和统一 Phi 计算 selected set 成本。"""
    return sum(
        recovery_cost_model.estimate(
            recovery_gap(continuation, selected)
        )
        for continuation in continuations
    )


def _scenario_for_case(
    case: RuntimeCorrectnessCase,
) -> ScalableScenario | SignalScenario:
    """按 case 身份重建未修改的冻结逻辑场景。"""
    if case.scenario_name == SCALABLE_SCENARIO_NAME:
        return build_scalable_scenario(16, case.budget_checkpoints)
    if case.scenario_name == SIGNAL_SCENARIO_NAME:
        return build_signal_scenario(case.budget_checkpoints)
    raise ValueError(f"未知场景：{case.scenario_name}")


def _pending_tokens(
    continuation: PendingContinuation,
    scenario: ScalableScenario | SignalScenario,
    runtime_workflows: Sequence[RuntimeWorkflow],
) -> tuple[int, ...]:
    """为单个隔离 case 构造确定且独立的待续分支令牌。"""
    runtime_by_workflow = {
        workflow.spec.workflow_id: workflow
        for workflow in runtime_workflows
    }
    continuation_index = next(
        index
        for index, item in enumerate(scenario.continuations)
        if item.continuation_id == continuation.continuation_id
    )
    runtime_workflow = runtime_by_workflow[continuation.workflow_id]
    suffix = make_tokens(
        541_019 + continuation_index * 10_007,
        PENDING_SUFFIX_LENGTH,
    )
    return (
        runtime_workflow.anchor_tokens
        + (runtime_workflow.anchor_output,)
        + suffix
    )


def _case_id(index: int, case: RuntimeCorrectnessCase) -> str:
    """生成本次单进程中唯一且确定的 case 标识。"""
    scenario = case.scenario_name.replace("_", "-")
    policy = case.policy_name.lower().replace("-", "_")
    continuation = case.continuation_id.lower().replace("-", "_")
    return (
        f"case_{index:04d}_{scenario}_k{case.budget_checkpoints}_"
        f"{policy}_{continuation}"
    )


def build_flowstate_oracle_report() -> dict[str, dict[str, bool]]:
    """分别记录 FlowState 与冻结 Oracle 的目标和选择是否一致。"""
    scalable = load_scalable_offline_artifact()
    signal = load_signal_offline_artifact()
    result: dict[str, dict[str, bool]] = {}
    for budget in SCALABLE_BUDGETS:
        rows = {
            row.policy_name: row
            for row in scalable.rows
            if row.workflow_count == 16
            and row.budget_checkpoints == budget
        }
        objective_match = (
            abs(
                rows["FlowState"].estimated_recovery_cost_ms
                - rows["Oracle"].estimated_recovery_cost_ms
            )
            <= 1e-9
        )
        selection_match = (
            rows["FlowState"].selected_checkpoint_ids
            == rows["Oracle"].selected_checkpoint_ids
        )
        result[f"{SCALABLE_SCENARIO_NAME}:K={budget}"] = {
            "oracle_objective_match": objective_match,
            "oracle_selection_match": selection_match,
            "multiple_optimal_selections": (
                objective_match and not selection_match
            ),
        }
    for budget in SIGNAL_BUDGETS:
        rows = {
            row.policy_name: row
            for row in signal.rows
            if row.budget_checkpoints == budget
        }
        objective_match = (
            abs(
                rows["FlowState"].estimated_recovery_cost_ms
                - rows["Oracle"].estimated_recovery_cost_ms
            )
            <= 1e-9
        )
        selection_match = (
            rows["FlowState"].selected_checkpoint_ids
            == rows["Oracle"].selected_checkpoint_ids
        )
        result[f"{SIGNAL_SCENARIO_NAME}:K={budget}"] = {
            "oracle_objective_match": objective_match,
            "oracle_selection_match": selection_match,
            "multiple_optimal_selections": (
                objective_match and not selection_match
            ),
        }
    return result


def main() -> int:
    """先审计等价类规模，再按两层协议执行真实运行时验证。"""
    cases = build_representative_cases()
    snapshot_plans = build_snapshot_audit_plans(cases)
    e2e_cases = build_e2e_equivalence_cases(cases)
    dry_run = build_equivalence_class_report(cases)
    oracle_report = build_flowstate_oracle_report()
    print(
        json.dumps(
            {
                "dry_run": dry_run,
                "oracle_report": oracle_report,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if len(e2e_cases) != EXPECTED_E2E_EQUIVALENCE_CASES:
        print(
            "等价类代表请求数量偏离冻结值，按协议在启动图形处理器前停止："
            f"{len(e2e_cases)} != {EXPECTED_E2E_EQUIVALENCE_CASES}",
            flush=True,
        )
        return 0

    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import (
        FormalEndToEndGateEngine,
        requested_control_port,
    )

    model = RecoveryCostModel()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    writer = RuntimeArtifactWriter.create(
        _RUNTIME_ARTIFACT_ROOT,
        "sota_correctness",
        timestamp,
    )
    snapshot_records_by_scenario: dict[str, list[dict]] = {
        SCALABLE_SCENARIO_NAME: [],
        SIGNAL_SCENARIO_NAME: [],
    }
    e2e_records_by_scenario: dict[str, list[dict]] = {
        SCALABLE_SCENARIO_NAME: [],
        SIGNAL_SCENARIO_NAME: [],
    }
    engine = None
    failure_stage = None
    error_text = None
    started = time.perf_counter()
    try:
        engine = FormalEndToEndGateEngine(
            **STEP8E_ENGINE_CONFIGURATION
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        for index, plan in enumerate(snapshot_plans, start=1):
            print(
                f"[STEP8E.1-A] SNAPSHOT {index}/{len(snapshot_plans)} "
                f"{plan.scenario_name} K={plan.budget_checkpoints} "
                f"{plan.policy_name}",
                flush=True,
            )
            record = run_snapshot_audit(
                index=index,
                plan=plan,
                engine=engine,
                client=client,
                recovery_cost_model=model,
            )
            snapshot_records_by_scenario[plan.scenario_name].append(record)
            writer.append_snapshot_case(record)
            if record["status"] != "PASS":
                failure_stage = str(record["failure_stage"])
                error_text = str(record.get("error"))
                break
        if error_text is None:
            for index, case in enumerate(e2e_cases, start=1):
                print(
                    f"[STEP8E.1-B] E2E {index}/{len(e2e_cases)} "
                    f"{case.scenario_name} K={case.budget_checkpoints} "
                    f"{case.policy_name} {case.continuation_id}",
                    flush=True,
                )
                record = run_runtime_case(
                    index=index,
                    case=case,
                    engine=engine,
                    client=client,
                    recovery_cost_model=model,
                )
                record["layer"] = "e2e_equivalence"
                e2e_records_by_scenario[case.scenario_name].append(record)
                writer.append_e2e_case(record)
                if record["status"] != "PASS":
                    failure_stage = str(record["failure_stage"])
                    error_text = str(record.get("error"))
                    break
    except Exception as error:
        failure_stage = failure_stage or "初始化 Step 8E 运行时"
        error_text = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as shutdown_error:
                if error_text is None:
                    failure_stage = "关闭 Step 8E 运行时"
                    error_text = repr(shutdown_error)

    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    scenario_summaries = {}
    for scenario_name in (
        SCALABLE_SCENARIO_NAME,
        SIGNAL_SCENARIO_NAME,
    ):
        expected = sum(
            case.scenario_name == scenario_name for case in e2e_cases
        )
        summary = build_runtime_summary(
            e2e_records_by_scenario[scenario_name],
            scenario_name=scenario_name,
            expected_cases=expected,
            artifact_directory=writer.directory,
            failure_stage=failure_stage,
            error=error_text,
        )
        snapshot_records = snapshot_records_by_scenario[scenario_name]
        summary["snapshot_audit"] = {
            "runtime_snapshots": len(snapshot_records),
            "runtime_snapshots_passed": sum(
                record.get("status") == "PASS"
                for record in snapshot_records
            ),
            "candidates_inspected": sum(
                int(record.get("candidates_inspected", 0))
                for record in snapshot_records
            ),
            "logical_continuations_checked": sum(
                int(record.get("logical_continuations_checked", 0))
                for record in snapshot_records
            ),
            "planning_runtime_frontier_mismatch": sum(
                int(record.get("planning_runtime_frontier_mismatch", 0))
                for record in snapshot_records
            ),
            "planning_runtime_gap_mismatch": sum(
                int(record.get("planning_runtime_gap_mismatch", 0))
                for record in snapshot_records
            ),
        }
        summary["equivalence_class_dry_run"] = dry_run
        summary["wall_time_ms_total_process"] = elapsed_ms
        summary["flowstate_oracle_report"] = {
            key: value
            for key, value in oracle_report.items()
            if key.startswith(scenario_name)
        }
        scenario_summaries[scenario_name] = summary

    passed = sum(
        record.get("status") == "PASS"
        for records in e2e_records_by_scenario.values()
        for record in records
    )
    snapshot_passed = sum(
        record.get("status") == "PASS"
        for records in snapshot_records_by_scenario.values()
        for record in records
    )
    snapshot_records = tuple(
        record
        for records in snapshot_records_by_scenario.values()
        for record in records
    )
    e2e_records = tuple(
        record
        for records in e2e_records_by_scenario.values()
        for record in records
    )
    combined_summary = {
        "schema_version": "flowstate.sota_runtime_correctness.v2",
        "status": (
            "PASS"
            if passed == len(e2e_cases)
            and snapshot_passed == len(snapshot_plans)
            else "FAIL"
        ),
        "snapshot_audit": {
            "snapshots_expected": len(snapshot_plans),
            "snapshots_completed": len(snapshot_records),
            "snapshots_passed": snapshot_passed,
            "candidates_inspected": sum(
                int(record.get("candidates_inspected", 0))
                for record in snapshot_records
            ),
            "logical_continuations_checked": sum(
                int(record.get("logical_continuations_checked", 0))
                for record in snapshot_records
            ),
            "selection_residency_mismatch": sum(
                not bool(
                    record.get("safety", {}).get(
                        "selection_residency_match",
                        False,
                    )
                )
                for record in snapshot_records
            ),
            "frontier_mismatch": sum(
                int(record.get("planning_runtime_frontier_mismatch", 0))
                for record in snapshot_records
            ),
            "gap_mismatch": sum(
                int(record.get("planning_runtime_gap_mismatch", 0))
                for record in snapshot_records
            ),
            "max_absolute_gap_difference": max(
                (
                    int(record.get("max_absolute_gap_difference", 0))
                    for record in snapshot_records
                ),
                default=0,
            ),
        },
        "e2e": {
            "equivalence_classes": len(e2e_cases),
            "completed": len(e2e_records),
            "passed": passed,
            "frontier_mismatch": sum(
                record.get("frontier_match") is not True
                for record in e2e_records
            ),
            "gap_mismatch": sum(
                record.get("gap_match") is not True
                for record in e2e_records
            ),
            "max_absolute_gap_difference": max(
                (
                    abs(
                        int(record.get("runtime_gap_tokens", 0))
                        - int(record["planning_gap_tokens"])
                    )
                    for record in e2e_records
                    if "runtime_gap_tokens" in record
                ),
                default=0,
            ),
        },
        "scenario_summaries": scenario_summaries,
        "groups": tuple(
            group
            for summary in scenario_summaries.values()
            for group in summary["groups"]
        ),
        "oracle_report": oracle_report,
        "equivalence_class_dry_run": dry_run,
        "safety": {
            "fa_preserved": bool(snapshot_records) and all(
                record.get("safety", {}).get("fa_preserved") is True
                for record in snapshot_records
            ),
            "mamba_selection_respected": bool(snapshot_records) and all(
                record.get("safety", {}).get(
                    "selection_residency_match"
                )
                is True
                for record in snapshot_records
            ),
            "allocator_unchanged": bool(snapshot_records) and all(
                record.get("safety", {}).get("allocator_invariant") is True
                for record in snapshot_records
            ),
            "node_prefix_tree_preserved": bool(snapshot_records) and all(
                record.get("safety", {}).get("path_invariant") is True
                and record.get("safety", {}).get("tree_invariant") is True
                for record in snapshot_records
            ),
            "sanity_check": bool(snapshot_records) and all(
                record.get("safety", {}).get("sanity_check") is True
                for record in snapshot_records
            ),
            "cascade_called": any(
                record.get("safety", {}).get("cascade_not_called") is False
                for record in snapshot_records
            ),
        },
        "formal_mutation_primitive": _FORMAL_PRIMITIVE,
        "timing_interpretation": "仅用于正确性诊断",
        "total_runtime_ms": elapsed_ms,
        "failure_stage": failure_stage,
        "error": error_text,
        "artifact_directory": str(writer.directory),
        "engine_configuration": STEP8E_ENGINE_CONFIGURATION,
    }
    writer.write_summary(combined_summary)
    print(
        json.dumps(
            {
                "status": (
                    "PASS"
                    if passed == len(e2e_cases)
                    and snapshot_passed == len(snapshot_plans)
                    else "FAIL"
                ),
                "snapshot_audits_passed": snapshot_passed,
                "snapshot_audits_expected": len(snapshot_plans),
                "e2e_cases_passed": passed,
                "e2e_cases_expected": len(e2e_cases),
                "artifacts": str(writer.directory),
                "failure_stage": failure_stage,
                "error": error_text,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return (
        0
        if passed == len(e2e_cases)
        and snapshot_passed == len(snapshot_plans)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
