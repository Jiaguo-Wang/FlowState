#!/usr/bin/env python3
"""执行受控多工作流固定快照的真实运行时比较。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"
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
    select_equal_share,
    select_global_lru,
    select_recovery_only,
)
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    ENGINE_CONFIGURATION,
    PENDING_SUFFIX_LENGTH,
    RuntimeWorkflow,
    build_runtime_handles,
    build_runtime_workflows,
    changed_mamba_nodes,
    generate,
    inspect_after_allocation,
    make_tokens,
    optional_latency_ms,
    query_runtime_metrics,
    validate_runtime_observation,
    wait_for_transport,
)
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
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.controller import StateController
from flowstate.executable_state import recovery_gap
from flowstate.optimizer import AllocationResult, GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate
from flowstate.workflow import PendingContinuation


_FORMAL_PRIMITIVE = (
    "flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only"
)


class PolicyOptimizerAdapter:
    """把冻结基线选择转换为 StateController 所需的优化器接口。"""

    def __init__(
        self,
        policy_name: str,
        scenario: ControlledScenario,
        recovery_cost_model: RecoveryCostModel,
    ) -> None:
        if policy_name not in POLICY_NAMES[1:]:
            raise ValueError(f"不支持的基线策略：{policy_name}")
        self._policy_name = policy_name
        self._scenario = scenario
        self._recovery_cost_model = recovery_cost_model

    def select(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        budget_bytes: int,
    ) -> AllocationResult:
        """调用已有基线并构造与 GlobalOptimizer 一致的分配结果。"""
        selected_ids = self._select_ids(
            continuations,
            candidates,
            budget_bytes,
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
            raise ValueError(f"基线策略恢复收益不能为负：{benefit} ms")
        if benefit < 0.0:
            benefit = 0.0
        return AllocationResult(
            selected=selected,
            total_benefit_ms=benefit,
            recovery_cost_before_ms=cost_before,
            recovery_cost_after_ms=cost_after,
            used_bytes=sum(
                candidate.memory_bytes for candidate in selected
            ),
        )

    def _select_ids(
        self,
        continuations: Sequence[PendingContinuation],
        candidates: Sequence[CheckpointCandidate],
        budget_bytes: int,
    ) -> tuple[str, ...]:
        """把策略名称分派到 Step 7C 已冻结的基线实现。"""
        if self._policy_name == "Global-LRU":
            return select_global_lru(
                candidates,
                self._scenario.metadata.checkpoint_recency,
                budget_bytes,
            )
        if self._policy_name == "Equal-Share":
            return select_equal_share(
                continuations,
                candidates,
                self._scenario.metadata.workflow_order,
                budget_bytes,
            )
        return select_recovery_only(
            continuations,
            candidates,
            budget_bytes,
            self._recovery_cost_model,
        )


class SnapshotSchedulerRuntimeAdapter:
    """使用 case 唯一标识把 controller 动作送到调度器安全时点。"""

    def __init__(self, client: object, case_id: str) -> None:
        self._client = client
        self._case_id = case_id
        self.evicted_checkpoint_ids: list[str] = []
        self.eviction_responses: list[dict] = []

    def evict_mamba_only(
        self,
        handle: RuntimeCheckpointHandle,
    ) -> None:
        """请求 scheduler 调用正式 FlowState Mamba-only primitive。"""
        response = self._client._call(
            {
                "op": "checkpoint_control",
                "nonce": (
                    f"flowstate_step7e:{self._case_id}:"
                    f"evict:{handle.checkpoint_id}"
                ),
                "label": f"flowstate_step7e:{handle.checkpoint_id}",
                "action": "flowstate_evict_mamba_only",
                "checkpoint_id": handle.checkpoint_id,
                "token_ids": list(handle.token_ids),
                "extra_key": handle.extra_key,
                "expected_node_id": handle.expected_node_id,
                "expected_prefix_sha256": handle.expected_prefix_digest,
            }
        )
        self.evicted_checkpoint_ids.append(handle.checkpoint_id)
        self.eviction_responses.append(response)


@dataclass
class CaseArtifactWriter:
    """增量保存 case 记录，并在结束时写出汇总文件。"""

    directory: Path

    @classmethod
    def create(cls, root: Path = _ARTIFACT_ROOT) -> "CaseArtifactWriter":
        """创建一个不会覆盖旧结果的时间戳目录。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = root / f"snapshot_runtime_{timestamp}"
        directory.mkdir(parents=True, exist_ok=False)
        return cls(directory)

    @property
    def cases_path(self) -> Path:
        """返回逐 case 记录路径。"""
        return self.directory / "cases.jsonl"

    def append_case(self, record: dict) -> None:
        """在一个 case 结束后立即追加持久化记录。"""
        with self.cases_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    def write_summary(self, summary: dict) -> None:
        """写出机器可读汇总与按策略展开的表格。"""
        summary_path = self.directory / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        csv_path = self.directory / "summary.csv"
        fieldnames = (
            "policy",
            "n_cases",
            "planning_total_gap",
            "runtime_total_gap",
            "mean_gap_per_request",
            "physical_hit_tokens",
            "executable_hit_tokens",
            "executable_prefix_ratio",
            "estimated_recovery_cost_ms",
            "mean_request_e2e_ms",
            "selected_checkpoint_ids",
            "per_workflow_runtime_gap",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for policy_name in POLICY_NAMES:
                policy = summary["runtime_summary"].get(policy_name)
                if policy is None:
                    continue
                row = dict(policy)
                row["policy"] = policy_name
                row["selected_checkpoint_ids"] = ";".join(
                    policy["selected_checkpoint_ids"]
                )
                row["per_workflow_runtime_gap"] = json.dumps(
                    policy["per_workflow_runtime_gap"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                writer.writerow(
                    {field: row.get(field) for field in fieldnames}
                )


def _total_recovery_cost(
    continuations: Sequence[PendingContinuation],
    selected: Sequence[CheckpointCandidate],
    recovery_cost_model: RecoveryCostModel,
) -> float:
    """用核心恢复间隔和真实 Phi 计算一个 selected set 的总成本。"""
    return sum(
        recovery_cost_model.estimate(
            recovery_gap(continuation, selected)
        )
        for continuation in continuations
    )


def validate_clean_cache(census: dict) -> dict[str, bool]:
    """验证 flush 后 radix tree、FA 与 Mamba 计数均已清空。"""
    tree = census["tree"]
    accounting = census["accounting"]
    flags = {
        "single_root_node": int(tree["node_count"]) == 1,
        "mamba_nodes_empty": int(tree["mamba_node_count"]) == 0,
        "full_values_empty": all(
            row[1] in (None, 0) for row in tree["full_rows"]
        ),
        "mamba_accounting_empty": (
            int(accounting["mamba_evictable"]) == 0
            and int(accounting["mamba_protected"]) == 0
        ),
        "full_accounting_empty": (
            int(accounting["full_evictable"]) == 0
            and int(accounting["full_protected"]) == 0
        ),
    }
    if not all(flags.values()):
        raise RuntimeError(f"flush 后 cache 未清空：{flags}")
    return flags


def validate_snapshot_runtime_observation(
    *,
    anchor_pos: int,
    planning_gap: int,
    metrics: dict,
) -> dict:
    """按至多一个令牌的边界容差验证固定快照运行时结果。"""
    observation = validate_runtime_observation(
        anchor_pos=anchor_pos,
        planning_gap=planning_gap,
        metrics=metrics,
    )
    runtime_gap = int(observation["recovery_gap"])
    if planning_gap == 0:
        agreement = runtime_gap in (0, 1)
    else:
        agreement = abs(runtime_gap - planning_gap) <= 1
    if not agreement:
        raise RuntimeError(
            "运行时恢复间隔超出固定快照容差："
            f"planning={planning_gap}，runtime={runtime_gap}"
        )
    return observation


def allocation_safety_snapshot(
    *,
    before_paths: dict[str, dict],
    after_paths: dict[str, dict],
    after_states: dict[str, dict],
    selected_ids: tuple[str, ...],
    evicted_ids: tuple[str, ...],
    eviction_responses: Sequence[dict],
) -> tuple[dict[str, bool], dict[str, int]]:
    """计算一次 controller 分配后的完整安全条件。"""
    if not eviction_responses:
        raise RuntimeError("本场景应至少执行一次未选状态驱逐")
    global_before = eviction_responses[0]["before"]
    global_after = eviction_responses[-1]["after"]
    allocator_before = int(
        global_before["accounting"]["full_allocator"]["available"]
    )
    allocator_after = int(
        global_after["accounting"]["full_allocator"]["available"]
    )
    expected_changed_nodes = {
        int(before_paths[checkpoint_id]["node_id"])
        for checkpoint_id in evicted_ids
    }
    formal_primitives = {
        response["formal_primitive"] for response in eviction_responses
    }
    flags = {
        "selected_mamba_resident": all(
            after_states[checkpoint_id]["mamba_resident"]
            for checkpoint_id in selected_ids
        ),
        "unselected_mamba_evicted": all(
            not after_states[checkpoint_id]["mamba_resident"]
            for checkpoint_id in evicted_ids
        ),
        "fa_preserved": (
            all(state["fa_resident"] for state in after_states.values())
            and global_before["tree"]["full_tree_sha256"]
            == global_after["tree"]["full_tree_sha256"]
        ),
        "allocator_invariant": allocator_before == allocator_after,
        "tree_invariant": (
            global_before["tree"]["structure_sha256"]
            == global_after["tree"]["structure_sha256"]
        ),
        "path_invariant": all(
            before_paths[checkpoint_id]["path_node_ids"]
            == after_paths[checkpoint_id]["path_node_ids"]
            and before_paths[checkpoint_id]["segment_lengths"]
            == after_paths[checkpoint_id]["segment_lengths"]
            and before_paths[checkpoint_id]["prefix_sha256"]
            == after_paths[checkpoint_id]["prefix_sha256"]
            for checkpoint_id in before_paths
        ),
        "fa_identity_invariant": all(
            response["proof"]["fa_identity_unchanged"]
            for response in eviction_responses
        ),
        "only_expected_mamba_changed": (
            changed_mamba_nodes(global_before, global_after)
            == expected_changed_nodes
        ),
        "sanity_check": all(
            response["proof"]["sanity_check"]
            for response in eviction_responses
        ),
        "cascade_not_called": not any(
            response["proof"]["cascade_called"]
            for response in eviction_responses
        ),
        "formal_primitive": formal_primitives == {_FORMAL_PRIMITIVE},
    }
    return flags, {
        "before_available_size": allocator_before,
        "after_available_size": allocator_after,
    }


def aggregate_case_records(
    records: Sequence[dict],
    planning_summaries: Sequence[PolicyPlanningSummary],
) -> dict[str, dict]:
    """按策略汇总已成功完成的隔离 case。"""
    planning_by_policy = {
        summary.policy_name: summary for summary in planning_summaries
    }
    aggregate = {}
    for policy_name in POLICY_NAMES:
        policy_records = tuple(
            record
            for record in records
            if record.get("policy") == policy_name
            and record.get("status") == "PASS"
        )
        if not policy_records:
            continue
        physical_tokens = sum(
            int(record["physical_hit"]) for record in policy_records
        )
        executable_tokens = sum(
            int(record["executable_prefix"]) for record in policy_records
        )
        runtime_gap = sum(
            int(record["runtime_gap"]) for record in policy_records
        )
        e2e_values = tuple(
            float(record["request_e2e_ms"])
            for record in policy_records
            if record.get("request_e2e_ms") is not None
        )
        per_workflow: dict[str, int] = {}
        for record in policy_records:
            workflow_id = str(record["workflow_id"])
            per_workflow[workflow_id] = (
                per_workflow.get(workflow_id, 0)
                + int(record["runtime_gap"])
            )
        planning = planning_by_policy[policy_name]
        aggregate[policy_name] = {
            "n_cases": len(policy_records),
            "planning_total_gap": planning.total_recovery_gap,
            "runtime_total_gap": runtime_gap,
            "mean_gap_per_request": runtime_gap / len(policy_records),
            "physical_hit_tokens": physical_tokens,
            "executable_hit_tokens": executable_tokens,
            "executable_prefix_ratio": (
                executable_tokens / physical_tokens
                if physical_tokens > 0
                else 0.0
            ),
            "estimated_recovery_cost_ms": sum(
                float(record["estimated_runtime_recovery_cost_ms"])
                for record in policy_records
            ),
            "mean_request_e2e_ms": (
                sum(e2e_values) / len(e2e_values)
                if e2e_values
                else None
            ),
            "selected_checkpoint_ids": list(
                planning.selected_checkpoint_ids
            ),
            "per_workflow_runtime_gap": per_workflow,
            "per_continuation_runtime_gap": {
                str(record["continuation_id"]): int(record["runtime_gap"])
                for record in policy_records
            },
        }
    return aggregate


def _case_id(index: int, case: SnapshotEvaluationCase) -> str:
    """生成跨整个图形处理器进程唯一且稳定的 case 标识。"""
    policy = case.policy_name.lower().replace("-", "_")
    continuation = case.continuation_id.lower().replace("-", "_")
    return f"case_{index:02d}_{policy}_{continuation}"


def _optimizer_for_case(
    case: SnapshotEvaluationCase,
    scenario: ControlledScenario,
    recovery_cost_model: RecoveryCostModel,
) -> object:
    """为 FlowState 或冻结基线建立 controller 优化器。"""
    if case.policy_name == "FlowState":
        return GlobalOptimizer(recovery_cost_model)
    return PolicyOptimizerAdapter(
        case.policy_name,
        scenario,
        recovery_cost_model,
    )


def _pending_tokens(
    case: SnapshotEvaluationCase,
    scenario: ControlledScenario,
    runtime_workflows: Sequence[RuntimeWorkflow],
) -> tuple[int, ...]:
    """根据冻结 scenario 构造当前隔离 case 的唯一待续分支。"""
    runtime_by_workflow = {
        workflow.spec.workflow_id: workflow
        for workflow in runtime_workflows
    }
    continuation_index = next(
        index
        for index, continuation in enumerate(scenario.continuations)
        if continuation.continuation_id == case.scenario_continuation_id
    )
    runtime_workflow = runtime_by_workflow[case.workflow_id]
    suffix = make_tokens(
        241_019 + continuation_index * 10_007,
        PENDING_SUFFIX_LENGTH,
    )
    return (
        runtime_workflow.anchor_tokens
        + (runtime_workflow.anchor_output,)
        + suffix
    )


def run_snapshot_case(
    *,
    index: int,
    case: SnapshotEvaluationCase,
    engine: object,
    client: object,
    scenario: ControlledScenario,
    recovery_cost_model: RecoveryCostModel,
) -> dict:
    """在一次独立 flush/rebuild 生命周期中执行一个 case。"""
    case_id = _case_id(index, case)
    stage = "flush 并验证空缓存"
    record: dict[str, object] = {
        "case_id": case_id,
        "policy": case.policy_name,
        "continuation_id": case.continuation_id,
        "scenario_continuation_id": case.scenario_continuation_id,
        "workflow_id": case.workflow_id,
        "selected_checkpoint_ids": list(case.expected_selected_ids),
        "planning_gap": case.expected_recovery_gap,
        "status": "FAIL",
    }
    try:
        engine.flush_cache()
        clean_census = client.census(
            f"flowstate_step7e:{case_id}:census:after_flush"
        )
        record["clean_cache"] = validate_clean_cache(clean_census)

        stage = "重建五个检查点"
        runtime_workflows, candidate_tokens = build_runtime_workflows(
            engine,
            scenario,
            request_namespace=f"flowstate_step7e_{case_id}_build",
        )

        stage = "验证 allocation 前状态"
        handles, before_paths, before_states = build_runtime_handles(
            client,
            scenario,
            candidate_tokens,
        )
        record["checkpoints_before_allocation"] = before_states

        stage = "执行策略与 controller"
        runtime_adapter = SnapshotSchedulerRuntimeAdapter(client, case_id)
        controller = StateController(
            _optimizer_for_case(case, scenario, recovery_cost_model),
            runtime_adapter,
        )
        allocation = controller.reconcile(
            scenario.continuations,
            scenario.candidates,
            handles,
            scenario.budget_bytes,
        )
        selected_ids = tuple(
            candidate.checkpoint_id for candidate in allocation.selected
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
        record["selected_checkpoint_ids"] = list(selected_ids)
        record["evicted_checkpoint_ids"] = list(evicted_ids)
        if selected_ids != case.expected_selected_ids:
            raise RuntimeError(
                f"策略选择与 case planner 不一致：{selected_ids}"
            )
        if evicted_ids != expected_evicted_ids:
            raise RuntimeError(
                f"controller 驱逐与 selected set 不一致：{evicted_ids}"
            )

        continuation = next(
            continuation
            for continuation in scenario.continuations
            if continuation.continuation_id
            == case.scenario_continuation_id
        )
        planning_gap = recovery_gap(continuation, allocation.selected)
        record["planning_gap"] = planning_gap
        if planning_gap != case.expected_recovery_gap:
            raise RuntimeError(
                "核心 recovery_gap 与 case planner 不一致："
                f"{planning_gap} != {case.expected_recovery_gap}"
            )

        stage = "验证 allocation 后状态与安全条件"
        after_paths, after_states = inspect_after_allocation(
            client,
            candidate_tokens,
        )
        record["checkpoints_after_allocation"] = after_states
        safety, allocator = allocation_safety_snapshot(
            before_paths=before_paths,
            after_paths=after_paths,
            after_states=after_states,
            selected_ids=selected_ids,
            evicted_ids=evicted_ids,
            eviction_responses=runtime_adapter.eviction_responses,
        )
        record["safety"] = safety
        record["fa_allocator"] = allocator
        if not all(safety.values()):
            raise RuntimeError(f"allocation 安全条件失败：{safety}")

        stage = "发送单个待续请求"
        request_id = f"flowstate_step7e_{case_id}_pending"
        _, metadata = generate(
            engine,
            request_id,
            _pending_tokens(case, scenario, runtime_workflows),
        )
        metrics = query_runtime_metrics(client, request_id)
        record.update(
            {
                "physical_hit": int(metrics["physical_fa_hit"]),
                "executable_prefix": int(metrics["executable_prefix"]),
                "runtime_gap": int(metrics["replay_gap"]),
                "request_e2e_ms": optional_latency_ms(
                    metadata,
                    "e2e_latency",
                ),
            }
        )
        record["estimated_runtime_recovery_cost_ms"] = (
            recovery_cost_model.estimate(int(record["runtime_gap"]))
        )
        validate_snapshot_runtime_observation(
            anchor_pos=continuation.planning_target,
            planning_gap=planning_gap,
            metrics=metrics,
        )
        record["planning_runtime_agreement"] = True
        record["status"] = "PASS"
        record["failure_stage"] = None
        return record
    except Exception as error:
        record["failure_stage"] = stage
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
        return record


def build_summary(
    *,
    records: Sequence[dict],
    planning_summaries: Sequence[PolicyPlanningSummary],
    artifact_directory: Path,
    failure_stage: str | None,
    error: str | None,
) -> dict:
    """构造成功或提前失败时都可持久化的运行时汇总。"""
    passed = sum(record.get("status") == "PASS" for record in records)
    failed = sum(record.get("status") == "FAIL" for record in records)
    status = "PASS" if passed == 28 and failed == 0 else "FAIL"
    completed_safety = tuple(
        record["safety"]
        for record in records
        if record.get("status") == "PASS"
    )
    safety = {
        "fa_preserved": bool(completed_safety) and all(
            item["fa_preserved"] for item in completed_safety
        ),
        "allocator_invariant": bool(completed_safety) and all(
            item["allocator_invariant"] for item in completed_safety
        ),
        "tree_path_invariant": bool(completed_safety) and all(
            item["tree_invariant"] and item["path_invariant"]
            for item in completed_safety
        ),
        "sanity_check": bool(completed_safety) and all(
            item["sanity_check"] for item in completed_safety
        ),
        "cascade_called": any(
            not item["cascade_not_called"] for item in completed_safety
        ),
    }
    return {
        "schema_version": "flowstate.controlled_multiworkflow.snapshot.v1",
        "status": status,
        "cases_passed": passed,
        "cases_failed": failed,
        "cases_expected": 28,
        "runtime_summary": aggregate_case_records(
            records,
            planning_summaries,
        ),
        "safety": safety,
        "planning_runtime_agreement": all(
            record.get("planning_runtime_agreement") is True
            for record in records
            if record.get("status") == "PASS"
        ) and status == "PASS",
        "formal_mutation_primitive": _FORMAL_PRIMITIVE,
        "performance_interpretation": "CORRECTNESS ONLY",
        "failure_stage": failure_stage,
        "error": error,
        "artifact_directory": str(artifact_directory),
        "engine_configuration": ENGINE_CONFIGURATION,
    }


def main() -> int:
    """在一个图形处理器进程内依次完成二十八个隔离 case。"""
    from targeted_probe import ControlClient
    from wp3b_end_to_end_transport import (
        FormalEndToEndGateEngine,
        requested_control_port,
    )

    writer = CaseArtifactWriter.create()
    scenario = build_scenario()
    recovery_cost_model = RecoveryCostModel()
    cases = build_snapshot_cases(scenario, recovery_cost_model)
    planning_summaries = build_planning_summaries(
        scenario,
        recovery_cost_model,
    )
    records: list[dict] = []
    engine = None
    failure_stage = None
    error_text = None
    started = time.perf_counter()
    try:
        if len(cases) != 28:
            raise RuntimeError(f"隔离 case 数量异常：{len(cases)}")
        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)

        for index, case in enumerate(cases, start=1):
            print(
                f"[STEP7E] CASE {index}/28 "
                f"{case.policy_name} {case.continuation_id}",
                flush=True,
            )
            record = run_snapshot_case(
                index=index,
                case=case,
                engine=engine,
                client=client,
                scenario=scenario,
                recovery_cost_model=recovery_cost_model,
            )
            records.append(record)
            writer.append_case(record)
            if record["status"] != "PASS":
                failure_stage = str(record["failure_stage"])
                error_text = str(record.get("error"))
                break
    except Exception as error:
        failure_stage = failure_stage or "初始化运行时比较"
        error_text = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as shutdown_error:
                if error_text is None:
                    failure_stage = "关闭运行时"
                    error_text = repr(shutdown_error)

    summary = build_summary(
        records=records,
        planning_summaries=planning_summaries,
        artifact_directory=writer.directory,
        failure_stage=failure_stage,
        error=error_text,
    )
    summary["wall_time_ms"] = (time.perf_counter() - started) * 1_000.0
    writer.write_summary(summary)
    print(
        "[STEP7E] RESULT="
        + json.dumps(summary, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
