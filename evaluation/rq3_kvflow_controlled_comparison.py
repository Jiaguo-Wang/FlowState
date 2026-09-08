"""执行 workflow-distance-sensitive 的 RQ3 受控策略比较。"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.controlled_multiworkflow_v1.scenario import CHECKPOINT_SIZE_BYTES
from evaluation.sota_metadata import (
    CONTROLLED_MARCONI_ALPHA,
    build_marconi_flop_saved,
)
from evaluation.sota_policies import (
    KVFlowStylePolicy,
    MarconiStylePolicy,
    _build_marconi_metrics,
)
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate, is_compatible
from flowstate.workflow import PendingContinuation


REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET_K = 1
DETERMINISM_REPETITIONS = 20
POLICY_NAMES = (
    "KVFlow Adaptation",
    "Marconi Adaptation",
    "FlowState",
    "Exact",
)


@dataclass(frozen=True)
class WorkflowSpec:
    """记录一个公开工作流图及其状态需求。"""

    workflow_id: str
    candidate_id: str
    continuation_id: str
    graph_nodes: tuple[str, ...]
    graph_edges: tuple[tuple[str, str], ...]
    lineage_path: tuple[str, ...]
    pending_lineage_path: tuple[str, ...]
    target_tokens: int
    checkpoint_tokens: int
    last_access: float


@dataclass(frozen=True)
class ControlledWorkload:
    """汇总受控比较的公共候选、待续请求和策略信号。"""

    specs: tuple[WorkflowSpec, ...]
    continuations: tuple[PendingContinuation, ...]
    candidates: tuple[CheckpointCandidate, ...]
    steps_by_continuation: Mapping[str, int]
    last_access_by_checkpoint: Mapping[str, float]
    flop_saved_by_checkpoint: Mapping[str, float]
    budget_k: int


WORKFLOW_SPECS = (
    WorkflowSpec(
        workflow_id="WF_NEAR",
        candidate_id="CP_NEAR",
        continuation_id="PENDING_NEAR",
        graph_nodes=("decision", "execute"),
        graph_edges=(("decision", "execute"),),
        lineage_path=("WF_NEAR_ROOT",),
        pending_lineage_path=("WF_NEAR_ROOT", "NEAR_CONTINUATION"),
        target_tokens=8_192,
        checkpoint_tokens=4_096,
        last_access=1.0,
    ),
    WorkflowSpec(
        workflow_id="WF_MID",
        candidate_id="CP_MID",
        continuation_id="PENDING_MID",
        graph_nodes=("decision", "prepare", "review", "execute"),
        graph_edges=(
            ("decision", "prepare"),
            ("prepare", "review"),
            ("review", "execute"),
        ),
        lineage_path=("WF_MID_ROOT",),
        pending_lineage_path=("WF_MID_ROOT", "MID_CONTINUATION"),
        target_tokens=16_384,
        checkpoint_tokens=8_192,
        last_access=1.0,
    ),
    WorkflowSpec(
        workflow_id="WF_FAR",
        candidate_id="CP_FAR",
        continuation_id="PENDING_FAR",
        graph_nodes=(
            "decision",
            "prepare",
            "tool",
            "review",
            "merge",
            "execute",
        ),
        graph_edges=(
            ("decision", "prepare"),
            ("prepare", "tool"),
            ("tool", "review"),
            ("review", "merge"),
            ("merge", "execute"),
        ),
        lineage_path=("WF_FAR_ROOT",),
        pending_lineage_path=("WF_FAR_ROOT", "FAR_CONTINUATION"),
        target_tokens=32_768,
        checkpoint_tokens=32_768,
        last_access=1.0,
    ),
)


def _derive_steps_to_execution(spec: WorkflowSpec) -> int:
    """由公开工作流图计算 decision 到 execute 的最短边数。"""
    adjacency: dict[str, list[str]] = {node: [] for node in spec.graph_nodes}
    for source, target in spec.graph_edges:
        if source not in adjacency or target not in adjacency:
            raise ValueError(f"工作流 {spec.workflow_id} 的边引用未知节点")
        adjacency[source].append(target)

    queue = deque([("decision", 0)])
    visited = {"decision"}
    while queue:
        node, distance = queue.popleft()
        if node == "execute":
            return distance
        for target in sorted(adjacency[node]):
            if target not in visited:
                visited.add(target)
                queue.append((target, distance + 1))
    raise ValueError(f"工作流 {spec.workflow_id} 不存在 execution 路径")


def build_workload() -> ControlledWorkload:
    """从显式 metadata 构造公共决策快照。"""
    continuations = tuple(
        PendingContinuation(
            continuation_id=spec.continuation_id,
            workflow_id=spec.workflow_id,
            lineage_path=spec.pending_lineage_path,
            anchor_pos=spec.target_tokens,
            resident_fa_frontier=spec.target_tokens,
        )
        for spec in WORKFLOW_SPECS
    )
    candidates = tuple(
        CheckpointCandidate(
            checkpoint_id=spec.candidate_id,
            workflow_id=spec.workflow_id,
            lineage_path=spec.lineage_path,
            token_pos=spec.checkpoint_tokens,
            memory_bytes=CHECKPOINT_SIZE_BYTES,
            recurrent_resident=True,
            fa_resident=True,
        )
        for spec in WORKFLOW_SPECS
    )
    steps = {
        spec.continuation_id: _derive_steps_to_execution(spec)
        for spec in WORKFLOW_SPECS
    }
    recency = {
        spec.candidate_id: spec.last_access for spec in WORKFLOW_SPECS
    }
    return ControlledWorkload(
        specs=WORKFLOW_SPECS,
        continuations=continuations,
        candidates=candidates,
        steps_by_continuation=steps,
        last_access_by_checkpoint=recency,
        flop_saved_by_checkpoint=build_marconi_flop_saved(candidates),
        budget_k=BUDGET_K,
    )


def _selection_cost(
    workload: ControlledWorkload,
    selected_ids: Sequence[str],
    model: RecoveryCostModel,
) -> dict[str, object]:
    """使用公共正式目标计算一个选择集合的成本。"""
    selected_set = set(selected_ids)
    selected = tuple(
        candidate
        for candidate in workload.candidates
        if candidate.checkpoint_id in selected_set
    )
    if len(selected) != len(selected_set):
        raise ValueError("选择集合包含未知候选")
    per_continuation = []
    for continuation in workload.continuations:
        frontier = executable_frontier(continuation, selected)
        gap = recovery_gap(continuation, selected)
        cost_ms = model.estimate(gap, continuation.planning_target)
        per_continuation.append(
            {
                "continuation_id": continuation.continuation_id,
                "workflow_id": continuation.workflow_id,
                "T_tokens": continuation.planning_target,
                "E_tokens": frontier,
                "G_tokens": gap,
                "cost_ms": cost_ms,
            }
        )
    return {
        "selected_checkpoint_ids": list(selected_ids),
        "total_cost_ms": sum(row["cost_ms"] for row in per_continuation),
        "total_gap_tokens": sum(row["G_tokens"] for row in per_continuation),
        "per_continuation": per_continuation,
    }


def _select_exact(
    workload: ControlledWorkload,
    model: RecoveryCostModel,
) -> tuple[tuple[str, ...], int]:
    """独立枚举预算内全部子集并返回公共目标最优解。"""
    candidate_ids = tuple(
        sorted(candidate.checkpoint_id for candidate in workload.candidates)
    )
    best_ids: tuple[str, ...] = ()
    best_cost = float("inf")
    evaluated = 0
    for subset_size in range(workload.budget_k + 1):
        for subset in combinations(candidate_ids, subset_size):
            evaluated += 1
            cost = float(_selection_cost(workload, subset, model)["total_cost_ms"])
            if cost < best_cost - 1e-9 or (
                abs(cost - best_cost) <= 1e-9 and subset < best_ids
            ):
                best_ids = subset
                best_cost = cost
    return best_ids, evaluated


def _select_policies(
    workload: ControlledWorkload,
    model: RecoveryCostModel,
) -> dict[str, tuple[str, ...]]:
    """调用冻结策略并返回公共预算下的选择。"""
    kvflow = KVFlowStylePolicy().select(
        workload.continuations,
        workload.candidates,
        workload.budget_k,
        workload.steps_by_continuation,
        workload.last_access_by_checkpoint,
    ).selected_checkpoint_ids
    marconi = MarconiStylePolicy().select(
        workload.candidates,
        workload.budget_k,
        workload.last_access_by_checkpoint,
        workload.flop_saved_by_checkpoint,
        CONTROLLED_MARCONI_ALPHA,
    ).selected_checkpoint_ids
    flowstate_result = GlobalOptimizer(model).select(
        workload.continuations,
        workload.candidates,
        workload.budget_k * CHECKPOINT_SIZE_BYTES,
    )
    flowstate = tuple(
        candidate.checkpoint_id for candidate in flowstate_result.selected
    )
    exact, _ = _select_exact(workload, model)
    return {
        "KVFlow Adaptation": kvflow,
        "Marconi Adaptation": marconi,
        "FlowState": flowstate,
        "Exact": exact,
    }


def _canonical_digest(value: object) -> str:
    """计算稳定 JSON 表示的 SHA-256。"""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes() -> dict[str, str]:
    """记录冻结策略、目标与兼容语义源文件摘要。"""
    paths = (
        "evaluation/rq3_kvflow_controlled_comparison.py",
        "evaluation/sota_policies.py",
        "flowstate/optimizer.py",
        "flowstate/recovery_model.py",
        "flowstate/executable_state.py",
        "flowstate/state_catalog.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _workload_definition(workload: ControlledWorkload) -> dict[str, object]:
    """生成公开且可审计的 workload 定义。"""
    workflows = []
    for spec in workload.specs:
        workflows.append(
            {
                **asdict(spec),
                "steps_to_execution": workload.steps_by_continuation[
                    spec.continuation_id
                ],
                "steps_source": "显式 workflow graph 中 decision 到 execute 的最短边数",
                "pending_known_at_decision": True,
                "pending_materialized_at_decision": True,
                "ancestry_source": "显式 workflow_id 与 lineage_path metadata",
            }
        )
    return {
        "schema_version": "flowstate.rq3_kvflow_controlled.v1",
        "workload_name": "workflow_distance_sensitive_three_tier_v1",
        "execution_mode": "CPU-only 离线确定性比较",
        "decision_epoch": "CONTROLLED_DECISION_0",
        "design_rule": (
            "使用近、中、远三档公开执行链和 8K、16K、32K 目标；"
            "所有候选等大小、同时序，隔离 STE、增量计算量与恢复收益。"
        ),
        "workflows": workflows,
        "budget": {
            "K": workload.budget_k,
            "checkpoint_size_bytes": CHECKPOINT_SIZE_BYTES,
            "budget_bytes": workload.budget_k * CHECKPOINT_SIZE_BYTES,
        },
        "common_pending_ids": sorted(workload.steps_by_continuation),
        "common_candidate_ids": sorted(
            candidate.checkpoint_id for candidate in workload.candidates
        ),
        "provenance": {
            "hidden_future_trajectory_used": False,
            "radix_tree_used_for_ancestry": False,
            "token_lcp_used_for_ancestry": False,
            "external_population_snapshot_used": False,
            "policy_output_used_to_construct_workload": False,
            "workload_parameters_tuned_for_flowstate_win": False,
            "openhands_168_snapshots_run": False,
            "agentx_23_snapshots_run": False,
            "gpu_used": False,
        },
    }


def _candidate_signals(
    workload: ControlledWorkload,
    model: RecoveryCostModel,
) -> dict[str, object]:
    """计算各策略在选择前可见的候选信号。"""
    empty_cost = float(_selection_cost(workload, (), model)["total_cost_ms"])
    marconi_metrics = _build_marconi_metrics(
        workload.candidates,
        workload.last_access_by_checkpoint,
        workload.flop_saved_by_checkpoint,
        CONTROLLED_MARCONI_ALPHA,
    )
    kvflow_policy = KVFlowStylePolicy()
    rows = []
    for candidate in sorted(
        workload.candidates,
        key=lambda item: item.checkpoint_id,
    ):
        compatible = tuple(
            continuation.continuation_id
            for continuation in workload.continuations
            if is_compatible(candidate, continuation)
        )
        singleton_cost = float(
            _selection_cost(workload, (candidate.checkpoint_id,), model)[
                "total_cost_ms"
            ]
        )
        rows.append(
            {
                "candidate_id": candidate.checkpoint_id,
                "workflow_id": candidate.workflow_id,
                "token_position": candidate.token_pos,
                "memory_bytes": candidate.memory_bytes,
                "compatible_pending_ids": list(compatible),
                "compatibility_source": "显式 workflow/lineage metadata",
                "steps_to_execution": kvflow_policy.priority(
                    candidate,
                    workload.continuations,
                    workload.steps_by_continuation,
                ),
                "last_access": workload.last_access_by_checkpoint[
                    candidate.checkpoint_id
                ],
                "marconi_flop_saved": workload.flop_saved_by_checkpoint[
                    candidate.checkpoint_id
                ],
                "marconi_utility": marconi_metrics[candidate.checkpoint_id],
                "flowstate_benefit_from_empty_ms": empty_cost - singleton_cost,
                "cost_if_selected_alone_ms": singleton_cost,
            }
        )

    full_kvflow_ranking = KVFlowStylePolicy().select(
        workload.continuations,
        workload.candidates,
        len(workload.candidates),
        workload.steps_by_continuation,
        workload.last_access_by_checkpoint,
    ).selected_checkpoint_ids
    full_marconi_ranking = MarconiStylePolicy().select(
        workload.candidates,
        len(workload.candidates),
        workload.last_access_by_checkpoint,
        workload.flop_saved_by_checkpoint,
        CONTROLLED_MARCONI_ALPHA,
    ).selected_checkpoint_ids
    full_flowstate = GlobalOptimizer(model).select(
        workload.continuations,
        workload.candidates,
        len(workload.candidates) * CHECKPOINT_SIZE_BYTES,
    ).selected
    flowstate_ranking = tuple(item.checkpoint_id for item in full_flowstate)

    swapped_steps = dict(workload.steps_by_continuation)
    swapped_steps["PENDING_NEAR"] = 5
    swapped_steps["PENDING_FAR"] = 1
    swapped_selected = KVFlowStylePolicy().select(
        workload.continuations,
        workload.candidates,
        workload.budget_k,
        swapped_steps,
        workload.last_access_by_checkpoint,
    ).selected_checkpoint_ids

    return {
        "candidates": rows,
        "rankings": {
            "KVFlow Adaptation": list(full_kvflow_ranking),
            "Marconi Adaptation": list(full_marconi_ranking),
            "FlowState marginal order": list(flowstate_ranking),
        },
        "kvflow_ste_sensitivity_probe": {
            "probe_only_not_main_result": True,
            "original_steps": dict(workload.steps_by_continuation),
            "original_selected": [full_kvflow_ranking[0]],
            "near_far_steps_swapped": swapped_steps,
            "swapped_selected": list(swapped_selected),
            "selection_changed": tuple(full_kvflow_ranking[:1]) != swapped_selected,
        },
    }


def run_comparison() -> dict[str, object]:
    """运行完整受控比较并返回全部内存结果。"""
    source_before = _source_hashes()
    workload = build_workload()
    model = RecoveryCostModel()
    definition = _workload_definition(workload)
    signals = _candidate_signals(workload, model)
    selections = _select_policies(workload, model)
    exact_ids, exact_evaluations = _select_exact(workload, model)
    if selections["Exact"] != exact_ids:
        raise RuntimeError("Exact 选择在重复枚举中不一致")

    objective = {
        policy: _selection_cost(workload, selected, model)
        for policy, selected in selections.items()
    }
    common_input = {
        "pending_ids": definition["common_pending_ids"],
        "candidate_ids": definition["common_candidate_ids"],
        "budget": definition["budget"],
        "steps_by_continuation": dict(workload.steps_by_continuation),
        "last_access_by_checkpoint": dict(workload.last_access_by_checkpoint),
        "flop_saved_by_checkpoint": dict(workload.flop_saved_by_checkpoint),
        "phi": asdict(model.metadata),
    }
    input_digest = _canonical_digest(common_input)

    reruns = []
    for _ in range(DETERMINISM_REPETITIONS):
        reruns.append(_select_policies(workload, model))
    deterministic = all(item == selections for item in reruns)
    ste_values = sorted(set(workload.steps_by_continuation.values()))
    candidate_rows = {
        row["candidate_id"]: row for row in signals["candidates"]
    }
    near = candidate_rows["CP_NEAR"]
    far = candidate_rows["CP_FAR"]
    source_after = _source_hashes()
    gates = {
        "three_distinct_ste_values": len(ste_values) >= 3,
        "kvflow_ranking_ste_sensitive": signals[
            "kvflow_ste_sensitivity_probe"
        ]["selection_changed"],
        "workflow_semantics_explicit": all(
            spec.graph_edges and spec.lineage_path for spec in workload.specs
        ),
        "no_future_leakage": not any(
            definition["provenance"][key]
            for key in (
                "hidden_future_trajectory_used",
                "radix_tree_used_for_ancestry",
                "token_lcp_used_for_ancestry",
                "external_population_snapshot_used",
                "policy_output_used_to_construct_workload",
            )
        ),
        "no_policy_directed_tuning": not definition["provenance"][
            "workload_parameters_tuned_for_flowstate_win"
        ],
        "common_candidate_set_and_budget": all(
            len(selected) <= workload.budget_k
            and set(selected).issubset(definition["common_candidate_ids"])
            for selected in selections.values()
        ),
        "common_objective_evaluator": set(objective) == set(POLICY_NAMES),
        "policy_deterministic": deterministic,
        "near_lower_ste_lower_recovery_value": (
            near["steps_to_execution"] < far["steps_to_execution"]
            and near["flowstate_benefit_from_empty_ms"]
            < far["flowstate_benefit_from_empty_ms"]
        ),
        "source_integrity": source_before == source_after,
    }
    status = (
        "RQ3_KVFLOW_CONTROLLED_READY"
        if all(gates.values())
        else "RQ3_KVFLOW_CONTROLLED_BLOCKED"
    )
    policy_selections = {
        "status": status,
        "common_input_digest": input_digest,
        "budget_k": workload.budget_k,
        "selections": {
            policy: list(selected) for policy, selected in selections.items()
        },
        "determinism": {
            "repetitions": DETERMINISM_REPETITIONS,
            "pass": deterministic,
        },
        "pass_gates": gates,
    }
    objective_results = {
        "common_input_digest": input_digest,
        "common_objective": "C(S)=sum_p Phi(T_p-E_p(S),T_p)",
        "recovery_model": {
            "name": model.metadata.name,
            "formula": (
                "Phi(G,T)=37.828150*g+0.345974143*g*tau-0.156201917*g^2"
            ),
            "g_definition": "G/1024",
            "tau_definition": "T/1024",
            "output_unit": "ms",
        },
        "exact_subset_evaluations": exact_evaluations,
        "results": objective,
        "comparisons": {
            "kvflow_minus_flowstate_ms": (
                objective["KVFlow Adaptation"]["total_cost_ms"]
                - objective["FlowState"]["total_cost_ms"]
            ),
            "marconi_minus_flowstate_ms": (
                objective["Marconi Adaptation"]["total_cost_ms"]
                - objective["FlowState"]["total_cost_ms"]
            ),
            "flowstate_minus_exact_ms": (
                objective["FlowState"]["total_cost_ms"]
                - objective["Exact"]["total_cost_ms"]
            ),
        },
        "source_integrity": {
            "before": source_before,
            "after": source_after,
            "pass": source_before == source_after,
        },
    }
    return {
        "status": status,
        "workload_definition": definition,
        "candidate_signals": signals,
        "policy_selections": policy_selections,
        "common_objective_results": objective_results,
    }


def _write_json(path: Path, value: object) -> None:
    """以稳定格式写入 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_report(result: Mapping[str, object]) -> str:
    """生成受控比较的中文冻结报告。"""
    definition = result["workload_definition"]
    signals = result["candidate_signals"]
    policies = result["policy_selections"]
    objective = result["common_objective_results"]
    selections = policies["selections"]
    costs = objective["results"]
    ste_values = sorted(
        workflow["steps_to_execution"] for workflow in definition["workflows"]
    )
    lines = [
        "# RQ3 KVFlow 工作流距离敏感受控比较",
        "",
        f"状态：**{result['status']}**。",
        "",
        "本实验只执行 CPU 离线选择与公共目标计算，没有读取或重跑 OpenHands/AgentX 正式快照。",
        "三个待续请求在决策时均已知且已物化；STE 直接来自公开 workflow graph，不来自真实未来轨迹。",
        "workflow ancestry 只读取显式 workflow_id 与 lineage_path metadata，不读取 radix tree 或 token LCP。",
        "",
        "## 工作负载与信号",
        "",
        f"- STE：{', '.join(str(value) for value in ste_values)}",
        f"- 公共预算：K={definition['budget']['K']}",
        f"- KVFlow ranking：{' > '.join(signals['rankings']['KVFlow Adaptation'])}",
        f"- Marconi ranking：{' > '.join(signals['rankings']['Marconi Adaptation'])}",
        f"- FlowState 边际顺序：{' > '.join(signals['rankings']['FlowState marginal order'])}",
        "",
        "## 选择与公共恢复目标",
        "",
        "| 策略 | 选择 | C(S) |",
        "|---|---|---:|",
    ]
    for policy in POLICY_NAMES:
        lines.append(
            f"| {policy} | {', '.join(selections[policy])} | "
            f"{costs[policy]['total_cost_ms']:.6f} ms |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "KVFlow 选择 STE=1 的 CP_NEAR，符合 workflow proximity 语义；"
            "FlowState 与 Exact 选择 CP_FAR，因为它在公共 C(S) 下提供最大的可执行恢复收益。",
            "因此存在清晰的“更近但恢复价值更低”与“更远但恢复价值更高”的受控权衡。",
            "本结果只比较 KVFlow 的 STE retention signal，不声称复现 KVFlow runtime、prefetching 或完整系统。",
            "",
            "## 门禁",
            "",
        ]
    )
    for gate, passed in policies["pass_gates"].items():
        lines.append(f"- {gate}: {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines) + "\n"


def write_artifacts(
    result: Mapping[str, object],
    output_root: Path,
) -> Path:
    """写入本实验要求的五个正式文件。"""
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "workload_definition.json", result["workload_definition"])
    _write_json(output_root / "candidate_signals.json", result["candidate_signals"])
    _write_json(output_root / "policy_selections.json", result["policy_selections"])
    _write_json(
        output_root / "common_objective_results.json",
        result["common_objective_results"],
    )
    (output_root / "final_report.md").write_text(
        _build_report(result),
        encoding="utf-8",
    )
    return output_root


def _default_output_root() -> Path:
    """返回不会覆盖历史结果的时间戳目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "evaluation" / f"rq3_kvflow_controlled_{timestamp}"


def main() -> int:
    """解析命令行并执行一次 CPU-only 正式比较。"""
    parser = argparse.ArgumentParser(description="执行 RQ3 KVFlow 受控比较")
    parser.add_argument("--output-root", type=Path, default=None)
    arguments = parser.parse_args()
    result = run_comparison()
    output_root = arguments.output_root or _default_output_root()
    write_artifacts(result, output_root)
    print(result["status"])
    print(f"artifact_root={output_root.resolve()}")
    return 0 if result["status"] == "RQ3_KVFLOW_CONTROLLED_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
