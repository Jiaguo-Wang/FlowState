#!/usr/bin/env python3
"""在真实四工作流 barrier 上执行一次纯 selector 的 K=2 门禁。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.barrier_fa_frontier_control import (
    BarrierFAControlClient,
    semantic_snapshot_differences,
)
from evaluation.barrier_fa_frontier_query_gate import _side_effect_checks
from evaluation.controlled_multiworkflow_v1.policies import select_global_lru
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    wait_for_transport,
)
from evaluation.controlled_multiworkflow_v1.scenario import CheckpointRecency
from evaluation.openhands_4workflow_occupancy_calibration import (
    PHYSICAL_MAX_MAMBA_CACHE_SIZE,
    WORKFLOWS,
    _environment,
    _failure_record,
    compact_census,
    execute_request,
)
from evaluation.openhands_common_barrier_snapshot_gate import (
    BUDGET_BYTES,
    CHECKPOINT_SIZE_BYTES,
    ENGINE_CONFIGURATION_COMMON_BARRIER,
    EXECUTED_TURN,
    LOGICAL_K,
    PENDING_TURN,
    SCHEDULE,
    build_pending_set,
    build_policy_metadata,
    common_candidate_universe,
    load_barrier_requests,
    locate_materialized_candidate,
    token_digest,
    validate_candidate_at_barrier,
)
from evaluation.openhands_single_workflow_baseline10 import (
    ArtifactLogCapture,
    _append_jsonl,
    _write_json,
)
from evaluation.openhands_single_workflow_smoke import (
    DATASET_PATH,
    TOKENIZER_PATH,
)
from evaluation.sota_metadata import CONTROLLED_MARCONI_ALPHA
from evaluation.sota_policies import MarconiStylePolicy
from flowstate.adapters.sglang import RuntimeCheckpointHandle
from flowstate.executable_state import executable_frontier, recovery_gap
from flowstate.optimizer import AllocationResult, GlobalOptimizer
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import (
    CheckpointCandidate,
    validate_unique_checkpoint_ids,
)
from flowstate.workflow import PendingContinuation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
ENGINE_CONFIGURATION_K2_SELECTION = dict(
    ENGINE_CONFIGURATION_COMMON_BARRIER
)


def _artifact_directory() -> Path:
    """创建不会覆盖已有结果的独立产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_frozen_barrier_k2_selection_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """对仓库内路径返回相对表示。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    """按现有 Marconi selector 的最小最大语义归一化。"""
    if not values:
        return {}
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return {checkpoint_id: 0.0 for checkpoint_id in values}
    scale = maximum - minimum
    return {
        checkpoint_id: (value - minimum) / scale
        for checkpoint_id, value in values.items()
    }


def marconi_score_rows(
    candidates: Sequence[CheckpointCandidate],
    metadata_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """复算现有 selector 使用的 Marconi 归一化分项与最终分数。"""
    metadata_by_id = {
        str(row["checkpoint_id"]): row for row in metadata_rows
    }
    raw_recency = {
        candidate.checkpoint_id: float(
            metadata_by_id[candidate.checkpoint_id]["marconi_recency"]
        )
        for candidate in candidates
    }
    raw_efficiency = {
        candidate.checkpoint_id: (
            float(
                metadata_by_id[candidate.checkpoint_id][
                    "marconi_incremental_span"
                ]
            )
            / candidate.memory_bytes
        )
        for candidate in candidates
    }
    normalized_recency = _normalize(raw_recency)
    normalized_efficiency = _normalize(raw_efficiency)
    return [
        {
            "checkpoint_id": candidate.checkpoint_id,
            "raw_recency": raw_recency[candidate.checkpoint_id],
            "normalized_recency": normalized_recency[
                candidate.checkpoint_id
            ],
            "incremental_span": float(
                metadata_by_id[candidate.checkpoint_id][
                    "marconi_incremental_span"
                ]
            ),
            "raw_flop_efficiency": raw_efficiency[
                candidate.checkpoint_id
            ],
            "normalized_flop_efficiency": normalized_efficiency[
                candidate.checkpoint_id
            ],
            "alpha": CONTROLLED_MARCONI_ALPHA,
            "final_score": (
                normalized_recency[candidate.checkpoint_id]
                + CONTROLLED_MARCONI_ALPHA
                * normalized_efficiency[candidate.checkpoint_id]
            ),
        }
        for candidate in candidates
    ]


def flowstate_continuation_rows(
    continuations: Sequence[PendingContinuation],
    result: AllocationResult,
    model: RecoveryCostModel,
) -> list[dict[str, object]]:
    """记录 FlowState 选择前后的 E、G 与正式恢复成本。"""
    rows = []
    for continuation in continuations:
        e_before = executable_frontier(continuation, ())
        g_before = recovery_gap(continuation, ())
        e_after = executable_frontier(continuation, result.selected)
        g_after = recovery_gap(continuation, result.selected)
        rows.append(
            {
                "continuation_id": continuation.continuation_id,
                "workflow_id": continuation.workflow_id,
                "planning_target": continuation.planning_target,
                "e_before": e_before,
                "g_before": g_before,
                "cost_before_ms": model.estimate(
                    g_before,
                    continuation.planning_target,
                ),
                "e_after": e_after,
                "g_after": g_after,
                "cost_after_ms": model.estimate(
                    g_after,
                    continuation.planning_target,
                ),
            }
        )
    return rows


def run_selectors(
    candidates: Sequence[CheckpointCandidate],
    continuations: Sequence[PendingContinuation],
    metadata_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """对同一不可变 candidate universe 依次调用三个现有 selector。"""
    frozen_candidates = tuple(candidates)
    frozen_continuations = tuple(continuations)
    validate_unique_checkpoint_ids(frozen_candidates)
    ordered_candidate_ids = tuple(
        candidate.checkpoint_id for candidate in frozen_candidates
    )
    metadata_by_id = {
        str(row["checkpoint_id"]): row for row in metadata_rows
    }
    recency_records = tuple(
        CheckpointRecency(
            checkpoint_id=candidate.checkpoint_id,
            creation_order=int(
                metadata_by_id[candidate.checkpoint_id]["creation_order"]
            ),
            last_access_order=int(
                metadata_by_id[candidate.checkpoint_id]["last_access_order"]
            ),
        )
        for candidate in frozen_candidates
    )

    lru_selected = select_global_lru(
        frozen_candidates,
        recency_records,
        BUDGET_BYTES,
    )

    last_access = {
        candidate.checkpoint_id: float(
            metadata_by_id[candidate.checkpoint_id]["marconi_recency"]
        )
        for candidate in frozen_candidates
    }
    flop_saved = {
        candidate.checkpoint_id: float(
            metadata_by_id[candidate.checkpoint_id][
                "marconi_incremental_span"
            ]
        )
        for candidate in frozen_candidates
    }
    marconi_result = MarconiStylePolicy().select(
        frozen_candidates,
        LOGICAL_K,
        last_access,
        flop_saved,
        CONTROLLED_MARCONI_ALPHA,
    )
    score_rows = marconi_score_rows(frozen_candidates, metadata_rows)

    recovery_model = RecoveryCostModel()
    flowstate_result = GlobalOptimizer(recovery_model).select(
        frozen_continuations,
        frozen_candidates,
        BUDGET_BYTES,
    )
    flow_rows = flowstate_continuation_rows(
        frozen_continuations,
        flowstate_result,
        recovery_model,
    )
    flowstate_selected = tuple(
        candidate.checkpoint_id for candidate in flowstate_result.selected
    )
    eligible = set(ordered_candidate_ids)
    selected_by_policy = {
        "LRU": list(lru_selected),
        "Marconi": list(marconi_result.selected_checkpoint_ids),
        "FlowState": list(flowstate_selected),
    }
    valid_selection = all(
        len(selected) <= LOGICAL_K
        and len(set(selected)) == len(selected)
        and set(selected).issubset(eligible)
        for selected in selected_by_policy.values()
    )
    return {
        "ordered_candidate_ids": list(ordered_candidate_ids),
        "candidate_ids_by_policy_input": {
            "LRU": list(ordered_candidate_ids),
            "Marconi": list(ordered_candidate_ids),
            "FlowState": list(ordered_candidate_ids),
        },
        "common_candidate_universe": True,
        "lru": {
            "selected_checkpoint_ids": list(lru_selected),
            "selected_count": len(lru_selected),
            "ranking_metadata": [
                {
                    "checkpoint_id": record.checkpoint_id,
                    "last_access_order": record.last_access_order,
                    "creation_order": record.creation_order,
                }
                for record in recency_records
            ],
        },
        "marconi": {
            "selected_checkpoint_ids": list(
                marconi_result.selected_checkpoint_ids
            ),
            "selected_count": len(
                marconi_result.selected_checkpoint_ids
            ),
            "alpha": CONTROLLED_MARCONI_ALPHA,
            "scores": score_rows,
            "normalization": "PASS",
        },
        "flowstate": {
            "selected_checkpoint_ids": list(flowstate_selected),
            "selected_count": len(flowstate_selected),
            "recovery_cost_before_ms": (
                flowstate_result.recovery_cost_before_ms
            ),
            "recovery_cost_after_ms": (
                flowstate_result.recovery_cost_after_ms
            ),
            "total_benefit_ms": flowstate_result.total_benefit_ms,
            "used_bytes": flowstate_result.used_bytes,
            "continuations": flow_rows,
            "per_step_marginal_trace": "NOT EXPOSED",
        },
        "selection_valid": valid_selection,
        "flowstate_budget_valid": (
            flowstate_result.used_bytes <= BUDGET_BYTES
        ),
        "selection_comparison": {
            "lru_vs_marconi_same": set(lru_selected)
            == set(marconi_result.selected_checkpoint_ids),
            "lru_vs_flowstate_same": set(lru_selected)
            == set(flowstate_selected),
            "marconi_vs_flowstate_same": set(
                marconi_result.selected_checkpoint_ids
            )
            == set(flowstate_selected),
            "at_least_two_policies_differ": len(
                {
                    frozenset(lru_selected),
                    frozenset(marconi_result.selected_checkpoint_ids),
                    frozenset(flowstate_selected),
                }
            )
            > 1,
        },
    }


def capture_runtime_snapshot(
    barrier_client: BarrierFAControlClient,
    token_ids: Sequence[int],
    *,
    nonce: str,
) -> dict[str, object]:
    """通过已验证只读接口冻结 selector 前或后的 runtime 状态。"""
    response = barrier_client.inspect_fa_frontier(
        token_ids,
        extra_key=None,
        limit=None,
        nonce=nonce,
    )
    if not response.get("state_equal"):
        raise RuntimeError(
            f"冻结 runtime 状态时查询产生变化：{response['changed_fields']}"
        )
    if response["scope_before"] != response["scope_after"]:
        raise RuntimeError("只读查询前后 runtime scope 不一致")
    return {
        "semantic_snapshot": response["semantic_snapshot_after"],
        "resident_fa_frontier": int(response["resident_fa_frontier"]),
        "query_state_equal": True,
        "query_changed_fields": list(response["changed_fields"]),
    }


def runtime_mutation_result(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    """比较 selector 前后全部 eviction-relevant runtime 状态。"""
    changed_fields = semantic_snapshot_differences(
        before["semantic_snapshot"],
        after["semantic_snapshot"],
    )
    checks = _side_effect_checks({"changed_fields": changed_fields})
    return {
        "state_equal": not changed_fields,
        "changed_fields": changed_fields,
        "tree_unchanged": checks["tree_structure_unchanged"],
        "fa_residency_unchanged": checks["full_residency_unchanged"],
        "mamba_residency_unchanged": checks[
            "mamba_residency_unchanged"
        ],
        "fa_allocator_unchanged": checks["fa_allocator_unchanged"],
        "mamba_allocator_unchanged": checks[
            "mamba_allocator_unchanged"
        ],
        "fa_recency_unchanged": checks["fa_recency_unchanged"],
        "mamba_lru_unchanged": checks["mamba_lru_unchanged"],
        "reference_state_unchanged": checks[
            "reference_state_unchanged"
        ],
    }


def build_summary(
    *,
    artifact: Path,
    records: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    pending_rows: Sequence[Mapping[str, object]],
    selector_result: Mapping[str, object] | None,
    mutation: Mapping[str, object] | None,
    boundary_audit: Sequence[Mapping[str, object]],
    native_eviction: bool,
    fa_cascade: bool,
    environment: Mapping[str, object] | None,
    fatal_error: str | None,
) -> dict[str, object]:
    """汇总 selector 合法性、无副作用与在线信息边界门禁。"""
    selection = dict(selector_result or {})
    mutation_result = dict(mutation or {})
    future_information_used = any(
        item.get("r_plus_2_message_consumed") is not False
        or item.get("r_plus_2_request_materialized") is not False
        or item.get("pending_assistant_output_read") is not False
        for item in boundary_audit
    )
    all_executed = bool(
        len(records) == len(WORKFLOWS)
        and all(record.get("status") == "PASS" for record in records)
    )
    passed = bool(
        fatal_error is None
        and all_executed
        and len(candidate_rows) == len(WORKFLOWS)
        and len(pending_rows) == len(WORKFLOWS)
        and selection.get("common_candidate_universe") is True
        and selection.get("selection_valid") is True
        and selection.get("flowstate_budget_valid") is True
        and mutation_result.get("state_equal") is True
        and not native_eviction
        and not fa_cascade
        and not future_information_used
    )
    return {
        "schema_version": "flowstate.openhands_frozen_barrier_k2_selection.v1",
        "status": "PASS" if passed else "FAIL",
        "verdict": "READY" if passed else "PARTIAL",
        "artifact": _display_path(artifact),
        "workflows": WORKFLOWS,
        "executed_schedule": [f"{label}1" for label in WORKFLOWS],
        "pending_schedule": [f"{label}2" for label in WORKFLOWS],
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_K2_SELECTION,
        "sampling_parameters": SAMPLING_PARAMETERS,
        "logical_k": LOGICAL_K,
        "checkpoint_size_bytes": CHECKPOINT_SIZE_BYTES,
        "budget_bytes": BUDGET_BYTES,
        "executed_requests": list(records),
        "candidate_count": len(candidate_rows),
        "candidates": list(candidate_rows),
        "pending_count": len(pending_rows),
        "pending": list(pending_rows),
        "selectors": selection,
        "runtime_mutation": mutation_result,
        "native_mamba_capacity_eviction_observed": native_eviction,
        "fa_kv_cascade_eviction_observed": fa_cascade,
        "pending_requests_executed": False,
        "recurrent_eviction_performed": False,
        "controller_reconcile_executed": False,
        "online_information_boundary": list(boundary_audit),
        "future_information_used": future_information_used,
        "future_leakage": future_information_used,
        "step12h3_artifact_used_as_policy_input": False,
        "fatal_error": fatal_error,
        "environment": dict(environment) if environment is not None else None,
    }


def _run(artifact: Path) -> dict[str, object]:
    """重建真实 barrier，仅调用 selector，并验证 runtime 状态不变。"""
    requests_path = artifact / "requests.jsonl"
    census_path = artifact / "census.jsonl"
    candidates_path = artifact / "candidates.jsonl"
    pending_path = artifact / "pending.jsonl"
    selections_path = artifact / "selections.json"
    for path in (requests_path, census_path, candidates_path, pending_path):
        path.write_text("", encoding="utf-8")

    records: list[dict[str, object]] = []
    censuses: list[dict[str, object]] = []
    candidates: list[CheckpointCandidate] = []
    handles: list[RuntimeCheckpointHandle] = []
    candidate_rows: list[dict[str, object]] = []
    pending: tuple[PendingContinuation, ...] = ()
    pending_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    boundary_audit: list[dict[str, object]] = []
    selector_result = None
    mutation = None
    native_eviction = False
    fa_cascade = False
    fatal_error = None
    environment = None
    engine = None
    try:
        environment = _environment()
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )
        requests, boundary_audit = load_barrier_requests(tokenizer)
        _write_json(
            artifact / "config.json",
            {
                "workflows": WORKFLOWS,
                "executed_schedule": [f"{label}1" for label in WORKFLOWS],
                "pending_schedule": [f"{label}2" for label in WORKFLOWS],
                "visible_request_metadata": [
                    {
                        "workflow": label,
                        "turn": turn,
                        "input_tokens": len(
                            requests[(label, turn)]["input_ids"]
                        ),
                        "input_token_digest": token_digest(
                            requests[(label, turn)]["input_ids"]
                        ),
                    }
                    for label in WORKFLOWS
                    for turn in (EXECUTED_TURN, PENDING_TURN)
                ],
                "engine_configuration": ENGINE_CONFIGURATION_K2_SELECTION,
                "sampling_parameters": SAMPLING_PARAMETERS,
                "dataset_path": str(DATASET_PATH),
                "tokenizer_path": str(TOKENIZER_PATH),
                "logical_k": LOGICAL_K,
                "budget_bytes": BUDGET_BYTES,
                "online_information_boundary": boundary_audit,
                "pending_requests_executed": False,
                "recurrent_eviction_performed": False,
                "controller_reconcile_executed": False,
                "environment": environment,
            },
        )

        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(
            **ENGINE_CONFIGURATION_K2_SELECTION
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        baseline = compact_census(
            client.census("openhands-k2-selection:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        baseline["event"] = "baseline"
        censuses.append(baseline)
        _append_jsonl(census_path, baseline)
        if int(baseline["mamba_node_count"]) != 0:
            raise RuntimeError("Engine 初始 census 含有 Mamba checkpoint")

        previous = baseline
        for ordinal, (label, turn) in enumerate(SCHEDULE, start=1):
            request = requests[(label, turn)]
            try:
                record = execute_request(engine, client, request, ordinal)
                census = compact_census(
                    client.census(
                        f"openhands-k2-selection:after:{label}{turn}"
                    ),
                    ordinal=ordinal,
                    request=request,
                    previous=previous,
                )
                census["event"] = f"after_{label}{turn}"
                candidate, handle, row = locate_materialized_candidate(
                    client,
                    request,
                    census,
                    event_order=ordinal,
                )
            except Exception as error:
                record = _failure_record(request, ordinal, error)
                records.append(record)
                _append_jsonl(requests_path, record)
                raise
            records.append(record)
            censuses.append(census)
            candidates.append(candidate)
            handles.append(handle)
            candidate_rows.append(row)
            _append_jsonl(requests_path, record)
            _append_jsonl(census_path, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{label}{turn} 请求失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                native_eviction = True
                raise RuntimeError("观察到原生 Mamba capacity eviction")
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
                raise RuntimeError("观察到 FA-KV cascade eviction")

        validate_unique_checkpoint_ids(candidates)
        for index, (candidate, handle) in enumerate(
            zip(candidates, handles)
        ):
            validation = validate_candidate_at_barrier(
                client,
                candidate,
                handle,
            )
            candidate_rows[index]["barrier_validation"] = validation
            if not validation["consistent"]:
                raise RuntimeError(
                    f"candidate {candidate.checkpoint_id} residency 不一致"
                )

        barrier_client = BarrierFAControlClient(client)
        pending, pending_rows = build_pending_set(barrier_client, requests)
        metadata_rows, _ = build_policy_metadata(candidates, candidate_rows)
        metadata_by_id = {
            str(row["checkpoint_id"]): row for row in metadata_rows
        }
        for row in candidate_rows:
            row["policy_metadata"] = metadata_by_id[row["checkpoint_id"]]

        universe = common_candidate_universe(candidates)
        if universe["all_equal"] is not True:
            raise RuntimeError("三个 policy 的 candidate universe 不一致")

        a2_input_ids = requests[("A", PENDING_TURN)]["input_ids"]
        before = capture_runtime_snapshot(
            barrier_client,
            a2_input_ids,
            nonce="openhands-k2-selection:before-selectors",
        )
        selector_result = run_selectors(
            tuple(candidates),
            pending,
            metadata_rows,
        )
        after = capture_runtime_snapshot(
            barrier_client,
            a2_input_ids,
            nonce="openhands-k2-selection:after-selectors",
        )
        mutation = runtime_mutation_result(before, after)
        if not mutation["state_equal"]:
            raise RuntimeError(
                f"selector 前后 runtime 发生变化：{mutation['changed_fields']}"
            )
        if selector_result["selection_valid"] is not True:
            raise RuntimeError("selector 输出包含超预算或非 eligible candidate")
        if selector_result["flowstate_budget_valid"] is not True:
            raise RuntimeError("FlowState used_bytes 超过 logical budget")

        for row in candidate_rows:
            _append_jsonl(candidates_path, row)
        for row in pending_rows:
            _append_jsonl(pending_path, row)
        _write_json(selections_path, selector_result)
    except Exception as error:
        fatal_error = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                if fatal_error is None:
                    fatal_error = f"关闭 Engine 失败：{error!r}"

    summary = build_summary(
        artifact=artifact,
        records=records,
        candidate_rows=candidate_rows,
        pending_rows=pending_rows,
        selector_result=selector_result,
        mutation=mutation,
        boundary_audit=boundary_audit,
        native_eviction=native_eviction,
        fa_cascade=fa_cascade,
        environment=environment,
        fatal_error=fatal_error,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行唯一一次 K=2 selector-only gate。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
