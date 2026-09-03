#!/usr/bin/env python3
"""验证 Barrier 2 第二次选择到循环状态驻留结果的精确映射。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.barrier_fa_frontier_control import BarrierFAControlClient
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    SchedulerRuntimeAdapter,
    wait_for_transport,
)
from evaluation.openhands_4workflow_occupancy_calibration import (
    WORKFLOWS,
    _environment,
    compact_census,
    execute_request,
)
from evaluation.openhands_common_barrier_snapshot_gate import (
    BUDGET_BYTES,
    LOGICAL_K,
    locate_materialized_candidate,
    validate_candidate_at_barrier,
)
from evaluation.openhands_policy_to_actuator_mapping_gate import (
    FrozenSelectedSetOptimizer,
    RecordingRuntimeAdapter,
    build_controller_report,
    evaluate_mapping_invariants,
    inspect_candidate_states,
)
from evaluation.openhands_round2_dynamic_registry_selection_gate import (
    ENGINE_CONFIGURATION_DYNAMIC_REGISTRY,
    EXPECTED_BARRIER_ONE_SELECTION,
    POLICY_ORDER,
    ROUND_ONE_SCHEDULE,
    ROUND_TWO_SCHEDULE,
    _boundary_has_future_leakage,
    build_dynamic_metadata,
    build_round_three_pending,
    build_round_three_pending_compatible,
    load_round2_visible_requests,
    refresh_registry,
    register_round_two_materialization,
    registry_candidates,
    registry_entry_from_round_one,
    run_policy_selector,
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
from flowstate.controller import StateController
from flowstate.state_catalog import CheckpointCandidate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
ENGINE_CONFIGURATION_SECOND_RECONCILE = dict(
    ENGINE_CONFIGURATION_DYNAMIC_REGISTRY
)
EXPECTED_SECOND_SELECTION = {
    "LRU": (
        "OPENHANDS_BARRIER_D_TURN_002",
        "OPENHANDS_BARRIER_D_TURN_001",
    ),
    "Marconi": (
        "OPENHANDS_BARRIER_D_TURN_001",
        "OPENHANDS_BARRIER_C_TURN_001",
    ),
    "FlowState": (
        "OPENHANDS_BARRIER_C_TURN_002",
        "OPENHANDS_BARRIER_A_TURN_002",
    ),
}
EXPECTED_SECOND_ELIGIBLE = {
    "LRU": (
        "OPENHANDS_BARRIER_A_TURN_001",
        "OPENHANDS_BARRIER_B_TURN_001",
        "OPENHANDS_BARRIER_C_TURN_001",
        "OPENHANDS_BARRIER_C_TURN_002",
        "OPENHANDS_BARRIER_D_TURN_001",
        "OPENHANDS_BARRIER_D_TURN_002",
    ),
    "Marconi": (
        "OPENHANDS_BARRIER_A_TURN_001",
        "OPENHANDS_BARRIER_B_TURN_001",
        "OPENHANDS_BARRIER_C_TURN_001",
        "OPENHANDS_BARRIER_C_TURN_002",
        "OPENHANDS_BARRIER_D_TURN_001",
        "OPENHANDS_BARRIER_D_TURN_002",
    ),
    "FlowState": (
        "OPENHANDS_BARRIER_A_TURN_001",
        "OPENHANDS_BARRIER_A_TURN_002",
        "OPENHANDS_BARRIER_B_TURN_001",
        "OPENHANDS_BARRIER_C_TURN_001",
        "OPENHANDS_BARRIER_C_TURN_002",
        "OPENHANDS_BARRIER_D_TURN_001",
    ),
}


def _artifact_directory() -> Path:
    """创建不会覆盖既有结果的时间戳产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / (
        f"openhands_barrier2_second_reconcile_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """对仓库内路径返回相对表示。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class PolicyPaths:
    """保存一个独立 policy 生命周期的审计产物路径。"""

    requests: Path
    censuses: Path
    registry: Path
    pending: Path
    selections: Path
    controllers: Path
    mapping: Path


def _policy_paths(directory: Path) -> PolicyPaths:
    """创建一个 policy 独占的产物目录和空日志文件。"""
    directory.mkdir(parents=True, exist_ok=False)
    paths = PolicyPaths(
        requests=directory / "requests.jsonl",
        censuses=directory / "census.jsonl",
        registry=directory / "registry.jsonl",
        pending=directory / "pending.jsonl",
        selections=directory / "selections.json",
        controllers=directory / "controllers.json",
        mapping=directory / "second_mapping.json",
    )
    for path in (
        paths.requests,
        paths.censuses,
        paths.registry,
        paths.pending,
    ):
        path.touch(exist_ok=False)
    return paths


def build_second_reconcile_plan(
    policy: str,
    candidates: Sequence[CheckpointCandidate],
    selection: Mapping[str, object],
) -> dict[str, object]:
    """冻结第二次合法选择，并计算当前驻留候选中的驱逐集合。"""
    if policy not in POLICY_ORDER:
        raise ValueError(f"未知 policy：{policy}")
    eligible_ids = tuple(
        sorted(
            candidate.checkpoint_id
            for candidate in candidates
            if candidate.recurrent_resident
        )
    )
    selected_ids = tuple(
        str(value) for value in selection["selected_checkpoint_ids"]
    )
    if selection.get("selection_valid") is not True:
        raise RuntimeError(f"{policy} 的第二次选择不是合法结果")
    if len(selected_ids) > LOGICAL_K:
        raise RuntimeError(f"{policy} 的第二次选择超过 K={LOGICAL_K}")
    if not set(selected_ids).issubset(eligible_ids):
        raise RuntimeError(f"{policy} 选择了非驻留候选")
    expected_evicted_ids = tuple(
        sorted(set(eligible_ids) - set(selected_ids))
    )
    return {
        "policy": policy,
        "eligible_candidate_ids": list(eligible_ids),
        "selected_checkpoint_ids": list(selected_ids),
        "expected_evicted_ids": list(expected_evicted_ids),
        "expected_eviction_count": len(expected_evicted_ids),
        "matches_step12h8_candidate_universe": (
            set(eligible_ids) == set(EXPECTED_SECOND_ELIGIBLE[policy])
        ),
        "matches_step12h8_selection": (
            set(selected_ids) == set(EXPECTED_SECOND_SELECTION[policy])
        ),
    }


def _query_changed_runtime(census: Mapping[str, object]) -> bool:
    """判断 Barrier 2 FA 查询是否改变任一 runtime 结构。"""
    return bool(
        census["added_mamba_node_ids"]
        or census["removed_mamba_node_ids"]
        or census["changed_existing_mamba_node_ids"]
        or census["removed_full_device_node_ids"]
        or census["removed_structure_node_ids"]
    )


def _run_policy_lifecycle(
    *,
    policy: str,
    engine_ordinal: int,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    boundary_audit: Sequence[Mapping[str, object]],
    paths: PolicyPaths,
) -> dict[str, object]:
    """执行至 Barrier 2 第二次 reconcile，并在 Round 3 前停止。"""
    engine = None
    records: list[dict[str, object]] = []
    registry = {}
    registry_events: list[dict[str, object]] = []
    first_selection = None
    second_selection = None
    first_controller_report = None
    second_controller_report = None
    first_mapping = None
    second_mapping = None
    second_plan = None
    pending_rows: list[dict[str, object]] = []
    native_eviction = False
    fa_cascade = False
    query_state_unchanged = False
    fatal_error = None
    shutdown_error = None
    try:
        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(
            **ENGINE_CONFIGURATION_SECOND_RECONCILE
        )
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        baseline = compact_census(
            client.census(f"openhands-second:{policy}:baseline"),
            ordinal=0,
            request=None,
            previous=None,
        )
        baseline["event"] = "baseline"
        _append_jsonl(paths.censuses, baseline)
        if int(baseline["mamba_node_count"]) != 0:
            raise RuntimeError("fresh Engine 初始含循环检查点")
        previous = baseline

        round_one_candidates = []
        round_one_handles = {}
        for ordinal, (label, turn) in enumerate(
            ROUND_ONE_SCHEDULE,
            start=1,
        ):
            request = requests[(label, turn)]
            record = execute_request(engine, client, request, ordinal)
            census = compact_census(
                client.census(
                    f"openhands-second:{policy}:after:{label}{turn}"
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
            entry = registry_entry_from_round_one(candidate, handle, row)
            registry[entry.checkpoint_id] = entry
            round_one_candidates.append(candidate)
            round_one_handles[candidate.checkpoint_id] = handle
            records.append({**record, "policy": policy})
            _append_jsonl(paths.requests, records[-1])
            _append_jsonl(paths.censuses, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{policy} 的 {label}1 请求失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                native_eviction = True
                raise RuntimeError("Round 1 发生原生 Mamba 驱逐")
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
                raise RuntimeError("Round 1 发生 FA 级联")

        for candidate in round_one_candidates:
            validation = validate_candidate_at_barrier(
                client,
                candidate,
                round_one_handles[candidate.checkpoint_id],
            )
            if not validation["consistent"]:
                raise RuntimeError("Barrier 1 candidate residency 不一致")
        barrier_client = BarrierFAControlClient(client)
        pending_two, _ = build_round_three_pending_compatible(
            barrier_client,
            requests,
            turn=2,
            policy=policy,
        )
        candidates_one = registry_candidates(registry)
        metadata_one, _ = build_dynamic_metadata(
            registry,
            candidates_one,
        )
        first_selection = run_policy_selector(
            policy,
            candidates_one,
            pending_two,
            metadata_one,
        )
        if set(first_selection["selected_checkpoint_ids"]) != set(
            EXPECTED_BARRIER_ONE_SELECTION[policy]
        ):
            raise RuntimeError(f"{policy} 的 Barrier 1 selected set 异常")

        before_first = compact_census(
            client.census(f"openhands-second:{policy}:before-first"),
            ordinal=4,
            request=None,
            previous=previous,
        )
        before_first_states, _ = inspect_candidate_states(
            client,
            candidates_one,
            round_one_handles,
            phase=f"{policy}_SECOND_FIRST_BEFORE",
        )
        first_adapter = RecordingRuntimeAdapter(
            SchedulerRuntimeAdapter(client)
        )
        first_controller = StateController(
            FrozenSelectedSetOptimizer(
                first_selection["selected_checkpoint_ids"]
            ),
            first_adapter,
        )
        first_allocation = None
        try:
            first_allocation = first_controller.reconcile(
                pending_two,
                candidates_one,
                round_one_handles,
                BUDGET_BYTES,
            )
        finally:
            first_controller_report = build_controller_report(
                allocation=first_allocation,
                adapter=first_adapter,
            )
        after_first = compact_census(
            client.census(f"openhands-second:{policy}:after-first"),
            ordinal=4,
            request=None,
            previous=before_first,
        )
        after_first_states, _ = inspect_candidate_states(
            client,
            candidates_one,
            round_one_handles,
            phase=f"{policy}_SECOND_FIRST_AFTER",
        )
        first_expected_evicted = sorted(
            set(round_one_handles)
            - set(first_selection["selected_checkpoint_ids"])
        )
        first_mapping = evaluate_mapping_invariants(
            candidate_ids=list(round_one_handles),
            selected_ids=first_selection["selected_checkpoint_ids"],
            expected_evicted_ids=first_expected_evicted,
            handles=round_one_handles,
            before_states=before_first_states,
            after_states=after_first_states,
            before_census=before_first,
            after_census=after_first,
            controller_report=first_controller_report,
        )
        if first_mapping["status"] != "PASS":
            raise RuntimeError(f"{policy} 的 Barrier 1 reconcile 失败")
        refresh_registry(
            client,
            registry,
            phase=f"{policy}_SECOND_POST_BARRIER1",
        )
        previous = after_first

        for offset, (label, turn) in enumerate(
            ROUND_TWO_SCHEDULE,
            start=1,
        ):
            ordinal = len(ROUND_ONE_SCHEDULE) + offset
            request = requests[(label, turn)]
            previously_resident = {
                checkpoint_id
                for checkpoint_id, entry in registry.items()
                if entry.recurrent_resident
            }
            record = execute_request(engine, client, request, ordinal)
            census = compact_census(
                client.census(
                    f"openhands-second:{policy}:after:{label}{turn}"
                ),
                ordinal=ordinal,
                request=request,
                previous=previous,
            )
            census["event"] = f"after_{label}{turn}"
            event = register_round_two_materialization(
                client,
                registry,
                request,
                record,
                census,
                event_order=ordinal,
                previously_resident_ids=previously_resident,
            )
            registry_events.append(event)
            refresh_registry(
                client,
                registry,
                phase=f"{policy}_SECOND_{label}{turn}",
            )
            records.append({**record, "policy": policy})
            _append_jsonl(paths.requests, records[-1])
            _append_jsonl(paths.censuses, census)
            previous = census
            if record["status"] != "PASS":
                raise RuntimeError(f"{policy} 的 {label}2 请求失败")
            if census["native_mamba_capacity_eviction_inferred"]:
                native_eviction = True
                raise RuntimeError("Round 2 发生原生 Mamba 驱逐")
            if census["fa_kv_cascade_eviction_inferred"]:
                fa_cascade = True
                raise RuntimeError("Round 2 发生 FA 级联")

        refresh_registry(
            client,
            registry,
            phase=f"{policy}_SECOND_BARRIER2",
        )
        candidates_two = registry_candidates(registry)
        metadata_two, _ = build_dynamic_metadata(
            registry,
            candidates_two,
        )
        pending_three, pending_rows = build_round_three_pending(
            barrier_client,
            requests,
            policy=policy,
        )
        post_query = compact_census(
            client.census(f"openhands-second:{policy}:after-query"),
            ordinal=8,
            request=None,
            previous=previous,
        )
        query_state_unchanged = not _query_changed_runtime(post_query)
        if not query_state_unchanged:
            raise RuntimeError("Barrier 2 FA frontier 查询改变 runtime")
        second_selection = run_policy_selector(
            policy,
            candidates_two,
            pending_three,
            metadata_two,
        )
        second_plan = build_second_reconcile_plan(
            policy,
            candidates_two,
            second_selection,
        )

        eligible_ids = set(second_plan["eligible_candidate_ids"])
        eligible_candidates = tuple(
            candidate
            for candidate in candidates_two
            if candidate.checkpoint_id in eligible_ids
        )
        eligible_handles = {
            candidate.checkpoint_id: registry[candidate.checkpoint_id].handle
            for candidate in eligible_candidates
        }
        for candidate in eligible_candidates:
            validation = validate_candidate_at_barrier(
                client,
                candidate,
                eligible_handles[candidate.checkpoint_id],
            )
            if not validation["consistent"]:
                raise RuntimeError("Barrier 2 candidate residency 不一致")
        before_second = compact_census(
            client.census(f"openhands-second:{policy}:before-second"),
            ordinal=8,
            request=None,
            previous=post_query,
        )
        before_second["event"] = "before_second_reconcile"
        _append_jsonl(paths.censuses, before_second)
        before_second_states, before_responses = inspect_candidate_states(
            client,
            eligible_candidates,
            eligible_handles,
            phase=f"{policy}_SECOND_BARRIER2_BEFORE",
        )
        if not all(
            state["fa_resident"] and state["recurrent_resident"]
            for state in before_second_states.values()
        ):
            raise RuntimeError("Barrier 2 reconcile 前 eligible 候选未全部驻留")

        second_adapter = RecordingRuntimeAdapter(
            SchedulerRuntimeAdapter(client)
        )
        second_controller = StateController(
            FrozenSelectedSetOptimizer(
                second_plan["selected_checkpoint_ids"]
            ),
            second_adapter,
        )
        second_allocation = None
        try:
            second_allocation = second_controller.reconcile(
                pending_three,
                eligible_candidates,
                eligible_handles,
                BUDGET_BYTES,
            )
        finally:
            second_controller_report = build_controller_report(
                allocation=second_allocation,
                adapter=second_adapter,
            )
        after_second = compact_census(
            client.census(f"openhands-second:{policy}:after-second"),
            ordinal=8,
            request=None,
            previous=before_second,
        )
        after_second["event"] = "after_second_reconcile"
        _append_jsonl(paths.censuses, after_second)
        after_second_states, after_responses = inspect_candidate_states(
            client,
            eligible_candidates,
            eligible_handles,
            phase=f"{policy}_SECOND_BARRIER2_AFTER",
        )
        second_mapping = evaluate_mapping_invariants(
            candidate_ids=second_plan["eligible_candidate_ids"],
            selected_ids=second_plan["selected_checkpoint_ids"],
            expected_evicted_ids=second_plan["expected_evicted_ids"],
            handles=eligible_handles,
            before_states=before_second_states,
            after_states=after_second_states,
            before_census=before_second,
            after_census=after_second,
            controller_report=second_controller_report,
        )
        if second_mapping["status"] != "PASS":
            raise RuntimeError(f"{policy} 的 Barrier 2 reconcile 失败")
        native_eviction = bool(
            native_eviction
            or second_mapping["native_mamba_capacity_eviction"]
        )
        fa_cascade = bool(fa_cascade or second_mapping["fa_kv_cascade"])
        if native_eviction or fa_cascade:
            raise RuntimeError("Barrier 2 reconcile 出现非预期 runtime 变化")

        refresh_registry(
            client,
            registry,
            phase=f"{policy}_SECOND_POST_RECONCILE",
        )
        final_resident = sorted(
            checkpoint_id
            for checkpoint_id, entry in registry.items()
            if entry.recurrent_resident and checkpoint_id in eligible_ids
        )
        if set(final_resident) != set(
            second_plan["selected_checkpoint_ids"]
        ):
            raise RuntimeError("最终 recurrent residency 与选择集合不一致")

        for entry in registry.values():
            row = entry.row()
            row["policy"] = policy
            _append_jsonl(paths.registry, row)
        for row in pending_rows:
            _append_jsonl(paths.pending, {**row, "policy": policy})
        _write_json(
            paths.selections,
            {
                "barrier_1": first_selection,
                "barrier_2": second_selection,
            },
        )
        _write_json(
            paths.controllers,
            {
                "barrier_1": first_controller_report,
                "barrier_2": second_controller_report,
            },
        )
        _write_json(
            paths.mapping,
            {
                "plan": second_plan,
                "invariants": second_mapping,
                "before_inspections": before_responses,
                "after_inspections": after_responses,
            },
        )
    except Exception as error:
        fatal_error = repr(error)
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as error:
                shutdown_error = repr(error)

    future_leakage = _boundary_has_future_leakage(boundary_audit)
    passed = bool(
        fatal_error is None
        and shutdown_error is None
        and len(records) == 8
        and all(record.get("status") == "PASS" for record in records)
        and first_mapping is not None
        and first_mapping.get("status") == "PASS"
        and second_mapping is not None
        and second_mapping.get("status") == "PASS"
        and query_state_unchanged
        and not native_eviction
        and not fa_cascade
        and not future_leakage
    )
    return {
        "policy": policy,
        "engine_ordinal": engine_ordinal,
        "engine_lifecycle": "independent_fresh",
        "status": "PASS" if passed else "FAIL",
        "round_1_selection": first_selection,
        "round_1_mapping": first_mapping,
        "round_2_requests": [
            record for record in records if record.get("turn") == 2
        ],
        "registry_events": registry_events,
        "barrier_2_pending": pending_rows,
        "round_2_selection": second_selection,
        "second_reconcile_plan": second_plan,
        "second_controller_report": second_controller_report,
        "second_mapping_invariants": second_mapping,
        "barrier_2_fa_query_state_unchanged": query_state_unchanged,
        "second_selection_applied": second_mapping is not None,
        "round_3_requests_executed": False,
        "native_mamba_capacity_eviction": native_eviction,
        "fa_kv_cascade": fa_cascade,
        "future_leakage": future_leakage,
        "fatal_error": fatal_error,
        "engine_shutdown_error": shutdown_error,
    }


def build_summary(
    *,
    artifact: Path,
    runs: Sequence[Mapping[str, object]],
    boundary_audit: Sequence[Mapping[str, object]],
    environment: Mapping[str, object] | None,
) -> dict[str, object]:
    """汇总三个 path-dependent 第二次 reconcile 门禁。"""
    complete = bool(
        len(runs) == len(POLICY_ORDER)
        and [run.get("policy") for run in runs] == list(POLICY_ORDER)
        and all(run.get("status") == "PASS" for run in runs)
    )
    future_leakage = _boundary_has_future_leakage(boundary_audit) or any(
        run.get("future_leakage") is True for run in runs
    )
    universes = {
        str(run["policy"]): list(
            run.get("second_reconcile_plan", {}).get(
                "eligible_candidate_ids", ()
            )
        )
        for run in runs
    }
    mappings_exact = bool(
        complete
        and all(
            run.get("second_mapping_invariants", {}).get("status")
            == "PASS"
            for run in runs
        )
    )
    passed = bool(
        complete
        and mappings_exact
        and all(
            run.get("barrier_2_fa_query_state_unchanged") is True
            for run in runs
        )
        and not any(
            run.get("native_mamba_capacity_eviction") for run in runs
        )
        and not any(run.get("fa_kv_cascade") for run in runs)
        and not future_leakage
    )
    return {
        "schema_version": "flowstate.openhands_barrier2_mapping.v1",
        "status": "PASS" if passed else "FAIL",
        "verdict": "READY" if passed else "PARTIAL",
        "artifact": _display_path(artifact),
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_SECOND_RECONCILE,
        "policy_order": list(POLICY_ORDER),
        "engine_lifecycle_count": len(runs),
        "fresh_engine_per_policy": True,
        "round_1_schedule": [f"{label}1" for label in WORKFLOWS],
        "round_2_schedule": [f"{label}2" for label in WORKFLOWS],
        "logical_k": LOGICAL_K,
        "budget_bytes": BUDGET_BYTES,
        "runs": list(runs),
        "eligible_registry_by_policy": universes,
        "candidate_universe_path_dependent": True,
        "candidate_universe_equality_required": False,
        "second_reconcile_mappings_exact": mappings_exact,
        "round_3_requests_executed": False,
        "native_mamba_capacity_eviction": any(
            run.get("native_mamba_capacity_eviction") for run in runs
        ),
        "fa_kv_cascade": any(
            run.get("fa_kv_cascade") for run in runs
        ),
        "future_leakage": future_leakage,
        "online_information_boundary": list(boundary_audit),
        "environment": dict(environment) if environment is not None else None,
    }


def _run(artifact: Path) -> dict[str, object]:
    """顺序执行三个 fresh policy 生命周期并在第二次 reconcile 后停止。"""
    environment = _environment()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )
    requests, boundary_audit = load_round2_visible_requests(tokenizer)
    _write_json(
        artifact / "config.json",
        {
            "policies": list(POLICY_ORDER),
            "engine_lifecycle_count": len(POLICY_ORDER),
            "fresh_engine_per_policy": True,
            "engine_configuration": ENGINE_CONFIGURATION_SECOND_RECONCILE,
            "sampling_parameters": SAMPLING_PARAMETERS,
            "logical_k": LOGICAL_K,
            "budget_bytes": BUDGET_BYTES,
            "dataset_path": str(DATASET_PATH),
            "tokenizer_path": str(TOKENIZER_PATH),
            "visible_turns": [1, 2, 3],
            "round_3_assistant_output_read": False,
            "round_4_or_later_materialized": False,
            "second_selection_applied": True,
            "round_3_requests_executed": False,
            "online_information_boundary": boundary_audit,
            "environment": environment,
        },
    )
    runs = []
    for ordinal, policy in enumerate(POLICY_ORDER, start=1):
        paths = _policy_paths(
            artifact / f"policy_{ordinal:02d}_{policy.lower()}"
        )
        result = _run_policy_lifecycle(
            policy=policy,
            engine_ordinal=ordinal,
            requests=requests,
            boundary_audit=boundary_audit,
            paths=paths,
        )
        runs.append(result)
        if result["status"] != "PASS":
            break
    summary = build_summary(
        artifact=artifact,
        runs=runs,
        boundary_audit=boundary_audit,
        environment=environment,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行 Barrier 2 第二次 reconcile 门禁。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
