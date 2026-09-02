#!/usr/bin/env python3
"""验证 barrier 时刻 FA 驻留前缀查询的正确性与无副作用语义。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

from evaluation.barrier_fa_frontier_control import BarrierFAControlClient
from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    generate,
    make_tokens,
    query_runtime_metrics,
    wait_for_transport,
)
from evaluation.openhands_4workflow_occupancy_calibration import _environment
from evaluation.openhands_single_workflow_baseline10 import (
    ArtifactLogCapture,
    _append_jsonl,
    _write_json,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
ENGINE_CONFIGURATION_BARRIER_QUERY = dict(ENGINE_CONFIGURATION_128K)
BOUNDARY_PREFIX_LENGTH = 320
PARTIAL_STORED_LENGTH = 512
PARTIAL_MATCH_LENGTH = 333


def _artifact_directory() -> Path:
    """创建不会覆盖已有结果的独立产物目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ARTIFACT_ROOT / f"barrier_fa_frontier_query_{timestamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """对仓库内文件返回相对路径。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _different_token(token_id: int) -> int:
    """构造一个确定且仍位于词表范围内的不同令牌。"""
    return (int(token_id) + 1) % 248_320


def build_cases() -> tuple[dict[str, object], ...]:
    """构造互不共享首令牌的 node-boundary 与 partial-node case。"""
    boundary_prefix = make_tokens(101_003, BOUNDARY_PREFIX_LENGTH)
    boundary_setup = boundary_prefix + (
        _different_token(boundary_prefix[-1]),
    )
    boundary_pending = (
        boundary_prefix
        + (_different_token(boundary_setup[-1]),)
        + make_tokens(151_007, 64)
    )

    partial_stored = make_tokens(201_011, PARTIAL_STORED_LENGTH)
    partial_setup = partial_stored + (
        _different_token(partial_stored[-1]),
    )
    divergence = _different_token(partial_stored[PARTIAL_MATCH_LENGTH])
    partial_pending = (
        partial_stored[:PARTIAL_MATCH_LENGTH]
        + (divergence,)
        + make_tokens(231_013, 96)
    )

    if boundary_setup[0] == partial_setup[0]:
        raise RuntimeError("两个 correctness case 的首令牌意外相同")
    return (
        {
            "case_id": "node_boundary",
            "setup_rid": "barrier-fa-boundary-setup",
            "pending_rid": "barrier-fa-boundary-pending",
            "setup_input_ids": boundary_setup,
            "pending_input_ids": boundary_pending,
            "expected_h": BOUNDARY_PREFIX_LENGTH,
            "expected_partial_match": False,
        },
        {
            "case_id": "partial_node",
            "setup_rid": "barrier-fa-partial-setup",
            "pending_rid": "barrier-fa-partial-pending",
            "setup_input_ids": partial_setup,
            "pending_input_ids": partial_pending,
            "expected_h": PARTIAL_MATCH_LENGTH,
            "expected_partial_match": True,
        },
    )


def _unchanged(changed_fields: Sequence[str], prefixes: Sequence[str]) -> bool:
    """判断指定稳定状态字段是否没有发生变化。"""
    return not any(
        field == prefix or field.startswith(f"{prefix}.")
        for field in changed_fields
        for prefix in prefixes
    )


def _side_effect_checks(response: Mapping[str, object]) -> dict[str, bool]:
    """把稳定状态差异归类为最终门所需的检查项。"""
    changed = [str(value) for value in response.get("changed_fields", ())]
    return {
        "fa_recency_unchanged": _unchanged(changed, ("recency_rows",)),
        "mamba_lru_unchanged": _unchanged(
            changed,
            ("mamba_lru_order_mru_to_lru",),
        ),
        "tree_structure_unchanged": _unchanged(
            changed,
            (
                "tree.structure_rows",
                "tree.structure_sha256",
                "tree.node_count",
                "full_evictable_leaf_ids",
            ),
        ),
        "full_residency_unchanged": _unchanged(
            changed,
            ("tree.full_rows", "tree.full_tree_sha256"),
        ),
        "mamba_residency_unchanged": _unchanged(
            changed,
            (
                "tree.mamba_rows",
                "tree.mamba_tree_sha256",
                "tree.mamba_node_count",
            ),
        ),
        "fa_allocator_unchanged": _unchanged(
            changed,
            (
                "accounting.full_allocator",
                "accounting.full_evictable",
                "accounting.full_protected",
            ),
        ),
        "mamba_allocator_unchanged": _unchanged(
            changed,
            (
                "accounting.mamba_available",
                "accounting.mamba_schedulable_available",
                "accounting.mamba_free_slots",
                "accounting.mamba_free_slots_sha256",
                "accounting.mamba_evictable",
                "accounting.mamba_protected",
            ),
        ),
        "reference_state_unchanged": _unchanged(
            changed,
            ("reference_rows",),
        ),
    }


def execute_case(
    *,
    engine: object,
    client: object,
    barrier_client: BarrierFAControlClient,
    case: Mapping[str, object],
) -> dict[str, object]:
    """建立固定缓存状态并立即验证一个尚未执行的 pending 请求。"""
    setup_ids = tuple(int(value) for value in case["setup_input_ids"])
    pending_ids = tuple(int(value) for value in case["pending_input_ids"])
    generate(engine, str(case["setup_rid"]), setup_ids)

    lookup = barrier_client.inspect_fa_frontier(
        pending_ids,
        nonce=f"barrier-fa:{case['case_id']}:lookup",
    )
    side_effects = _side_effect_checks(lookup)
    if not lookup.get("state_equal"):
        raise RuntimeError(
            f"{case['case_id']} 查询改变稳定状态：{lookup['changed_fields']}"
        )

    _, metadata = generate(engine, str(case["pending_rid"]), pending_ids)
    metrics = query_runtime_metrics(client, str(case["pending_rid"]))
    h_pred = int(lookup["resident_fa_frontier"])
    h_actual = int(metrics["physical_fa_hit"])
    executable = int(metrics["executable_prefix"])
    gap = int(metrics["replay_gap"])
    expected_h = int(case["expected_h"])
    expected_partial = bool(case["expected_partial_match"])
    state_equal = bool(lookup["state_equal"] and all(side_effects.values()))
    passed = bool(
        h_pred == h_actual == expected_h
        and bool(lookup["partial_match"]) is expected_partial
        and state_equal
        and lookup["scope_before"] == lookup["scope_after"]
        and int(metadata.get("prompt_tokens", -1)) == len(pending_ids)
        and gap == h_actual - executable
    )
    return {
        "case_id": case["case_id"],
        "setup_rid": case["setup_rid"],
        "pending_rid": case["pending_rid"],
        "setup_input_tokens": len(setup_ids),
        "pending_input_tokens": len(pending_ids),
        "effective_lookup_limit": int(lookup["effective_lookup_limit"]),
        "expected_h": expected_h,
        "h_pred": h_pred,
        "h_actual": h_actual,
        "e": executable,
        "g": gap,
        "h_pred_equals_h_actual": h_pred == h_actual,
        "partial_match": bool(lookup["partial_match"]),
        "expected_partial_match": expected_partial,
        "traversed_node_ids": list(lookup["traversed_node_ids"]),
        "matched_segments": list(lookup["matched_segments"]),
        "stop_reason": lookup["stop_reason"],
        "state_equal": state_equal,
        "changed_fields": list(lookup["changed_fields"]),
        "side_effect_checks": side_effects,
        "tree_identity_unchanged": side_effects["tree_structure_unchanged"],
        "scope_unchanged": lookup["scope_before"] == lookup["scope_after"],
        "server_prompt_tokens": int(metadata.get("prompt_tokens", -1)),
        "runtime_metrics": metrics,
        "excluded_nonsemantic_fields": lookup["excluded_nonsemantic_fields"],
        "status": "PASS" if passed else "FAIL",
    }


def build_summary(
    *,
    records: Sequence[Mapping[str, object]],
    artifact: Path,
    environment: Mapping[str, object] | None,
    fatal_error: str | None,
) -> dict[str, object]:
    """汇总两个 correctness case 与全局无副作用检查。"""
    by_id = {str(record["case_id"]): record for record in records}
    boundary = by_id.get("node_boundary", {})
    partial = by_id.get("partial_node", {})
    all_side_effects = [
        value
        for record in records
        for value in record.get("side_effect_checks", {}).values()
    ]
    passed = bool(
        fatal_error is None
        and len(records) == 2
        and boundary.get("status") == "PASS"
        and partial.get("status") == "PASS"
        and partial.get("partial_match") is True
        and partial.get("tree_identity_unchanged") is True
        and all(all_side_effects)
    )
    return {
        "schema_version": "flowstate.barrier_fa_frontier_query.v1",
        "status": "PASS" if passed else "FAIL",
        "verdict": "READY" if passed else "PARTIAL",
        "interface": "checkpoint_control(action='inspect_fa_frontier')",
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_BARRIER_QUERY,
        "sampling_parameters": SAMPLING_PARAMETERS,
        "effective_request_prefix_limit": "len(input_ids) - 1",
        "enable_hicache": False,
        "page_size": 1,
        "cases": list(records),
        "recurrent_absence_case": "NOT RUN",
        "future_leakage_safe": True,
        "policy_executed": False,
        "artifact": _display_path(artifact),
        "environment": dict(environment) if environment is not None else None,
        "fatal_error": fatal_error,
    }


def _run(artifact: Path) -> dict[str, object]:
    """启动一次 Engine，并顺序执行两个最小 correctness case。"""
    cases_path = artifact / "cases.jsonl"
    cases_path.write_text("", encoding="utf-8")
    records: list[dict[str, object]] = []
    fatal_error = None
    environment = None
    engine = None
    try:
        environment = _environment()
        cases = build_cases()
        _write_json(
            artifact / "config.json",
            {
                "engine_configuration": ENGINE_CONFIGURATION_BARRIER_QUERY,
                "sampling_parameters": SAMPLING_PARAMETERS,
                "effective_request_prefix_limit": "len(input_ids) - 1",
                "cases": [
                    {
                        "case_id": case["case_id"],
                        "setup_rid": case["setup_rid"],
                        "pending_rid": case["pending_rid"],
                        "setup_input_tokens": len(case["setup_input_ids"]),
                        "pending_input_tokens": len(case["pending_input_ids"]),
                        "expected_h": case["expected_h"],
                        "expected_partial_match": case[
                            "expected_partial_match"
                        ],
                    }
                    for case in cases
                ],
                "environment": environment,
                "policy_executed": False,
            },
        )

        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(
            **ENGINE_CONFIGURATION_BARRIER_QUERY
        )
        client = ControlClient(requested_control_port())
        barrier_client = BarrierFAControlClient(client)
        wait_for_transport(client)

        for case in cases:
            record = execute_case(
                engine=engine,
                client=client,
                barrier_client=barrier_client,
                case=case,
            )
            records.append(record)
            _append_jsonl(cases_path, record)
            if record["status"] != "PASS":
                raise RuntimeError(f"{case['case_id']} correctness gate 失败")
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
        records=records,
        artifact=artifact,
        environment=environment,
        fatal_error=fatal_error,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """保存完整日志并执行唯一一次 H100 correctness gate。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
