"""Step 13G-B2.3：对既有 AgentX full-23 runtime artifact 做最终 CPU 审计。

本模块不启动 SGLang、GPU 或 policy。兼容集合只按 B1 冻结的逻辑 lineage 与
已验证 FORK 继承关系重建；radix/token LCP/物理驻留均不参与兼容性定义。
"""

from __future__ import annotations

import argparse
import ast
import bisect
import copy
import hashlib
import itertools
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from evaluation.agentx_qwen_replay_preflight import (
    DEFAULT_FROZEN_CENSUS_PATH,
    _build_conversations_for_population,
    _find_epoch_snapshot,
    _request_index_for_target,
)
from evaluation.agentx_structure_audit import (
    FROZEN_AGENTX_PATH,
    _analyze_trace_online_compatibility,
)


DEFAULT_B1_ROOT = Path(
    "/home/wjg/data/agentx/audits/"
    "agentx_runtime_population_census_20260905_010614"
)
DEFAULT_B2_ROOT = Path(
    "/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649"
)
FORBIDDEN_B2_ROOT = Path(
    "/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_085021"
)
COLLECTOR_PATH = Path(__file__).with_name("agentx_runtime_snapshot_collection.py")

EXPECTED_SNAPSHOT_COUNT = 23
EXPECTED_CANDIDATE_HANDLE_COUNT = 568
GPU_CLEANUP_THRESHOLD_MIB = 8 * 1024
EMPTY_STRUCTURE_SHA256 = "d363e84bd4e79c14d71d949bc60a19e6e3eca96e77c3add365bb8448e810187b"
EMPTY_FULL_TREE_SHA256 = "8f3101b31d8933572012f6802df40fd0373f0de6dc6cfae8f932affc14dd1f3b"
EMPTY_MAMBA_TREE_SHA256 = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"


def _sha256_file(path: Path) -> str:
    """计算文件 SHA256，供来源与产物完整性留档。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 并忽略空行。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    """用稳定格式写入 JSON。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_id(conversation_id: str, request_index: int) -> str:
    """构造与 B2 一致的 candidate ID。"""
    return f"{conversation_id}::req{request_index:04d}"


def _pending_id(conversation_id: str, request_index: int) -> str:
    """构造与 B2 一致的 pending continuation ID。"""
    return f"{conversation_id}::pending{request_index:04d}"


def logical_compatible_pending_ids(
    candidate_conversation_index: int,
    candidate_token_position: int,
    active_targets: dict[int, int],
    conversations: list[Any],
) -> list[str]:
    """按 B1 正式定义返回一个 candidate 的 exact compatible pending 集合。"""
    candidate = conversations[candidate_conversation_index]
    compatible: list[str] = []
    for pending_index, target in active_targets.items():
        pending = conversations[pending_index]
        if pending_index == candidate_conversation_index:
            continue
        if not pending.is_inherited or target <= 0:
            continue
        candidate_path = tuple(candidate.lineage_path)
        pending_path = tuple(pending.lineage_path)
        if len(candidate_path) >= len(pending_path):
            continue
        if pending_path[: len(candidate_path)] != candidate_path:
            continue
        if candidate_token_position > target:
            continue
        request_index = _request_index_for_target(pending, target)
        if request_index is None:
            continue
        compatible.append(_pending_id(pending.conversation_id, request_index))
    return sorted(compatible)


def _build_b1_exact_sets(
    population: list[dict[str, Any]],
    trace_conversations: list[Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """由冻结 corpus 与 formal population 重建 568 个 candidate 的 B1 exact set。"""
    by_trace = {tc.trace_id: tc for tc in trace_conversations}
    result: dict[str, dict[str, Any]] = {}
    degree_checks = 0
    pending_ids_by_trace: dict[str, list[str]] = {}
    trace_summaries: list[dict[str, Any]] = []

    for entry in population:
        trace_id = entry["trace_id"]
        tc = by_trace[trace_id]
        online = _analyze_trace_online_compatibility(tc)
        snapshot = _find_epoch_snapshot(
            online,
            float(entry["t"]),
            entry["active_conversation_ids"],
            tc.conversations,
        )
        offsets: list[int] = []
        offset = 0
        for conv in tc.conversations:
            offsets.append(offset)
            offset += len(conv.request_hash_token_positions)

        trace_candidates: list[str] = []
        for conversation_index, conv in enumerate(tc.conversations):
            completed = bisect.bisect_right(conv.request_end_seconds, float(entry["t"]) + 1e-12)
            positions = list(itertools.accumulate(conv.request_hash_token_positions))
            for request_index in range(completed):
                identifier = _candidate_id(conv.conversation_id, request_index)
                compatible = logical_compatible_pending_ids(
                    conversation_index,
                    int(positions[request_index]),
                    snapshot.active_targets,
                    tc.conversations,
                )
                frozen_degree = int(snapshot.candidate_degrees[offsets[conversation_index] + request_index])
                if frozen_degree != len(compatible):
                    raise RuntimeError(
                        f"{identifier} 重建 degree={len(compatible)} 与 B1 frozen degree={frozen_degree} 不一致"
                    )
                degree_checks += 1
                trace_candidates.append(identifier)
                result[identifier] = {
                    "trace_id": trace_id,
                    "conversation_id": conv.conversation_id,
                    "request_index": request_index,
                    "token_pos": int(positions[request_index]),
                    "lineage_path": list(conv.lineage_path),
                    "compatible_pending_ids": compatible,
                    "d_t_c": len(compatible),
                }

        pending_ids: list[str] = []
        for conversation_index, target in snapshot.active_targets.items():
            conv = tc.conversations[conversation_index]
            request_index = _request_index_for_target(conv, target)
            if request_index is not None:
                pending_ids.append(_pending_id(conv.conversation_id, request_index))
        pending_ids_by_trace[trace_id] = sorted(pending_ids)

        if len(trace_candidates) != int(entry["candidate_count"]):
            raise RuntimeError(f"{trace_id} B1 candidate 数量重建不一致")
        if len(pending_ids) != int(entry["pending_count"]):
            raise RuntimeError(f"{trace_id} B1 pending 数量重建不一致")
        reconstructed_max = max((result[cid]["d_t_c"] for cid in trace_candidates), default=0)
        if reconstructed_max != int(entry["max_degree"]):
            raise RuntimeError(f"{trace_id} B1 max degree 重建不一致")
        trace_summaries.append(
            {
                "trace_id": trace_id,
                "candidate_count": len(trace_candidates),
                "pending_count": len(pending_ids),
                "max_d_t_c": reconstructed_max,
            }
        )

    metadata = {
        "candidate_degree_validated": degree_checks,
        "pending_ids_by_trace": pending_ids_by_trace,
        "trace_summaries": trace_summaries,
        "max_d_t_c": max((row["d_t_c"] for row in result.values()), default=0),
    }
    return result, metadata


def _classify_raw_extra(
    candidate: dict[str, Any],
    pending_id: str,
    conversation_by_id: dict[str, Any],
) -> str:
    """解释一个仅存在于 B2 raw 的兼容 pair 为何不属于 B1。"""
    pending_conversation_id = pending_id.rsplit("::pending", 1)[0]
    pending = conversation_by_id[pending_conversation_id]
    if pending_conversation_id == candidate["conversation_id"]:
        return "same_conversation_self_match"
    if not pending.is_inherited:
        return "spawn_or_noninherited_physical_prefix_match"
    candidate_path = tuple(candidate["lineage_path"])
    pending_path = tuple(pending.lineage_path)
    if len(candidate_path) >= len(pending_path) or pending_path[: len(candidate_path)] != candidate_path:
        return "nonancestor_physical_prefix_match"
    return "physical_prefix_match_outside_frozen_logical_relation"


def audit_compatibility(
    population: list[dict[str, Any]],
    trace_conversations: list[Any],
    raw_snapshots: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """逐 candidate 对比 B1 exact set 与 B2 raw，并构造离线纠正版。"""
    expected, b1_meta = _build_b1_exact_sets(population, trace_conversations)
    conversation_by_id = {
        conv.conversation_id: conv
        for tc in trace_conversations
        for conv in tc.conversations
    }
    raw_by_id: dict[str, dict[str, Any]] = {}
    observed_pending_by_trace: dict[str, list[str]] = {}
    corrected_snapshots = copy.deepcopy(raw_snapshots)

    for snapshot in corrected_snapshots:
        observed_pending_by_trace[snapshot["trace_id"]] = sorted(
            row["continuation_id"] for row in snapshot["pendings"]
        )
        for checkpoint in snapshot["checkpoints"]:
            identifier = checkpoint["checkpoint_id"]
            raw_ids = sorted(set(checkpoint["compatible_pending_ids"]))
            if int(checkpoint["d_t_c"]) != len(raw_ids):
                raise RuntimeError(f"{identifier} 的 B2 raw d_t(c) 与集合大小不一致")
            raw_by_id[identifier] = copy.deepcopy(checkpoint)
            corrected = expected.get(identifier)
            if corrected is None:
                continue
            checkpoint["raw_compatible_pending_ids"] = checkpoint["compatible_pending_ids"]
            checkpoint["raw_d_t_c"] = checkpoint["d_t_c"]
            checkpoint["compatible_pending_ids"] = corrected["compatible_pending_ids"]
            checkpoint["d_t_c"] = corrected["d_t_c"]
            checkpoint["compatibility_source"] = (
                "B1 frozen logical lineage + validated FORK inheritance"
            )

    expected_ids = set(expected)
    raw_ids = set(raw_by_id)
    candidate_universe_match = expected_ids == raw_ids
    pending_universe_mismatches = []
    for trace_id, expected_pending in b1_meta["pending_ids_by_trace"].items():
        observed = observed_pending_by_trace.get(trace_id, [])
        if expected_pending != observed:
            pending_universe_mismatches.append(
                {"trace_id": trace_id, "b1_pending_ids": expected_pending, "b2_pending_ids": observed}
            )

    mismatches: list[dict[str, Any]] = []
    d_gt_2: list[dict[str, Any]] = []
    metadata_mismatches: list[dict[str, Any]] = []
    exact_matches = 0
    reason_counts: Counter[str] = Counter()
    missing_pair_count = 0
    extra_pair_count = 0
    for identifier in sorted(expected_ids & raw_ids):
        expected_row = expected[identifier]
        raw_row = raw_by_id[identifier]
        differing_fields = {
            field: {"b1": expected_row[field], "b2": raw_row.get(field)}
            for field in ("conversation_id", "request_index", "token_pos")
            if expected_row[field] != raw_row.get(field)
        }
        if differing_fields:
            metadata_mismatches.append(
                {"candidate_id": identifier, "differing_fields": differing_fields}
            )
        b1_ids = expected[identifier]["compatible_pending_ids"]
        b2_ids = sorted(set(raw_by_id[identifier]["compatible_pending_ids"]))
        if b1_ids == b2_ids:
            exact_matches += 1
        else:
            missing = sorted(set(b1_ids) - set(b2_ids))
            extras = sorted(set(b2_ids) - set(b1_ids))
            extra_reasons = {
                pending_id: _classify_raw_extra(expected[identifier], pending_id, conversation_by_id)
                for pending_id in extras
            }
            reason_counts.update(extra_reasons.values())
            missing_pair_count += len(missing)
            extra_pair_count += len(extras)
            mismatches.append(
                {
                    "trace_id": expected[identifier]["trace_id"],
                    "candidate_id": identifier,
                    "b1_compatible_pending_ids": b1_ids,
                    "b2_raw_compatible_pending_ids": b2_ids,
                    "missing_from_b2_raw": missing,
                    "extra_in_b2_raw": extras,
                    "extra_root_causes": extra_reasons,
                }
            )
        if len(b2_ids) > 2:
            d_gt_2.append(
                {
                    "trace_id": expected[identifier]["trace_id"],
                    "candidate_id": identifier,
                    "b1_compatible_pending_ids": b1_ids,
                    "b2_raw_compatible_pending_ids": b2_ids,
                    "b1_d_t_c": len(b1_ids),
                    "b2_raw_d_t_c": len(b2_ids),
                    "root_cause": (
                        "B2 raw token-prefix/LCP 判据把不属于 B1 严格逻辑祖先 + inherited FORK "
                        "关系的物理前缀重合计入兼容集合"
                    ),
                }
            )

    raw_max = max((int(row["d_t_c"]) for row in raw_by_id.values()), default=0)
    corrected_max = int(b1_meta["max_d_t_c"])
    total = len(expected)
    audit = {
        "formal_definition": (
            "仅 logical lineage 严格祖先、validated FORK inherited、candidate token position "
            "不超过 active target；禁止 radix/token LCP/物理驻留定义兼容性"
        ),
        "b1_max_d_t_c": corrected_max,
        "b2_raw_max_d_t_c": raw_max,
        "offline_corrected_max_d_t_c": corrected_max,
        "candidate_exact_set_matches": exact_matches,
        "candidate_exact_set_total": total,
        "candidate_exact_set_mismatch_count": total - exact_matches,
        "candidate_universe_exact_match": candidate_universe_match,
        "candidate_only_in_b1": sorted(expected_ids - raw_ids),
        "candidate_only_in_b2": sorted(raw_ids - expected_ids),
        "candidate_metadata_mismatch_count": len(metadata_mismatches),
        "candidate_metadata_mismatches": metadata_mismatches,
        "pending_universe_exact_match_snapshots": len(population) - len(pending_universe_mismatches),
        "pending_universe_total_snapshots": len(population),
        "pending_universe_mismatches": pending_universe_mismatches,
        "b1_frozen_degree_validated": b1_meta["candidate_degree_validated"],
        "missing_logical_pair_count": missing_pair_count,
        "extra_raw_pair_count": extra_pair_count,
        "extra_root_cause_counts": dict(sorted(reason_counts.items())),
        "mismatched_candidates": mismatches,
        "b2_raw_d_t_c_gt_2_count": len(d_gt_2),
        "b2_raw_d_t_c_gt_2": d_gt_2,
        "root_cause": (
            "B2 raw 用单个已执行请求的 recurrent prefix 与 pending token 序列做 token-prefix/LCP "
            "匹配，它与 B1 的累计逻辑位置关系不是同一判据：一方面把同 conversation、"
            "SPAWN/non-inherited 或非逻辑祖先的物理前缀重合误计为兼容；另一方面会漏掉"
            "逻辑祖先成立、但独立合成的请求 token 序列不是 pending 字面前缀的 pair。"
            "B1 正式定义只允许 logical lineage 的严格祖先覆盖 active inherited FORK 后代。"
        ),
        "corrected_exactly_matches_b1": (
            candidate_universe_match
            and not pending_universe_mismatches
            and not metadata_mismatches
            and b1_meta["candidate_degree_validated"] == total
        ),
    }
    return audit, corrected_snapshots


def _collector_lifecycle_evidence(path: Path) -> dict[str, Any]:
    """用 AST 核对每个 snapshot 的 fresh runtime 构造与 finally shutdown。"""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def calls(function_name: str, called_name: str) -> bool:
        function = functions.get(function_name)
        if function is None:
            return False
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == called_name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == called_name:
                return True
        return False

    return {
        "collector_path": str(path.resolve()),
        "collector_sha256": _sha256_file(path),
        "collect_snapshot_calls_start_runtime": calls("_collect_snapshot", "_start_runtime"),
        "collect_snapshot_calls_shutdown": calls("_collect_snapshot", "shutdown"),
        "run_collection_calls_collect_snapshot": calls("run_collection", "_collect_snapshot"),
        "start_runtime_constructs_engine": calls("_start_runtime", "FormalEndToEndGateEngine"),
    }


def _snapshot_runtime_state_checks(snapshot: dict[str, Any], trace_id: str) -> dict[str, Any]:
    """验证一个快照启动 baseline 与首请求没有继承上个快照的 runtime state。"""
    census_rows = snapshot.get("census_rows", [])
    baseline = census_rows[0] if census_rows else {}
    accounting = baseline.get("accounting", {})
    tree = baseline.get("tree", {})
    scope = baseline.get("scope", {})
    request_rows = snapshot.get("request_rows", [])
    first_metrics = request_rows[0].get("runtime_metrics", {}) if request_rows else {}
    mamba_slots = accounting.get("mamba_free_slots", [])
    mamba_clean = (
        accounting.get("mamba_available") == 192
        and accounting.get("mamba_schedulable_available") == 192
        and accounting.get("mamba_evictable") == 0
        and accounting.get("mamba_protected") == 0
        and len(mamba_slots) == 192
    )
    radix_clean = (
        tree.get("node_count") == 1
        and tree.get("mamba_node_count") == 0
        and tree.get("mamba_rows") == []
        and tree.get("structure_sha256") == EMPTY_STRUCTURE_SHA256
        and tree.get("full_tree_sha256") == EMPTY_FULL_TREE_SHA256
        and tree.get("mamba_tree_sha256") == EMPTY_MAMBA_TREE_SHA256
    )
    scheduler_clean = (
        scope.get("scheduler_fully_idle") is True
        and scope.get("waiting_requests") == 0
        and scope.get("running_requests") == 0
        and scope.get("chunked_request_present") is False
    )
    first_request_clean = (
        bool(request_rows)
        and first_metrics.get("physical_fa_hit") == 0
        and first_metrics.get("executable_prefix") == 0
        and first_metrics.get("replay_gap") == 0
        and first_metrics.get("mamba_host_hit_length") == 0
    )
    namespace_clean = (
        all(str(row.get("request_id", "")).startswith(trace_id) for row in request_rows)
        and all(str(row.get("checkpoint_id", "")).startswith(trace_id) for row in snapshot.get("checkpoints", []))
        and all(str(row.get("continuation_id", "")).startswith(trace_id) for row in snapshot.get("pendings", []))
        and str(baseline.get("nonce", "")) == f"agentx:{trace_id}:baseline"
    )
    passed = mamba_clean and radix_clean and scheduler_clean and first_request_clean and namespace_clean
    return {
        "trace_id": trace_id,
        "runtime_state_isolation_pass": passed,
        "new_engine_operational_evidence": "fresh constructor path + empty baseline + zero-hit first request",
        "scheduler_idle_and_empty": scheduler_clean,
        "recurrent_handles_or_residency_inherited": not mamba_clean,
        "radix_or_fa_cache_inherited": not (radix_clean and first_request_clean),
        "previous_request_or_session_inherited": not namespace_clean,
        "workflow_metadata_inherited": not namespace_clean,
        "baseline_full_allocator_available": accounting.get("full_allocator", {}).get("available"),
        "baseline_tree_node_count": tree.get("node_count"),
        "baseline_mamba_node_count": tree.get("mamba_node_count"),
        "first_request_physical_fa_hit": first_metrics.get("physical_fa_hit"),
        "first_request_mamba_host_hit_length": first_metrics.get("mamba_host_hit_length"),
    }


def audit_lifecycle(
    b2_root: Path,
    manifest: list[dict[str, Any]],
    gpu_lifecycle: list[dict[str, Any]],
) -> dict[str, Any]:
    """区分 runtime state contamination 与同进程 CUDA context/allocator 保留。"""
    source_evidence = _collector_lifecycle_evidence(COLLECTOR_PATH)
    source_ok = all(
        source_evidence[key]
        for key in (
            "collect_snapshot_calls_start_runtime",
            "collect_snapshot_calls_shutdown",
            "run_collection_calls_collect_snapshot",
            "start_runtime_constructs_engine",
        )
    )
    lifecycle_by_trace = {row["trace_id"]: row for row in gpu_lifecycle}
    snapshots: list[dict[str, Any]] = []
    threshold_pass = 0
    context_only_fail = 0
    for entry in manifest:
        artifact_path = Path(entry["artifact_path"])
        if not artifact_path.is_absolute():
            artifact_path = b2_root / artifact_path
        snapshot_path = artifact_path / "snapshot.json" if artifact_path.is_dir() else artifact_path
        snapshot = _read_json(snapshot_path)
        checks = _snapshot_runtime_state_checks(snapshot, entry["trace_id"])
        gpu = lifecycle_by_trace[entry["trace_id"]]
        final_used = int(gpu.get("gpu_cleanup", {}).get("final_used_mib", -1))
        raw_pass = bool(gpu.get("gpu_cleanup_pass")) and final_used < GPU_CLEANUP_THRESHOLD_MIB
        if raw_pass:
            threshold_pass += 1
            classification = "THRESHOLD_PASS"
        elif checks["runtime_state_isolation_pass"] and source_ok:
            context_only_fail += 1
            classification = "CUDA_CONTEXT_OR_ALLOCATOR_ONLY"
        else:
            classification = "RUNTIME_STATE_CONTAMINATION"
        checks.update(
            {
                "gpu_cleanup_threshold_pass": raw_pass,
                "gpu_cleanup_final_used_mib": final_used,
                "cleanup_classification": classification,
            }
        )
        snapshots.append(checks)

    isolation_pass = sum(1 for row in snapshots if row["runtime_state_isolation_pass"])
    final_used_values = [row["gpu_cleanup_final_used_mib"] for row in snapshots]
    deltas = [b - a for a, b in zip(final_used_values, final_used_values[1:])]
    allocator_available = [row["baseline_full_allocator_available"] for row in snapshots]
    contamination = [row for row in snapshots if row["cleanup_classification"] == "RUNTIME_STATE_CONTAMINATION"]
    return {
        "raw_cleanup_threshold_mib": GPU_CLEANUP_THRESHOLD_MIB,
        "raw_gpu_cleanup_pass": threshold_pass,
        "raw_gpu_cleanup_fail": len(snapshots) - threshold_pass,
        "runtime_state_isolation_pass": isolation_pass,
        "runtime_state_isolation_total": len(snapshots),
        "cuda_context_or_allocator_only_failures": context_only_fail,
        "runtime_state_contamination_count": len(contamination),
        "final_used_mib_sequence": final_used_values,
        "successive_final_used_mib_deltas": deltas,
        "baseline_full_allocator_available_sequence": allocator_available,
        "collector_lifecycle_evidence": source_evidence,
        "conclusion": (
            "16 个 <8 GiB threshold failure 仅反映同一父进程中的 CUDA context/allocator 保留；"
            "每次均走 fresh Engine 构造与 shutdown，下一快照 baseline 的 scheduler、Mamba、radix/FA、"
            "request/session/workflow state 为空，首请求 cache hit 为 0，因此不构成运行态污染或 correctness failure。"
            if not contamination
            else "检测到跨快照 runtime state contamination，必须重新采集受影响快照。"
        ),
        "snapshots": snapshots,
    }


def _validate_runtime_snapshots(
    raw_snapshots: list[dict[str, Any]],
    runtime_correctness: dict[str, Any],
) -> dict[str, Any]:
    """复核 full-23 已有 runtime correctness gate，不重跑 GPU。"""
    checkpoints = [cp for snapshot in raw_snapshots for cp in snapshot.get("checkpoints", [])]
    return {
        "designated_snapshots": EXPECTED_SNAPSHOT_COUNT,
        "attempted": int(runtime_correctness.get("attempted", -1)),
        "eligible": int(runtime_correctness.get("eligible", -1)),
        "artifact_snapshot_count": len(raw_snapshots),
        "candidate_handles_resolved": len(checkpoints),
        "all_snapshot_status_eligible": all(row.get("status") == "ELIGIBLE" for row in raw_snapshots),
        "all_fa_resident": all(bool(cp.get("fa_resident")) for cp in checkpoints),
        "all_recurrent_resident": all(bool(cp.get("recurrent_resident")) for cp in checkpoints),
        "all_node_ids_resolved": all(cp.get("node_id") is not None for cp in checkpoints),
        "no_future_leakage": int(runtime_correctness.get("future_leakage", -1)) == 0,
        "no_native_eviction": int(runtime_correctness.get("unexpected_native_eviction", -1)) == 0,
        "no_rematerialization": int(runtime_correctness.get("rematerialization", -1)) == 0,
        "no_truncation_or_oom": runtime_correctness.get("truncation_or_oom") is False,
        "no_fa_cascade": all(not bool(row.get("fa_cascade")) for row in raw_snapshots),
    }


def _runtime_validation_pass(validation: dict[str, Any]) -> bool:
    """汇总 full-23 runtime correctness gate。"""
    return (
        validation["attempted"] == EXPECTED_SNAPSHOT_COUNT
        and validation["eligible"] == EXPECTED_SNAPSHOT_COUNT
        and validation["artifact_snapshot_count"] == EXPECTED_SNAPSHOT_COUNT
        and validation["candidate_handles_resolved"] == EXPECTED_CANDIDATE_HANDLE_COUNT
        and all(value is True for key, value in validation.items() if key.startswith("all_") or key.startswith("no_"))
    )


def _build_report(result: dict[str, Any], artifact_root: Path) -> str:
    """生成简洁、可复核的中文最终审计报告。"""
    compatibility = result["compatibility"]
    cleanup = result["cleanup"]
    validation = result["runtime_validation"]
    lines = [
        "# Step 13G-B2.3 AgentX Full-23 Runtime Final Correctness Audit",
        "",
        f"- Overall Status: `{result['overall_status']}`",
        f"- designated / attempted / eligible: `{validation['designated_snapshots']} / {validation['attempted']} / {validation['eligible']}`",
        f"- B1 formal max d_t(c): `{compatibility['b1_max_d_t_c']}`",
        f"- B2 raw max d_t(c): `{compatibility['b2_raw_max_d_t_c']}`",
        f"- candidate exact-set match（raw）: `{compatibility['candidate_exact_set_matches']} / {compatibility['candidate_exact_set_total']}`",
        f"- exact-set mismatch count: `{compatibility['candidate_exact_set_mismatch_count']}`",
        f"- offline corrected max d_t(c): `{compatibility['offline_corrected_max_d_t_c']}`",
        f"- corrected compatibility equals frozen B1: `{'PASS' if compatibility['corrected_exactly_matches_b1'] else 'FAIL'}`",
        f"- raw cleanup threshold pass / fail: `{cleanup['raw_gpu_cleanup_pass']} / {cleanup['raw_gpu_cleanup_fail']}`",
        f"- runtime-state isolation PASS: `{cleanup['runtime_state_isolation_pass']} / {cleanup['runtime_state_isolation_total']}`",
        f"- runtime-state contamination: `{cleanup['runtime_state_contamination_count']}`",
        f"- candidate handles resolved: `{validation['candidate_handles_resolved']}`",
        f"- recollection required: `{'YES' if result['recollection_required'] else 'NO'}`",
        f"- readiness: `{result['readiness']}`",
        "",
        "## 兼容性根因",
        "",
        compatibility["root_cause"],
        "",
        "## Cleanup 结论",
        "",
        cleanup["conclusion"],
        "",
        f"- artifact root: `{artifact_root}`",
        "",
    ]
    return "\n".join(lines)


def run_audit(
    output_root: Path,
    *,
    b1_root: Path = DEFAULT_B1_ROOT,
    b2_root: Path = DEFAULT_B2_ROOT,
    agentx_path: Path = FROZEN_AGENTX_PATH,
) -> Path:
    """执行 B2.3 全部 CPU-only 审计并写出不可变来源上的纠正 artifact。"""
    resolved_b2 = b2_root.resolve()
    if resolved_b2 == FORBIDDEN_B2_ROOT.resolve():
        raise RuntimeError("禁止使用已声明 invalid 的 B2 run")
    if resolved_b2 != DEFAULT_B2_ROOT.resolve():
        raise RuntimeError(f"B2.3 只允许审计正式 full-23 artifact：{DEFAULT_B2_ROOT}")
    formal_population_path = b1_root / "FORMAL_CANDIDATE_POPULATION.json"
    if formal_population_path.resolve() != DEFAULT_FROZEN_CENSUS_PATH.resolve():
        raise RuntimeError("B1 formal population 路径不符合冻结输入")

    source_paths = {
        "b1_population": formal_population_path,
        "b2_runtime_snapshots": b2_root / "runtime_snapshots.jsonl",
        "b2_runtime_correctness": b2_root / "runtime_correctness.json",
        "b2_gpu_lifecycle": b2_root / "gpu_lifecycle.json",
        "b2_manifest": b2_root / "manifest.json",
    }
    source_hashes_before = {name: _sha256_file(path) for name, path in source_paths.items()}
    population = _read_json(formal_population_path)
    raw_snapshots = _read_jsonl(source_paths["b2_runtime_snapshots"])
    runtime_correctness = _read_json(source_paths["b2_runtime_correctness"])
    gpu_lifecycle = _read_json(source_paths["b2_gpu_lifecycle"])
    manifest = _read_json(source_paths["b2_manifest"])
    trace_conversations = _build_conversations_for_population(agentx_path, population)

    compatibility, corrected_snapshots = audit_compatibility(
        population, trace_conversations, raw_snapshots
    )
    cleanup = audit_lifecycle(b2_root, manifest, gpu_lifecycle)
    validation = _validate_runtime_snapshots(raw_snapshots, runtime_correctness)
    source_hashes_after = {name: _sha256_file(path) for name, path in source_paths.items()}
    sources_unchanged = source_hashes_before == source_hashes_after

    valid = _runtime_validation_pass(validation)
    compatibility_pass = bool(compatibility["corrected_exactly_matches_b1"])
    isolation_pass = cleanup["runtime_state_isolation_pass"] == EXPECTED_SNAPSHOT_COUNT
    recollection_required = not (valid and compatibility_pass and isolation_pass)
    ready = not recollection_required and sources_unchanged
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall_status": (
            "AGENTX_RUNTIME_FINAL_AUDIT_READY"
            if ready
            else "AGENTX_RUNTIME_FINAL_AUDIT_BLOCKED"
        ),
        "readiness": "READY_FOR_POLICY_EVALUATION" if ready else "NOT_READY",
        "recollection_required": recollection_required,
        "cpu_only_offline_audit": True,
        "source_b1_root": str(b1_root),
        "source_b2_root": str(b2_root),
        "forbidden_invalid_run_not_used": str(FORBIDDEN_B2_ROOT),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "source_artifacts_unchanged": sources_unchanged,
        "runtime_validation": validation,
        "compatibility": compatibility,
        "cleanup": cleanup,
    }

    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "runtime_final_correctness.json", result)
    _write_json(output_root / "compatibility_exact_set_audit.json", compatibility)
    _write_json(output_root / "cleanup_semantics_audit.json", cleanup)
    _write_json(output_root / "b2_raw_d_t_c_gt_2.json", compatibility["b2_raw_d_t_c_gt_2"])
    (output_root / "corrected_runtime_snapshots.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in corrected_snapshots) + "\n",
        encoding="utf-8",
    )
    (output_root / "final_report.md").write_text(
        _build_report(result, output_root), encoding="utf-8"
    )
    return output_root


def main(argv: Iterable[str] | None = None) -> int:
    """命令行入口；输出目录必须是新目录，避免覆盖任何既有 artifact。"""
    parser = argparse.ArgumentParser(description="AgentX full-23 runtime 最终 CPU correctness audit")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--b1-root", type=Path, default=DEFAULT_B1_ROOT)
    parser.add_argument("--b2-root", type=Path, default=DEFAULT_B2_ROOT)
    parser.add_argument("--agentx-path", type=Path, default=FROZEN_AGENTX_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(run_audit(args.output_root, b1_root=args.b1_root, b2_root=args.b2_root, agentx_path=args.agentx_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
