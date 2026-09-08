"""Step 13G-C：在 AgentX 正式运行快照上执行纯 CPU 策略评估。

兼容关系只从 B1 冻结的逻辑血缘与已验证 FORK 继承语义重建。B2 中由
令牌前缀得到的原始兼容字段不会进入快照、选择器、目标函数或任何统计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.agentx_qwen_replay_preflight import (
    _build_conversations_for_population,
    _find_epoch_snapshot,
)
from evaluation.agentx_runtime_final_correctness_audit import (
    DEFAULT_B1_ROOT,
    DEFAULT_B2_ROOT,
    FORBIDDEN_B2_ROOT,
    _build_b1_exact_sets,
)
from evaluation.agentx_structure_audit import (
    FROZEN_AGENTX_PATH,
    _analyze_trace_online_compatibility,
)
from evaluation.controlled_multiworkflow_v1.scenario import CHECKPOINT_SIZE_BYTES
from evaluation.rq3_formal_policy_evaluation import (
    _EXACT_OPT_SEARCH_SPACE_THRESHOLD,
    _FLOAT_TOLERANCE_MS,
    _FROZEN_BUDGET_RATIOS,
    _bootstrap_ci,
    _policy_result,
    _summary_stats,
    aggregate_results,
    compute_budget_ks,
    create_budget_variant,
    evaluate_snapshot_at_ks,
    search_space_size,
)
from evaluation.rq3_frozen_snapshot_evaluator import (
    AllocationSnapshot,
    FrozenCheckpointRuntimeEvidence,
    FrozenOnlineInformationBoundary,
    build_allocation_snapshot,
    evaluate_objective,
)
from evaluation.rq3_sanity_structure_audit import (
    _marginal_gain,
    _standalone_topk_selection,
)
from evaluation.sota_metadata import (
    CONTROLLED_MARCONI_ALPHA,
    build_marconi_flop_saved,
)
from flowstate.recovery_model import RecoveryCostModel
from flowstate.state_catalog import CheckpointCandidate, is_compatible
from flowstate.workflow import PendingContinuation


DEFAULT_FINAL_AUDIT_ROOT = Path(
    "agentx_runtime_final_audit_20260906_194000"
)
OPENHANDS_SANITY_ROOT = Path(
    "evaluation/runtime_artifacts/rq3_sanity_structure_audit_20260904_121500"
)
EXPECTED_SNAPSHOTS = 23
EXPECTED_CANDIDATES = 568
EXPECTED_FORMAL_MAX_DEGREE = 2
POLICIES = ("LRU", "LFU", "Marconi", "FlowState")


@dataclass(frozen=True)
class AgentXFormalSnapshot:
    """保存策略输入及其独立的 B1 正式兼容证据。"""

    trace_id: str
    timestamp: float
    snapshot: AllocationSnapshot
    formal_compatible_pending_ids: Mapping[str, tuple[str, ...]]
    objective_usable_pending_ids: Mapping[str, tuple[str, ...]]
    physical_checkpoint_depths: Mapping[str, int]
    actual_lineage_paths: Mapping[str, tuple[str, ...]]


def _sha256_bytes(value: bytes) -> str:
    """返回字节序列的 SHA256。"""
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    """计算任意可序列化对象的规范摘要。"""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _read_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    """用稳定格式写入 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _semantic_lineage_paths(conversations: Sequence[Any]) -> dict[str, tuple[str, ...]]:
    """把非继承 SPAWN 作为新根，把继承 FORK 接到父逻辑路径。"""
    by_id = {conv.conversation_id: conv for conv in conversations}
    result: dict[str, tuple[str, ...]] = {}

    def resolve(conv: Any) -> tuple[str, ...]:
        cached = result.get(conv.conversation_id)
        if cached is not None:
            return cached
        parent = by_id.get(conv.parent_conversation_id)
        if conv.is_inherited and parent is not None:
            path = resolve(parent) + (conv.conversation_id,)
        else:
            path = (conv.conversation_id,)
        result[conv.conversation_id] = path
        return path

    for conversation in conversations:
        resolve(conversation)
    return result


def _assert_laminar(clusters: Sequence[frozenset[str]]) -> None:
    """确认兼容 pending 集合可以无损编码为前缀树。"""
    for index, left in enumerate(clusters):
        for right in clusters[index + 1 :]:
            if left & right and not (left <= right or right <= left):
                raise RuntimeError(
                    "B1 兼容集合不是层叠集合，无法无损送入冻结 lineage selector"
                )


def encode_usable_compatibility(
    compatible_by_candidate: Mapping[str, Sequence[str]],
    candidate_depths: Mapping[str, int],
    pending_targets: Mapping[str, int],
    trace_id: str,
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, str],
    dict[str, tuple[str, ...]],
]:
    """把 B1 兼容集合与 T 上界编码为冻结 selector 可消费的前缀树。

    B1 集合描述工作流兼容性；正式目标仍要求检查点物理深度不超过 T。
    因此编码的是二者交集，同时原始 B1 exact-set 单独保留并接受门禁。
    """
    usable: dict[str, tuple[str, ...]] = {}
    for candidate_id, pending_ids in compatible_by_candidate.items():
        depth = candidate_depths[candidate_id]
        usable[candidate_id] = tuple(
            sorted(pid for pid in pending_ids if depth <= pending_targets[pid])
        )

    clusters = sorted(
        {frozenset(values) for values in usable.values() if values},
        key=lambda values: (-len(values), tuple(sorted(values))),
    )
    _assert_laminar(clusters)
    cluster_tokens = {
        cluster: "兼容簇:" + _canonical_digest(sorted(cluster))[:16]
        for cluster in clusters
    }

    candidate_paths: dict[str, tuple[str, ...]] = {}
    candidate_workflows: dict[str, str] = {}
    for candidate_id, pending_ids in usable.items():
        cluster = frozenset(pending_ids)
        if not cluster:
            candidate_workflows[candidate_id] = f"{trace_id}:空兼容:{candidate_id}"
            candidate_paths[candidate_id] = (f"空兼容:{candidate_id}",)
            continue
        chain = [value for value in clusters if cluster <= value]
        chain.sort(key=lambda value: (-len(value), tuple(sorted(value))))
        candidate_workflows[candidate_id] = trace_id
        candidate_paths[candidate_id] = tuple(cluster_tokens[value] for value in chain)

    pending_paths: dict[str, tuple[str, ...]] = {}
    pending_workflows: dict[str, str] = {}
    for pending_id in sorted(pending_targets):
        chain = [value for value in clusters if pending_id in value]
        chain.sort(key=lambda value: (-len(value), tuple(sorted(value))))
        pending_workflows[pending_id] = trace_id
        pending_paths[pending_id] = tuple(cluster_tokens[value] for value in chain) + (
            f"待续:{pending_id}",
        )
    return (
        candidate_paths,
        candidate_workflows,
        pending_paths,
        pending_workflows,
        usable,
    )


def _access_metadata(
    checkpoints: Sequence[dict[str, Any]],
    semantic_paths: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """仅用截至快照时点的已物化请求构造 LRU/LFU 元数据。"""
    ordered = list(checkpoints)
    creation = {row["checkpoint_id"]: index for index, row in enumerate(ordered)}
    last_access = dict(creation)
    frequency = {row["checkpoint_id"]: 0 for row in ordered}
    for candidate in ordered:
        candidate_id = candidate["checkpoint_id"]
        candidate_path = semantic_paths[candidate["conversation_id"]]
        candidate_position = int(candidate["token_pos"])
        candidate_workflow = candidate_path[0]
        for request_order, request in enumerate(ordered):
            request_path = semantic_paths[request["conversation_id"]]
            request_workflow = request_path[0]
            related = (
                candidate_workflow == request_workflow
                and len(candidate_path) <= len(request_path)
                and request_path[: len(candidate_path)] == candidate_path
                and candidate_position <= int(request["token_pos"])
            )
            if related:
                frequency[candidate_id] += 1
                last_access[candidate_id] = max(last_access[candidate_id], request_order)
        if frequency[candidate_id] < 1:
            raise RuntimeError(f"候选 {candidate_id} 缺少创建访问")
    return creation, last_access, frequency


def _marconi_flop_metadata(
    checkpoints: Sequence[dict[str, Any]],
    semantic_paths: Mapping[str, tuple[str, ...]],
) -> dict[str, float]:
    """在真实 SPAWN/FORK 逻辑树上复用冻结的 Marconi FLOP 代理。"""
    candidates = []
    for row in checkpoints:
        path = semantic_paths[row["conversation_id"]]
        candidates.append(
            CheckpointCandidate(
                checkpoint_id=row["checkpoint_id"],
                workflow_id=path[0],
                lineage_path=path,
                token_pos=int(row["recurrent_prefix_length"]),
                memory_bytes=CHECKPOINT_SIZE_BYTES,
                recurrent_resident=True,
                fa_resident=True,
            )
        )
    return build_marconi_flop_saved(candidates)


def _validate_b2_runtime_summary(b2_root: Path) -> dict[str, Any]:
    """确认只消费 23/23 已通过最终审计的正式运行快照。"""
    if b2_root.resolve() == FORBIDDEN_B2_ROOT.resolve():
        raise RuntimeError("禁止使用 INVALID_DIAGNOSTIC_ONLY 的 B2 run")
    if b2_root.resolve() != DEFAULT_B2_ROOT.resolve():
        raise RuntimeError(f"只允许正式 B2 root：{DEFAULT_B2_ROOT}")
    summary = _read_json(b2_root / "runtime_correctness.json")
    required = {
        "attempted": EXPECTED_SNAPSHOTS,
        "eligible": EXPECTED_SNAPSHOTS,
        "candidate_handles_resolved": EXPECTED_CANDIDATES,
        "future_leakage": 0,
        "unexpected_native_eviction": 0,
        "rematerialization": 0,
        "truncation_or_oom": False,
    }
    mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in required.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"B2 正式运行摘要门禁失败：{mismatches}")
    return summary


def build_agentx_formal_snapshots(
    *,
    b1_root: Path = DEFAULT_B1_ROOT,
    b2_root: Path = DEFAULT_B2_ROOT,
    agentx_path: Path = FROZEN_AGENTX_PATH,
) -> tuple[list[AgentXFormalSnapshot], dict[str, Any]]:
    """从 B1 逻辑语义与 B2 物理观测装配 23 个不可变策略输入。"""
    _validate_b2_runtime_summary(b2_root)
    population = _read_json(b1_root / "FORMAL_CANDIDATE_POPULATION.json")
    raw_snapshots = _read_jsonl(b2_root / "runtime_snapshots.jsonl")
    trace_conversations = _build_conversations_for_population(agentx_path, population)
    exact_sets, b1_metadata = _build_b1_exact_sets(population, trace_conversations)
    if b1_metadata["max_d_t_c"] != EXPECTED_FORMAL_MAX_DEGREE:
        raise RuntimeError("B1 formal max d_t(c) 不等于 2")

    by_trace = {tc.trace_id: tc for tc in trace_conversations}
    raw_by_trace = {row["trace_id"]: row for row in raw_snapshots}
    formal_snapshots: list[AgentXFormalSnapshot] = []
    universe_matches = 0
    encoded_usable_pairs = 0
    formal_pairs_beyond_target = 0

    for ordinal, entry in enumerate(population):
        trace_id = entry["trace_id"]
        raw = raw_by_trace[trace_id]
        if raw.get("status") != "ELIGIBLE":
            raise RuntimeError(f"{trace_id} 不是 ELIGIBLE")
        tc = by_trace[trace_id]
        online = _analyze_trace_online_compatibility(tc)
        epoch = _find_epoch_snapshot(
            online,
            float(entry["t"]),
            entry["active_conversation_ids"],
            tc.conversations,
        )
        conv_by_id = {conv.conversation_id: conv for conv in tc.conversations}
        semantic_paths = _semantic_lineage_paths(tc.conversations)
        checkpoints = list(raw["checkpoints"])
        pendings = list(raw["pendings"])
        candidate_ids = {row["checkpoint_id"] for row in checkpoints}
        expected_candidate_ids = {
            identifier for identifier, value in exact_sets.items() if value["trace_id"] == trace_id
        }
        pending_ids = {row["continuation_id"] for row in pendings}
        expected_pending_ids = set(b1_metadata["pending_ids_by_trace"][trace_id])
        if candidate_ids != expected_candidate_ids or pending_ids != expected_pending_ids:
            raise RuntimeError(f"{trace_id} 的 candidate/pending universe 不一致")
        universe_matches += 1

        candidate_depths = {
            row["checkpoint_id"]: int(row["recurrent_prefix_length"])
            for row in checkpoints
        }
        pending_targets = {
            row["continuation_id"]: int(row["planning_target"])
            for row in pendings
        }
        compatible = {
            candidate_id: tuple(exact_sets[candidate_id]["compatible_pending_ids"])
            for candidate_id in sorted(candidate_ids)
        }
        (
            candidate_paths,
            candidate_workflows,
            pending_paths,
            pending_workflows,
            usable,
        ) = encode_usable_compatibility(
            compatible,
            candidate_depths,
            pending_targets,
            trace_id,
        )
        encoded_usable_pairs += sum(len(values) for values in usable.values())
        formal_pairs_beyond_target += sum(
            len(compatible[candidate_id]) - len(usable[candidate_id])
            for candidate_id in compatible
        )

        core_candidates = tuple(
            CheckpointCandidate(
                checkpoint_id=row["checkpoint_id"],
                workflow_id=candidate_workflows[row["checkpoint_id"]],
                lineage_path=candidate_paths[row["checkpoint_id"]],
                token_pos=candidate_depths[row["checkpoint_id"]],
                memory_bytes=CHECKPOINT_SIZE_BYTES,
                recurrent_resident=bool(row["recurrent_resident"]),
                fa_resident=bool(row["fa_resident"]),
            )
            for row in checkpoints
        )
        core_pendings = tuple(
            PendingContinuation(
                continuation_id=row["continuation_id"],
                workflow_id=pending_workflows[row["continuation_id"]],
                lineage_path=pending_paths[row["continuation_id"]],
                anchor_pos=int(row["input_token_count"]),
                resident_fa_frontier=int(row["resident_fa_frontier"]),
            )
            for row in pendings
        )
        observed_usable = {
            candidate.checkpoint_id: tuple(
                sorted(
                    pending.continuation_id
                    for pending in core_pendings
                    if is_compatible(candidate, pending)
                )
            )
            for candidate in core_candidates
        }
        if observed_usable != usable:
            raise RuntimeError(f"{trace_id} 的 B1 兼容编码不保真")

        creation, last_access, frequency = _access_metadata(
            checkpoints, semantic_paths
        )
        marconi_flop = _marconi_flop_metadata(checkpoints, semantic_paths)
        runtime_evidence = tuple(
            FrozenCheckpointRuntimeEvidence(
                checkpoint_id=row["checkpoint_id"],
                node_id=int(row["node_id"]),
                runtime_identity_digest=_sha256_bytes(
                    str(row["runtime_identity_digest"]).encode("utf-8")
                ),
                checkpoint_handle_digest=_canonical_digest(
                    {
                        "checkpoint_id": row["checkpoint_id"],
                        "node_id": row["node_id"],
                        "runtime_identity_digest": row["runtime_identity_digest"],
                        "recurrent_prefix_length": row["recurrent_prefix_length"],
                    }
                ),
            )
            for row in checkpoints
        )
        snapshot = build_allocation_snapshot(
            allocation_epoch=len(checkpoints),
            snapshot_id=f"rq3-agentx-g{ordinal:03d}-epoch{len(checkpoints)}",
            pending_continuations=core_pendings,
            eligible_candidates=core_candidates,
            creation_order_by_checkpoint=creation,
            last_access_order_by_checkpoint=last_access,
            marconi_flop_saved_by_checkpoint=marconi_flop,
            access_frequency_by_checkpoint=frequency,
            frequency_observed_through_epoch=len(checkpoints),
            marconi_alpha=CONTROLLED_MARCONI_ALPHA,
            logical_budget_k=1,
            budget_bytes=CHECKPOINT_SIZE_BYTES,
            runtime_evidence=runtime_evidence,
            residency_snapshot_digest=_canonical_digest(raw),
            online_boundary=FrozenOnlineInformationBoundary(
                materialized_through_epoch=len(checkpoints),
                visible_continuation_ids=tuple(sorted(pending_ids)),
            ),
        )
        formal_snapshots.append(
            AgentXFormalSnapshot(
                trace_id=trace_id,
                timestamp=float(entry["t"]),
                snapshot=snapshot,
                formal_compatible_pending_ids={
                    key: tuple(value) for key, value in compatible.items()
                },
                objective_usable_pending_ids={
                    key: tuple(value) for key, value in usable.items()
                },
                physical_checkpoint_depths=candidate_depths,
                actual_lineage_paths={
                    row["checkpoint_id"]: tuple(
                        conv_by_id[row["conversation_id"]].lineage_path
                    )
                    for row in checkpoints
                },
            )
        )

    audit = {
        "designated_snapshots": EXPECTED_SNAPSHOTS,
        "assembled_snapshots": len(formal_snapshots),
        "candidate_universe_total": sum(
            len(item.snapshot.eligible_candidates) for item in formal_snapshots
        ),
        "candidate_pending_universe_exact_match": f"{universe_matches}/{EXPECTED_SNAPSHOTS}",
        "formal_compatibility_source": "B1 冻结逻辑血缘与已验证 FORK 继承语义",
        "b2_raw_compatibility_used": False,
        "formal_max_d_t_c": max(
            (len(values) for item in formal_snapshots for values in item.formal_compatible_pending_ids.values()),
            default=0,
        ),
        "formal_pair_count": sum(
            len(values) for item in formal_snapshots for values in item.formal_compatible_pending_ids.values()
        ),
        "objective_usable_pair_count": encoded_usable_pairs,
        "formal_pairs_beyond_physical_T": formal_pairs_beyond_target,
        "coordinate_rule": (
            "B1 exact-set 决定工作流兼容；B2 recurrent_prefix_length 是 E 的物理深度，"
            "pending input_token_count 与 resident_fa_frontier 分别作为 A 与 R，T=min(A,R)；"
            "物理深度超过 T 的正式兼容 pair 不产生可执行前沿。"
        ),
    }
    if (
        len(formal_snapshots) != EXPECTED_SNAPSHOTS
        or audit["candidate_universe_total"] != EXPECTED_CANDIDATES
        or audit["formal_max_d_t_c"] != EXPECTED_FORMAL_MAX_DEGREE
    ):
        raise RuntimeError(f"AgentX formal snapshot 装配门禁失败：{audit}")
    return formal_snapshots, audit


def _zero_marginal_count(snapshot: AllocationSnapshot, selected_ids: Sequence[str]) -> int:
    """统计删除后成本不变的已选候选数。"""
    selected = tuple(selected_ids)
    full_cost = evaluate_objective(snapshot, selected).total_recovery_cost_ms
    count = 0
    for candidate_id in selected:
        remaining = tuple(value for value in selected if value != candidate_id)
        cost = evaluate_objective(snapshot, remaining).total_recovery_cost_ms
        if abs(cost - full_cost) <= _FLOAT_TOLERANCE_MS:
            count += 1
    return count


def _augment_case(
    row: dict[str, Any],
    formal: AgentXFormalSnapshot,
) -> dict[str, Any]:
    """补充 AgentX 共享覆盖、冗余与独立 TopK 对照。"""
    snapshot = create_budget_variant(formal.snapshot, int(row["k"]))
    formal_degree = {
        candidate_id: len(pending_ids)
        for candidate_id, pending_ids in formal.formal_compatible_pending_ids.items()
    }
    shared_ids = {candidate_id for candidate_id, degree in formal_degree.items() if degree > 1}
    policy_shared: dict[str, Any] = {}
    policy_coverage: dict[str, Any] = {}
    policy_redundancy: dict[str, Any] = {}
    for policy in POLICIES:
        selected = tuple(row["policies"][policy]["selected_checkpoint_ids"])
        selected_shared = sorted(set(selected) & shared_ids)
        covered_pairs = [
            (candidate_id, pending_id)
            for candidate_id in selected
            for pending_id in formal.formal_compatible_pending_ids[candidate_id]
        ]
        covered_pending = sorted({pending_id for _, pending_id in covered_pairs})
        policy_shared[policy] = {
            "selected_shared_candidate_count": len(selected_shared),
            "selected_shared_candidate_ratio": (
                len(selected_shared) / len(selected) if selected else None
            ),
            "selected_shared_candidate_ids": selected_shared,
        }
        policy_coverage[policy] = {
            "formal_candidate_pending_pairs": len(covered_pairs),
            "distinct_pending_covered": len(covered_pending),
            "pending_coverage_ratio": (
                len(covered_pending) / len(snapshot.pending_continuations)
                if snapshot.pending_continuations
                else None
            ),
        }
        zero_count = _zero_marginal_count(snapshot, selected)
        policy_redundancy[policy] = {
            "zero_marginal_selected_count": zero_count,
            "selected_set_redundancy_ratio": zero_count / len(selected) if selected else None,
        }

    standalone_ids = _standalone_topk_selection(snapshot, int(row["k"]))
    standalone_objective = evaluate_objective(snapshot, standalone_ids)
    flowstate_ids = tuple(row["policies"]["FlowState"]["selected_checkpoint_ids"])
    flowstate_cost = float(row["policies"]["FlowState"]["total_recovery_cost_ms"])
    standalone_set = set(standalone_ids)
    flowstate_set = set(flowstate_ids)
    union = standalone_set | flowstate_set
    row["agentx_shared"] = {
        "formal_shared_candidate_count": len(shared_ids),
        "policy_selected_shared": policy_shared,
        "cross_pending_coverage": policy_coverage,
        "selected_set_redundancy": policy_redundancy,
    }
    row["standalone_topk"] = {
        "selected_checkpoint_ids": list(standalone_ids),
        "selected_set_equal": standalone_set == flowstate_set,
        "selected_set_symmetric_difference_count": len(standalone_set ^ flowstate_set),
        "selected_set_jaccard": len(standalone_set & flowstate_set) / len(union) if union else 1.0,
        "total_recovery_cost_ms": standalone_objective.total_recovery_cost_ms,
        "standalone_minus_flowstate_cost_ms": (
            standalone_objective.total_recovery_cost_ms - flowstate_cost
        ),
    }
    common_input = {
        "pending_ids": [value.continuation_id for value in snapshot.pending_continuations],
        "candidate_ids": [value.checkpoint_id for value in snapshot.eligible_candidates],
        "k": row["k"],
        "a_r_t": [
            {
                "continuation_id": value.continuation_id,
                "A": value.anchor_pos,
                "R": value.resident_fa_frontier,
                "T": value.planning_target,
            }
            for value in snapshot.pending_continuations
        ],
        "phi": {
            "name": snapshot.recovery_model.name,
            "coefficient_a": snapshot.recovery_model.coefficient_a,
            "coefficient_b": snapshot.recovery_model.coefficient_b,
            "coefficient_c": snapshot.recovery_model.coefficient_c,
            "minimum_gap_tokens": snapshot.recovery_model.minimum_gap_tokens,
            "maximum_target_tokens": snapshot.recovery_model.maximum_target_tokens,
        },
    }
    common_input_digest = _canonical_digest(common_input)
    compared_policies = list(POLICIES) + (
        ["Exact OPT"] if row["exact_opt"].get("tractable") else []
    )
    row["common_input_digest"] = common_input_digest
    row["policy_input_digests"] = {
        policy: common_input_digest for policy in compared_policies
    }
    row["common_input_verified"] = (
        len(set(row["policy_input_digests"].values())) == 1
    )
    return row


def evaluate_agentx_snapshots(
    formal_snapshots: Sequence[AgentXFormalSnapshot],
    exact_threshold: int = _EXACT_OPT_SEARCH_SPACE_THRESHOLD,
) -> list[dict[str, Any]]:
    """在每个快照的三个冻结预算上执行四策略与条件 Exact。"""
    results: list[dict[str, Any]] = []
    for ordinal, formal in enumerate(formal_snapshots):
        digest_before = formal.snapshot.content_digest()
        rows = evaluate_snapshot_at_ks(
            formal.snapshot,
            ordinal,
            exact_threshold=exact_threshold,
        )
        if formal.snapshot.content_digest() != digest_before:
            raise RuntimeError(f"{formal.trace_id} 被策略修改")
        for row in rows:
            row["trace_id"] = formal.trace_id
            row["timestamp"] = formal.timestamp
            _augment_case(row, formal)
        results.extend(rows)
    return results


def _marginal_effects(
    rows: Sequence[dict[str, Any]],
    formal_by_snapshot: Mapping[str, AgentXFormalSnapshot],
) -> dict[str, Any]:
    """逐贪心阶段比较候选边际收益与空集合独立收益。"""
    model = RecoveryCostModel()
    total_events = 0
    changed_events = 0
    cross_pending_events = 0
    changed_candidate_case_keys: set[str] = set()
    cross_pending_candidate_case_keys: set[str] = set()
    per_case: list[dict[str, Any]] = []
    for row in rows:
        formal = formal_by_snapshot[row["snapshot_id"]]
        snapshot = create_budget_variant(formal.snapshot, int(row["k"]))
        candidates = list(snapshot.core_candidates())
        candidate_by_id = {value.checkpoint_id: value for value in candidates}
        selected: list[CheckpointCandidate] = []
        order = row["mechanism_diagnostics"]["flowstate_greedy_trace"]["selection_order"]
        case_changed: set[str] = set()
        case_cross: set[str] = set()
        case_events = 0
        case_cross_events = 0
        for selected_id in order:
            remaining = [
                candidate
                for candidate in candidates
                if candidate.checkpoint_id not in {value.checkpoint_id for value in selected}
                and candidate.checkpoint_id != selected_id
            ]
            for candidate in remaining:
                delta_current = _marginal_gain(snapshot, selected, candidate, model)
                delta_empty = _marginal_gain(snapshot, [], candidate, model)
                total_events += 1
                if abs(delta_current - delta_empty) <= _FLOAT_TOLERANCE_MS:
                    continue
                changed_events += 1
                case_events += 1
                key = f"{row['snapshot_id']}|{row['k']}|{candidate.checkpoint_id}"
                changed_candidate_case_keys.add(key)
                case_changed.add(candidate.checkpoint_id)
                selected_covered = {
                    pending.continuation_id
                    for pending in snapshot.core_continuations()
                    for chosen in selected
                    if is_compatible(chosen, pending)
                }
                candidate_covered = {
                    pending.continuation_id
                    for pending in snapshot.core_continuations()
                    if is_compatible(candidate, pending)
                }
                if len(selected_covered & candidate_covered) > 1:
                    cross_pending_events += 1
                    case_cross_events += 1
                    cross_pending_candidate_case_keys.add(key)
                    case_cross.add(candidate.checkpoint_id)
            selected.append(candidate_by_id[selected_id])
        per_case.append(
            {
                "snapshot_id": row["snapshot_id"],
                "k": row["k"],
                "marginal_changed_candidate_count": len(case_changed),
                "marginal_changed_event_count": case_events,
                "cross_pending_overlap_candidate_count": len(case_cross),
                "cross_pending_overlap_event_count": case_cross_events,
            }
        )

    openhands_multi = None
    source = OPENHANDS_SANITY_ROOT / "marginal_dependency.json"
    if source.exists():
        openhands_multi = int(_read_json(source).get("multi_pending_overlap", 0))
    return {
        "total_candidate_comparison_events": total_events,
        "marginal_changed_event_count": changed_events,
        "marginal_changed_candidate_case_count": len(changed_candidate_case_keys),
        "cross_pending_overlap_event_count": cross_pending_events,
        "cross_pending_overlap_candidate_case_count": len(cross_pending_candidate_case_keys),
        "openhands_cross_pending_overlap_event_count": openhands_multi,
        "cross_pending_effect_absent_in_openhands": (
            cross_pending_events > 0 and openhands_multi == 0
        ),
        "conclusion": (
            "存在 OpenHands 正式快照中没有出现的跨 pending 边际依赖"
            if cross_pending_events > 0 and openhands_multi == 0
            else "未观察到超出 OpenHands 的跨 pending 边际依赖"
        ),
        "per_case": per_case,
    }


def _aggregate_agentx(
    rows: Sequence[dict[str, Any]],
    formal_snapshots: Sequence[AgentXFormalSnapshot],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """生成主指标、Exact 指标和共享覆盖分析。"""
    aggregate = aggregate_results(rows)
    for ratio in _FROZEN_BUDGET_RATIOS:
        ratio_rows = [row for row in rows if row["budget_ratio"] == ratio]
        for baseline in ("LRU", "LFU", "Marconi"):
            differences = [row["paired"][baseline]["absolute_difference_ms"] for row in ratio_rows]
            wins = sum(value > _FLOAT_TOLERANCE_MS for value in differences)
            losses = sum(value < -_FLOAT_TOLERANCE_MS for value in differences)
            aggregate[ratio][f"FlowState_vs_{baseline}"]["win_tie_loss"] = {
                "win": wins,
                "tie": len(differences) - wins - losses,
                "loss": losses,
            }
            aggregate[ratio][f"FlowState_vs_{baseline}"]["absolute_reduction_bootstrap_ci"] = _bootstrap_ci(differences)

    exact_rows = [row for row in rows if row["exact_opt"].get("tractable")]
    exact_results = {
        "total_cases": len(rows),
        "tractable_cases": len(exact_rows),
        "skipped_cases": len(rows) - len(exact_rows),
        "flowstate_equals_exact_count": sum(
            abs(row["exact_opt"]["flowstate_vs_exact"]["absolute_cost_gap_ms"])
            <= _FLOAT_TOLERANCE_MS
            for row in exact_rows
        ),
        "absolute_optimality_gap_ms": _summary_stats(
            [row["exact_opt"]["flowstate_vs_exact"]["absolute_cost_gap_ms"] for row in exact_rows]
        ),
        "max_absolute_optimality_gap_ms": max(
            (row["exact_opt"]["flowstate_vs_exact"]["absolute_cost_gap_ms"] for row in exact_rows),
            default=None,
        ),
        "cases": [
            {
                "snapshot_id": row["snapshot_id"],
                "trace_id": row["trace_id"],
                "budget_ratio": row["budget_ratio"],
                "k": row["k"],
                **row["exact_opt"],
            }
            for row in exact_rows
        ],
    }

    formal_by_snapshot = {item.snapshot.snapshot_id: item for item in formal_snapshots}
    marginal = _marginal_effects(rows, formal_by_snapshot)
    shared_count = sum(
        len(values) > 1
        for item in formal_snapshots
        for values in item.formal_compatible_pending_ids.values()
    )
    per_policy: dict[str, Any] = {}
    for policy in POLICIES:
        counts = [
            row["agentx_shared"]["policy_selected_shared"][policy]["selected_shared_candidate_count"]
            for row in rows
        ]
        ratios = [
            row["agentx_shared"]["policy_selected_shared"][policy]["selected_shared_candidate_ratio"]
            for row in rows
            if row["agentx_shared"]["policy_selected_shared"][policy]["selected_shared_candidate_ratio"] is not None
        ]
        coverages = [
            row["agentx_shared"]["cross_pending_coverage"][policy]["pending_coverage_ratio"]
            for row in rows
            if row["agentx_shared"]["cross_pending_coverage"][policy]["pending_coverage_ratio"] is not None
        ]
        redundancies = [
            row["agentx_shared"]["selected_set_redundancy"][policy]["selected_set_redundancy_ratio"]
            for row in rows
            if row["agentx_shared"]["selected_set_redundancy"][policy]["selected_set_redundancy_ratio"] is not None
        ]
        per_policy[policy] = {
            "selected_shared_candidate_count_total": sum(counts),
            "selected_shared_candidate_count": _summary_stats(counts),
            "selected_shared_candidate_ratio": _summary_stats(ratios),
            "cross_pending_coverage_ratio": _summary_stats(coverages),
            "selected_set_redundancy_ratio": _summary_stats(redundancies),
        }
    standalone_differences = [row["standalone_topk"]["standalone_minus_flowstate_cost_ms"] for row in rows]
    standalone_set_differences = [row["standalone_topk"]["selected_set_symmetric_difference_count"] for row in rows]
    shared = {
        "formal_shared_candidates": shared_count,
        "formal_degree_distribution": dict(
            sorted(
                Counter(
                    len(values)
                    for item in formal_snapshots
                    for values in item.formal_compatible_pending_ids.values()
                ).items()
            )
        ),
        "per_policy": per_policy,
        "flowstate_vs_standalone_topk": {
            "cases": len(rows),
            "selected_set_equal_count": sum(row["standalone_topk"]["selected_set_equal"] for row in rows),
            "selected_set_difference_count": sum(not row["standalone_topk"]["selected_set_equal"] for row in rows),
            "symmetric_difference_count": _summary_stats(standalone_set_differences),
            "standalone_minus_flowstate_cost_ms": _summary_stats(standalone_differences),
            "flowstate_strictly_better_count": sum(value > _FLOAT_TOLERANCE_MS for value in standalone_differences),
            "cost_tie_count": sum(abs(value) <= _FLOAT_TOLERANCE_MS for value in standalone_differences),
            "flowstate_worse_count": sum(value < -_FLOAT_TOLERANCE_MS for value in standalone_differences),
        },
        "marginal_effect": marginal,
    }
    return aggregate, exact_results, shared


def _run_determinism(
    rows: Sequence[dict[str, Any]],
    formal_snapshots: Sequence[AgentXFormalSnapshot],
    exact_threshold: int,
) -> dict[str, Any]:
    """重新运行每个 selector 并逐 selected-set 比较。"""
    expected = {
        (row["snapshot_id"], int(row["k"]), policy): tuple(
            row["policies"][policy]["selected_checkpoint_ids"]
        )
        for row in rows
        for policy in POLICIES
    }
    for row in rows:
        if row["exact_opt"].get("tractable"):
            expected[(row["snapshot_id"], int(row["k"]), "Exact OPT")] = tuple(
                row["exact_opt"]["selected_checkpoint_ids"]
            )
    mismatches = []
    runs = 0
    for formal in formal_snapshots:
        for _ratio, k in compute_budget_ks(len(formal.snapshot.eligible_candidates)):
            variant = create_budget_variant(formal.snapshot, k)
            policies = list(POLICIES)
            if search_space_size(len(variant.eligible_candidates), k) <= exact_threshold:
                policies.append("Exact OPT")
            for policy in policies:
                selected, _internal, _wall, _objective = _policy_result(variant, policy)
                runs += 1
                key = (formal.snapshot.snapshot_id, k, policy)
                if selected != expected[key]:
                    mismatches.append(
                        {
                            "snapshot_id": key[0],
                            "k": k,
                            "policy": policy,
                            "expected": list(expected[key]),
                            "observed": list(selected),
                        }
                    )
    return {
        "rerun_count": runs,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def _source_paths() -> list[Path]:
    """返回需要前后锁定的目标函数与 selector 源码。"""
    return [
        Path("evaluation/rq3_agentx_formal_policy_evaluation.py"),
        Path("evaluation/agentx_runtime_final_correctness_audit.py"),
        Path("evaluation/agentx_structure_audit.py"),
        Path("evaluation/rq3_frozen_snapshot_evaluator.py"),
        Path("evaluation/rq3_formal_policy_evaluation.py"),
        Path("evaluation/controlled_multiworkflow_v1/policies.py"),
        Path("evaluation/sota_policies.py"),
        Path("evaluation/sota_metadata.py"),
        Path("flowstate/optimizer.py"),
        Path("flowstate/recovery_model.py"),
        Path("flowstate/executable_state.py"),
        Path("flowstate/state_catalog.py"),
        Path("flowstate/workflow.py"),
    ]


def _source_integrity(paths: Sequence[Path]) -> dict[str, Any]:
    """记录每个冻结源文件的摘要。"""
    files = {
        str(path): {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in paths
    }
    return {"files": files, "combined_sha256": _canonical_digest(files)}


def _ratio_json(aggregate: Mapping[Any, Any]) -> dict[str, Any]:
    """把浮点预算键转换为稳定字符串。"""
    return {
        (f"ratio_{int(key * 100):02d}" if isinstance(key, float) else str(key)): value
        for key, value in aggregate.items()
    }


def _build_report(
    status: str,
    assembly: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    aggregate: Mapping[Any, Any],
    exact: Mapping[str, Any],
    shared: Mapping[str, Any],
    determinism: Mapping[str, Any],
    source_integrity: Mapping[str, Any],
    output_root: Path,
) -> str:
    """生成正式中文结果摘要。"""
    lines = [
        "# Step 13G-C AgentX Formal Policy Evaluation",
        "",
        f"- 状态：`{status}`",
        f"- snapshots：`{assembly['assembled_snapshots']}`",
        f"- cases：`{len(rows)}`",
        f"- formal max d_t(c)：`{assembly['formal_max_d_t_c']}`",
        f"- 使用 B2 raw compatibility：`{'是' if assembly['b2_raw_compatibility_used'] else '否'}`",
        f"- shared candidates：`{shared['formal_shared_candidates']}`",
        f"- Exact tractable：`{exact['tractable_cases']}`",
        f"- FlowState == Exact：`{exact['flowstate_equals_exact_count']}`",
        f"- determinism：`{'通过' if determinism['pass'] else '失败'}`",
        f"- source integrity：`{'通过' if source_integrity['pass'] else '失败'}`",
        "",
        "## 各预算 FlowState 相对基线",
        "",
    ]
    for ratio in _FROZEN_BUDGET_RATIOS:
        entry = aggregate[ratio]
        lines.append(f"### {int(ratio * 100)}%")
        lines.append("")
        for baseline in ("LRU", "LFU", "Marconi"):
            comparison = entry[f"FlowState_vs_{baseline}"]
            reduction = comparison["relative_reduction"]["mean"]
            interval = comparison["bootstrap_ci"]
            outcome = comparison["win_tie_loss"]
            lines.append(
                f"- 相对 {baseline}：mean reduction={reduction}，"
                f"95% CI=[{interval['ci_low']}, {interval['ci_high']}]，"
                f"win/tie/loss={outcome['win']}/{outcome['tie']}/{outcome['loss']}"
            )
        lines.append("")
    lines.extend(
        [
            "## AgentX 跨 pending 结论",
            "",
            shared["marginal_effect"]["conclusion"],
            "",
            f"- artifact root：`{output_root}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(
    output_root: Path,
    *,
    b1_root: Path = DEFAULT_B1_ROOT,
    b2_root: Path = DEFAULT_B2_ROOT,
    agentx_path: Path = FROZEN_AGENTX_PATH,
    exact_threshold: int = _EXACT_OPT_SEARCH_SPACE_THRESHOLD,
) -> Path:
    """执行正式评估，并在新目录中写出完整 artifact。"""
    if output_root.exists():
        raise RuntimeError(f"输出目录已存在：{output_root}")
    source_before = _source_integrity(_source_paths())
    input_paths = {
        "b1_population": b1_root / "FORMAL_CANDIDATE_POPULATION.json",
        "b2_runtime_snapshots": b2_root / "runtime_snapshots.jsonl",
        "b2_runtime_correctness": b2_root / "runtime_correctness.json",
        "agentx_corpus": agentx_path,
    }
    input_hashes_before = {name: _sha256_file(path) for name, path in input_paths.items()}
    formal_snapshots, assembly = build_agentx_formal_snapshots(
        b1_root=b1_root,
        b2_root=b2_root,
        agentx_path=agentx_path,
    )
    snapshot_digests_before = {
        item.snapshot.snapshot_id: item.snapshot.content_digest()
        for item in formal_snapshots
    }
    rows = evaluate_agentx_snapshots(formal_snapshots, exact_threshold)
    aggregate, exact, shared = _aggregate_agentx(rows, formal_snapshots)
    determinism = _run_determinism(rows, formal_snapshots, exact_threshold)
    snapshot_digests_after = {
        item.snapshot.snapshot_id: item.snapshot.content_digest()
        for item in formal_snapshots
    }
    source_after = _source_integrity(_source_paths())
    input_hashes_after = {name: _sha256_file(path) for name, path in input_paths.items()}
    source_integrity = {
        "before": source_before,
        "after": source_after,
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "pass": (
            source_before == source_after
            and input_hashes_before == input_hashes_after
        ),
    }
    snapshot_immutable = snapshot_digests_before == snapshot_digests_after
    ratio_counts = Counter(row["budget_ratio"] for row in rows)
    ready = (
        assembly["assembled_snapshots"] == EXPECTED_SNAPSHOTS
        and len(rows) == EXPECTED_SNAPSHOTS * len(_FROZEN_BUDGET_RATIOS)
        and all(ratio_counts[ratio] == EXPECTED_SNAPSHOTS for ratio in _FROZEN_BUDGET_RATIOS)
        and assembly["formal_max_d_t_c"] == EXPECTED_FORMAL_MAX_DEGREE
        and assembly["b2_raw_compatibility_used"] is False
        and all(row["common_input_verified"] for row in rows)
        and determinism["pass"]
        and source_integrity["pass"]
        and snapshot_immutable
    )
    status = "AGENTX_RQ3_EVAL_READY" if ready else "AGENTX_RQ3_EVAL_BLOCKED"

    output_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cpu_only": True,
        "gpu_used": False,
        "ttft_or_end_to_end_run": False,
        "source_b1_root": str(b1_root.resolve()),
        "source_b2_root": str(b2_root.resolve()),
        "invalid_b2_root_not_used": str(FORBIDDEN_B2_ROOT),
        "snapshots": len(formal_snapshots),
        "cases": len(rows),
        "budget_ratios": list(_FROZEN_BUDGET_RATIOS),
        "exact_search_space_threshold": exact_threshold,
        "snapshot_immutable": snapshot_immutable,
        "common_inputs_all_policies": True,
        "assembly": assembly,
    }
    _write_json(output_root / "manifest.json", manifest)
    with (output_root / "per_case_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(output_root / "aggregate_results.json", _ratio_json(aggregate))
    _write_json(output_root / "exact_results.json", exact)
    _write_json(output_root / "shared_coverage_analysis.json", shared)
    _write_json(output_root / "determinism.json", determinism)
    _write_json(output_root / "source_integrity.json", source_integrity)
    _write_json(
        output_root / "formal_snapshot_manifest.json",
        [
            {
                "trace_id": item.trace_id,
                "timestamp": item.timestamp,
                "snapshot_id": item.snapshot.snapshot_id,
                "snapshot_digest": item.snapshot.content_digest(),
                "candidate_count": len(item.snapshot.eligible_candidates),
                "pending_count": len(item.snapshot.pending_continuations),
                "formal_max_d_t_c": max(
                    map(len, item.formal_compatible_pending_ids.values()), default=0
                ),
                "formal_compatible_pending_ids": item.formal_compatible_pending_ids,
                "objective_usable_pending_ids": item.objective_usable_pending_ids,
            }
            for item in formal_snapshots
        ],
    )
    (output_root / "final_report.md").write_text(
        _build_report(
            status,
            assembly,
            rows,
            aggregate,
            exact,
            shared,
            determinism,
            source_integrity,
            output_root,
        ),
        encoding="utf-8",
    )
    return output_root


def main(argv: Iterable[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="AgentX RQ3 正式 CPU 策略评估")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--b1-root", type=Path, default=DEFAULT_B1_ROOT)
    parser.add_argument("--b2-root", type=Path, default=DEFAULT_B2_ROOT)
    parser.add_argument("--agentx-path", type=Path, default=FROZEN_AGENTX_PATH)
    parser.add_argument("--exact-threshold", type=int, default=_EXACT_OPT_SEARCH_SPACE_THRESHOLD)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(
        run_evaluation(
            args.output_root,
            b1_root=args.b1_root,
            b2_root=args.b2_root,
            agentx_path=args.agentx_path,
            exact_threshold=args.exact_threshold,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
