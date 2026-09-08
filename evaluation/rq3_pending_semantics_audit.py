"""Step 13G-B2 pending semantics CPU audit。

验证 B1 / B1.1 / B2 对冻结的 23 个 formal snapshot 解析出的 pending set 完全一致，
并明确 active_target == 0 的正式语义。

约束：
- CPU only，不调用 GPU/SGLang。
- 直接消费 B1 冻结 population 与 B0.1 online epoch reconstruction。
- 不修改 formal 23 population 的选择。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.agentx_qwen_replay_preflight import (
    QWEN_MODEL_PATH,
    _build_conversations_for_population,
    _build_replay_snapshot,
    _load_formal_population,
    _load_tokenizer,
    _request_index_for_target,
)
from evaluation.agentx_runtime_snapshot_collection import _build_replay_plan

FROZEN_B1_CENSUS_PATH = Path(
    "/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614"
    "/FORMAL_CANDIDATE_POPULATION.json"
)
FROZEN_AGENTX_PATH = Path(
    "/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl"
)


@dataclass
class _PendingSemanticsAuditResult:
    """单个 snapshot 的 pending 语义审计结果。"""

    trace_id: str
    t: float
    b1_pending_count: int
    b1_pending_ids: set[str]
    b11_pending_ids: set[str]
    b2_pending_ids: set[str]
    active_targets: dict[int, int]
    target_zero_conv_ids: list[str]
    mismatch_detail: str = ""


def _pending_id(conversation_id: str, request_index: int) -> str:
    """构造与 B2 replay plan 一致的 pending ID。"""
    return f"{conversation_id}::pending{request_index:04d}"


def _old_request_index_for_target(conv: Any, target: int) -> int | None:
    """修复前语义：target == 0 返回 None，导致该 conversation 被排除在 pending set 外。"""
    positions = conv.request_hash_token_positions
    if not positions:
        return None
    if target <= 0:
        return None
    cum = list(__import__("itertools").accumulate(positions))
    idx = __import__("bisect").bisect_right(cum, target) - 1
    if idx >= 0 and cum[idx] == target:
        return idx
    return max(idx, 0)


def _old_b1_pending_ids(
    active_targets: dict[int, int], convs: list[Any]
) -> tuple[set[str], list[str]]:
    """用修复前语义推导的 B1 pending ID 集合（用于统计修复前 mismatch）。"""
    pending_ids: set[str] = set()
    zero_ids: list[str] = []
    for conv_idx, target in active_targets.items():
        conv = convs[conv_idx]
        req_idx = _old_request_index_for_target(conv, target)
        if req_idx is None:
            continue
        if target == 0:
            zero_ids.append(conv.conversation_id)
        pending_ids.add(_pending_id(conv.conversation_id, req_idx))
    return pending_ids, zero_ids


def _b1_pending_ids(
    active_targets: dict[int, int], convs: list[Any]
) -> tuple[set[str], list[str]]:
    """由 B0.1 active_targets 推导 B1 的 pending ID 集合，并返回 target==0 的 conversation ID 列表。"""
    pending_ids: set[str] = set()
    zero_ids: list[str] = []
    for conv_idx, target in active_targets.items():
        conv = convs[conv_idx]
        req_idx = _request_index_for_target(conv, target)
        if req_idx is None:
            continue
        if target == 0:
            zero_ids.append(conv.conversation_id)
        pending_ids.add(_pending_id(conv.conversation_id, req_idx))
    return pending_ids, zero_ids


def _b11_pending_ids(snapshot: Any) -> set[str]:
    """由 B1.1 replay snapshot 推导 pending ID 集合。"""
    return {
        _pending_id(req.conversation_id, req.request_index)
        for req in snapshot.requests
    }


def _b2_pending_ids(plan: Any) -> set[str]:
    """由 B2 replay plan 推导 pending ID 集合。"""
    return {_pending_id(p.conversation_id, p.request_index) for p in plan.pendings}


def run_audit() -> dict[str, Any]:
    """执行 23 snapshot pending 语义审计。"""
    tokenizer = _load_tokenizer(QWEN_MODEL_PATH)
    population = _load_formal_population(FROZEN_B1_CENSUS_PATH)
    trace_conversations = _build_conversations_for_population(
        FROZEN_AGENTX_PATH, population
    )
    conv_by_trace = {tc.trace_id: tc for tc in trace_conversations}

    results: list[_PendingSemanticsAuditResult] = []
    snapshots_with_target_zero = 0
    total_target_zero_conversations = 0
    b1_vs_b11_mismatches = 0
    b1_vs_b2_mismatches = 0

    for entry in population:
        trace_id = entry["trace_id"]
        tc = conv_by_trace[trace_id]

        # B1：由 online epoch reconstruction 的 active_targets 推导。
        online = _build_replay_snapshot(tc, entry, tokenizer)
        active_targets = dict(online.active_targets)
        b1_ids, zero_ids = _b1_pending_ids(active_targets, tc.conversations)

        # 修复前语义：B1.1 与 B2 均调用同一 _request_index_for_target，因此旧 pending set
        # 等价于用 _old_request_index_for_target 重新推导的 B1 pending set。
        old_b1_ids, _ = _old_b1_pending_ids(active_targets, tc.conversations)

        # B1.1：replay preflight 合成的 pending requests。
        b11_ids = _b11_pending_ids(online)

        # B2：runtime collection 的 replay plan。
        plan = _build_replay_plan(tc, entry, tokenizer)
        b2_ids = _b2_pending_ids(plan)

        if zero_ids:
            snapshots_with_target_zero += 1
            total_target_zero_conversations += len(zero_ids)

        mismatch_parts: list[str] = []
        # 修复前，B1.1 与 B2 的 pending set 均等于 old_b1_ids。
        if b1_ids != old_b1_ids:
            b1_vs_b11_mismatches += 1
            b1_vs_b2_mismatches += 1
            mismatch_parts.append(
                f"B1 vs old B1.1/B2: only_in_b1={sorted(b1_ids - old_b1_ids)} "
                f"only_in_old={sorted(old_b1_ids - b1_ids)}"
            )
        # 修复后三者应完全一致。
        if b1_ids != b11_ids:
            mismatch_parts.append(
                f"B1 vs B1.1: only_in_b1={sorted(b1_ids - b11_ids)} "
                f"only_in_b11={sorted(b11_ids - b1_ids)}"
            )
        if b1_ids != b2_ids:
            mismatch_parts.append(
                f"B1 vs B2: only_in_b1={sorted(b1_ids - b2_ids)} "
                f"only_in_b2={sorted(b2_ids - b1_ids)}"
            )

        results.append(
            _PendingSemanticsAuditResult(
                trace_id=trace_id,
                t=float(entry["t"]),
                b1_pending_count=len(b1_ids),
                b1_pending_ids=b1_ids,
                b11_pending_ids=b11_ids,
                b2_pending_ids=b2_ids,
                active_targets=active_targets,
                target_zero_conv_ids=zero_ids,
                mismatch_detail="; ".join(mismatch_parts),
            )
        )

    consistent_count = sum(
        1
        for r in results
        if r.b1_pending_ids == r.b11_pending_ids == r.b2_pending_ids
    )

    return {
        "target_zero_formal_semantics": (
            "active_target == 0 表示 conversation 已启动但尚无 completed request，"
            "当前 pending continuation 为 request_index == 0。"
        ),
        "snapshots_containing_target_zero": snapshots_with_target_zero,
        "total_target_zero_conversations": total_target_zero_conversations,
        "b1_vs_b11_mismatches_before_fix": b1_vs_b11_mismatches,
        "b1_vs_b2_mismatches_before_fix": b1_vs_b2_mismatches,
        "after_repair_b1_b11_b2_consistent": f"{consistent_count}/{len(results)}",
        "affected_b11_snapshots": sorted(
            {r.trace_id for r in results if r.target_zero_conv_ids}
        ),
        "per_snapshot": [
            {
                "trace_id": r.trace_id,
                "t": r.t,
                "b1_pending_count": r.b1_pending_count,
                "target_zero_conv_ids": r.target_zero_conv_ids,
                "b1_pending_ids": sorted(r.b1_pending_ids),
                "b11_pending_ids": sorted(r.b11_pending_ids),
                "b2_pending_ids": sorted(r.b2_pending_ids),
                "consistent": (
                    r.b1_pending_ids == r.b11_pending_ids == r.b2_pending_ids
                ),
                "mismatch_detail": r.mismatch_detail,
            }
            for r in results
        ],
    }


def main() -> None:
    report = run_audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
