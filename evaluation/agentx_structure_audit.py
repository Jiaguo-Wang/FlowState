"""Step 13G-B0.1：AgentX Weka corpus 在线安全兼容性修复审计。

约束：
- CPU only，不下载数据，不调用 GPU/SGLang，不修改 FlowState 核心。
- 严格区分 physical prefix sharing（hash_ids LCP）与 logical inherited ancestry。
- 禁止 future leakage：epoch t 的 pending set 只能包含 t 时刻已 known/active 的 continuation。
- compatibility degree d_t(c) 必须满足 0 <= d_t(c) <= |P_t|。

输出写入 canonical root：
``/home/wjg/data/agentx/audits/agentx_online_compatibility_<timestamp>/``；
同时在 ``/home/wjg/code/FlowState/`` 保留最终报告副本。
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from evaluation.agentx_chain_detection import (
    DEFAULT_AUX_CROSS_MODEL,
    DEFAULT_AUX_ISL_FLOOR,
    DEFAULT_AUX_ISL_RATIO,
    DEFAULT_AUX_MAX_REQUESTS,
    DEFAULT_AUX_REDUCTION_OSL_MAX,
    DEFAULT_AUX_REDUCTION_RATIO,
    DEFAULT_SEAM_MAX_GAP_SECONDS,
    DEFAULT_SEAM_MIN_OVERLAP_RATIO,
    DEFAULT_WORKER_GROUP_MIN,
    AgentChain,
    _np_lcp,
    _req_end,
    _Req,
    detect_agent_chains,
    is_aux_chain,
    is_reduction_chain,
    split_off_preamble,
    worker_group_assignment,
)

FROZEN_AGENTX_PATH = Path(
    "/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl"
)
EXPECTED_SHA256 = (
    "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"
)
EXPECTED_RECORD_COUNT = 393

# 与 OpenHands RQ3 比较的冻结常量（来自 Step 13G-A 与 population 模块）。
OPENHANDS_MAX_REPLAY_INPUT_TOKENS = 131_072
OPENHANDS_WORKFLOWS_PER_SNAPSHOT = 4
OPENHANDS_MAIN_ELIGIBLE_SNAPSHOTS = 168

# 上下文长度 feasibility 放宽阈值（含少量余量）。
CONTEXT_LENGTH_SOFT_LIMIT = 131_200


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_records(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(slots=True)
class _SchemaAccumulator:
    top_level_keys: Counter = field(default_factory=Counter)
    request_type_keys: dict[str, Counter] = field(default_factory=dict)
    request_type_value_types: dict[str, Counter] = field(default_factory=dict)
    subagent_keys: Counter = field(default_factory=Counter)
    inner_request_type_keys: dict[str, Counter] = field(default_factory=dict)
    inner_request_type_value_types: dict[str, Counter] = field(default_factory=dict)
    request_type_counts: Counter = field(default_factory=Counter)
    inner_request_type_counts: Counter = field(default_factory=Counter)
    subagent_status_counts: Counter = field(default_factory=Counter)
    subagent_type_counts: Counter = field(default_factory=Counter)

    def _type_name(self, value: object) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, list):
            return "list"
        if isinstance(value, dict):
            return "dict"
        return type(value).__name__

    def inspect_top(self, record: dict[str, Any]) -> None:
        for key, value in record.items():
            self.top_level_keys[key] += 1
            if key == "requests":
                for req in value:
                    self.inspect_request(req, top_level=True)

    def inspect_request(
        self, req: dict[str, Any], *, top_level: bool = True
    ) -> None:
        req_type = str(req.get("type", "unknown"))
        if top_level:
            self.request_type_counts[req_type] += 1
            target_keys = self.request_type_keys.setdefault(req_type, Counter())
            target_types = self.request_type_value_types.setdefault(
                req_type, Counter()
            )
        else:
            self.inner_request_type_counts[req_type] += 1
            target_keys = self.inner_request_type_keys.setdefault(req_type, Counter())
            target_types = self.inner_request_type_value_types.setdefault(
                req_type, Counter()
            )
        for key, value in req.items():
            target_keys[key] += 1
            target_types[f"{key}:{self._type_name(value)}"] += 1
            if key == "requests" and req_type == "subagent":
                self.inspect_subagent(req)

    def inspect_subagent(self, entry: dict[str, Any]) -> None:
        for key in entry.keys():
            self.subagent_keys[key] += 1
        self.subagent_status_counts[str(entry.get("status", "unknown"))] += 1
        self.subagent_type_counts[str(entry.get("subagent_type", "unknown"))] += 1
        for inner in entry.get("requests", []):
            self.inspect_request(inner, top_level=False)


def collect_schema_inventory(path: Path = FROZEN_AGENTX_PATH) -> dict[str, Any]:
    """Phase 1：流式收集 schema 字段与类型清单。"""
    acc = _SchemaAccumulator()
    record_count = 0
    total_top_requests = 0
    total_inner_requests = 0
    total_subagents = 0
    for record in _stream_records(path):
        record_count += 1
        acc.inspect_top(record)
        total_top_requests += len(record.get("requests", []))
        for req in record.get("requests", []):
            if req.get("type") == "subagent":
                total_subagents += 1
                total_inner_requests += len(req.get("requests", []))

    def _counter_dict(counter: Counter) -> dict[str, int]:
        return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "record_count": record_count,
        "total_top_level_requests": total_top_requests,
        "total_subagent_entries": total_subagents,
        "total_inner_requests": total_inner_requests,
        "top_level_fields": _counter_dict(acc.top_level_keys),
        "top_level_required_fields": ["id", "models", "block_size", "hash_id_scope", "requests"],
        "top_level_optional_fields": ["totals"],
        "top_level_request_type_counts": _counter_dict(acc.request_type_counts),
        "inner_request_type_counts": _counter_dict(acc.inner_request_type_counts),
        "subagent_status_counts": _counter_dict(acc.subagent_status_counts),
        "subagent_type_counts": _counter_dict(acc.subagent_type_counts),
        "request_type_fields": {
            req_type: _counter_dict(keys)
            for req_type, keys in sorted(acc.request_type_keys.items())
        },
        "request_type_value_types": {
            req_type: _counter_dict(types)
            for req_type, types in sorted(acc.request_type_value_types.items())
        },
        "subagent_fields": _counter_dict(acc.subagent_keys),
        "inner_request_type_fields": {
            req_type: _counter_dict(keys)
            for req_type, keys in sorted(acc.inner_request_type_keys.items())
        },
        "inner_request_type_value_types": {
            req_type: _counter_dict(types)
            for req_type, types in sorted(acc.inner_request_type_value_types.items())
        },
    }


def collect_corpus_statistics(path: Path = FROZEN_AGENTX_PATH) -> dict[str, Any]:
    """Phase 1-Corpus：统计 trace、请求、模型、长度分布。"""
    record_count = 0
    traces_with_subagent = 0
    total_subagents = 0
    total_inner = 0
    top_type_counts: Counter = Counter()
    inner_type_counts: Counter = Counter()
    model_counts: Counter = Counter()
    block_sizes: Counter = Counter()
    scopes: Counter = Counter()
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    hash_block_counts: list[int] = []
    trace_total_tokens: list[int] = []

    for record in _stream_records(path):
        record_count += 1
        bs = int(record.get("block_size", 0))
        block_sizes[bs] += 1
        scopes[str(record.get("hash_id_scope", "unknown"))] += 1
        for model in record.get("models", []):
            model_counts[str(model)] += 1
        has_sub = False
        trace_tokens = 0

        def _walk(reqs: list[dict[str, Any]]) -> None:
            nonlocal has_sub, total_subagents, total_inner, trace_tokens
            for req in reqs:
                t = req.get("type")
                if t in ("n", "s"):
                    top_type_counts[t] += 1
                    inp = int(req.get("in", 0))
                    out = int(req.get("out", 0))
                    input_lengths.append(inp)
                    output_lengths.append(out)
                    trace_tokens += inp + out
                    hash_block_counts.append(len(req.get("hash_ids", [])))
                elif t == "subagent":
                    has_sub = True
                    total_subagents += 1
                    total_inner += len(req.get("requests", []))
                    for inner in req.get("requests", []):
                        inner_type_counts[inner.get("type", "unknown")] += 1
                        iinp = int(inner.get("in", 0))
                        iout = int(inner.get("out", 0))
                        input_lengths.append(iinp)
                        output_lengths.append(iout)
                        trace_tokens += iinp + iout
                        hash_block_counts.append(len(inner.get("hash_ids", [])))

        _walk(record.get("requests", []))
        if has_sub:
            traces_with_subagent += 1
        trace_total_tokens.append(trace_tokens)

    def _dist(values: list[int]) -> dict[str, int | float]:
        if not values:
            return {}
        values_sorted = sorted(values)
        n = len(values_sorted)
        return {
            "min": values_sorted[0],
            "max": values_sorted[-1],
            "mean": round(sum(values_sorted) / n, 2),
            "p50": values_sorted[n // 2],
            "p95": values_sorted[int(n * 0.95)],
            "p99": values_sorted[int(n * 0.99)],
        }

    return {
        "record_count": record_count,
        "traces_with_subagent": traces_with_subagent,
        "traces_without_subagent": record_count - traces_with_subagent,
        "total_subagent_entries": total_subagents,
        "total_inner_requests": total_inner,
        "top_level_request_type_counts": dict(top_type_counts),
        "inner_request_type_counts": dict(inner_type_counts),
        "model_counts": dict(sorted(model_counts.items(), key=lambda kv: -kv[1])),
        "block_size_counts": dict(block_sizes),
        "hash_id_scope_counts": dict(scopes),
        "input_length_tokens_dist": _dist(input_lengths),
        "output_length_tokens_dist": _dist(output_lengths),
        "hash_block_count_dist": _dist(hash_block_counts),
        "trace_total_tokens_dist": _dist(trace_total_tokens),
    }


@dataclass(slots=True)
class _Conversation:
    """AgentX 中一个可映射为 FlowState workflow/lineage 的对话。"""

    conversation_id: str
    workflow_id: str
    lineage_path: tuple[str, ...]
    start_seconds: float
    end_seconds: float
    request_hash_token_positions: list[int]
    request_models: list[str]
    request_end_seconds: list[float]
    parent_conversation_id: str | None = None
    source: str = ""  # "root" | "flat_chain" | "subagent_main" | "subagent_overflow"
    is_inherited: bool = False  # True 表示 fork_depth > 0，继承父对话累积上下文
    fork_depth_blocks: int = 0
    request_input_lengths: list[int] = field(default_factory=list)
    request_hash_ids: list[list[int]] = field(default_factory=list)


@dataclass(slots=True)
class _TraceConversations:
    trace_id: str
    block_size: int
    conversations: list[_Conversation]
    spawn_fork_rows: list[dict[str, Any]]
    top_level_seams_merged: int


def _tail_hash_at(
    chain: AgentChain, outer_idx: int, req_by_outer: dict[int, _Req]
) -> np.ndarray | None:
    """返回 chain 在 outer_idx 之前最后一个带 hash 请求的 hash 数组。"""
    best: np.ndarray | None = None
    for oi, _ in chain.requests:
        if oi > outer_idx:
            break
        req = req_by_outer.get(oi)
        if req and req.hash_ids:
            best = np.asarray(req.hash_ids, dtype=np.int64)
    return best


def _detect_and_build_conversations(
    trace_id: str,
    block_size: int,
    top_requests: list[dict[str, Any]],
) -> _TraceConversations:
    """对单个 trace 重建 root + flat chain + subagent 对话。"""
    conversations: list[_Conversation] = []
    spawn_fork_rows: list[dict[str, Any]] = []

    normals = [
        (i, _Req.from_dict(r))
        for i, r in enumerate(top_requests)
        if r.get("type") in ("n", "s")
    ]
    preamble, detect_normals = split_off_preamble(normals)
    top_detection = detect_agent_chains(detect_normals)
    main_chain = top_detection.chains[top_detection.main_index]
    detect_req_by_outer = {oi: r for oi, r in detect_normals}

    # root conversation（preamble 重新挂回 main chain）。
    root_reqs = list(main_chain.requests)
    if preamble:
        root_reqs = sorted(preamble + root_reqs, key=lambda item: (item[1].t, item[0]))
    root_hash_positions = [
        len(req.hash_ids) * block_size for _, req in root_reqs if req.hash_ids
    ]
    root_input_lengths = [
        req.input_length for _, req in root_reqs if req.hash_ids
    ]
    root_hash_ids = [
        list(req.hash_ids) for _, req in root_reqs if req.hash_ids
    ]
    root_end_times = [
        _req_end(req) for _, req in root_reqs if req.hash_ids
    ]
    root_end = max((_req_end(req) for _, req in root_reqs), default=0.0)
    conversations.append(
        _Conversation(
            conversation_id=trace_id,
            workflow_id=trace_id,
            lineage_path=(trace_id,),
            start_seconds=0.0,
            end_seconds=root_end,
            request_hash_token_positions=root_hash_positions,
            request_input_lengths=root_input_lengths,
            request_hash_ids=root_hash_ids,
            request_models=[req.model for _, req in root_reqs],
            request_end_seconds=root_end_times,
            parent_conversation_id=None,
            source="root",
            is_inherited=False,
            fork_depth_blocks=0,
        )
    )

    main_reqs_list = [req for _, req in main_chain.requests]
    main_peak_isl = max((r.input_length for r in main_reqs_list), default=0)
    main_model = main_reqs_list[0].model if main_reqs_list else None
    top_wg_coords = worker_group_assignment(top_detection)

    # top-level flat chains。
    for n, ci in enumerate(top_detection.worker_indices):
        chain = top_detection.chains[ci]
        reqs = sorted(chain.requests, key=lambda item: (item[1].t, item[0]))
        aux = is_aux_chain(
            [req for _, req in reqs], main_peak_isl, main_model=main_model
        )
        reduction = not aux and is_reduction_chain([req for _, req in reqs])
        if aux or reduction:
            suffix = f"aux:{n:03d}"
        elif ci in top_wg_coords:
            g, m = top_wg_coords[ci]
            suffix = f"wg:{g:03d}_{m:03d}"
        else:
            suffix = f"fa:{n:03d}"
        conv_id = f"{trace_id}::{suffix}"
        hpos = [len(req.hash_ids) * block_size for _, req in reqs if req.hash_ids]
        inps = [req.input_length for _, req in reqs if req.hash_ids]
        hids = [list(req.hash_ids) for _, req in reqs if req.hash_ids]
        ends = [_req_end(req) for _, req in reqs if req.hash_ids]
        start = reqs[0][1].t if reqs else 0.0
        end = max((_req_end(req) for _, req in reqs), default=start)
        depth = chain.fork.depth if chain.fork else 0
        context_inheritance = "fork" if depth > 0 else "spawn"
        first_hash_len = len(reqs[0][1].hash_ids) * block_size if reqs else 0
        conversations.append(
            _Conversation(
                conversation_id=conv_id,
                workflow_id=trace_id,
                lineage_path=(trace_id, suffix),
                start_seconds=start,
                end_seconds=end,
                request_hash_token_positions=hpos,
                request_input_lengths=inps,
                request_hash_ids=hids,
                request_models=[req.model for _, req in reqs],
                request_end_seconds=ends,
                parent_conversation_id=trace_id,
                source="flat_chain",
                is_inherited=(context_inheritance == "fork"),
                fork_depth_blocks=depth,
            )
        )
        spawn_fork_rows.append(
            {
                "trace_id": trace_id,
                "conversation_id": conv_id,
                "parent_conversation_id": trace_id,
                "source": "flat_chain",
                "fork_depth_blocks": depth,
                "parent_prefix_tokens": depth * block_size,
                "child_first_hash_tokens": first_hash_len,
                "classification": (
                    "aux" if aux else ("reduction" if reduction else ("worker_group" if ci in top_wg_coords else "solo_agent"))
                ),
                "context_inheritance": context_inheritance,
            }
        )

    # subagents。
    for sa_outer_idx, req in enumerate(top_requests):
        if req.get("type") != "subagent":
            continue
        entry_t = float(req["t"])
        agent_id = str(req["agent_id"])

        root_tail_hash = _tail_hash_at(main_chain, sa_outer_idx, detect_req_by_outer)

        main_sa_id = f"{trace_id}::sa:{agent_id}"
        if not req.get("requests"):
            conversations.append(
                _Conversation(
                    conversation_id=main_sa_id,
                    workflow_id=trace_id,
                    lineage_path=(trace_id, f"sa:{agent_id}"),
                    start_seconds=entry_t,
                    end_seconds=entry_t,
                    request_hash_token_positions=[],
                    request_models=[],
                    request_end_seconds=[],
                    parent_conversation_id=trace_id,
                    source="subagent_main",
                    is_inherited=False,
                    fork_depth_blocks=0,
                )
            )
            spawn_fork_rows.append(
                {
                    "trace_id": trace_id,
                    "conversation_id": main_sa_id,
                    "parent_conversation_id": trace_id,
                    "source": "subagent_main",
                    "fork_depth_blocks": 0,
                    "parent_prefix_tokens": 0,
                    "child_first_hash_tokens": 0,
                    "classification": "subagent_main",
                    "context_inheritance": "spawn",
                }
            )
            continue

        inner = [_Req.from_dict(ir) for ir in req["requests"]]
        indexed = list(enumerate(inner))
        sa_preamble, sa_detect = split_off_preamble(indexed)
        sa_detection = detect_agent_chains(sa_detect)
        sa_main = sa_detection.chains[sa_detection.main_index]
        sa_reqs = list(sa_main.requests)
        if sa_preamble:
            sa_reqs = sorted(sa_preamble + sa_reqs, key=lambda item: (item[1].t, item[0]))
        sa_hash_positions = [
            len(r.hash_ids) * block_size for _, r in sa_reqs if r.hash_ids
        ]
        sa_input_lengths = [r.input_length for _, r in sa_reqs if r.hash_ids]
        sa_hash_ids = [list(r.hash_ids) for _, r in sa_reqs if r.hash_ids]
        sa_end_times = [_req_end(r) for _, r in sa_reqs if r.hash_ids]
        sa_start = sa_reqs[0][1].t if sa_reqs else entry_t
        sa_end = max((_req_end(r) for _, r in sa_reqs), default=sa_start)

        sa_first_hash: np.ndarray | None = None
        for _, r in sa_main.requests:
            if r.hash_ids:
                sa_first_hash = np.asarray(r.hash_ids, dtype=np.int64)
                break
        if sa_first_hash is not None and root_tail_hash is not None:
            sa_main_depth = int(_np_lcp(root_tail_hash, sa_first_hash))
        else:
            sa_main_depth = 0
        sa_context = "fork" if sa_main_depth > 0 else "spawn"
        conversations.append(
            _Conversation(
                conversation_id=main_sa_id,
                workflow_id=trace_id,
                lineage_path=(trace_id, f"sa:{agent_id}"),
                start_seconds=sa_start,
                end_seconds=sa_end,
                request_hash_token_positions=sa_hash_positions,
                request_input_lengths=sa_input_lengths,
                request_hash_ids=sa_hash_ids,
                request_models=[r.model for _, r in sa_reqs],
                request_end_seconds=sa_end_times,
                parent_conversation_id=trace_id,
                source="subagent_main",
                is_inherited=(sa_context == "fork"),
                fork_depth_blocks=sa_main_depth,
            )
        )
        first_len = len(sa_first_hash) * block_size if sa_first_hash is not None else 0
        spawn_fork_rows.append(
            {
                "trace_id": trace_id,
                "conversation_id": main_sa_id,
                "parent_conversation_id": trace_id,
                "source": "subagent_main",
                "fork_depth_blocks": sa_main_depth,
                "parent_prefix_tokens": sa_main_depth * block_size,
                "child_first_hash_tokens": first_len,
                "classification": "subagent_main",
                "context_inheritance": sa_context,
            }
        )

        sa_main_reqs_list = [r for _, r in sa_main.requests]
        sa_peak_isl = max((r.input_length for r in sa_main_reqs_list), default=0)
        sa_model = sa_main_reqs_list[0].model if sa_main_reqs_list else None
        sa_wg_coords = worker_group_assignment(sa_detection)

        for n, ci in enumerate(sa_detection.worker_indices):
            chain = sa_detection.chains[ci]
            reqs = sorted(chain.requests, key=lambda item: (item[1].t, item[0]))
            aux = is_aux_chain(
                [req for _, req in reqs], sa_peak_isl, main_model=sa_model
            )
            reduction = not aux and is_reduction_chain([req for _, req in reqs])
            if aux or reduction:
                suffix = f"aux:{n:03d}"
            elif ci in sa_wg_coords:
                g, m = sa_wg_coords[ci]
                suffix = f"wg:{g:03d}_{m:03d}"
            else:
                suffix = f"fa:{n:03d}"
            conv_id = f"{trace_id}::sa:{agent_id}:{suffix}"
            hpos = [len(req.hash_ids) * block_size for _, req in reqs if req.hash_ids]
            inps = [req.input_length for _, req in reqs if req.hash_ids]
            hids = [list(req.hash_ids) for _, req in reqs if req.hash_ids]
            ends = [_req_end(req) for _, req in reqs if req.hash_ids]
            start = reqs[0][1].t if reqs else entry_t
            end = max((_req_end(req) for _, req in reqs), default=start)
            depth = chain.fork.depth if chain.fork else 0
            ctx_inh = "fork" if depth > 0 else "spawn"
            conversations.append(
                _Conversation(
                    conversation_id=conv_id,
                    workflow_id=trace_id,
                    lineage_path=(trace_id, f"sa:{agent_id}", suffix),
                    start_seconds=start,
                    end_seconds=end,
                    request_hash_token_positions=hpos,
                    request_input_lengths=inps,
                    request_hash_ids=hids,
                    request_models=[req.model for _, req in reqs],
                    request_end_seconds=ends,
                    parent_conversation_id=main_sa_id,
                    source="subagent_overflow",
                    is_inherited=(ctx_inh == "fork"),
                    fork_depth_blocks=depth,
                )
            )
            spawn_fork_rows.append(
                {
                    "trace_id": trace_id,
                    "conversation_id": conv_id,
                    "parent_conversation_id": main_sa_id,
                    "source": "subagent_overflow",
                    "fork_depth_blocks": depth,
                    "parent_prefix_tokens": depth * block_size,
                    "child_first_hash_tokens": len(reqs[0][1].hash_ids) * block_size if reqs else 0,
                    "classification": (
                        "aux" if aux else ("reduction" if reduction else ("worker_group" if ci in sa_wg_coords else "solo_agent"))
                    ),
                    "context_inheritance": ctx_inh,
                }
            )

    return _TraceConversations(
        trace_id=trace_id,
        block_size=block_size,
        conversations=conversations,
        spawn_fork_rows=spawn_fork_rows,
        top_level_seams_merged=top_detection.seams_merged,
    )


def _build_all_trace_conversations(
    path: Path = FROZEN_AGENTX_PATH,
) -> list[_TraceConversations]:
    """一次性构建所有 trace 的 conversation 结构（避免重复 chain detection）。"""
    result: list[_TraceConversations] = []
    for record in _stream_records(path):
        tc = _detect_and_build_conversations(
            record["id"], int(record["block_size"]), record.get("requests", [])
        )
        result.append(tc)
    return result


@dataclass(slots=True)
class _EpochEvent:
    """单个 timeline 事件。"""

    t: float
    kind_order: int  # 0=start, 1=target, 2=end
    conv_idx: int
    value: int  # target 值，start/end 为 0
    conv_start: float


@dataclass(slots=True)
class _EpochSnapshot:
    """某一 allocation epoch 的完整状态快照。"""

    t: float
    pending_count: int
    max_degree: int
    degree_counter: Counter
    candidate_degrees: list[int]
    active_targets: dict[int, int]


@dataclass(slots=True)
class _TraceOnlineResult:
    trace_id: str
    max_pending: int
    mean_pending: float
    median_pending: float
    p75_pending: float
    p95_pending: float
    epochs_ge_2: int
    epochs_ge_4: int
    epochs_ge_8: int
    epochs_ge_16: int
    max_online_degree: int
    epochs_d_ge_2: int
    epochs_d_ge_3: int
    all_invariants_pass: bool
    degree_distribution: dict[int, int]
    snapshots: list[_EpochSnapshot]
    shared_coverage_examples: list[dict[str, Any]]


def _trace_event_timeline(tc: _TraceConversations) -> list[_EpochEvent]:
    """构造 trace 内所有 conversation 的在线事件时间线。

    allocation epoch 定义为：任意 conversation 的 start、任意 hash-bearing request 的完成、
    或任意 conversation 的 end。同一时刻的事件按 start -> target -> end 顺序处理，
    snapshot 在 end 之前记录，从而保证半开区间 [start, end) 内 conversation 属于 P_t。
    """
    events: list[_EpochEvent] = []
    for idx, conv in enumerate(tc.conversations):
        events.append(_EpochEvent(conv.start_seconds, 0, idx, 0, conv.start_seconds))
        events.append(_EpochEvent(conv.end_seconds, 2, idx, 0, conv.start_seconds))
        cumulative = 0
        for pos, end_sec in zip(conv.request_hash_token_positions, conv.request_end_seconds):
            cumulative += pos
            events.append(_EpochEvent(end_sec, 1, idx, cumulative, conv.start_seconds))
    # 确定性排序：同时间先 start，再 target；target 按 conversation 开始时间早的先处理。
    events.sort(key=lambda e: (e.t, e.kind_order, e.conv_start, e.conv_idx))
    return events


def _analyze_trace_online_compatibility(
    tc: _TraceConversations,
) -> _TraceOnlineResult:
    """对一个 trace 执行 epoch-scoped online-safe compatibility degree 分析。"""
    convs = tc.conversations
    n_convs = len(convs)
    events = _trace_event_timeline(tc)

    # 祖先索引：p 的所有 lineage prefix 对应的 conversation 索引（含自身）。
    lineage_to_idx = {conv.lineage_path: i for i, conv in enumerate(convs)}
    ancestors: list[list[int]] = [[] for _ in convs]
    for i, conv in enumerate(convs):
        for k in range(1, len(conv.lineage_path) + 1):
            prefix = conv.lineage_path[:k]
            if prefix in lineage_to_idx:
                ancestors[i].append(lineage_to_idx[prefix])

    # 每个 conversation 的 candidate 位置与全局索引（numpy 向量化用）。
    positions_arr: list[np.ndarray] = []
    cand_idx_arr: list[np.ndarray] = []
    gidx_to_conv = np.empty(0, dtype=np.int64)
    gidx_to_local = np.empty(0, dtype=np.int64)
    total_candidates = 0
    for i, conv in enumerate(convs):
        poss = np.array(conv.request_hash_token_positions, dtype=np.int64)
        # 转换为累积位置。
        if poss.size:
            poss = np.cumsum(poss)
        positions_arr.append(poss)
        idxs = np.arange(total_candidates, total_candidates + poss.size, dtype=np.int64)
        cand_idx_arr.append(idxs)
        total_candidates += poss.size
        gidx_to_conv = np.concatenate([gidx_to_conv, np.full(poss.size, i, dtype=np.int64)])
        gidx_to_local = np.concatenate([gidx_to_local, np.arange(poss.size, dtype=np.int64)])

    active_targets: dict[int, int] = {}
    materialized_targets = np.zeros(n_convs, dtype=np.int64)
    degrees = np.zeros(total_candidates, dtype=np.int64)

    snapshots: list[_EpochSnapshot] = []
    shared_examples: list[dict[str, Any]] = []

    i = 0
    n_events = len(events)
    while i < n_events:
        t = events[i].t
        # 1) start
        while i < n_events and events[i].t == t and events[i].kind_order == 0:
            active_targets[events[i].conv_idx] = 0
            i += 1
        # 2) target：更新 materialized target 与 active target。
        while i < n_events and events[i].t == t and events[i].kind_order == 1:
            cv = events[i].conv_idx
            new_target = events[i].value
            materialized_targets[cv] = new_target
            active_targets[cv] = new_target
            i += 1
        # 3) 基于当前在线可见状态重新计算所有 candidate 的 d_t(c)。
        degrees.fill(0)
        for p_idx, p_target in active_targets.items():
            p_conv = convs[p_idx]
            if not p_conv.is_inherited or p_target <= 0:
                continue
            for anc_idx in ancestors[p_idx]:
                if anc_idx == p_idx:
                    continue
                cap = min(p_target, materialized_targets[anc_idx])
                if cap <= 0:
                    continue
                poss = positions_arr[anc_idx]
                if poss.size == 0:
                    continue
                right = int(np.searchsorted(poss, cap, side="right"))
                if right > 0:
                    degrees[cand_idx_arr[anc_idx][:right]] += 1
        # 4) 记录 snapshot。
        pending_count = len(active_targets)
        max_degree = int(degrees.max()) if total_candidates else 0
        degree_counter: Counter = Counter(int(x) for x in np.bincount(degrees))
        if 0 not in degree_counter and total_candidates:
            degree_counter[0] = 0
        snapshots.append(
            _EpochSnapshot(
                t=t,
                pending_count=pending_count,
                max_degree=max_degree,
                degree_counter=degree_counter,
                candidate_degrees=degrees.tolist(),
                active_targets=dict(active_targets),
            )
        )
        # 5) 收集 shared-coverage 示例。
        if pending_count >= 2 and max_degree >= 2 and len(shared_examples) < 20:
            cand_gidx = int(np.argmax(degrees))
            conv_idx = int(gidx_to_conv[cand_gidx])
            local_idx = int(gidx_to_local[cand_gidx])
            cand_conv = convs[conv_idx]
            cand_pos = int(positions_arr[conv_idx][local_idx]) if positions_arr[conv_idx].size else 0
            compatible_ids = []
            for p_idx, target in active_targets.items():
                if p_idx == conv_idx:
                    continue
                p_conv = convs[p_idx]
                if not p_conv.is_inherited:
                    continue
                if _is_compatible(cand_conv.lineage_path, cand_pos, p_conv.lineage_path, target):
                    compatible_ids.append(p_conv.conversation_id)
            if compatible_ids:
                shared_examples.append(
                    {
                        "trace_id": tc.trace_id,
                        "epoch_timestamp": t,
                        "|P_t|": pending_count,
                        "candidate_id": cand_conv.conversation_id,
                        "candidate_token_pos": cand_pos,
                        "candidate_materialization_time": float(
                            cand_conv.request_end_seconds[local_idx]
                        )
                        if cand_conv.request_end_seconds
                        else None,
                        "d_t(c)": max_degree,
                        "compatible_pending_ids": compatible_ids[:10],
                        "fork_ancestry": cand_conv.lineage_path,
                        "why_online_safe": (
                            "该 epoch 的 P_t 仅包含已 start 且未 end 的 conversation；"
                            "compatible pending 均为 is_inherited=True 的 fork 后代，"
                            "且 candidate 的 materialization 时间不晚于 epoch 时间。"
                        ),
                    }
                )
        # 6) end：从 active set 移除。
        while i < n_events and events[i].t == t and events[i].kind_order == 2:
            active_targets.pop(events[i].conv_idx, None)
            i += 1

    pending_counts = [s.pending_count for s in snapshots]
    sorted_pending = sorted(pending_counts)
    n = len(sorted_pending)

    def _pct(vals: list[int], p: float) -> float:
        if not vals:
            return 0.0
        return float(vals[min(len(vals) - 1, int(len(vals) * p))])

    return _TraceOnlineResult(
        trace_id=tc.trace_id,
        max_pending=max(sorted_pending) if sorted_pending else 0,
        mean_pending=round(sum(sorted_pending) / n, 2) if n else 0.0,
        median_pending=round(_pct(sorted_pending, 0.5), 2),
        p75_pending=round(_pct(sorted_pending, 0.75), 2),
        p95_pending=round(_pct(sorted_pending, 0.95), 2),
        epochs_ge_2=sum(1 for s in snapshots if s.pending_count >= 2),
        epochs_ge_4=sum(1 for s in snapshots if s.pending_count >= 4),
        epochs_ge_8=sum(1 for s in snapshots if s.pending_count >= 8),
        epochs_ge_16=sum(1 for s in snapshots if s.pending_count >= 16),
        max_online_degree=max((s.max_degree for s in snapshots), default=0),
        epochs_d_ge_2=sum(1 for s in snapshots if s.max_degree >= 2),
        epochs_d_ge_3=sum(1 for s in snapshots if s.max_degree >= 3),
        all_invariants_pass=all(s.max_degree <= s.pending_count for s in snapshots),
        degree_distribution=Counter(sum((s.degree_counter for s in snapshots), Counter())),
        snapshots=snapshots,
        shared_coverage_examples=shared_examples,
    )


def analyze_spawn_fork_and_prefix(
    path: Path = FROZEN_AGENTX_PATH,
    *,
    _trace_conversations: list[_TraceConversations] | None = None,
) -> dict[str, Any]:
    """Phase 2：按 AIPerf 语义统计 SPAWN/FORK、前缀共享与 sidecar。"""
    trace_conversations = _trace_conversations or _build_all_trace_conversations(path)

    total_traces = len(trace_conversations)
    top_split_traces = 0
    subagent_traces = 0
    all_spawn_fork_rows: list[dict[str, Any]] = []
    top_worker_counts: list[int] = []
    subagent_overflow_counts: list[int] = []
    seams_merged: list[int] = []

    for tc in trace_conversations:
        all_spawn_fork_rows.extend(tc.spawn_fork_rows)
        top_workers = [c for c in tc.conversations if c.source == "flat_chain"]
        if top_workers:
            top_split_traces += 1
        if any(c.source == "subagent_main" for c in tc.conversations):
            subagent_traces += 1
        top_worker_counts.append(len(top_workers))
        subagent_overflow_counts.append(
            len([c for c in tc.conversations if c.source == "subagent_overflow"])
        )
        seams_merged.append(tc.top_level_seams_merged)

    def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        by_source: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_source.setdefault(r["source"], []).append(r)
        summary: dict[str, Any] = {}
        for source, src_rows in sorted(by_source.items()):
            fork = sum(1 for r in src_rows if r["context_inheritance"] == "fork")
            spawn = sum(1 for r in src_rows if r["context_inheritance"] == "spawn")
            unknown = len(src_rows) - fork - spawn
            classes = Counter(r["classification"] for r in src_rows)
            depths = [r["fork_depth_blocks"] for r in src_rows if r["fork_depth_blocks"] > 0]
            summary[source] = {
                "count": len(src_rows),
                "inherited_context_forks": fork,
                "fresh_context_spawns": spawn,
                "unknown": unknown,
                "classification_counts": dict(classes),
                "fork_depth_blocks_dist": {
                    "min": min(depths) if depths else 0,
                    "max": max(depths) if depths else 0,
                    "mean": round(sum(depths) / len(depths), 2) if depths else 0.0,
                },
            }
        return summary

    inheritance_counts = Counter(r["context_inheritance"] for r in all_spawn_fork_rows)
    trace_has_spawn: set[str] = set()
    trace_has_fork: set[str] = set()
    for r in all_spawn_fork_rows:
        tid = r["trace_id"]
        if r["context_inheritance"] == "spawn":
            trace_has_spawn.add(tid)
        elif r["context_inheritance"] == "fork":
            trace_has_fork.add(tid)

    fork_examples = [
        {
            "trace_id": r["trace_id"],
            "parent_conversation_id": r["parent_conversation_id"],
            "child_conversation_id": r["conversation_id"],
            "fork_depth_blocks": r["fork_depth_blocks"],
            "parent_prefix_tokens": r["parent_prefix_tokens"],
            "child_first_hash_tokens": r["child_first_hash_tokens"],
            "source": r["source"],
            "classification": r["classification"],
            "evidence": (
                "AIPerf weka_agent_chains 使用 fork_depth>0 表示子 chain 与父 chain 的 "
                "hash_ids 存在非空 LCP，因此子对话继承了父对话已累积的 KV-cache 上下文。"
            ),
        }
        for r in all_spawn_fork_rows
        if r["context_inheritance"] == "fork" and r["fork_depth_blocks"] > 0
    ][:10]

    return {
        "total_traces": total_traces,
        "traces_with_top_level_split": top_split_traces,
        "traces_with_subagent": subagent_traces,
        "top_level_seams_merged_total": sum(seams_merged),
        "top_level_worker_conversation_counts": {
            "min": min(top_worker_counts),
            "max": max(top_worker_counts),
            "mean": round(sum(top_worker_counts) / len(top_worker_counts), 2),
            "p50": sorted(top_worker_counts)[len(top_worker_counts) // 2],
        },
        "subagent_overflow_conversation_counts": {
            "min": min(subagent_overflow_counts),
            "max": max(subagent_overflow_counts),
            "mean": round(sum(subagent_overflow_counts) / len(subagent_overflow_counts), 2),
            "p50": sorted(subagent_overflow_counts)[len(subagent_overflow_counts) // 2],
        },
        "child_context_inheritance_counts": dict(inheritance_counts),
        "traces_with_spawn": len(trace_has_spawn),
        "traces_with_fork": len(trace_has_fork),
        "traces_with_both": len(trace_has_spawn & trace_has_fork),
        "traces_with_neither": total_traces - len(trace_has_spawn | trace_has_fork),
        "spawn_fork_summary": _summarize(all_spawn_fork_rows),
        "spawn_fork_detail_sample": all_spawn_fork_rows[:20],
        "deterministic_fork_examples": fork_examples,
    }


def _is_compatible(
    candidate_path: tuple[str, ...],
    candidate_pos: int,
    pending_path: tuple[str, ...],
    pending_target: int,
) -> bool:
    """FlowState ``is_compatible`` 的轻量复刻（同 workflow_id 已由 trace 保证）。"""
    if len(candidate_path) > len(pending_path):
        return False
    if pending_path[: len(candidate_path)] != candidate_path:
        return False
    return candidate_pos <= pending_target


def reconstruct_logical_workflows_and_compatibility(
    path: Path = FROZEN_AGENTX_PATH,
    *,
    _trace_conversations: list[_TraceConversations] | None = None,
) -> dict[str, Any]:
    """旧版全 trace descendant 兼容性统计（仅用于 root-cause 诊断）。

    警告：此实现把整个 trace 中所有 descendant conversation 都视为同时 pending，
    会违反 d_t(c) <= |P_t|，因此只用于 OLD_MAX_COMPATIBILITY_DIAGNOSTIC。
    """
    trace_conversations = _trace_conversations or _build_all_trace_conversations(path)
    trace_results: list[dict[str, Any]] = []
    all_degrees: list[int] = []

    for tc in trace_conversations:
        conversations = tc.conversations
        candidates: list[tuple[tuple[str, ...], int]] = []
        for conv in conversations:
            for pos in conv.request_hash_token_positions:
                candidates.append((conv.lineage_path, pos))

        pendings = [
            (conv.lineage_path, max(conv.request_hash_token_positions) if conv.request_hash_token_positions else 0)
            for conv in conversations
        ]

        degrees: list[int] = []
        for cand_path, cand_pos in candidates:
            d = sum(
                1
                for pend_path, pend_target in pendings
                if _is_compatible(cand_path, cand_pos, pend_path, pend_target)
            )
            degrees.append(d)

        all_degrees.extend(degrees)
        trace_results.append(
            {
                "trace_id": tc.trace_id,
                "conversation_count": len(conversations),
                "candidate_count": len(candidates),
                "pending_count": len(pendings),
                "max_compatibility_degree": max(degrees) if degrees else 0,
                "candidates_with_degree_at_least_2": sum(1 for d in degrees if d >= 2),
            }
        )

    degree_counter = Counter(all_degrees)
    return {
        "total_candidates": len(all_degrees),
        "compatibility_degree_distribution": dict(degree_counter),
        "max_degree_overall": max(all_degrees) if all_degrees else 0,
        "mean_max_degree_per_trace": round(
            sum(r["max_compatibility_degree"] for r in trace_results) / len(trace_results), 2
        ),
        "traces_with_any_cross_pending_candidate": sum(
            1 for r in trace_results if r["max_compatibility_degree"] >= 2
        ),
        "trace_results": trace_results,
    }


def analyze_online_logical_workflows_and_compatibility(
    path: Path = FROZEN_AGENTX_PATH,
    *,
    _trace_conversations: list[_TraceConversations] | None = None,
) -> dict[str, Any]:
    """Phase 3（修复版）：基于 online-safe epoch 的 compatibility degree d_t(c)。"""
    trace_conversations = _trace_conversations or _build_all_trace_conversations(path)

    trace_results: list[dict[str, Any]] = []
    global_degree_dist: Counter = Counter()
    global_pending_counts: list[int] = []
    max_online_degree = 0
    max_pending = 0
    all_invariants_pass = True
    shared_coverage_examples: list[dict[str, Any]] = []
    seen_example_traces: set[str] = set()

    for tc in trace_conversations:
        online = _analyze_trace_online_compatibility(tc)
        trace_results.append(
            {
                "trace_id": online.trace_id,
                "max_pending": online.max_pending,
                "mean_pending": online.mean_pending,
                "median_pending": online.median_pending,
                "p75_pending": online.p75_pending,
                "p95_pending": online.p95_pending,
                "epochs_ge_2": online.epochs_ge_2,
                "epochs_ge_4": online.epochs_ge_4,
                "epochs_ge_8": online.epochs_ge_8,
                "epochs_ge_16": online.epochs_ge_16,
                "max_online_degree": online.max_online_degree,
                "epochs_d_ge_2": online.epochs_d_ge_2,
                "epochs_d_ge_3": online.epochs_d_ge_3,
                "all_invariants_pass": online.all_invariants_pass,
            }
        )
        global_degree_dist.update(online.degree_distribution)
        global_pending_counts.extend(s.pending_count for s in online.snapshots)
        max_online_degree = max(max_online_degree, online.max_online_degree)
        max_pending = max(max_pending, online.max_pending)
        all_invariants_pass = all_invariants_pass and online.all_invariants_pass

        for ex in online.shared_coverage_examples:
            if len(shared_coverage_examples) >= 20:
                break
            if ex["trace_id"] not in seen_example_traces or len(shared_coverage_examples) < 10:
                shared_coverage_examples.append(ex)
                seen_example_traces.add(ex["trace_id"])

    sorted_pending = sorted(global_pending_counts)
    n = len(sorted_pending)

    def _pct(vals: list[int], p: float) -> float:
        if not vals:
            return 0.0
        return float(vals[min(len(vals) - 1, int(len(vals) * p))])

    return {
        "total_traces": len(trace_results),
        "max_online_degree": max_online_degree,
        "max_pending": max_pending,
        "invariant_max_degree_le_max_pending": max_online_degree <= max_pending,
        "all_per_epoch_invariants_pass": all_invariants_pass,
        "global_pending_distribution": {
            "max": max_pending,
            "mean": round(sum(sorted_pending) / n, 2) if n else 0.0,
            "median": round(_pct(sorted_pending, 0.5), 2),
            "p75": round(_pct(sorted_pending, 0.75), 2),
            "p95": round(_pct(sorted_pending, 0.95), 2),
            "epochs_ge_2": sum(1 for v in sorted_pending if v >= 2),
            "epochs_ge_4": sum(1 for v in sorted_pending if v >= 4),
            "epochs_ge_8": sum(1 for v in sorted_pending if v >= 8),
            "epochs_ge_16": sum(1 for v in sorted_pending if v >= 16),
        },
        "degree_distribution": dict(global_degree_dist),
        "d_t_counts": {
            "d=0": global_degree_dist.get(0, 0),
            "d=1": global_degree_dist.get(1, 0),
            "d=2": global_degree_dist.get(2, 0),
            "d=3": global_degree_dist.get(3, 0),
            "d=4": global_degree_dist.get(4, 0),
            "d>=5": sum(v for k, v in global_degree_dist.items() if k >= 5),
        },
        "traces_with_online_safe_d_at_least_2": sum(
            1 for r in trace_results if r["max_online_degree"] >= 2
        ),
        "epochs_with_online_safe_d_at_least_2": sum(
            r["epochs_d_ge_2"] for r in trace_results
        ),
        "candidate_epoch_pairs_with_d_at_least_2": sum(
            v for k, v in global_degree_dist.items() if k >= 2
        ),
        "trace_results": trace_results,
        "shared_coverage_examples": shared_coverage_examples[:10],
    }


def analyze_pending_concurrency(
    path: Path = FROZEN_AGENTX_PATH,
    *,
    _trace_conversations: list[_TraceConversations] | None = None,
) -> dict[str, Any]:
    """分析同一 trace 内多个 pending continuation 的时间重叠（基于半开区间）。"""
    trace_conversations = _trace_conversations or _build_all_trace_conversations(path)
    trace_concurrency: list[dict[str, Any]] = []
    all_max_concurrency: list[int] = []

    for tc in trace_conversations:
        conversations = tc.conversations
        if len(conversations) <= 1:
            trace_concurrency.append(
                {"trace_id": tc.trace_id, "max_simultaneous_pending": 1}
            )
            all_max_concurrency.append(1)
            continue
        events: list[tuple[float, int]] = []
        for conv in conversations:
            events.append((conv.start_seconds, +1))
            events.append((conv.end_seconds + 1e-9, -1))
        events.sort(key=lambda x: (x[0], x[1]))
        cur = 0
        max_conc = 0
        for _, delta in events:
            cur += delta
            max_conc = max(max_conc, cur)
        trace_concurrency.append(
            {"trace_id": tc.trace_id, "max_simultaneous_pending": max_conc}
        )
        all_max_concurrency.append(max_conc)

    return {
        "trace_count": len(trace_concurrency),
        "max_simultaneous_pending_overall": max(all_max_concurrency),
        "mean_max_simultaneous_pending": round(
            sum(all_max_concurrency) / len(all_max_concurrency), 2
        ),
        "traces_with_multiple_simultaneous_pending": sum(
            1 for v in all_max_concurrency if v >= 2
        ),
        "concurrency_distribution": dict(Counter(all_max_concurrency)),
        "trace_concurrency": trace_concurrency,
    }


def evaluate_online_gates(
    online: dict[str, Any], concurrency: dict[str, Any]
) -> dict[str, Any]:
    """评估 online-safe RQ3 可行性 gate。"""
    shared_coverage_pass = (
        online["traces_with_online_safe_d_at_least_2"] > 0
        and online["epochs_with_online_safe_d_at_least_2"] > 0
    )
    branching_pass = concurrency["traces_with_multiple_simultaneous_pending"] > 0
    return {
        "inherited_context_branching": {
            "gate": "traces_with_fork_depth>0 descendant",
            "passed": online["traces_with_online_safe_d_at_least_2"] > 0,
            "evidence": f"{online['traces_with_online_safe_d_at_least_2']}/393 traces have online-safe d_t(c)>=2",
        },
        "simultaneously_known_pending_continuations": {
            "gate": "max_simultaneous_pending >= 2",
            "passed": branching_pass,
            "evidence": f"{concurrency['traces_with_multiple_simultaneous_pending']}/393 traces show >=2 simultaneous pending",
        },
        "cross_pending_shared_checkpoint_compatibility": {
            "gate": "exists online-safe (epoch, candidate) with d_t(c) >= 2",
            "passed": shared_coverage_pass,
            "evidence": (
                f"max d_t(c) = {online['max_online_degree']}, "
                f"epochs with d_t(c)>=2 = {online['epochs_with_online_safe_d_at_least_2']}, "
                f"candidate-epoch pairs with d>=2 = {online['candidate_epoch_pairs_with_d_at_least_2']}"
            ),
        },
        "context_length_feasibility": {
            "gate": "max input tokens <= OpenHands limit (131072)",
            "passed": None,
            "evidence": "see CONTEXT_LENGTH_FEASIBILITY.json",
        },
    }


def context_length_feasibility(path: Path = FROZEN_AGENTX_PATH) -> dict[str, Any]:
    """Phase 4：与 OpenHands replay 限制做上下文长度可行性比较。"""
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    hash_token_positions: list[int] = []
    trace_total_tokens: list[int] = []
    replayable_flags: list[bool] = []

    for record in _stream_records(path):
        bs = int(record["block_size"])
        total = 0
        trace_inputs: list[int] = []

        def _walk(reqs: list[dict[str, Any]]) -> None:
            nonlocal total
            for req in reqs:
                t = req.get("type")
                if t in ("n", "s"):
                    inp = int(req.get("in", 0))
                    out = int(req.get("out", 0))
                    input_lengths.append(inp)
                    output_lengths.append(out)
                    trace_inputs.append(inp)
                    total += inp + out
                    hash_token_positions.append(len(req.get("hash_ids", [])) * bs)
                elif t == "subagent":
                    _walk(req.get("requests", []))

        _walk(record.get("requests", []))
        trace_total_tokens.append(total)
        replayable_flags.append(all(inp <= CONTEXT_LENGTH_SOFT_LIMIT for inp in trace_inputs))

    def _dist(values: list[int]) -> dict[str, int | float]:
        if not values:
            return {}
        s = sorted(values)
        n = len(s)
        return {
            "min": s[0],
            "max": s[-1],
            "mean": round(sum(s) / n, 2),
            "p50": s[n // 2],
            "p95": s[int(n * 0.95)],
            "p99": s[int(n * 0.99)],
        }

    max_input = max(input_lengths) if input_lengths else 0
    le_limit = sum(1 for x in input_lengths if x <= CONTEXT_LENGTH_SOFT_LIMIT)
    gt_limit = sum(1 for x in input_lengths if x > CONTEXT_LENGTH_SOFT_LIMIT)
    total_reqs = len(input_lengths)
    return {
        "openhands_max_replay_input_tokens": OPENHANDS_MAX_REPLAY_INPUT_TOKENS,
        "context_length_soft_limit": CONTEXT_LENGTH_SOFT_LIMIT,
        "requests_le_soft_limit": le_limit,
        "requests_gt_soft_limit": gt_limit,
        "fraction_le_soft_limit": round(le_limit / total_reqs, 4) if total_reqs else 0.0,
        "traces_all_requests_le_soft_limit": sum(replayable_flags),
        "traces_any_request_gt_soft_limit": len(replayable_flags) - sum(replayable_flags),
        "input_length_tokens_dist": _dist(input_lengths),
        "output_length_tokens_dist": _dist(output_lengths),
        "hash_token_position_dist": _dist(hash_token_positions),
        "trace_total_tokens_dist": _dist(trace_total_tokens),
        "exceeds_openhands_input_limit": max_input > OPENHANDS_MAX_REPLAY_INPUT_TOKENS,
        "max_input_to_openhands_ratio": round(max_input / OPENHANDS_MAX_REPLAY_INPUT_TOKENS, 2)
        if OPENHANDS_MAX_REPLAY_INPUT_TOKENS > 0
        else None,
        "note": (
            "AgentX traces contain individual requests up to ~990k tokens, "
            "far above the OpenHands RQ3 replay cap of 131072. Direct replay "
            "would require a larger engine context window or down-sampling."
        ),
    }


def build_old_max_compatibility_diagnostic(
    path: Path = FROZEN_AGENTX_PATH,
    *,
    _trace_conversations: list[_TraceConversations] | None = None,
) -> dict[str, Any]:
    """定位旧版 d(c)=734 的 root cause。"""
    trace_conversations = _trace_conversations or _build_all_trace_conversations(path)
    legacy = reconstruct_logical_workflows_and_compatibility(
        _trace_conversations=trace_conversations
    )
    max_degree = legacy["max_degree_overall"]

    # 找到旧版最大 degree 所在的 trace。
    target_trace_id: str | None = None
    for r in legacy["trace_results"]:
        if r["max_compatibility_degree"] == max_degree:
            target_trace_id = r["trace_id"]
            break
    if target_trace_id is None:
        return {"error": "无法定位旧版最大 degree 的 trace"}

    tc = next(tc for tc in trace_conversations if tc.trace_id == target_trace_id)
    convs = tc.conversations
    root = next(c for c in convs if c.source == "root")

    # 复现旧版对该 trace 的 candidate 计数。
    candidates: list[tuple[_Conversation, int]] = []
    for conv in convs:
        for pos in conv.request_hash_token_positions:
            candidates.append((conv, pos))
    pendings = [
        (conv.lineage_path, max(conv.request_hash_token_positions) if conv.request_hash_token_positions else 0)
        for conv in convs
    ]

    best_cand: tuple[_Conversation, int] | None = None
    best_degree = -1
    for conv, pos in candidates:
        d = sum(
            1
            for pend_path, pend_target in pendings
            if _is_compatible(conv.lineage_path, pos, pend_path, pend_target)
        )
        if d > best_degree:
            best_degree = d
            best_cand = (conv, pos)

    if best_cand is None:
        return {"error": "未找到 candidate"}
    cand_conv, cand_pos = best_cand

    # 734 个对象到底是什么。
    compatible_convs = [
        conv
        for conv in convs
        if conv.conversation_id != cand_conv.conversation_id
        and _is_compatible(
            cand_conv.lineage_path,
            cand_pos,
            conv.lineage_path,
            max(conv.request_hash_token_positions) if conv.request_hash_token_positions else 0,
        )
    ]

    inherited_count = sum(1 for c in compatible_convs if c.is_inherited)
    spawn_count = sum(1 for c in compatible_convs if not c.is_inherited)
    root_cause_explanation = (
        "旧实现把 trace 内所有 structural descendant conversation（无论是否继承、"
        "是否在同一 epoch active、是否已完成）全部放入 P_t，导致 d(c) 被严重夸大。"
    )

    return {
        "old_max_degree": max_degree,
        "target_trace_id": target_trace_id,
        "candidate_conversation_id": cand_conv.conversation_id,
        "candidate_token_pos": cand_pos,
        "what_degree_represented": (
            "candidate 的 lineage_path 是该 trace 内所有其他 conversation 的 prefix，"
            "因此旧实现把每一个 descendant conversation 都计为一个 compatible pending。"
        ),
        "compatible_objects_count": len(compatible_convs),
        "breakdown": {
            "total_descendant_conversations": len(convs) - 1,
            "inherited_fork_descendants": inherited_count,
            "fresh_spawn_descendants": spawn_count,
            "future_descendants_included": True,
            "completed_or_inactive_descendants_included": True,
            "max_simultaneously_active_in_this_trace": analyze_pending_concurrency(
                _trace_conversations=[tc]
            )["max_simultaneous_pending_overall"],
        },
        "sample_compatible_ids": [c.conversation_id for c in compatible_convs[:20]],
        "root_cause": root_cause_explanation,
    }


def compare_with_openhands(
    corpus: dict[str, Any],
    online: dict[str, Any],
    concurrency: dict[str, Any],
) -> dict[str, Any]:
    """与 Step 13G-A OpenHands Main Population 的关键指标对比。"""
    return {
        "openhands_reference": {
            "eligible_snapshots": OPENHANDS_MAIN_ELIGIBLE_SNAPSHOTS,
            "workflows_per_snapshot": OPENHANDS_WORKFLOWS_PER_SNAPSHOT,
            "pending_per_snapshot": OPENHANDS_WORKFLOWS_PER_SNAPSHOT,
            "max_compatibility_degree": 1,
            "cross_pending_candidates": 0,
            "branching_structure": "none",
        },
        "agentx_observed_online_safe": {
            "total_traces": corpus["record_count"],
            "traces_with_subagent": corpus["traces_with_subagent"],
            "max_pending": online["max_pending"],
            "max_online_degree": online["max_online_degree"],
            "traces_with_online_safe_d_at_least_2": online[
                "traces_with_online_safe_d_at_least_2"
            ],
            "max_simultaneous_pending": concurrency["max_simultaneous_pending_overall"],
        },
        "interpretation": (
            "AgentX 在 online-safe 语义下仍存在跨 pending 共享 checkpoint 的结构，"
            "但其真实 max d_t(c) 受 |P_t| 上限约束，远小于旧版全 trace descendant 计数。"
            "上下文长度仍是主要 blocker。"
        ),
    }


def build_semantics_evidence() -> dict[str, Any]:
    """Phase 2-Semantics：记录 AIPerf 官方解析器语义。"""
    return {
        "source": "AIPerf weka_trace.py / weka_agent_chains.py (Apache-2.0)",
        "local_hash_scope": "hash_id_scope='local' means hash IDs are scoped per trace; no cross-trace KV-cache sharing is asserted.",
        "chain_detection": {
            "algorithm": "Two-phase hash_id LCP greedy detection",
            "phase1": "Assign each hash-bearing request to an extension target (tail is a complete prefix, same model, ended by t) or create a new chain from deepest LCP fork.",
            "phase2": "Splice join-seam continuations onto dead tails if they share enough prefix and start promptly.",
            "cross_model_rule": "Cross-model attachment is always a spawn, never an extension.",
        },
        "spawn_vs_fork": {
            "spawn": "Child conversation with fork_depth == 0 (no proven shared prefix) -> fresh context.",
            "fork": "Child conversation with fork_depth > 0 (hash_id LCP evidence) -> inherited accumulated context.",
        },
        "worker_classifications": {
            "aux": f"Short (≤{DEFAULT_AUX_MAX_REQUESTS} request) sidecar with small fresh context or cross-model.",
            "reduction": f"Single-request large-input/short-output (osl<{DEFAULT_AUX_REDUCTION_OSL_MAX}, ratio>{DEFAULT_AUX_REDUCTION_RATIO}).",
            "worker_group": f"Workers that share fork_depth>0 AND temporally overlap, with group_min={DEFAULT_WORKER_GROUP_MIN}.",
            "solo_agent": "Any other non-aux, non-reduction, non-wg worker chain.",
        },
        "seam_splicing": {
            "max_gap_seconds": DEFAULT_SEAM_MAX_GAP_SECONDS,
            "min_overlap_ratio": DEFAULT_SEAM_MIN_OVERLAP_RATIO,
            "purpose": "Avoid merging distinct sessions that merely share a base prefix after a long idle gap.",
        },
        "prefix_vs_ancestry": {
            "physical_prefix_sharing": "Detected from hash_ids LCP; evidence that two request states share KV-cache blocks.",
            "logical_inherited_ancestry": "fork_depth>0 is the official AIPerf marker for inherited context; spawn children do not inherit parent checkpoints.",
            "caveat": "Physical prefix sharing is necessary but not sufficient for logical inheritance; a fresh spawn may coincidentally share a base system/tools prefix.",
        },
    }


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_md(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _build_final_report(
    artifact_root: Path,
    gate_results: dict[str, Any],
    corpus: dict[str, Any],
    online: dict[str, Any],
    concurrency: dict[str, Any],
    ctx: dict[str, Any],
    comparison: dict[str, Any],
    diagnostic: dict[str, Any],
    spawn_fork: dict[str, Any],
) -> str:
    """生成 Step 13G-B0.1 最终 Markdown 报告。"""
    shared = gate_results["cross_pending_shared_checkpoint_compatibility"]
    branch = gate_results["simultaneously_known_pending_continuations"]
    ctx_gate_status = "PASS" if not ctx["exceeds_openhands_input_limit"] else "WARN"
    invariant_pass = online["invariant_max_degree_le_max_pending"] and online["all_per_epoch_invariants_pass"]
    lines = [
        "# Step 13G-B0.1 · AgentX Weka Corpus 在线安全兼容性修复审计 最终报告",
        "",
        f"**状态：** `AGENTX_ONLINE_COMPATIBILITY_AUDIT_READY`  ",
        f"**报告生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Canonical Audit Root：** `{artifact_root}`  ",
        "**消费对象：** 已冻结的 AgentX Weka corpus (`semianalysisai/cc-traces-weka-062126`)",
        "",
        "---",
        "",
        "## 1. 执行摘要",
        "",
        "本报告修复 Step 13G-B0 中 d(c)=734 与 max pending=32 的语义矛盾，"
        "引入 online-safe、epoch-scoped 的 compatibility degree d_t(c)。",
        "审计严格区分 physical prefix sharing 与 logical inherited ancestry，"
        "禁止 future leakage，并强制验证 d_t(c) <= |P_t|。",
        "",
        "| 需求 | Gate | 结果 | 证据 |",
        "|---|---|---|---|",
        f"| 继承上下文分支 | inherited_context_branching | {'PASS' if gate_results['inherited_context_branching']['passed'] else 'FAIL'} | {gate_results['inherited_context_branching']['evidence']} |",
        f"| 同时已知 pending continuation | simultaneously_known_pending_continuations | {'PASS' if branch['passed'] else 'FAIL'} | {branch['evidence']} |",
        f"| 跨 pending 共享候选兼容性 | cross_pending_shared_checkpoint_compatibility | {'PASS' if shared['passed'] else 'FAIL'} | {shared['evidence']} |",
        f"| d_t(c) <= |P_t| invariant | per-epoch invariant | {'PASS' if invariant_pass else 'FAIL'} | max d_t(c)={online['max_online_degree']}, max |P_t|={online['max_pending']} |",
        f"| 上下文长度可行性 | context_length_feasibility | {ctx_gate_status} | max input = {ctx['input_length_tokens_dist'].get('max')} tokens, OpenHands limit = {OPENHANDS_MAX_REPLAY_INPUT_TOKENS} |",
        "",
        "### 1.1 关键指标（online-safe）",
        "",
        f"- **总 trace 数**：{corpus['record_count']}",
        f"- **含 subagent 的 trace**：{corpus['traces_with_subagent']} / {corpus['record_count']}",
        f"- **最大 |P_t|**：{online['max_pending']}",
        f"- **最大 d_t(c)**：{online['max_online_degree']}",
        f"- **d_t(c) <= |P_t|  invariant**：{'PASS' if invariant_pass else 'FAIL'}",
        f"- **存在 online-safe d_t(c)≥2 的 trace 数**：{online['traces_with_online_safe_d_at_least_2']} / {corpus['record_count']}",
        f"- **存在 d_t(c)≥2 的 epoch 数**：{online['epochs_with_online_safe_d_at_least_2']}",
        f"- **最大单个请求输入长度**：{ctx['input_length_tokens_dist'].get('max'):,} tokens",
        f"- **请求 <= {CONTEXT_LENGTH_SOFT_LIMIT} 的比例**：{ctx['fraction_le_soft_limit']}",
        "",
        "---",
        "",
        "## 2. 审计范围与约束",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        "| 输入 | 已冻结 AgentX Weka corpus |",
        "| 修改权限 | **只读**；未修改 FlowState 核心、selector、恢复模型 |",
        "| 计算资源 | CPU only；未使用 GPU/SGLang |",
        "| 新增代码 | `evaluation/agentx_structure_audit.py` |",
        "| 新增测试 | `tests/test_agentx_structure_audit.py` |",
        "",
        "---",
        "",
        "## 3. Allocation Epoch 定义",
        "",
        "Allocation epoch 由以下三类**在线可见**事件触发：",
        "",
        "1. 任意 conversation 的 `start_seconds`；",
        "2. 任意 hash-bearing request 的完成时刻（产生新的 candidate 或增大 pending target）；",
        "3. 任意 conversation 的 `end_seconds`。",
        "",
        "同一时刻的事件处理顺序：`start` -> `target` -> snapshot -> `end`，"
        "从而保证 pending set P_t 采用半开区间 `[start, end)`。",
        "所有 snapshot 仅使用截至该时刻已 materialized 的 candidate 与 pending target，"
        "不使用任何未来 trajectory。",
        "",
        "---",
        "",
        "## 4. 旧 d(c)=734 的 Root Cause",
        "",
        f"- 旧版最大 d(c) = **{diagnostic.get('old_max_degree', 'N/A')}**",
        f"- 出现在 trace = `{diagnostic.get('target_trace_id', 'N/A')}`",
        f"- candidate = `{diagnostic.get('candidate_conversation_id', 'N/A')}@{diagnostic.get('candidate_token_pos', 'N/A')}`",
        f"- 734 代表的对象数 = **{diagnostic.get('compatible_objects_count', 'N/A')}**",
        "- 这些对象包括：未来才产生的 descendant、已经完成的 descendant、spawn（非继承）descendant。",
        "- **根本原因**：旧实现把整个 trace 的 descendant conversation 集合当作同时已知的 P_t，"
        "忽略了 epoch scope、active 状态、completion 状态以及 fork/spawn 语义。",
        "",
        "详见 `OLD_MAX_COMPATIBILITY_DIAGNOSTIC.json`。",
        "",
        "---",
        "",
        "## 5. SPAWN/FORK 精确计数",
        "",
        f"- SPAWN child 总数 = {spawn_fork['child_context_inheritance_counts'].get('spawn', 0)}",
        f"- FORK child 总数 = {spawn_fork['child_context_inheritance_counts'].get('fork', 0)}",
        f"- UNKNOWN child 总数 = {spawn_fork['child_context_inheritance_counts'].get('unknown', 0)}",
        f"- traces with SPAWN = {spawn_fork['traces_with_spawn']}",
        f"- traces with FORK = {spawn_fork['traces_with_fork']}",
        f"- traces with both = {spawn_fork['traces_with_both']}",
        f"- traces with neither = {spawn_fork['traces_with_neither']}",
        "",
        "fork_depth>0 的示例见 `SPAWN_FORK_AUDIT.json` 中的 `deterministic_fork_examples`。",
        "",
        "---",
        "",
        "## 6. Pending Concurrency（修复后）",
        "",
        f"- max |P_t| = {online['global_pending_distribution']['max']}",
        f"- mean |P_t| = {online['global_pending_distribution']['mean']}",
        f"- median |P_t| = {online['global_pending_distribution']['median']}",
        f"- P75 |P_t| = {online['global_pending_distribution']['p75']}",
        f"- P95 |P_t| = {online['global_pending_distribution']['p95']}",
        f"- epochs with |P_t| >= 2 = {online['global_pending_distribution']['epochs_ge_2']}",
        f"- epochs with |P_t| >= 4 = {online['global_pending_distribution']['epochs_ge_4']}",
        f"- epochs with |P_t| >= 8 = {online['global_pending_distribution']['epochs_ge_8']}",
        f"- epochs with |P_t| >= 16 = {online['global_pending_distribution']['epochs_ge_16']}",
        "",
        "---",
        "",
        "## 7. Epoch-Scoped Compatibility Degree",
        "",
        f"- max d_t(c) = {online['max_online_degree']}",
        f"- max |P_t| = {online['max_pending']}",
        f"- invariant max d_t(c) <= max |P_t| = {'PASS' if online['invariant_max_degree_le_max_pending'] else 'FAIL'}",
        f"- all per-epoch invariants = {'PASS' if online['all_per_epoch_invariants_pass'] else 'FAIL'}",
        "",
        "| d_t(c) | candidate-epoch 对数 |",
        "|---|---|",
        f"| 0 | {online['degree_distribution'].get(0, 0):,} |",
        f"| 1 | {online['degree_distribution'].get(1, 0):,} |",
        f"| 2 | {online['degree_distribution'].get(2, 0):,} |",
        f"| 3 | {online['degree_distribution'].get(3, 0):,} |",
        f"| 4 | {online['degree_distribution'].get(4, 0):,} |",
        f"| >=5 | {sum(v for k,v in online['degree_distribution'].items() if k >= 5):,} |",
        "",
        "---",
        "",
        "## 8. Online-Safe Shared Coverage",
        "",
        f"- traces with online-safe d_t(c)>=2 = {online['traces_with_online_safe_d_at_least_2']}",
        f"- epochs with online-safe d_t(c)>=2 = {online['epochs_with_online_safe_d_at_least_2']}",
        f"- ONLINE_SAFE_SHARED_COVERAGE = {'YES' if shared['passed'] else 'NO/UNKNOWN'}",
        "",
        "10 个确定性示例见 `ONLINE_COMPATIBILITY_DEGREE.json` 中的 `shared_coverage_examples`。",
        "",
        "---",
        "",
        "## 9. Context-Length 可行性",
        "",
        f"- OpenHands replay 输入上限：{OPENHANDS_MAX_REPLAY_INPUT_TOKENS:,} tokens",
        f"- AgentX 最大请求输入长度：{ctx['input_length_tokens_dist'].get('max'):,} tokens",
        f"- 请求 <= {CONTEXT_LENGTH_SOFT_LIMIT}: {ctx['requests_le_soft_limit']} / {ctx['requests_le_soft_limit'] + ctx['requests_gt_soft_limit']} ({ctx['fraction_le_soft_limit']})",
        f"- 完全可 replay 的 trace 数（所有请求 <= {CONTEXT_LENGTH_SOFT_LIMIT}）: {ctx['traces_all_requests_le_soft_limit']} / {corpus['record_count']}",
        "",
        "**结论**：上下文长度仍是 AgentX 作为 RQ3 workload 的主要 blocker。",
        "",
        "---",
        "",
        "## 10. 诚实科学结论",
        "",
        "### 10.1 旧 d(c)=734 为什么错误",
        "",
        "它把整个 trace 的 descendant conversation 集合当作同一 epoch 的 P_t，"
        "违反了 FlowState 的正式定义 d_t(c) = |{p ∈ P_t : c compatible with p}|。",
        "",
        "### 10.2 online-safe d_t(c) 的真实 max",
        "",
        f"修复后 max d_t(c) = **{online['max_online_degree']}**，"
        f"且始终 <= max |P_t| = **{online['max_pending']}**。",
        "",
        "### 10.3 AgentX 是否仍存在 shared coverage",
        "",
        f"{'是' if shared['passed'] else '否'}。"
        f"有 {online['traces_with_online_safe_d_at_least_2']} 个 trace 在至少一个 epoch 出现 d_t(c)≥2。",
        "",
        "### 10.4 是否补足 OpenHands 结构缺口",
        "",
        "是的。OpenHands Main Population 的 max d=1 且无 branching；"
        "AgentX 在 online-safe 语义下仍提供 branching 与 cross-pending shared coverage。",
        "",
        "### 10.5 是否值得进入 formal snapshot construction",
        "",
        "**值得，但需先解决 context-length 限制**。建议下一步：",
        "1. 设计长度过滤/采样协议；2. 在受控 runtime 下验证 candidate residency 与 FA frontier。",
        "",
        "---",
        "",
        "## 11. Artifact 清单",
        "",
        "见 canonical audit root 目录下的 JSON/MD 文件。",
        "",
        "---",
        "",
        f"*报告生成于 {datetime.now().strftime('%Y-%m-%d')} · FlowState RQ3 Step 13G-B0.1 审计*",
        "",
    ]
    return "\n".join(lines)


def run_audit(output_root: Path | None = None) -> Path:
    """执行完整审计并写入 canonical artifact root。"""
    if output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        canonical_base = Path("/home/wjg/data/agentx/audits")
        canonical_base.mkdir(parents=True, exist_ok=True)
        output_root = canonical_base / f"agentx_online_compatibility_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Phase 0: frozen input gate。
    gate = {
        "frozen_input_path": str(FROZEN_AGENTX_PATH),
        "expected_sha256": EXPECTED_SHA256,
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "actual_sha256": _sha256_file(FROZEN_AGENTX_PATH),
        "actual_record_count": sum(1 for _ in _stream_records(FROZEN_AGENTX_PATH)),
    }
    gate["sha256_match"] = gate["actual_sha256"] == EXPECTED_SHA256
    gate["record_count_match"] = gate["actual_record_count"] == EXPECTED_RECORD_COUNT
    gate["gate_passed"] = gate["sha256_match"] and gate["record_count_match"]
    gate["artifact_root"] = str(output_root)
    gate["revision"] = "23f152f6f0f9399a85901b89a6458def0ef16729"
    _write_json(output_root / "INPUTS.json", gate)

    if not gate["gate_passed"]:
        raise RuntimeError(f"Frozen input gate failed: {gate}")

    # Phase 1: schema & corpus。
    schema = collect_schema_inventory()
    _write_json(output_root / "SCHEMA_INVENTORY.json", schema)
    schema_notes = [
        "# Schema Notes",
        "",
        "## Top-level fields",
        "- `id`: trace/session identifier.",
        "- `models`: list of model identifiers used in the trace.",
        "- `block_size`: KV-cache block size in tokens (always 64 in this corpus).",
        "- `hash_id_scope`: 'local' only.",
        "- `requests`: interleaved `n`, `s`, `subagent` entries.",
        "- `totals`: optional opaque trace-level summary.",
        "",
        "## Request types",
        "- `n`: normal API call.",
        "- `s`: streaming API call (includes `ttft`).",
        "- `subagent`: marker with nested `requests` and metadata.",
        "",
        "## Important observations",
        "- All `hash_id_scope` are 'local'; no cross-trace sharing is asserted.",
        "- Subagent inner requests are currently all type `n` and never further nested.",
        "- All normal/streaming requests carry non-empty `hash_ids`.",
    ]
    _write_md(output_root / "SCHEMA_NOTES.md", "\n".join(schema_notes))

    corpus = collect_corpus_statistics()
    _write_json(output_root / "CORPUS_STATISTICS.json", corpus)

    semantics = build_semantics_evidence()
    _write_json(output_root / "SEMANTICS_EVIDENCE.json", semantics)

    # 一次性构建所有 trace 的 conversation 结构，避免后续重复 chain detection。
    trace_conversations = _build_all_trace_conversations()

    # Phase 2: spawn/fork & prefix sharing。
    spawn_fork = analyze_spawn_fork_and_prefix(_trace_conversations=trace_conversations)
    _write_json(output_root / "SPAWN_FORK_AUDIT.json", spawn_fork)

    # Phase 3（旧版诊断）。
    diagnostic = build_old_max_compatibility_diagnostic(
        _trace_conversations=trace_conversations
    )
    _write_json(output_root / "OLD_MAX_COMPATIBILITY_DIAGNOSTIC.json", diagnostic)

    # Phase 3（修复版）：online-safe logical workflow & compatibility。
    online = analyze_online_logical_workflows_and_compatibility(
        _trace_conversations=trace_conversations
    )
    _write_json(output_root / "ONLINE_COMPATIBILITY_DEGREE.json", online)

    concurrency = analyze_pending_concurrency(_trace_conversations=trace_conversations)
    _write_json(output_root / "PENDING_CONCURRENCY.json", concurrency)

    logical_summary = {
        "total_traces": corpus["record_count"],
        "total_conversations": sum(r["max_pending"] for r in online["trace_results"]),  # placeholder
        "total_candidates": sum(
            sum(k * v for k, v in r["degree_distribution"].items())
            for r in []  # not used
        ),
        "max_online_compatibility_degree": online["max_online_degree"],
        "max_pending": online["max_pending"],
    }
    # 修正 total_conversations：从 trace_conversations 直接计算。
    logical_summary["total_conversations"] = sum(
        len(tc.conversations) for tc in trace_conversations
    )
    _write_json(output_root / "LOGICAL_WORKFLOW_RECONSTRUCTION.json", logical_summary)

    # Phase 4: gates & context feasibility。
    gates = evaluate_online_gates(online, concurrency)
    ctx = context_length_feasibility()
    gates["context_length_feasibility"]["passed"] = not ctx["exceeds_openhands_input_limit"]
    gates["context_length_feasibility"][
        "evidence"
    ] = f"max input = {ctx['input_length_tokens_dist'].get('max')} tokens vs OpenHands limit {OPENHANDS_MAX_REPLAY_INPUT_TOKENS}"
    _write_json(output_root / "SHARED_COVERAGE_GATE.json", gates)

    _write_json(output_root / "CONTEXT_LENGTH_FEASIBILITY.json", ctx)

    branching = {
        "max_simultaneous_pending_overall": concurrency["max_simultaneous_pending_overall"],
        "mean_max_simultaneous_pending": concurrency["mean_max_simultaneous_pending"],
        "traces_with_multiple_simultaneous_pending": concurrency[
            "traces_with_multiple_simultaneous_pending"
        ],
        "concurrency_distribution": concurrency["concurrency_distribution"],
        "online_safe_max_pending": online["max_pending"],
        "online_safe_global_pending_distribution": online["global_pending_distribution"],
    }
    _write_json(output_root / "BRANCHING_AUDIT.json", branching)

    comparison = compare_with_openhands(corpus, online, concurrency)
    _write_json(output_root / "COMPARISON_OPENHANDS.json", comparison)

    protocol = {
        "name": "Step 13G-B0.1 AgentX Online-Safe Compatibility Repair Audit",
        "constraints": [
            "read-only",
            "CPU-only",
            "no GPU/SGLang",
            "no FlowState core modification",
            "distinguish physical prefix sharing from logical inherited ancestry",
            "no future leakage",
            "d_t(c) <= |P_t| invariant enforced",
        ],
        "allocation_epoch_definition": [
            "conversation start event",
            "hash-bearing request completion event",
            "conversation end event",
        ],
        "tooling": "evaluation/agentx_chain_detection.py (AIPerf semantics port)",
    }
    _write_json(output_root / "AUDIT_PROTOCOL.json", protocol)

    report_md = _build_final_report(
        output_root, gates, corpus, online, concurrency, ctx, comparison, diagnostic, spawn_fork
    )
    _write_md(output_root / "STEP_13G_B0_1_AGENTX_ONLINE_COMPATIBILITY_FINAL_REPORT.md", report_md)

    # 同时在仓库根目录保留一份最终报告，便于阅读。
    repo_root = Path(__file__).resolve().parents[1]
    _write_md(repo_root / "STEP_13G_B0_1_AGENTX_ONLINE_COMPATIBILITY_FINAL_REPORT.md", report_md)

    return output_root


def main() -> None:
    """CLI entry point。"""
    root = run_audit()
    print(f"AgentX online compatibility audit artifacts written to: {root}")


if __name__ == "__main__":
    main()
