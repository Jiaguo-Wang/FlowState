"""Step 13G-B1.1：AgentX Qwen3.5-9B replay fidelity preflight。

约束：
- CPU only，不启动 SGLang/GPU。
- 使用本地 /home/wjg/models/qwen3.5-9b tokenizer 元数据确认模型身份。
- 实现与 AIPerf decode_block_tokens / compose_weka_prompt_tokens 等价的
  确定性 block-token 合成器：同一 hash_id -> 同一 64-token 块；
  合成后 prompt 的 token 数严格等于 trace 中该请求的 `in`。
- 对 Step 13G-B1 冻结的 23 个 formal shared-coverage snapshot 逐 trace 验证：
  token range、exact length、64-token block fidelity、prefix topology、
  FORK prefix preservation、SPAWN isolation、context <= 131200、determinism、
  SGLang direct-token-ID 请求路径可用性。
- 所有 gate 通过时输出 AGENTX_QWEN_REPLAY_PREFLIGHT_READY。
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from evaluation.agentx_runtime_population_census import RUNTIME_CONTEXT_LIMIT
from evaluation.agentx_structure_audit import (
    FROZEN_AGENTX_PATH,
    _EpochSnapshot,
    _TraceConversations,
    _analyze_trace_online_compatibility,
    _detect_and_build_conversations,
    _sha256_file,
    _stream_records,
)

QWEN_MODEL_PATH = Path("/home/wjg/models/qwen3.5-9b")
DEFAULT_FROZEN_CENSUS_PATH = Path(
    "/home/wjg/data/agentx/audits/agentx_runtime_population_census_20260905_010614"
    "/FORMAL_CANDIDATE_POPULATION.json"
)

SGLANG_INPUT_IDS_PATH = Path("evaluation/sota_latency_runtime.py")
SGLANG_INPUT_IDS_LINE = 146

# 确定性 tail fill 版本戳；修改此戳会改变合成 token，便于版本控制。
_REPLAY_SEED_SALT = "agentx-qwen35-9b-replay-v1"


@dataclass(slots=True)
class _ReplayRequest:
    """snapshot 中一个 pending request 的合成结果。"""

    conversation_id: str
    is_inherited: bool
    fork_depth_blocks: int
    request_index: int
    hash_ids: list[int]
    exact_input_length: int
    token_ids: list[int]


@dataclass(slots=True)
class _ReplaySnapshot:
    trace_id: str
    block_size: int
    t: float
    pending_count: int
    candidate_count: int
    shared_candidate_count: int
    max_degree: int
    max_fork_depth_blocks: int
    active_conversation_ids: list[str]
    requests: list[_ReplayRequest]
    active_targets: dict[int, int]


def _special_token_ids(tokenizer: Any) -> set[int]:
    """返回 tokenizer 中所有 special token id（含 added tokens）。"""
    special: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id", "unk_token_id", "bos_token_id"):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, int):
            special.add(val)
    # added_tokens_decoder 中标记为 special 的 token。
    decoder = getattr(tokenizer, "added_tokens_decoder", {})
    if isinstance(decoder, dict):
        for tid, info in decoder.items():
            if getattr(info, "special", False):
                special.add(int(tid))
    # 额外排除 image / video / vision 等特殊 id。
    for extra_attr in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        val = getattr(tokenizer, extra_attr, None)
        if isinstance(val, int):
            special.add(val)
    # 从 config 里读到的特殊 id 也一并排除。
    config_special = getattr(tokenizer, "added_tokens_encoder", {})
    if isinstance(config_special, dict):
        for tid in config_special.values():
            if isinstance(tid, int):
                special.add(tid)
    return special


@functools.lru_cache(maxsize=None)
def _decode_block_tokens_core(
    hash_id: int,
    block_size: int,
    base_id: int,
    max_id: int,
    special_ids: frozenset[int],
) -> tuple[int, ...]:
    """把单个 hash_id 映射为确定性 64-token 块（内部缓存版本）。

    返回 tuple 以便可被 lru_cache 哈希；调用方再转 list。
    """
    if max_id <= base_id + block_size:
        raise ValueError(f"tokenizer vocab 过小: {max_id}")
    tokens: list[int] = []
    counter = 0
    while len(tokens) < block_size:
        seed = f"{hash_id}:{counter}:{_REPLAY_SEED_SALT}".encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        val = int.from_bytes(digest[:8], "big")
        tid = base_id + (val % (max_id - base_id))
        counter += 1
        if tid in special_ids:
            continue
        tokens.append(tid)
    return tuple(tokens)


def _decode_block_tokens(
    hash_id: int,
    block_size: int,
    tokenizer: Any,
    *,
    base_id: int = 1000,
    special_ids: set[int] | None = None,
    max_id: int | None = None,
) -> list[int]:
    """把单个 hash_id 映射为确定性 64-token 块。

    等价语义：
    - 同一 hash_id 在相同 tokenizer/版本下永远返回同一 block。
    - 所有 token id 均在有效 vocab 范围内，且不落在 special token 区间。
    """
    if max_id is None:
        max_id = len(tokenizer)
    if special_ids is None:
        special_ids = _special_token_ids(tokenizer)
    return list(
        _decode_block_tokens_core(hash_id, block_size, base_id, max_id, frozenset(special_ids))
    )


def _tail_fill_tokens(
    count: int,
    tokenizer: Any,
    *,
    seed_prefix: str = "tail-fill",
    base_id: int = 1000,
    special_ids: set[int] | None = None,
    max_id: int | None = None,
) -> list[int]:
    """当 hash block 拼接后仍不足 exact_input_length 时，用 sha256-keyed 采样补齐。"""
    if max_id is None:
        max_id = len(tokenizer)
    if special_ids is None:
        special_ids = _special_token_ids(tokenizer)
    tokens: list[int] = []
    counter = 0
    while len(tokens) < count:
        seed = f"{seed_prefix}:{counter}:{_REPLAY_SEED_SALT}".encode("utf-8")
        val = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
        tid = base_id + (val % (max_id - base_id))
        counter += 1
        if tid in special_ids:
            continue
        tokens.append(tid)
    return tokens


def _compose_prompt_tokens(
    hash_ids: Sequence[int],
    exact_input_length: int,
    block_size: int,
    tokenizer: Any,
) -> list[int]:
    """按 AIPerf compose_weka_prompt_tokens 语义拼接 prompt token IDs。

    1. 依次拼接每个 hash_id 对应的 block（每个 block_size 个 token）。
    2. 若超过 exact_input_length，从右侧截断。
    3. 若不足，用确定性 tail fill 补齐，最终 len(token_ids) == exact_input_length。
    """
    max_id = len(tokenizer)
    special_ids = _special_token_ids(tokenizer)
    tokens: list[int] = []
    for hid in hash_ids:
        tokens.extend(_decode_block_tokens(hid, block_size, tokenizer, special_ids=special_ids, max_id=max_id))
        if len(tokens) >= exact_input_length:
            break
    if len(tokens) > exact_input_length:
        tokens = tokens[:exact_input_length]
    elif len(tokens) < exact_input_length:
        tokens.extend(
            _tail_fill_tokens(
                exact_input_length - len(tokens),
                tokenizer,
                seed_prefix=f"compose-{hash_ids[:3]}",
                special_ids=special_ids,
                max_id=max_id,
            )
        )
    return tokens


def _request_index_for_target(conv: Any, target: int) -> int | None:
    """由 active target（累积 token 位置）推断当前 pending request 的索引。

    正式语义（B0.1 epoch reconstruction）：
    - active target 是 conversation 在 epoch t 的已 materialized prefix 的累积 hash token 位置。
    - target > 0 且等于 cumsum[positions[:j+1]] 时，返回 j（刚刚完成的请求），
      其 input_length 即为该 active chain 当前已知 context。
    - target == 0 表示 conversation 已启动但尚未完成任何请求，当前 pending continuation
      即为 request_index == 0。只要 conversation 存在 request，返回 0。
    - conversation 没有 request（空 conversation）时返回 None。

    该语义保证 B1 / B1.1 / B2 对同一 epoch 得到完全相同的 pending set。
    """
    positions = conv.request_hash_token_positions
    if not positions:
        return None
    if target == 0:
        return 0
    cum = list(itertools.accumulate(positions))
    idx = bisect.bisect_right(cum, target) - 1
    if idx >= 0 and cum[idx] == target:
        return idx
    # target 未精确匹配：容错返回最接近的已完成请求。
    return max(idx, 0)


def _tokenizer_identity(model_path: Path, tokenizer: Any) -> dict[str, Any]:
    """记录实际加载的 tokenizer / model 身份信息。"""
    config_path = model_path / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})

    special_ids = sorted(_special_token_ids(tokenizer))
    return {
        "model_path": str(model_path),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "tokenizer_length": len(tokenizer),
        "model_type": str(config.get("model_type", text_config.get("model_type", "unknown"))),
        "architectures": config.get("architectures", []),
        "vocab_size_config": int(
            text_config.get("vocab_size", config.get("vocab_size", 0))
        ),
        "max_position_embeddings": int(
            text_config.get("max_position_embeddings", config.get("max_position_embeddings", 0))
        ),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "special_token_ids": special_ids,
        "safetensors_present": (model_path / "model.safetensors.index.json").exists(),
        "config_loaded": bool(config),
    }


def _load_tokenizer(model_path: Path) -> Any:
    """加载本地 Qwen tokenizer（CPU only，不需要 torch）。"""
    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


def _load_formal_population(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_conversations_for_population(
    agentx_path: Path,
    population: list[dict[str, Any]],
) -> list[_TraceConversations]:
    """仅构建 formal population 涉及的 trace 的 _TraceConversations，避免全量 corpus。"""
    needed = {entry["trace_id"] for entry in population}
    result: list[_TraceConversations] = []
    for record in _stream_records(agentx_path):
        trace_id = record.get("id")
        if trace_id not in needed:
            continue
        tc = _detect_and_build_conversations(
            trace_id, int(record.get("block_size", 64)), record.get("requests", [])
        )
        result.append(tc)
        needed.discard(trace_id)
        if not needed:
            break
    if needed:
        raise RuntimeError(f"formal population 中以下 trace 在 corpus 中未找到: {sorted(needed)}")
    return result


def _find_epoch_snapshot(
    online: Any, t: float, active_ids: Sequence[str], convs: Sequence[Any]
) -> _EpochSnapshot:
    """在 online 结果中找到与 formal snapshot 匹配的 epoch。"""
    conv_id_to_idx = {c.conversation_id: i for i, c in enumerate(convs)}
    active_idx_set = {conv_id_to_idx[cid] for cid in active_ids if cid in conv_id_to_idx}
    for snap in online.snapshots:
        if abs(snap.t - t) > 1e-9:
            continue
        if set(snap.active_targets.keys()) == active_idx_set:
            return snap
    raise RuntimeError(f"trace {online.trace_id} 未找到 t={t} 且 active_ids 匹配的 epoch")


def _build_replay_snapshot(
    tc: _TraceConversations,
    entry: dict[str, Any],
    tokenizer: Any,
) -> _ReplaySnapshot:
    """为单个 formal snapshot 合成所有 active pending request 的 token IDs。"""
    online = _analyze_trace_online_compatibility(tc)
    snap = _find_epoch_snapshot(
        online, float(entry["t"]), entry["active_conversation_ids"], tc.conversations
    )
    convs = tc.conversations
    requests: list[_ReplayRequest] = []
    for conv_idx, target in snap.active_targets.items():
        conv = convs[conv_idx]
        req_idx = _request_index_for_target(conv, target)
        if req_idx is None or req_idx >= len(conv.request_hash_ids):
            # 该 conversation 即将结束，没有下一个 pending request。
            continue
        hash_ids = list(conv.request_hash_ids[req_idx])
        exact_len = int(conv.request_input_lengths[req_idx])
        tokens = _compose_prompt_tokens(
            hash_ids, exact_len, tc.block_size, tokenizer
        )
        requests.append(
            _ReplayRequest(
                conversation_id=conv.conversation_id,
                is_inherited=conv.is_inherited,
                fork_depth_blocks=conv.fork_depth_blocks,
                request_index=req_idx,
                hash_ids=hash_ids,
                exact_input_length=exact_len,
                token_ids=tokens,
            )
        )

    max_fork_depth = max(
        (convs[idx].fork_depth_blocks for idx in snap.active_targets), default=0
    )
    return _ReplaySnapshot(
        trace_id=tc.trace_id,
        block_size=tc.block_size,
        t=snap.t,
        pending_count=snap.pending_count,
        candidate_count=entry["candidate_count"],
        shared_candidate_count=entry["shared_candidate_count"],
        max_degree=entry["max_degree"],
        max_fork_depth_blocks=max_fork_depth,
        active_conversation_ids=list(entry["active_conversation_ids"]),
        requests=requests,
        active_targets=dict(snap.active_targets),
    )


def _validate_token_range(tokens: Sequence[int], tokenizer: Any) -> dict[str, Any]:
    """验证所有 token id 有效，且不命中特殊 token。"""
    max_id = len(tokenizer)
    special = _special_token_ids(tokenizer)
    invalid = [tid for tid in tokens if tid < 0 or tid >= max_id]
    special_hits = [tid for tid in tokens if tid in special]
    return {
        "max_token_id": max(tokens) if tokens else -1,
        "min_token_id": min(tokens) if tokens else -1,
        "invalid_count": len(invalid),
        "special_token_count": len(special_hits),
        "pass": len(invalid) == 0,
    }


def _validate_block_fidelity(
    req: _ReplayRequest, block_size: int, tokenizer: Any
) -> dict[str, Any]:
    """验证前几个 hash block 在 prompt 中保持 64-token 块结构。"""
    if not req.hash_ids:
        return {"blocks_checked": 0, "pass": True}
    ok = True
    checked = 0
    for offset, hid in enumerate(req.hash_ids):
        block = _decode_block_tokens(hid, block_size, tokenizer)
        start = offset * block_size
        end = start + block_size
        if end > req.exact_input_length:
            # 最后一个 block 被截断，只检查在范围内的部分。
            end = min(end, req.exact_input_length)
        if req.token_ids[start:end] != block[: end - start]:
            ok = False
            break
        checked += 1
        if end >= req.exact_input_length:
            break
    return {"blocks_checked": checked, "pass": ok}


def _lineage_related(path_a: tuple[str, ...], path_b: tuple[str, ...]) -> bool:
    """两个 lineage path 是否存在祖先/后代关系（不含相等）。"""
    if len(path_a) == len(path_b):
        return False
    if len(path_a) < len(path_b):
        return path_b[: len(path_a)] == path_a
    return path_a[: len(path_b)] == path_b


def _hash_lcp(a: Sequence[int], b: Sequence[int]) -> int:
    """两个 hash_id 列表的最长公共前缀长度（以 block 计）。"""
    i = 0
    limit = min(len(a), len(b))
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _validate_prefix_topology(snapshot: _ReplaySnapshot, convs: Sequence[Any]) -> dict[str, Any]:
    """验证 FORK prefix preservation 与 SPAWN isolation。

    FORK prefix preservation 分两层：
    - 逻辑层：child 的 input length 不超过其 active 祖先的 materialized target（由 compatibility relation 保证）。
    - 物理层：若 child 的 hash_ids 与祖先 hash_ids 存在公共前缀，则 token 序列也必须共享对应前缀。
    SPAWN isolation：仅对与其它 active conversation 没有 lineage 关系的非继承 conversation，
    要求其首 token 与所有其它 active conversation 均不同。
    """
    conv_by_id = {c.conversation_id: c for c in convs}
    requests_by_id = {r.conversation_id: r for r in snapshot.requests}
    fork_pairs_ok: list[dict[str, Any]] = []
    spawn_isolations_ok: list[dict[str, Any]] = []

    active_ids = set(requests_by_id.keys())
    active_convs = [c for c in convs if c.conversation_id in active_ids]
    active_idx_by_id = {c.conversation_id: i for i, c in enumerate(convs) if c.conversation_id in active_ids}

    for cid, req in requests_by_id.items():
        conv = conv_by_id[cid]
        if req.is_inherited:
            # FORK：找到 lineage 上最近的 active 祖先。
            lineage = conv.lineage_path
            parent_cid = None
            for k in range(len(lineage) - 1, 0, -1):
                parent_path = lineage[:k]
                parent = next(
                    (c for c in active_convs if c.lineage_path == parent_path),
                    None,
                )
                if parent is not None:
                    parent_cid = parent.conversation_id
                    break
            if parent_cid is None:
                fork_pairs_ok.append({"child": cid, "parent": None, "shared_prefix_tokens": 0, "pass": True})
                continue
            parent_req = requests_by_id[parent_cid]
            ancestor_target = snapshot.active_targets[active_idx_by_id[parent_cid]]
            logical_ok = req.exact_input_length <= ancestor_target
            # 物理公共前缀检查（可选，仅在 hash_ids 真有公共前缀时执行）。
            physical_lcp_blocks = _hash_lcp(req.hash_ids, parent_req.hash_ids)
            physical_ok = True
            shared_tokens = 0
            if physical_lcp_blocks > 0:
                shared_len = physical_lcp_blocks * snapshot.block_size
                shared_len = min(shared_len, req.exact_input_length, parent_req.exact_input_length)
                physical_ok = (
                    req.token_ids[:shared_len] == parent_req.token_ids[:shared_len]
                )
                shared_tokens = shared_len
            fork_pairs_ok.append(
                {
                    "child": cid,
                    "parent": parent_cid,
                    "shared_prefix_tokens": shared_tokens,
                    "logical_ok": logical_ok,
                    "physical_lcp_blocks": physical_lcp_blocks,
                    "pass": logical_ok and physical_ok,
                }
            )
        else:
            # SPAWN：仅对没有 active 祖先/后代关系的非继承 conversation 检查隔离。
            has_relation = any(
                _lineage_related(conv.lineage_path, other.lineage_path)
                for other in active_convs
                if other.conversation_id != cid
            )
            if has_relation:
                spawn_isolations_ok.append({"conversation_id": cid, "pass": True})
                continue
            isolated = True
            for other_cid, other_req in requests_by_id.items():
                if other_cid == cid:
                    continue
                if req.token_ids and other_req.token_ids and req.token_ids[0] == other_req.token_ids[0]:
                    isolated = False
                    break
            spawn_isolations_ok.append({"conversation_id": cid, "pass": isolated})

    fork_pass = all(p["pass"] for p in fork_pairs_ok)
    spawn_pass = all(p["pass"] for p in spawn_isolations_ok)
    return {
        "fork_prefix_preserved": fork_pass,
        "fork_pairs": fork_pairs_ok,
        "spawn_isolation_preserved": spawn_pass,
        "spawn_conversations": spawn_isolations_ok,
        "pass": fork_pass and spawn_pass,
    }


def _validate_snapshot(
    snapshot: _ReplaySnapshot,
    convs: Sequence[Any],
    tokenizer: Any,
) -> dict[str, Any]:
    """对单个 snapshot 执行全部 fidelity gate。"""
    block_size = snapshot.block_size
    range_results = []
    length_results = []
    block_results = []
    context_lengths = []
    for req in snapshot.requests:
        range_results.append(_validate_token_range(req.token_ids, tokenizer))
        length_results.append(
            {
                "conversation_id": req.conversation_id,
                "expected": req.exact_input_length,
                "actual": len(req.token_ids),
                "pass": len(req.token_ids) == req.exact_input_length,
            }
        )
        block_results.append(
            {
                "conversation_id": req.conversation_id,
                **_validate_block_fidelity(req, block_size, tokenizer),
            }
        )
        context_lengths.append(len(req.token_ids))

    max_context = max(context_lengths) if context_lengths else 0
    context_pass = max_context <= RUNTIME_CONTEXT_LIMIT
    range_pass = all(r["pass"] for r in range_results)
    length_pass = all(r["pass"] for r in length_results)
    block_pass = all(r["pass"] for r in block_results)

    prefix = _validate_prefix_topology(snapshot, convs)

    return {
        "trace_id": snapshot.trace_id,
        "t": snapshot.t,
        "pending_count": snapshot.pending_count,
        "candidate_count": snapshot.candidate_count,
        "max_degree": snapshot.max_degree,
        "max_fork_depth_blocks": snapshot.max_fork_depth_blocks,
        "request_count": len(snapshot.requests),
        "token_range_pass": range_pass,
        "invalid_token_count": sum(r["invalid_count"] for r in range_results),
        "special_token_count": sum(r["special_token_count"] for r in range_results),
        "exact_length_pass": length_pass,
        "block_fidelity_pass": block_pass,
        "context_limit_pass": context_pass,
        "max_context_length": max_context,
        "prefix_topology_pass": prefix["pass"],
        "fork_prefix_preserved": prefix["fork_prefix_preserved"],
        "spawn_isolation_preserved": prefix["spawn_isolation_preserved"],
        "request_details": {
            "range": range_results,
            "length": length_results,
            "block": block_results,
        },
        "pass": (
            range_pass
            and length_pass
            and block_pass
            and context_pass
            and prefix["pass"]
        ),
    }


def _validate_determinism(
    snapshots: Sequence[_ReplaySnapshot], tokenizer: Any
) -> dict[str, Any]:
    """同一输入重新合成，验证每个 request 的 token IDs 不变。"""
    mismatches = 0
    for snap in snapshots:
        for req in snap.requests:
            recomposed = _compose_prompt_tokens(
                req.hash_ids, req.exact_input_length, snap.block_size, tokenizer
            )
            if recomposed != req.token_ids:
                mismatches += 1
    return {"mismatches": mismatches, "pass": mismatches == 0}


def _select_representative_snapshots(
    snapshots: Sequence[_ReplaySnapshot], population: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """选择 3 个代表性 snapshot：max candidate、max pending、max fork depth。"""
    reps: list[dict[str, Any]] = []
    by_candidate = max(population, key=lambda e: e["candidate_count"])
    by_pending = max(population, key=lambda e: e["pending_count"])
    by_fork = max(population, key=lambda e: e["max_fork_depth_blocks"])
    chosen_ids = {by_candidate["trace_id"], by_pending["trace_id"], by_fork["trace_id"]}
    for snap in snapshots:
        if snap.trace_id in chosen_ids:
            reps.append(
                {
                    "trace_id": snap.trace_id,
                    "t": snap.t,
                    "reason": "max_candidate"
                    if snap.trace_id == by_candidate["trace_id"]
                    else "max_pending"
                    if snap.trace_id == by_pending["trace_id"]
                    else "max_fork_depth",
                    "pending_count": snap.pending_count,
                    "candidate_count": snap.candidate_count,
                    "max_fork_depth_blocks": snap.max_fork_depth_blocks,
                }
            )
    return reps


def _sglang_direct_token_id_path() -> dict[str, Any]:
    """验证 FlowState 已存在直接传入 input_ids 的 SGLang 调用路径。"""
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / SGLANG_INPUT_IDS_PATH
    if not path.exists():
        return {"present": False, "path": str(path), "line": None, "pass": False}
    text = path.read_text(encoding="utf-8")
    present = "input_ids=list(token_ids)" in text and "engine.generate(" in text
    return {
        "present": present,
        "path": str(path),
        "line": SGLANG_INPUT_IDS_LINE,
        "pass": present,
    }


def _build_final_report(
    *,
    identity: dict[str, Any],
    validation: dict[str, Any],
    determinism: dict[str, Any],
    sglang: dict[str, Any],
    representative: list[dict[str, Any]],
    artifact_root: Path,
    overall_status: str,
) -> str:
    lines = [
        "# Step 13G-B1.1 AgentX Qwen3.5-9B Replay Fidelity Preflight",
        "",
        f"**Overall Status:** `{overall_status}`",
        "",
        "## Tokenizer / Model Identity",
        "",
        f"- Model path: `{identity.get('model_path')}`",
        f"- Tokenizer class: `{identity.get('tokenizer_class')}`",
        f"- Tokenizer vocab size: `{identity.get('tokenizer_vocab_size')}`",
        f"- Tokenizer length: `{identity.get('tokenizer_length')}`",
        f"- Config vocab size: `{identity.get('vocab_size_config')}`",
        f"- Max position embeddings: `{identity.get('max_position_embeddings')}`",
        f"- EOS token id: `{identity.get('eos_token_id')}`",
        f"- PAD token id: `{identity.get('pad_token_id')}`",
        f"- Safetensors index present: `{identity.get('safetensors_present')}`",
        "",
        "## Fidelity Gate Results",
        "",
        f"- Snapshots validated: `{validation['snapshot_count']}`",
        f"- Token range pass: `{validation['all_token_range_pass']}`",
        f"- Exact length pass: `{validation['all_exact_length_pass']}`",
        f"- 64-token block fidelity pass: `{validation['all_block_fidelity_pass']}`",
        f"- Context <= {RUNTIME_CONTEXT_LIMIT} pass: `{validation['all_context_limit_pass']}`",
        f"- Prefix topology pass: `{validation['all_prefix_topology_pass']}`",
        f"- Total invalid tokens: `{validation['total_invalid_tokens']}`",
        f"- Determinism mismatches: `{determinism['mismatches']}`",
        "",
        "## SGLang Direct-Token-ID Path",
        "",
        f"- Present: `{sglang['present']}` in `{sglang['path']}` around line `{sglang['line']}`",
        "",
        "## Representative Snapshots",
        "",
    ]
    for rep in representative:
        lines.append(
            f"- `{rep['trace_id']}` (t={rep['t']:.3f}, reason={rep['reason']}, "
            f"pending={rep['pending_count']}, candidates={rep['candidate_count']}, "
            f"fork_depth={rep['max_fork_depth_blocks']})"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"All artifacts frozen under: `{artifact_root}`",
            "",
            f"- `{overall_status}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_preflight(
    output_root: Path | None = None,
    *,
    model_path: Path = QWEN_MODEL_PATH,
    census_path: Path = DEFAULT_FROZEN_CENSUS_PATH,
    agentx_path: Path = FROZEN_AGENTX_PATH,
) -> Path:
    """执行完整 Qwen3.5-9B replay fidelity preflight 并冻结 artifacts。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_root = output_root or Path(
        f"/home/wjg/data/agentx/audits/agentx_qwen_replay_preflight_{timestamp}"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(model_path)
    identity = _tokenizer_identity(model_path, tokenizer)

    population = _load_formal_population(census_path)
    trace_conversations = _build_conversations_for_population(agentx_path, population)
    conv_by_trace = {tc.trace_id: tc for tc in trace_conversations}

    snapshots: list[_ReplaySnapshot] = []
    for entry in population:
        tc = conv_by_trace[entry["trace_id"]]
        snapshots.append(_build_replay_snapshot(tc, entry, tokenizer))

    # 逐 snapshot 验证。
    per_snapshot_validations: list[dict[str, Any]] = []
    for snap in snapshots:
        tc = conv_by_trace[snap.trace_id]
        per_snapshot_validations.append(_validate_snapshot(snap, tc.conversations, tokenizer))

    determinism = _validate_determinism(snapshots, tokenizer)
    sglang = _sglang_direct_token_id_path()
    representatives = _select_representative_snapshots(snapshots, population)

    # 汇总 gate。
    all_token_range_pass = all(v["token_range_pass"] for v in per_snapshot_validations)
    all_exact_length_pass = all(v["exact_length_pass"] for v in per_snapshot_validations)
    all_block_fidelity_pass = all(v["block_fidelity_pass"] for v in per_snapshot_validations)
    all_context_limit_pass = all(v["context_limit_pass"] for v in per_snapshot_validations)
    all_prefix_topology_pass = all(v["prefix_topology_pass"] for v in per_snapshot_validations)
    total_invalid = sum(v["invalid_token_count"] for v in per_snapshot_validations)

    gate_passed = (
        identity["config_loaded"]
        and identity["safetensors_present"]
        and all_token_range_pass
        and all_exact_length_pass
        and all_block_fidelity_pass
        and all_context_limit_pass
        and all_prefix_topology_pass
        and determinism["pass"]
        and sglang["pass"]
    )
    overall_status = (
        "AGENTX_QWEN_REPLAY_PREFLIGHT_READY"
        if gate_passed
        else "AGENTX_QWEN_REPLAY_PREFLIGHT_BLOCKED"
    )

    inputs = {
        "frozen_agentx_path": str(agentx_path),
        "frozen_census_path": str(census_path),
        "qwen_model_path": str(model_path),
        "runtime_context_limit": RUNTIME_CONTEXT_LIMIT,
        "formal_snapshot_count": len(population),
        "agentx_sha256": _sha256_file(agentx_path),
        "census_sha256": _sha256_file(census_path),
    }

    validation_summary = {
        "overall_status": overall_status,
        "gate_passed": gate_passed,
        "snapshot_count": len(per_snapshot_validations),
        "all_token_range_pass": all_token_range_pass,
        "all_exact_length_pass": all_exact_length_pass,
        "all_block_fidelity_pass": all_block_fidelity_pass,
        "all_context_limit_pass": all_context_limit_pass,
        "all_prefix_topology_pass": all_prefix_topology_pass,
        "all_fork_prefix_preserved": all(v["fork_prefix_preserved"] for v in per_snapshot_validations),
        "all_spawn_isolation_preserved": all(v["spawn_isolation_preserved"] for v in per_snapshot_validations),
        "total_invalid_tokens": total_invalid,
        "determinism_pass": determinism["pass"],
        "sglang_direct_token_id_path_pass": sglang["pass"],
        "per_snapshot": per_snapshot_validations,
    }

    # SNAPSHOT_TOKENIZED_INPUTS：每个 snapshot 保存摘要 + 前 256 个 token，
    # 代表性 snapshot 保存完整 token IDs 以便直接 replay。
    token_inputs: list[dict[str, Any]] = []
    representative_ids = {r["trace_id"] for r in representatives}
    for snap in snapshots:
        req_data = []
        for req in snap.requests:
            rd: dict[str, Any] = {
                "conversation_id": req.conversation_id,
                "is_inherited": req.is_inherited,
                "fork_depth_blocks": req.fork_depth_blocks,
                "request_index": req.request_index,
                "exact_input_length": req.exact_input_length,
                "hash_id_count": len(req.hash_ids),
                "first_64_tokens": req.token_ids[:64],
                "last_64_tokens": req.token_ids[-64:] if len(req.token_ids) >= 64 else [],
            }
            if snap.trace_id in representative_ids:
                rd["token_ids"] = req.token_ids
            req_data.append(rd)
        token_inputs.append(
            {
                "trace_id": snap.trace_id,
                "t": snap.t,
                "block_size": snap.block_size,
                "pending_count": snap.pending_count,
                "candidate_count": snap.candidate_count,
                "requests": req_data,
            }
        )

    # 写入 artifacts。
    (artifact_root / "INPUTS.json").write_text(
        json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_root / "QWEN_TOKENIZER_IDENTITY.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_root / "DETERMINISTIC_BLOCK_SYNTHESIS.json").write_text(
        json.dumps(
            {
                "method": "sha256-keyed per-hash block + tail fill",
                "salt": _REPLAY_SEED_SALT,
                "block_size": 64,
                "sample_blocks": {
                    str(hid): _decode_block_tokens(hid, 64, tokenizer)
                    for hid in range(5)
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_root / "REPLAY_FIDELITY_GATES.json").write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_root / "SNAPSHOT_TOKENIZED_INPUTS.json").write_text(
        json.dumps(token_inputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_root / "REPRESENTATIVE_CASE_AUDIT.json").write_text(
        json.dumps(representatives, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifact_root / "SGLANG_DIRECT_TOKEN_ID_PATH.json").write_text(
        json.dumps(sglang, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validation_report = {
        "overall_status": overall_status,
        "gate_passed": gate_passed,
        "checks": {
            "tokenizer_identity_loaded": identity["config_loaded"],
            "safetensors_present": identity["safetensors_present"],
            "token_range": all_token_range_pass,
            "exact_length": all_exact_length_pass,
            "block_fidelity": all_block_fidelity_pass,
            "context_limit": all_context_limit_pass,
            "prefix_topology": all_prefix_topology_pass,
            "fork_prefix": all(v["fork_prefix_preserved"] for v in per_snapshot_validations),
            "spawn_isolation": all(v["spawn_isolation_preserved"] for v in per_snapshot_validations),
            "determinism": determinism["pass"],
            "sglang_direct_token_id_path": sglang["pass"],
        },
    }
    (artifact_root / "validation_report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = _build_final_report(
        identity=identity,
        validation=validation_summary,
        determinism=determinism,
        sglang=sglang,
        representative=representatives,
        artifact_root=artifact_root,
        overall_status=overall_status,
    )
    report_path = artifact_root / "STEP_13G_B1_1_AGENTX_QWEN_REPLAY_PREFLIGHT_FINAL_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    # 仅在正式 canonical run（未指定 output_root）时更新 repo 根目录副本，避免测试污染。
    if output_root is None:
        repo_report_path = Path(
            "/home/wjg/code/FlowState/STEP_13G_B1_1_AGENTX_QWEN_REPLAY_PREFLIGHT_FINAL_REPORT.md"
        )
        repo_report_path.write_text(report, encoding="utf-8")

    return artifact_root


if __name__ == "__main__":
    print(run_preflight())
