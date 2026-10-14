#!/usr/bin/env python3
"""构造最小合法共享样本（Task A2）。

样本结构（Shared Prefix H → Checkpoint C0 → Fork → P1 / P2）：

- H 的 token 序列来自**真实 AgentX 冻结语料**（cc-traces-weka-062126 traces.jsonl）
  中某条真实请求的 hash_ids 前缀，经与正式 B2 采集**完全相同**的构造语义
  （evaluation.agentx_qwen_replay_preflight._compose_prompt_tokens，
  Qwen3.5-9B 真实 tokenizer）物化——即 H 就是正式 573 案例回放时会物化的
  那类 token 序列；
- C0 = materialize(H) 完成后 radix 树在 **H 输入边界（L = 8192）** 处的
  exact 节点。与正式采集的 candidate 位置约定一致（checkpoint 位于已完成
  请求的输入末尾；该 runtime 不把生成 token 插入树路径——首次冒烟实测
  8192/8193 处无子边，正是 collection 用 token_pos 边界 inspect 的原因）；
- P1 / P2 = H + 各自不同的 CONTROLLED 后缀（63 token，与正式采集
  PENDING_SUFFIX_LENGTH 一致）；两个后缀为确定性合成序列，
  **显式标记 CONTROLLED**，不冒充原生 AgentX 分支；
- P_bad = 在 H 内部（L 之前，块对齐位置 L0//2）翻转一个 token 的非法
  continuation，用于验证谓词在准入阶段即拒绝历史分歧的请求。

产物：evaluation/submission_p0_executable_recovery/compatibility_cases.json

本脚本只读语料、不做任何 GPU 操作；需在具备 transformers 与冻结语料
挂载的环境中执行（见 run_minimal_recovery_smoke.sh --build-only）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPOSITORY_ROOT))

from evaluation.agentx_qwen_replay_preflight import (  # noqa: E402
    QWEN_MODEL_PATH,
    _compose_prompt_tokens,
)

# 冻结语料与正式采集脚本（agentx_runtime_snapshot_collection.py）保持一致。
FROZEN_AGENTX_PATH = Path(
    "/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl"
)
# H 取 128 个 hash 块 × 64 token/块 = 8192 token：
# 足够非平凡（远超 mamba_track_interval=256），又保证最小 runtime 冒烟可在
# 45056 context 的受控引擎配置内快速完成。
PREFIX_BLOCK_COUNT = 128
BLOCK_SIZE = 64
# 分支后缀长度 256（而非正式采集的 63）：使 continuation 的输入终止在
# 8192+256=8448 —— 恰为 mamba track_interval=256 的边界。首次冒烟实测：
# 63+1 token 的分支终止在 8256（非边界），终止 Mamba 状态不落盘、census
# 无新行；对齐边界后各分支终止状态才会物化为独立节点，供隔离性断言使用。
PENDING_SUFFIX_LENGTH = 256
OUTPUT_PATH = (
    _REPOSITORY_ROOT
    / "evaluation"
    / "submission_p0_executable_recovery"
    / "compatibility_cases.json"
)


def _sha256_file(path: Path) -> str:
    """计算文件内容摘要，用于模型/tokenizer 身份记录。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer():
    """加载真实 Qwen3.5-9B tokenizer（与正式回放同一本地路径）。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(QWEN_MODEL_PATH))


def _first_hashed_request(trace: dict) -> tuple[int, dict]:
    """返回 trace 中第一条携带 hash_ids 的普通（非 subagent）请求。"""
    for index, request in enumerate(trace.get("requests", [])):
        if request.get("type") == "subagent":
            continue
        hash_ids = request.get("hash_ids") or []
        if len(hash_ids) >= PREFIX_BLOCK_COUNT:
            return index, request
    raise RuntimeError("首条 trace 中没有足够长的普通请求")


def _controlled_suffix(
    label: str,
    count: int,
    vocab_size: int,
    special_ids: frozenset[int],
) -> list[int]:
    """构造确定性 CONTROLLED 后缀：sha256(keyed) 采样，避开 special token。

    与语料内容无关，仅用于在 L 之后制造两个互不相同的合法分支。
    """
    tokens: list[int] = []
    counter = 0
    base_id = 1000  # 与 _tail_fill_tokens 相同的非 special 起始区
    while len(tokens) < count:
        seed = f"p0-minimal-suffix-{label}:{counter}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
        token_id = base_id + (value % (vocab_size - base_id))
        counter += 1
        if token_id in special_ids:
            continue
        tokens.append(token_id)
    return tokens


def token_digest(token_ids) -> str:
    """与正式运行时句柄相同的 token 摘要算法。"""
    return hashlib.sha256(array("q", token_ids).tobytes()).hexdigest()


def main() -> int:
    tokenizer = _load_tokenizer()
    vocab_size = len(tokenizer)
    special_ids = frozenset(int(x) for x in tokenizer.all_special_ids)

    # ---- 读取冻结语料首条 trace，取真实请求的 hash_ids 前缀 ----
    with open(FROZEN_AGENTX_PATH, "r", encoding="utf-8") as handle:
        first_line = handle.readline()
    trace = json.loads(first_line)
    request_index, request = _first_hashed_request(trace)
    hash_slice = list(request["hash_ids"])[:PREFIX_BLOCK_COUNT]

    # ---- 用正式合成语义物化 H（8192 token，恰好 128 块，无 tail-fill）----
    h_tokens = _compose_prompt_tokens(
        hash_slice,
        PREFIX_BLOCK_COUNT * BLOCK_SIZE,
        BLOCK_SIZE,
        tokenizer,
    )
    if len(h_tokens) != PREFIX_BLOCK_COUNT * BLOCK_SIZE:
        raise RuntimeError(f"H 长度异常: {len(h_tokens)}")

    # ---- 三个互不相同的 CONTROLLED 分支后缀 ----
    # P1/P2 为合法分支；P3 用于"驱逐反事实"：evict C0 的 Mamba 状态后，
    # P3 的可执行前缀必须塌缩，证明 P1/P2 的恢复确实依赖 Recurrent State。
    suffix1 = _controlled_suffix("P1", PENDING_SUFFIX_LENGTH, vocab_size, special_ids)
    suffix2 = _controlled_suffix("P2", PENDING_SUFFIX_LENGTH, vocab_size, special_ids)
    suffix3 = _controlled_suffix("P3", PENDING_SUFFIX_LENGTH, vocab_size, special_ids)
    if len({tuple(s) for s in (suffix1, suffix2, suffix3)}) != 3:
        raise RuntimeError("P1/P2/P3 后缀不应相同")

    # ---- 非法样本：在 L 之前（块对齐的 L0//2 处）翻转一个 token ----
    mutation_position = (len(h_tokens) // 2 // BLOCK_SIZE) * BLOCK_SIZE
    original_token = int(h_tokens[mutation_position])
    mutated_token = original_token + 1
    if mutated_token in special_ids:
        mutated_token += 1
    if not (0 <= mutated_token < vocab_size):
        raise RuntimeError("翻转后的 token 超出词表")
    h_mutated = (
        list(h_tokens[:mutation_position])
        + [mutated_token]
        + list(h_tokens[mutation_position + 1 :])
    )

    model_directory = Path(QWEN_MODEL_PATH)
    identity_files = {
        "config.json": _sha256_file(model_directory / "config.json"),
        "tokenizer_config.json": _sha256_file(
            model_directory / "tokenizer_config.json"
        ),
        "tokenizer.json": _sha256_file(model_directory / "tokenizer.json"),
    }
    model_identity = f"qwen3.5-9b@{identity_files['config.json'][:16]}"
    tokenizer_identity = (
        f"qwen3.5-9b-tokenizer@"
        f"{identity_files['tokenizer_config.json'][:16]}"
        f":vocab={vocab_size}"
    )

    l_zero = len(h_tokens)  # H 的长度；真实状态位置 L = L0 + 1（含 anchor_output）

    cases = {
        "schema_version": "1",
        "task": "FlowState 投稿冲刺任务 A2：最小合法共享样本",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "agentx_corpus": str(FROZEN_AGENTX_PATH),
            "trace_index": 0,
            "trace_id": trace.get("id"),
            "request_index": request_index,
            "request_t": request.get("t"),
            "request_model": request.get("model"),
            "hash_block_range": [0, PREFIX_BLOCK_COUNT],
            "block_size": BLOCK_SIZE,
            "compose_semantics": (
                "evaluation.agentx_qwen_replay_preflight._compose_prompt_tokens"
                "（与正式 B2 采集相同；128 块恰好 8192 token，无 tail-fill）"
            ),
            "h_digest": token_digest(h_tokens),
            "h_mutated_digest": token_digest(h_mutated),
        },
        "identity": {
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "model_path": str(QWEN_MODEL_PATH),
            "identity_file_sha256": identity_files,
            "tokenizer_vocab_size": vocab_size,
        },
        "checkpoint_template": {
            "checkpoint_id": "minimal_C0",
            "role": "共享祖先 H 的物化 checkpoint",
            "prefix_tokens": h_tokens,
            "prefix_length_L0": l_zero,
            "state_position_note": (
                "L = L0 = 8192：materialize(H) 完成后，radix 树在 H 输入边界"
                "持有 exact 节点（与正式采集 candidate 的 token_pos 约定一致；"
                "生成 token 不进入树路径）"
            ),
        },
        "continuations": [
            {
                "continuation_id": "P1",
                "kind": "legal_fork",
                "suffix_origin": "CONTROLLED",
                "template": "H + suffix1",
                "mutation": None,
                "expected_predicate": "ADMISSIBLE（C1–C6 无 FAIL，C7 待 runtime 认证）",
                "suffix_tokens": suffix1,
            },
            {
                "continuation_id": "P2",
                "kind": "legal_fork",
                "suffix_origin": "CONTROLLED",
                "template": "H + suffix2",
                "mutation": None,
                "expected_predicate": "ADMISSIBLE（C1–C6 无 FAIL，C7 待 runtime 认证）",
                "suffix_tokens": suffix2,
            },
            {
                "continuation_id": "P_bad",
                "kind": "illegal_prefix_mutation",
                "suffix_origin": "CONTROLLED（复用 P1 后缀）",
                "template": "H_mutated + suffix1",
                "mutation": {
                    "position": mutation_position,
                    "original_token": original_token,
                    "mutated_token": mutated_token,
                    "note": "分歧点严格在 L 之前（块对齐位置）",
                },
                "expected_predicate": "REJECTED（C2 FAIL：token 历史在分歧点之前不一致）",
                "suffix_tokens": suffix1,
                "prefix_tokens": h_mutated,
            },
            {
                "continuation_id": "P3",
                "kind": "legal_fork_post_eviction_probe",
                "suffix_origin": "CONTROLLED",
                "template": "H + suffix3",
                "mutation": None,
                "expected_predicate": (
                    "不参与合同判定；用于驱逐反事实——evict C0 的 Mamba 状态后，"
                    "P3 的 executable_prefix 必须塌缩、replay_gap 增大，"
                    "证明 P1/P2 的恢复依赖 Recurrent State 而非仅 FA-KV"
                ),
                "suffix_tokens": suffix3,
            },
        ],
        "declarations": {
            "controlled_suffixes": (
                "P1/P2/P3 的 256-token 后缀为确定性合成（sha256 keyed 采样），"
                "显式标记 CONTROLLED，不是原生 AgentX 分支内容；长度 256 是为"
                "使分支终止点对齐 mamba track 边界（8448），让各分支终止 Mamba "
                "状态物化为可观测的独立 radix 节点（首次冒烟实测 63-token 后缀"
                "终止在非边界 8256，状态不落盘）"
            ),
            "real_prefix": (
                "H 的 8192 token 全部来自真实语料请求的 hash_ids 前缀块，"
                "经正式 compose 语义与真实 tokenizer 物化"
            ),
            "checkpoint_position_convention": (
                "C0 位于 materialize(H) 的输入边界（L=8192）。首次冒烟实测："
                "inspect 在 8193（H+生成 token）处失败（exact prefix path "
                "missing after 8192/8193），证明该 runtime 不把生成 token 插入"
                "树路径——正式采集同样以请求输入边界为 checkpoint 位置"
            ),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[P0-CASES] 写入 {OUTPUT_PATH}")
    print(
        f"[P0-CASES] H={l_zero} tokens（trace={trace.get('id')}, "
        f"request#{request_index}），分歧点={mutation_position}，"
        f"H digest={cases['provenance']['h_digest'][:16]}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
