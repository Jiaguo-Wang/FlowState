"""Step 13G-B1.1 单元测试：AgentX Qwen3.5-9B replay fidelity preflight。

CPU only；集成测试在本地 model/tokenizer 存在时运行。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from evaluation.agentx_qwen_replay_preflight import (
    QWEN_MODEL_PATH,
    RUNTIME_CONTEXT_LIMIT,
    _compose_prompt_tokens,
    _decode_block_tokens,
    _request_index_for_target,
    _sglang_direct_token_id_path,
    _special_token_ids,
    _tokenizer_identity,
    _validate_block_fidelity,
    _validate_prefix_topology,
    _validate_snapshot,
    run_preflight,
)


class _FakeTokenizer:
    """用于不依赖 transformers 的单元测试的最小 tokenizer stub。"""

    vocab_size = 248_320
    eos_token_id = 248_044
    pad_token_id = 248_044
    unk_token_id = None
    bos_token_id = None
    image_token_id = 248_056
    added_tokens_decoder = {
        248_044: type("T", (), {"special": True}),
        248_045: type("T", (), {"special": True}),
        248_056: type("T", (), {"special": True}),
    }
    added_tokens_encoder = {}

    def __len__(self) -> int:
        return 248_077


def _fake_conversation(
    *,
    conversation_id: str = "conv",
    positions: list[int] | None = None,
    lengths: list[int] | None = None,
    hash_ids: list[list[int]] | None = None,
    is_inherited: bool = False,
    lineage_path: tuple[str, ...] | None = None,
    fork_depth: int = 0,
) -> Any:
    """构造最小 conversation-like 对象。"""
    positions = positions or []
    lengths = lengths or []
    hash_ids = hash_ids or []
    return type(
        "C",
        (),
        {
            "conversation_id": conversation_id,
            "lineage_path": lineage_path or (conversation_id,),
            "request_hash_token_positions": positions,
            "request_input_lengths": lengths,
            "request_hash_ids": hash_ids,
            "is_inherited": is_inherited,
            "fork_depth_blocks": fork_depth,
        },
    )


def _fake_replay_request(
    conversation_id: str,
    token_ids: list[int],
    *,
    is_inherited: bool = False,
    hash_ids: list[int] | None = None,
    exact_len: int | None = None,
    fork_depth: int = 0,
) -> Any:
    hash_ids = hash_ids or []
    return type(
        "R",
        (),
        {
            "conversation_id": conversation_id,
            "is_inherited": is_inherited,
            "fork_depth_blocks": fork_depth,
            "token_ids": token_ids,
            "hash_ids": hash_ids,
            "exact_input_length": exact_len or len(token_ids),
        },
    )


def _fake_replay_snapshot(requests: list[Any]) -> Any:
    return type(
        "S",
        (),
        {
            "trace_id": "trace",
            "block_size": 64,
            "t": 1.0,
            "pending_count": len(requests),
            "candidate_count": len(requests),
            "shared_candidate_count": 0,
            "max_degree": 0,
            "max_fork_depth_blocks": 0,
            "active_conversation_ids": [r.conversation_id for r in requests],
            "requests": requests,
            "active_targets": {},
        },
    )


def test_decode_block_tokens_length_and_range() -> None:
    """单个 hash block 必须恰好 block_size 个 token，且均在有效范围内。"""
    tok = _FakeTokenizer()
    block = _decode_block_tokens(42, 64, tok)
    assert len(block) == 64
    assert all(0 <= tid < len(tok) for tid in block)


def test_decode_block_tokens_same_hash_id_same_block() -> None:
    """同一 hash_id 两次解码必须得到同一 block。"""
    tok = _FakeTokenizer()
    b1 = _decode_block_tokens(12345, 64, tok)
    b2 = _decode_block_tokens(12345, 64, tok)
    assert b1 == b2


def test_decode_block_tokens_different_hash_ids_differ() -> None:
    """不同 hash_id 应产生不同 block（概率极高）。"""
    tok = _FakeTokenizer()
    b1 = _decode_block_tokens(1, 64, tok)
    b2 = _decode_block_tokens(2, 64, tok)
    assert b1 != b2


def test_compose_prompt_exact_length() -> None:
    """compose 后的 prompt 长度必须严格等于 exact_input_length。"""
    tok = _FakeTokenizer()
    tokens = _compose_prompt_tokens([1, 2, 3], 150, 64, tok)
    assert len(tokens) == 150


def test_compose_prompt_no_special_tokens() -> None:
    """合成的 token 不应落在 special token id 中。"""
    tok = _FakeTokenizer()
    special = _special_token_ids(tok)
    tokens = _compose_prompt_tokens([10, 20, 30], 200, 64, tok)
    assert not any(tid in special for tid in tokens)


def test_block_fidelity_alignment() -> None:
    """prompt 的前几个 hash block 必须与 decode_block_tokens 结果一致。"""
    tok = _FakeTokenizer()
    hash_ids = [7, 8, 9]
    exact_len = 64 * len(hash_ids)
    tokens = _compose_prompt_tokens(hash_ids, exact_len, 64, tok)
    req = _fake_replay_request("r", tokens, hash_ids=hash_ids, exact_len=exact_len)
    snap = _fake_replay_snapshot([req])
    res = _validate_block_fidelity(req, 64, tok)
    assert res["blocks_checked"] == 3
    assert res["pass"]


def test_request_index_for_target_zero_and_cumulative() -> None:
    """target 映射到当前 pending request 索引；target==0 对应第一个 pending 请求。"""
    conv = _fake_conversation(positions=[64, 64, 64])
    assert _request_index_for_target(conv, 0) == 0
    assert _request_index_for_target(conv, 64) == 0
    assert _request_index_for_target(conv, 128) == 1
    assert _request_index_for_target(conv, 192) == 2


def test_spawn_isolation_prefix_zero() -> None:
    """SPAWN conversation 与其他 conversation 的公共前缀长度应为 0。"""
    spawn_req = _fake_replay_request("spawn", [1000] * 64, is_inherited=False)
    root_req = _fake_replay_request("root", [2000] * 64, is_inherited=False)
    snap = _fake_replay_snapshot([spawn_req, root_req])
    # 构造 lineage 不互为祖先。
    convs = [
        _fake_conversation(conversation_id="spawn", lineage_path=("trace", "spawn")),
        _fake_conversation(conversation_id="root", lineage_path=("trace",)),
    ]
    res = _validate_prefix_topology(snap, convs)
    assert res["spawn_isolation_preserved"]
    assert res["pass"]


def test_fork_prefix_preservation() -> None:
    """FORK child 与 parent 共享前缀。"""
    parent_tokens = [1000] * 128
    child_tokens = [1000] * 128 + [2000] * 64
    parent_req = _fake_replay_request("parent", parent_tokens, is_inherited=False)
    child_req = _fake_replay_request(
        "child", child_tokens, is_inherited=True, fork_depth=1
    )
    snap = _fake_replay_snapshot([parent_req, child_req])
    # active_targets 使用 conversation 在 convs 列表中的原始索引；祖先 target 需覆盖 child input length。
    snap.active_targets = {0: len(child_tokens), 1: len(child_tokens)}
    convs = [
        _fake_conversation(
            conversation_id="parent", lineage_path=("trace",), is_inherited=False
        ),
        _fake_conversation(
            conversation_id="child",
            lineage_path=("trace", "child"),
            is_inherited=True,
            fork_depth=1,
        ),
    ]
    res = _validate_prefix_topology(snap, convs)
    assert res["fork_prefix_preserved"]
    assert res["pass"]


def test_context_limit_gate() -> None:
    """超过 131200 token 的 snapshot 应被标记为 context_limit 失败。"""
    tok = _FakeTokenizer()
    over = RUNTIME_CONTEXT_LIMIT + 10
    tokens = _compose_prompt_tokens([1], over, 64, tok)
    req = _fake_replay_request("r", tokens, exact_len=over)
    snap = _fake_replay_snapshot([req])
    conv = _fake_conversation(conversation_id="r")
    val = _validate_snapshot(snap, [conv], tok)
    assert not val["context_limit_pass"]


def test_determinism_via_recompose() -> None:
    """同一 hash_ids + exact length 两次 compose 结果相同。"""
    tok = _FakeTokenizer()
    t1 = _compose_prompt_tokens([5, 6, 7], 180, 64, tok)
    t2 = _compose_prompt_tokens([5, 6, 7], 180, 64, tok)
    assert t1 == t2


def test_sglang_direct_token_id_path_present() -> None:
    """验证 SGLang direct-token-ID 调用路径存在。"""
    res = _sglang_direct_token_id_path()
    assert res["pass"]
    assert res["present"]


def test_tokenizer_identity_on_real_model() -> None:
    """实际加载 Qwen tokenizer 并核对关键身份字段。"""
    if not QWEN_MODEL_PATH.exists():
        pytest.skip("本地 Qwen3.5-9B 模型不存在")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN_MODEL_PATH), trust_remote_code=True)
    identity = _tokenizer_identity(QWEN_MODEL_PATH, tok)
    assert identity["tokenizer_class"] == "Qwen2Tokenizer"
    assert identity["model_type"] == "qwen3_5"
    assert identity["max_position_embeddings"] == 262_144
    assert identity["safetensors_present"] is True


@pytest.mark.skipif(
    not QWEN_MODEL_PATH.exists(),
    reason="本地 Qwen3.5-9B 模型不存在",
)
def test_run_preflight_integration(tmp_path: Path) -> None:
    """端到端 preflight：artifact 齐全且 validation gate 通过。"""
    artifact_root = run_preflight(tmp_path / "preflight")
    required = {
        "INPUTS.json",
        "QWEN_TOKENIZER_IDENTITY.json",
        "DETERMINISTIC_BLOCK_SYNTHESIS.json",
        "REPLAY_FIDELITY_GATES.json",
        "SNAPSHOT_TOKENIZED_INPUTS.json",
        "REPRESENTATIVE_CASE_AUDIT.json",
        "SGLANG_DIRECT_TOKEN_ID_PATH.json",
        "validation_report.json",
        "STEP_13G_B1_1_AGENTX_QWEN_REPLAY_PREFLIGHT_FINAL_REPORT.md",
    }
    found = {p.name for p in artifact_root.iterdir()}
    assert required <= found, f"Missing artifacts: {required - found}"

    validation = json.loads((artifact_root / "validation_report.json").read_text())
    assert validation["gate_passed"] is True
    assert validation["overall_status"] == "AGENTX_QWEN_REPLAY_PREFLIGHT_READY"
    assert validation["checks"]["token_range"] is True
    assert validation["checks"]["exact_length"] is True
    assert validation["checks"]["sglang_direct_token_id_path"] is True
