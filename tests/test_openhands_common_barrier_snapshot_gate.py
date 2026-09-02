from __future__ import annotations

from types import SimpleNamespace

from evaluation.controlled_multiworkflow_v1.scenario import (
    CHECKPOINT_SIZE_BYTES,
)
from evaluation.openhands_common_barrier_snapshot_gate import (
    BUDGET_BYTES,
    ENGINE_CONFIGURATION_COMMON_BARRIER,
    LOGICAL_K,
    build_policy_metadata,
    common_candidate_universe,
    materialize_visible_requests,
)
from evaluation.openhands_4workflow_occupancy_calibration import (
    PHYSICAL_MAX_MAMBA_CACHE_SIZE,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K
from flowstate.state_catalog import CheckpointCandidate


class FakeTokenizer:
    """记录 chat template 实际看到的消息历史。"""

    def __init__(self) -> None:
        self.histories = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        self.histories.append(messages)
        return [len(messages), 7]


def candidate(checkpoint_id: str, workflow_id: str, token_pos: int):
    """构造 metadata 单测所需的常驻候选。"""
    return CheckpointCandidate(
        checkpoint_id=checkpoint_id,
        workflow_id=workflow_id,
        lineage_path=("openhands", workflow_id),
        token_pos=token_pos,
        memory_bytes=CHECKPOINT_SIZE_BYTES,
        recurrent_resident=True,
        fa_resident=True,
    )


def test_visible_builder_stops_before_pending_output_and_future_turns() -> None:
    tokenizer = FakeTokenizer()
    raw_messages = [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "请求一"},
        {"role": "assistant", "content": "回答一"},
        {"role": "tool", "content": "结果一", "tool_call_id": "一"},
        {"role": "assistant", "content": "回答二禁止读取"},
        {"role": "tool", "content": "结果二禁止读取", "tool_call_id": "二"},
        {"role": "assistant", "content": "回答三禁止读取"},
    ]

    requests, audit = materialize_visible_requests(
        tokenizer,
        raw_messages,
        workflow_label="A",
        workflow_id="workflow-a",
    )

    assert tuple(requests) == (1, 2)
    assert len(tokenizer.histories) == 2
    assert [item["role"] for item in tokenizer.histories[0]] == [
        "system",
        "user",
    ]
    assert [item["role"] for item in tokenizer.histories[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert audit["pending_assistant_output_read"] is False
    assert audit["r_plus_2_message_consumed"] is False
    assert audit["r_plus_2_request_materialized"] is False


def test_marconi_first_checkpoint_uses_zero_parent_position() -> None:
    candidates = (
        candidate("A1", "A", 3_968),
        candidate("B1", "B", 2_432),
    )
    rows = (
        {
            "checkpoint_id": "A1",
            "creation_order": 1,
            "last_access_order": 1,
        },
        {
            "checkpoint_id": "B1",
            "creation_order": 2,
            "last_access_order": 2,
        },
    )

    metadata, recency = build_policy_metadata(candidates, rows)

    assert [item.creation_order for item in recency] == [1, 2]
    assert [item.last_access_order for item in recency] == [1, 2]
    assert [item["marconi_incremental_span"] for item in metadata] == [
        3_968.0,
        2_432.0,
    ]
    assert all(item["marconi_parent_position"] == 0 for item in metadata)
    assert all(item["marconi_first_checkpoint"] is True for item in metadata)


def test_common_universe_is_identical_without_running_selectors() -> None:
    candidates = (
        candidate("B1", "B", 2_432),
        candidate("A1", "A", 3_968),
    )
    universe = common_candidate_universe(candidates)
    assert universe["all_equal"] is True
    assert universe["candidate_ids_by_policy"] == {
        "LRU": ["A1", "B1"],
        "Marconi": ["A1", "B1"],
        "FlowState": ["A1", "B1"],
    }


def test_logical_budget_is_two_exact_checkpoint_sizes() -> None:
    assert LOGICAL_K == 2
    assert CHECKPOINT_SIZE_BYTES == 51_511_296
    assert BUDGET_BYTES == 103_022_592


def test_engine_configuration_only_overrides_physical_pool() -> None:
    expected = {
        **ENGINE_CONFIGURATION_128K,
        "max_mamba_cache_size": PHYSICAL_MAX_MAMBA_CACHE_SIZE,
    }
    assert ENGINE_CONFIGURATION_COMMON_BARRIER == expected
    assert ENGINE_CONFIGURATION_COMMON_BARRIER is not ENGINE_CONFIGURATION_128K


def test_snapshot_module_does_not_import_policy_selector() -> None:
    module = __import__(
        "evaluation.openhands_common_barrier_snapshot_gate",
        fromlist=["unused"],
    )
    forbidden = (
        "GlobalOptimizer",
        "StateController",
        "MarconiStylePolicy",
        "select_global_lru",
    )
    assert all(not hasattr(module, name) for name in forbidden)
