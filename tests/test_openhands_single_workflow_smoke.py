from __future__ import annotations

import json

import pytest

from evaluation.openhands_single_workflow_smoke import (
    _parse_tool_calls,
    _template_input_ids,
    build_request_inputs,
    normalize_message,
)


class _FakeTokenizer:
    """记录 chat template 输入并返回可预测 token。"""

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return list(range(len(messages) + 1))


def test_normalize_message_parses_tool_arguments() -> None:
    raw = {
        "role": "assistant",
        "content": "",
        "tool_calls_json": json.dumps(
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": "pwd"}),
                    },
                }
            ]
        ),
        "tool_call_id": None,
    }

    message = normalize_message(raw)

    assert message == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": {"command": "pwd"},
                },
            }
        ],
    }


def test_normalize_message_drops_null_fields() -> None:
    message = normalize_message(
        {
            "role": "tool",
            "content": "完成",
            "tool_calls_json": None,
            "tool_call_id": "call-1",
        }
    )

    assert message == {
        "role": "tool",
        "content": "完成",
        "tool_call_id": "call-1",
    }


def test_build_request_inputs_uses_history_before_assistant() -> None:
    tokenizer = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": "一"},
        {"role": "tool", "content": "结果一"},
        {"role": "assistant", "content": "二"},
        {"role": "tool", "content": "结果二"},
        {"role": "assistant", "content": "三"},
    ]

    requests = build_request_inputs(tokenizer, messages)

    assert [request["turn"] for request in requests] == [1, 2, 3]
    assert [len(call[0]) for call in tokenizer.calls] == [2, 4, 6]
    assert all(
        call[1]
        == {"tokenize": True, "add_generation_prompt": True}
        for call in tokenizer.calls
    )
    assert [request["rid"] for request in requests] == [
        "openhands-smoke-a-turn-001",
        "openhands-smoke-a-turn-002",
        "openhands-smoke-a-turn-003",
    ]


def test_parse_tool_calls_requires_mapping_arguments() -> None:
    value = json.dumps(
        [{"function": {"name": "terminal", "arguments": "[1, 2]"}}]
    )

    with pytest.raises(ValueError, match="arguments"):
        _parse_tool_calls(value)


def test_template_input_ids_accepts_mapping_result() -> None:
    assert _template_input_ids(
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    ) == [1, 2, 3]
