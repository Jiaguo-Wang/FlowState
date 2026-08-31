#!/usr/bin/env python3
"""执行单个 OpenHands workflow 前三轮的真实 Engine smoke。"""

from __future__ import annotations

from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import traceback
from typing import Mapping, Sequence

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K
from evaluation.sota_latency_runtime import measure_streaming_request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
DATASET_PATH = Path(
    "/home/wjg/data/agentic_coding_trajectories/sessions.parquet"
)
TOKENIZER_PATH = Path("/home/wjg/models/qwen3.5-9b")
WORKFLOW_ID = (
    "nebius-swe-rebench-openhands::"
    "chatcmpl-0063c3ccef5e68d790c496c97203112c"
)
TARGET_TURNS = (1, 2, 3)
RID_TEMPLATE = "openhands-smoke-a-turn-{turn:03d}"


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """用稳定格式写出一个 JSON 文件。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    """追加一个请求记录并立即刷新文件。"""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()


def _artifact_directory() -> Path:
    """创建不会覆盖既有产物的时间戳目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = (
        ARTIFACT_ROOT
        / f"openhands_single_workflow_smoke_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _parse_tool_calls(value: object) -> list[dict[str, object]]:
    """解析数据集中的工具调用，并规范化函数参数。"""
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError("tool_calls_json 反序列化后必须是列表")

    normalized = []
    for raw_call in parsed:
        if not isinstance(raw_call, Mapping):
            raise ValueError("每个 tool call 必须是对象")
        call = dict(raw_call)
        raw_function = call.get("function")
        if raw_function is not None:
            if not isinstance(raw_function, Mapping):
                raise ValueError("tool call 的 function 必须是对象")
            function = dict(raw_function)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
                if not isinstance(arguments, Mapping):
                    raise ValueError(
                        "function.arguments 反序列化后必须是对象"
                    )
                function["arguments"] = dict(arguments)
            call["function"] = function
        normalized.append(call)
    return normalized


def normalize_message(raw_message: Mapping[str, object]) -> dict[str, object]:
    """按冻结语义把一条数据集消息转换为 chat template 消息。"""
    message: dict[str, object] = {}
    for field in ("role", "content"):
        value = raw_message.get(field)
        if value is not None:
            message[field] = value

    tool_calls_json = raw_message.get("tool_calls_json")
    if tool_calls_json not in (None, ""):
        message["tool_calls"] = _parse_tool_calls(tool_calls_json)

    tool_call_id = raw_message.get("tool_call_id")
    if tool_call_id not in (None, ""):
        message["tool_call_id"] = tool_call_id
    return message


def load_workflow_messages(
    dataset_path: Path = DATASET_PATH,
    workflow_id: str = WORKFLOW_ID,
) -> tuple[list[dict[str, object]], int]:
    """只读取目标 workflow，并返回规范化消息与记录轮数。"""
    table = pq.read_table(
        dataset_path,
        filters=[("session_id", "=", workflow_id)],
        columns=["session_id", "n_turns", "messages_json"],
    )
    if table.num_rows != 1:
        raise RuntimeError(
            f"目标 workflow 应唯一命中一行，实际为 {table.num_rows} 行"
        )
    row = table.to_pylist()[0]
    raw_messages = json.loads(row["messages_json"])
    if not isinstance(raw_messages, list):
        raise ValueError("messages_json 反序列化后必须是列表")
    messages = [normalize_message(message) for message in raw_messages]
    return messages, int(row["n_turns"])


def build_request_inputs(
    tokenizer: object,
    messages: Sequence[Mapping[str, object]],
    target_turns: Sequence[int] = TARGET_TURNS,
) -> list[dict[str, object]]:
    """构造各 assistant turn 之前的完整 recorded history 输入。"""
    targets = set(target_turns)
    requests = []
    assistant_turn = 0
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_turn += 1
        if assistant_turn not in targets:
            continue
        template_output = tokenizer.apply_chat_template(
            list(messages[:message_index]),
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = _template_input_ids(template_output)
        requests.append(
            {
                "turn": assistant_turn,
                "rid": RID_TEMPLATE.format(turn=assistant_turn),
                "input_ids": [int(token_id) for token_id in input_ids],
            }
        )
    found = tuple(int(request["turn"]) for request in requests)
    if found != tuple(target_turns):
        raise RuntimeError(f"目标 assistant turns 不完整：{found}")
    return requests


def _template_input_ids(template_output: object) -> list[int]:
    """兼容 tokenizer 返回 token 列表或 BatchEncoding 的情况。"""
    value = template_output
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("chat template 返回对象缺少 input_ids")
        value = value["input_ids"]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, list):
        raise TypeError("chat template 的 input_ids 必须是列表")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat template 不应返回多条批量输入")
        value = value[0]
    if any(isinstance(token_id, (list, dict)) for token_id in value):
        raise TypeError("chat template 的 input_ids 必须是一维序列")
    return [int(token_id) for token_id in value]


def _finish_reason(metadata: Mapping[str, object]) -> object:
    """保留服务端 finish_reason 的原始结构。"""
    return metadata.get("finish_reason")


def execute_request(
    engine: object,
    request: Mapping[str, object],
) -> dict[str, object]:
    """执行一个无重试的单 token 流式请求。"""
    input_ids = request["input_ids"]
    if not isinstance(input_ids, list):
        raise TypeError("input_ids 必须是列表")
    timing = measure_streaming_request(
        engine,
        request_id=str(request["rid"]),
        token_ids=input_ids,
    )
    metadata = timing["server_metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("server_metadata 必须是对象")
    offline_tokens = len(input_ids)
    server_tokens = int(metadata.get("prompt_tokens", -1))
    return {
        "workflow_id": WORKFLOW_ID,
        "turn": int(request["turn"]),
        "rid": str(request["rid"]),
        "offline_input_tokens": offline_tokens,
        "server_prompt_tokens": server_tokens,
        "cached_tokens": int(metadata.get("cached_tokens", 0) or 0),
        "completion_tokens": int(
            metadata.get("completion_tokens", 0) or 0
        ),
        "ttft_ms": float(timing["ttft_ms"]),
        "request_latency_ms": float(timing["request_latency_ms"]),
        "finish_reason": _finish_reason(metadata),
        "request_completed": True,
        "token_count_exact": server_tokens == offline_tokens,
        "oom": False,
        "truncation_or_clipping": server_tokens != offline_tokens,
        "status": "PASS" if server_tokens == offline_tokens else "FAIL",
        "error": None,
    }


def _environment() -> dict[str, object]:
    """采集解释本次运行所需的最小环境信息。"""
    import torch

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "sglang_version": importlib.metadata.version("sglang"),
        "transformers_version": importlib.metadata.version("transformers"),
        "pyarrow_version": importlib.metadata.version("pyarrow"),
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
    }


def _summary(
    *,
    records: Sequence[Mapping[str, object]],
    artifact: Path,
    n_turns: int | None,
    fatal_error: str | None,
    environment: Mapping[str, object] | None,
) -> dict[str, object]:
    """根据已经落盘的请求记录构建最终门禁结果。"""
    all_completed = (
        len(records) == len(TARGET_TURNS)
        and all(record.get("request_completed") is True for record in records)
    )
    token_counts_exact = (
        len(records) == len(TARGET_TURNS)
        and all(record.get("token_count_exact") is True for record in records)
    )
    lengths = [
        int(record["offline_input_tokens"])
        for record in records
        if "offline_input_tokens" in record
    ]
    growing_input = (
        len(lengths) == len(TARGET_TURNS)
        and lengths[0] < lengths[1] < lengths[2]
    )
    fatal_lowered = (fatal_error or "").lower()
    oom = (
        any(record.get("oom") is True for record in records)
        or "out of memory" in fatal_lowered
        or "oom" in fatal_lowered
    )
    clipping = any(
        record.get("truncation_or_clipping") is True for record in records
    )
    passed = (
        fatal_error is None
        and all_completed
        and token_counts_exact
        and growing_input
        and not oom
        and not clipping
    )
    return {
        "schema_version": "flowstate.openhands_single_workflow_smoke.v1",
        "status": "PASS" if passed else "FAIL",
        "workflow_id": WORKFLOW_ID,
        "target_turns": list(TARGET_TURNS),
        "recorded_n_turns": n_turns,
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_128K,
        "sampling_parameters": SAMPLING_PARAMETERS,
        "dataset_path": str(DATASET_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "artifact": str(artifact.relative_to(REPOSITORY_ROOT)),
        "request_count": len(records),
        "requests": list(records),
        "all_requests_completed": all_completed,
        "offline_server_token_counts_exact": token_counts_exact,
        "growing_input": growing_input,
        "oom": oom,
        "truncation_or_clipping": clipping,
        "fatal_error": fatal_error,
        "environment": dict(environment) if environment is not None else None,
    }


def main() -> int:
    """准备三个真实请求，在单个 128K Engine 中依次执行一次。"""
    artifact = _artifact_directory()
    requests_path = artifact / "requests.jsonl"
    requests_path.write_text("", encoding="utf-8")
    records: list[dict[str, object]] = []
    fatal_error = None
    engine = None
    n_turns = None
    environment = None
    try:
        environment = _environment()
        messages, n_turns = load_workflow_messages()
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )
        requests = build_request_inputs(tokenizer, messages)
        _write_json(
            artifact / "config.json",
            {
                "workflow_id": WORKFLOW_ID,
                "target_turns": list(TARGET_TURNS),
                "recorded_n_turns": n_turns,
                "offline_input_tokens": [
                    len(request["input_ids"]) for request in requests
                ],
                "engine_configuration": ENGINE_CONFIGURATION_128K,
                "sampling_parameters": SAMPLING_PARAMETERS,
                "dataset_path": str(DATASET_PATH),
                "tokenizer_path": str(TOKENIZER_PATH),
                "environment": environment,
            },
        )

        from wp3b_end_to_end_transport import FormalEndToEndGateEngine

        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION_128K)
        for request in requests:
            try:
                record = execute_request(engine, request)
            except Exception as error:
                message = repr(error)
                lowered = message.lower()
                record = {
                    "workflow_id": WORKFLOW_ID,
                    "turn": int(request["turn"]),
                    "rid": str(request["rid"]),
                    "offline_input_tokens": len(request["input_ids"]),
                    "server_prompt_tokens": None,
                    "cached_tokens": None,
                    "completion_tokens": None,
                    "ttft_ms": None,
                    "request_latency_ms": None,
                    "finish_reason": None,
                    "request_completed": False,
                    "token_count_exact": False,
                    "oom": "out of memory" in lowered or "oom" in lowered,
                    "truncation_or_clipping": False,
                    "status": "FAIL",
                    "error": message,
                }
                records.append(record)
                _append_jsonl(requests_path, record)
                fatal_error = message
                break
            records.append(record)
            _append_jsonl(requests_path, record)
            if record["status"] != "PASS":
                fatal_error = "服务端 prompt_tokens 与离线长度不一致"
                break
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

    summary = _summary(
        records=records,
        artifact=artifact,
        n_turns=n_turns,
        fatal_error=fatal_error,
        environment=environment,
    )
    _write_json(artifact / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
