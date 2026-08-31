#!/usr/bin/env python3
"""执行 OpenHands 单 workflow 前十轮 Hybrid Runtime baseline。"""

from __future__ import annotations

from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Mapping, Sequence

from transformers import AutoTokenizer

from evaluation.controlled_multiworkflow_v1.runtime_gate import (
    SAMPLING_PARAMETERS,
    query_runtime_metrics,
    wait_for_transport,
)
from evaluation.openhands_single_workflow_smoke import (
    DATASET_PATH,
    TOKENIZER_PATH,
    WORKFLOW_ID,
    build_request_inputs,
    load_workflow_messages,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K
from evaluation.sota_latency_runtime import measure_streaming_request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "evaluation" / "runtime_artifacts"
TARGET_TURNS = tuple(range(1, 11))
RID_TEMPLATE = "openhands-baseline-a-turn-{turn:03d}"
ALIGNMENT_TOKENS = 64


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """用稳定格式写出一个 JSON 文件。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    """追加并刷新一条请求记录。"""
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
        / f"openhands_single_workflow_baseline10_{timestamp}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _display_path(path: Path) -> str:
    """仓库内路径使用相对形式，其他路径保留绝对形式。"""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


class ArtifactLogCapture:
    """把当前进程及其子进程的标准输出和错误保存到产物目录。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.stdout_handle = None
        self.stderr_handle = None
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self) -> "ArtifactLogCapture":
        """在启动 Engine 前切换底层文件描述符。"""
        sys.stdout.flush()
        sys.stderr.flush()
        self.original_stdout = os.dup(1)
        self.original_stderr = os.dup(2)
        self.stdout_handle = (self.directory / "stdout.log").open(
            "w", encoding="utf-8"
        )
        self.stderr_handle = (self.directory / "stderr.log").open(
            "w", encoding="utf-8"
        )
        os.dup2(self.stdout_handle.fileno(), 1)
        os.dup2(self.stderr_handle.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """刷新日志并恢复调用者的标准输出和错误。"""
        sys.stdout.flush()
        sys.stderr.flush()
        if self.original_stdout is not None:
            os.dup2(self.original_stdout, 1)
            os.close(self.original_stdout)
        if self.original_stderr is not None:
            os.dup2(self.original_stderr, 2)
            os.close(self.original_stderr)
        if self.stdout_handle is not None:
            self.stdout_handle.close()
        if self.stderr_handle is not None:
            self.stderr_handle.close()


def exact_lcp(left: Sequence[int], right: Sequence[int]) -> int:
    """返回两个 token 序列从起点开始的连续相同长度。"""
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def prepare_requests(tokenizer: object) -> tuple[list[dict[str, object]], int]:
    """按冻结消息语义准备前十个 recorded-history 请求。"""
    messages, n_turns = load_workflow_messages()
    requests = build_request_inputs(
        tokenizer,
        messages,
        target_turns=TARGET_TURNS,
    )
    previous_ids = None
    for request in requests:
        turn = int(request["turn"])
        request["rid"] = RID_TEMPLATE.format(turn=turn)
        input_ids = request["input_ids"]
        if not isinstance(input_ids, list):
            raise TypeError("input_ids 必须是列表")
        lcp_tokens = (
            None if previous_ids is None else exact_lcp(previous_ids, input_ids)
        )
        request["adjacent_lcp_tokens"] = lcp_tokens
        request["expected_aligned_cache"] = (
            None
            if lcp_tokens is None
            else lcp_tokens // ALIGNMENT_TOKENS * ALIGNMENT_TOKENS
        )
        previous_ids = input_ids
    return requests, n_turns


def _optional_int(value: object) -> int | None:
    """把可用指标转换为整数，不可用时返回空值。"""
    if value is None:
        return None
    return int(value)


def _runtime_fields(
    client: object,
    rid: str,
    input_tokens: int,
) -> dict[str, object]:
    """使用现有只读控制接口采集同一 rid 的 Hybrid 指标。"""
    try:
        metrics = query_runtime_metrics(client, rid)
        h_value = _optional_int(metrics.get("physical_fa_hit"))
        e_value = _optional_int(metrics.get("executable_prefix"))
        g_value = _optional_int(metrics.get("replay_gap"))
        available = all(
            value is not None for value in (h_value, e_value, g_value)
        )
        bounds_valid = (
            available
            and 0 <= int(e_value) <= int(h_value) <= input_tokens
        )
        gap_valid = available and int(g_value) == int(h_value) - int(e_value)
        return {
            "runtime_metrics_available": available,
            "fa_kv_hit_frontier_h": h_value,
            "executable_frontier_e": e_value,
            "recovery_gap_g": g_value,
            "full_kv_hit_length": h_value,
            "mamba_branching_seqlen": _optional_int(
                metrics.get("mamba_branching_seqlen")
            ),
            "mamba_host_hit_length": _optional_int(
                metrics.get("mamba_host_hit_length")
            ),
            "recurrent_checkpoint_info": "unavailable",
            "runtime_bounds_valid": bounds_valid,
            "runtime_gap_valid": gap_valid,
            "runtime_metrics_error": None,
        }
    except Exception as error:
        return {
            "runtime_metrics_available": False,
            "fa_kv_hit_frontier_h": None,
            "executable_frontier_e": None,
            "recovery_gap_g": None,
            "full_kv_hit_length": None,
            "mamba_branching_seqlen": None,
            "mamba_host_hit_length": None,
            "recurrent_checkpoint_info": "unavailable",
            "runtime_bounds_valid": None,
            "runtime_gap_valid": None,
            "runtime_metrics_error": repr(error),
        }


def execute_request(
    engine: object,
    client: object,
    request: Mapping[str, object],
) -> dict[str, object]:
    """执行一个无重试请求并采集现有运行时指标。"""
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
    server_tokens = _optional_int(metadata.get("prompt_tokens"))
    cached_tokens = _optional_int(metadata.get("cached_tokens"))
    expected_cache = _optional_int(request.get("expected_aligned_cache"))
    token_count_exact = server_tokens == offline_tokens
    prefix_reuse_exact = (
        None
        if expected_cache is None or cached_tokens is None
        else cached_tokens == expected_cache
    )
    runtime = _runtime_fields(
        client,
        str(request["rid"]),
        offline_tokens,
    )
    return {
        "workflow_id": WORKFLOW_ID,
        "turn": int(request["turn"]),
        "rid": str(request["rid"]),
        "offline_input_tokens": offline_tokens,
        "adjacent_lcp_tokens": _optional_int(
            request.get("adjacent_lcp_tokens")
        ),
        "expected_aligned_cache": expected_cache,
        "server_prompt_tokens": server_tokens,
        "cached_tokens": cached_tokens,
        "cached_tokens_details": metadata.get("cached_tokens_details"),
        "completion_tokens": _optional_int(metadata.get("completion_tokens")),
        "ttft_ms": float(timing["ttft_ms"]),
        "request_latency_ms": float(timing["request_latency_ms"]),
        "finish_reason": metadata.get("finish_reason"),
        "request_completed": True,
        "token_count_exact": token_count_exact,
        "prefix_reuse_exact": prefix_reuse_exact,
        "oom": False,
        "truncation_or_clipping": not token_count_exact,
        **runtime,
        "status": "PASS" if token_count_exact else "FAIL",
        "error": None,
    }


def _failure_record(
    request: Mapping[str, object],
    error: Exception,
) -> dict[str, object]:
    """为首个 Engine 异常构造完整且不可误解的失败记录。"""
    message = repr(error)
    lowered = message.lower()
    input_ids = request.get("input_ids")
    input_tokens = len(input_ids) if isinstance(input_ids, list) else None
    return {
        "workflow_id": WORKFLOW_ID,
        "turn": int(request["turn"]),
        "rid": str(request["rid"]),
        "offline_input_tokens": input_tokens,
        "adjacent_lcp_tokens": request.get("adjacent_lcp_tokens"),
        "expected_aligned_cache": request.get("expected_aligned_cache"),
        "server_prompt_tokens": None,
        "cached_tokens": None,
        "cached_tokens_details": None,
        "completion_tokens": None,
        "ttft_ms": None,
        "request_latency_ms": None,
        "finish_reason": None,
        "request_completed": False,
        "token_count_exact": False,
        "prefix_reuse_exact": False,
        "oom": "out of memory" in lowered or "oom" in lowered,
        "truncation_or_clipping": False,
        "runtime_metrics_available": False,
        "fa_kv_hit_frontier_h": None,
        "executable_frontier_e": None,
        "recovery_gap_g": None,
        "full_kv_hit_length": None,
        "mamba_branching_seqlen": None,
        "mamba_host_hit_length": None,
        "recurrent_checkpoint_info": "unavailable",
        "runtime_bounds_valid": None,
        "runtime_gap_valid": None,
        "runtime_metrics_error": None,
        "status": "FAIL",
        "error": message,
    }


def _environment() -> dict[str, object]:
    """采集解释本次 baseline 所需的最小环境信息。"""
    import torch

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "sglang_version": importlib.metadata.version("sglang"),
        "transformers_version": importlib.metadata.version("transformers"),
        "pyarrow_version": importlib.metadata.version("pyarrow"),
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
    }


def build_summary(
    *,
    records: Sequence[Mapping[str, object]],
    artifact: Path,
    n_turns: int | None,
    fatal_error: str | None,
    environment: Mapping[str, object] | None,
) -> dict[str, object]:
    """构建 baseline 的正确性、复用和 Hybrid 语义门禁。"""
    all_completed = (
        len(records) == len(TARGET_TURNS)
        and all(record.get("request_completed") is True for record in records)
    )
    token_correctness = (
        len(records) == len(TARGET_TURNS)
        and all(record.get("token_count_exact") is True for record in records)
    )
    lengths = [
        int(record["offline_input_tokens"])
        for record in records
        if record.get("offline_input_tokens") is not None
    ]
    growing_history = (
        len(lengths) == len(TARGET_TURNS)
        and all(left <= right for left, right in zip(lengths, lengths[1:]))
    )
    comparable = [record for record in records if int(record["turn"]) >= 2]
    prefix_reuse_consistency = sum(
        record.get("prefix_reuse_exact") is True for record in comparable
    )
    metrics_available = (
        len(records) == len(TARGET_TURNS)
        and all(
            record.get("runtime_metrics_available") is True
            for record in records
        )
    )
    if metrics_available:
        runtime_semantic_status = (
            "PASS"
            if all(
                record.get("runtime_bounds_valid") is True
                and record.get("runtime_gap_valid") is True
                for record in records
            )
            else "FAIL"
        )
    else:
        runtime_semantic_status = "NOT_AVAILABLE"
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
        and token_correctness
        and growing_history
        and prefix_reuse_consistency == len(TARGET_TURNS) - 1
        and runtime_semantic_status != "FAIL"
        and not oom
        and not clipping
    )
    return {
        "schema_version": "flowstate.openhands_single_workflow_baseline10.v1",
        "status": "PASS" if passed else "FAIL",
        "workflow_id": WORKFLOW_ID,
        "target_turns": list(TARGET_TURNS),
        "recorded_n_turns": n_turns,
        "engine": "FormalEndToEndGateEngine",
        "engine_configuration": ENGINE_CONFIGURATION_128K,
        "sampling_parameters": SAMPLING_PARAMETERS,
        "dataset_path": str(DATASET_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "artifact": _display_path(artifact),
        "request_count": len(records),
        "requests": list(records),
        "all_requests_completed": all_completed,
        "token_correctness": token_correctness,
        "growing_history": growing_history,
        "alignment_tokens": ALIGNMENT_TOKENS,
        "alignment_interpretation": (
            "Hybrid Runtime 的 Mamba/FLA chunk，不是 FA-KV page size"
        ),
        "prefix_reuse_consistency": prefix_reuse_consistency,
        "prefix_reuse_comparisons": len(comparable),
        "h_available": metrics_available,
        "e_available": metrics_available,
        "g_available": metrics_available,
        "runtime_semantic_status": runtime_semantic_status,
        "warmup_jit_cause_proven": False,
        "ttft_cause": "CAUSE_NOT_PROVEN",
        "oom": oom,
        "truncation_or_clipping": clipping,
        "fatal_error": fatal_error,
        "environment": dict(environment) if environment is not None else None,
        "policy_executed": False,
        "eviction_executed": False,
        "concurrency": 1,
    }


def _run(artifact: Path) -> dict[str, object]:
    """在单个 Engine 中顺序执行十轮，不重试也不主动驱逐。"""
    requests_path = artifact / "requests.jsonl"
    requests_path.write_text("", encoding="utf-8")
    records: list[dict[str, object]] = []
    fatal_error = None
    engine = None
    n_turns = None
    environment = None
    try:
        environment = _environment()
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
        )
        requests, n_turns = prepare_requests(tokenizer)
        _write_json(
            artifact / "config.json",
            {
                "workflow_id": WORKFLOW_ID,
                "target_turns": list(TARGET_TURNS),
                "recorded_n_turns": n_turns,
                "offline_input_tokens": [
                    len(request["input_ids"]) for request in requests
                ],
                "adjacent_lcp_tokens": [
                    request["adjacent_lcp_tokens"] for request in requests
                ],
                "expected_aligned_cache": [
                    request["expected_aligned_cache"] for request in requests
                ],
                "alignment_tokens": ALIGNMENT_TOKENS,
                "engine_configuration": ENGINE_CONFIGURATION_128K,
                "sampling_parameters": SAMPLING_PARAMETERS,
                "dataset_path": str(DATASET_PATH),
                "tokenizer_path": str(TOKENIZER_PATH),
                "environment": environment,
                "policy_executed": False,
                "eviction_executed": False,
                "concurrency": 1,
            },
        )

        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )

        engine = FormalEndToEndGateEngine(**ENGINE_CONFIGURATION_128K)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        for request in requests:
            try:
                record = execute_request(engine, client, request)
            except Exception as error:
                record = _failure_record(request, error)
                records.append(record)
                _append_jsonl(requests_path, record)
                fatal_error = repr(error)
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

    summary = build_summary(
        records=records,
        artifact=artifact,
        n_turns=n_turns,
        fatal_error=fatal_error,
        environment=environment,
    )
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    """创建完整日志产物并执行唯一一次 baseline。"""
    artifact = _artifact_directory()
    with ArtifactLogCapture(artifact):
        summary = _run(artifact)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
