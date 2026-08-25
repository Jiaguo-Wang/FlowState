from __future__ import annotations

import json

import pytest

from evaluation.sota_latency_benchmark import build_benchmark_cases
from evaluation.sota_latency_runtime import (
    CaseAttemptError,
    LatencyArtifactWriter,
    build_gap_group_summary,
    execute_with_deterministic_retry,
    measure_streaming_request,
)


class _FakeStreamEngine:
    """提供一个只产生单 token 的确定性流式引擎。"""

    def generate(self, **kwargs):
        assert kwargs["stream"] is True
        yield {
            "output_ids": [7],
            "meta_info": {
                "completion_tokens": 1,
                "num_retractions": 0,
            },
        }


def test_streaming_measurement_uses_first_token_boundary() -> None:
    timestamps = iter((1_000_000, 11_000_000, 16_000_000))
    result = measure_streaming_request(
        _FakeStreamEngine(),
        request_id="request",
        token_ids=(1, 2, 3),
        clock_ns=lambda: next(timestamps),
    )

    assert result["output_token_id"] == 7
    assert result["ttft_ms"] == 10.0
    assert result["request_latency_ms"] == 15.0


def test_transient_failure_is_retried_at_most_once() -> None:
    case = next(item for item in build_benchmark_cases() if item.is_warmup)
    attempts = []

    def execute_once(active_case, retry_count):
        attempts.append(retry_count)
        if retry_count == 0:
            raise CaseAttemptError(
                "发送请求",
                RuntimeError("临时错误"),
                correctness_failure=False,
                partial_record={
                    "case_id": active_case.case_id,
                    "status": "FAIL",
                },
            )
        return {
            "case_id": active_case.case_id,
            "status": "PASS",
            "correctness_pass": True,
        }

    record = execute_with_deterministic_retry(case, execute_once)

    assert attempts == [0, 1]
    assert record["status"] == "PASS"
    assert record["retry_count"] == 1
    assert len(record["attempt_errors"]) == 1


def test_correctness_failure_is_not_retried() -> None:
    case = next(item for item in build_benchmark_cases() if item.is_warmup)
    attempts = []

    def execute_once(active_case, retry_count):
        attempts.append(retry_count)
        raise CaseAttemptError(
            "H/E/G 门禁",
            RuntimeError("恢复间隔不一致"),
            correctness_failure=True,
            partial_record={
                "case_id": active_case.case_id,
                "status": "FAIL",
                "correctness_failure": True,
            },
        )

    record = execute_with_deterministic_retry(case, execute_once)

    assert attempts == [0]
    assert record["status"] == "FAIL"
    assert record["correctness_failure"] is True
    assert record["retry_count"] == 0


def test_terminal_transient_failure_keeps_both_attempts() -> None:
    case = next(item for item in build_benchmark_cases() if item.is_warmup)
    attempts = []

    def execute_once(active_case, retry_count):
        attempts.append(retry_count)
        raise CaseAttemptError(
            "发送请求",
            RuntimeError("服务端错误"),
            correctness_failure=False,
            partial_record={
                "case_id": active_case.case_id,
                "status": "FAIL",
            },
        )

    record = execute_with_deterministic_retry(case, execute_once)

    assert attempts == [0, 1]
    assert record["status"] == "FAIL"
    assert record["retry_count"] == 1
    assert len(record["attempt_errors"]) == 2


def test_gap_group_summary_uses_multiplicity_and_excludes_invalid() -> None:
    records = (
        _sample(gap=0, ttft=10.0, multiplicity=1),
        _sample(gap=0, ttft=20.0, multiplicity=3),
        _sample(gap=8_192, ttft=100.0, multiplicity=2),
        _sample(
            gap=8_192,
            ttft=0.0,
            multiplicity=100,
            warmup=True,
        ),
        _sample(
            gap=8_192,
            ttft=0.0,
            multiplicity=100,
            correctness_pass=False,
        ),
    )

    rows = build_gap_group_summary(records)
    by_gap = {row["gap_tokens"]: row for row in rows}

    assert set(by_gap) == {0, 8_192}
    assert by_gap[0]["sample_count"] == 2
    assert by_gap[0]["weighted_request_count"] == 4.0
    assert by_gap[0]["ttft_weighted_mean_ms"] == pytest.approx(17.5)
    assert by_gap[8_192]["sample_count"] == 1
    assert by_gap[8_192]["weighted_request_count"] == 2.0


def test_artifact_writer_emits_all_frozen_files(tmp_path) -> None:
    writer = LatencyArtifactWriter(tmp_path / "artifact")
    writer.directory.mkdir()
    writer.append_raw_sample({"case_id": "case", "status": "PASS"})
    writer.write_metadata({"status": "RUNNING"})
    writer.write_summary(
        {
            "policy_summaries": (),
            "gap_group_summary": (),
            "status": "PASS",
        },
        (),
    )

    assert {
        path.name for path in writer.directory.iterdir()
    } == {
        "raw_samples.jsonl",
        "summary.json",
        "summary.csv",
        "gap_group_summary.csv",
        "run_metadata.json",
    }
    assert json.loads(
        writer.raw_samples_path.read_text(encoding="utf-8")
    )["status"] == "PASS"


def _sample(
    *,
    gap: int,
    ttft: float,
    multiplicity: int,
    warmup: bool = False,
    correctness_pass: bool = True,
) -> dict[str, object]:
    """构造 recovery-gap 统计测试使用的样本。"""
    return {
        "planning_gap": gap,
        "ttft_ms": ttft,
        "class_multiplicity": multiplicity,
        "is_warmup": warmup,
        "correctness_pass": correctness_pass,
    }
