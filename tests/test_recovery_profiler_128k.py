"""验证 128K 独立恢复 profiler 的冻结协议与离线审计。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evaluation.recovery_profiler_128k import (
    ANCHOR_POS,
    EFFECTIVE_CONTEXT_LENGTH,
    ENGINE_CONFIGURATION_128K,
    FIXED_GAP_POINTS,
    MEASURED_REPETITIONS,
    ORDER_SEED,
    PIECEWISE_DIAGNOSTIC_KNOTS,
    TARGET_REQUEST_INPUT_TOKENS,
    WARMUP_REPETITIONS,
    ProfilerArtifactWriter,
    audit_long_gap_shape,
    build_profile_scenario,
    build_profile_schedule,
    diagnostic_fits,
    old_phi_audit,
    summarize_trials,
    validate_context_capabilities,
    validate_feasibility_response,
    validate_runtime_gap,
)
from flowstate.recovery_model import RecoveryCostModel


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "evaluation" / "recovery_profiler_128k.py"
PROTECTED_HASHES = {
    "flowstate/recovery_model.py": (
        "9a13bc4f7778b9e1835ddb04237d54815ff86c7e9c57b42d293e73c5bb404082"
    ),
    "evaluation/controlled_multiworkflow_v1/scenario.py": (
        "608f729c2670f249201402063bc2d354d85bc7a43657d4be5f77c13ff6fe5909"
    ),
    "evaluation/controlled_multiworkflow_v1/policies.py": (
        "8df5a1391b651f3a55090e13b8abb9d2a520de0a94abeb6a7339fdcb49445a24"
    ),
    "evaluation/scalable_multiworkflow_v2/scenario.py": (
        "a39ec5a1a9761ccefcefb4763eb10ce142895fc53197bd0f4d66746cc71e5bdd"
    ),
    "evaluation/sota_policies.py": (
        "b276aff22d2dc1adcdb33b15a7a94dc608fa916789ba0f2e5d5fbe0b3189d212"
    ),
    "evaluation/sota_metadata.py": (
        "df6582dd9a5dd15e984e9cefdd899e1d0b8bc9292399e7862608d04162a283c2"
    ),
    "evaluation/public_agent_trace/TRACELAB_NONTRIVIAL_PROTOCOL.md": (
        "dddcdf05f985f6cbd38acf97282095a9e09536c60878c709cc92ce564132a1c7"
    ),
    "evaluation/public_agent_trace/tracelab_nontrivial_protocol.json": (
        "2307a3d2cd5c77e27f3a3f4d525b4455e8804bfba46fab75ec5d8e37ec119269"
    ),
    "evaluation/public_agent_trace/tracelab_nontrivial_protocol.py": (
        "bc39435266dc97fd9c55e9cf13e215f1ece2481e8f0cee85c36512c3a08688d4"
    ),
    "motivation/README.md": (
        "a066a70f1fb13bba472147fc6847ec8b80f6d7dd8d02fa3d698677abced659a8"
    ),
}


def _synthetic_records(
    slope_ms_per_token: float = 0.045,
) -> tuple[dict[str, object], ...]:
    """构造完整且逐 token 正确的合成 trial。"""
    records = []
    index = 0
    for gap in FIXED_GAP_POINTS:
        for repetition in range(WARMUP_REPETITIONS):
            records.append(
                {
                    "case_id": f"warmup-{index}",
                    "target_gap": gap,
                    "is_warmup": True,
                    "status": "PASS",
                    "correctness_pass": True,
                    "gap_match": True,
                    "ttft_ms": 99_999.0 + repetition,
                }
            )
            index += 1
        for repetition in range(MEASURED_REPETITIONS):
            records.append(
                {
                    "case_id": f"measured-{index}",
                    "target_gap": gap,
                    "is_warmup": False,
                    "status": "PASS",
                    "correctness_pass": True,
                    "gap_match": True,
                    "ttft_ms": (
                        30.0 + slope_ms_per_token * gap + repetition / 10
                    ),
                }
            )
            index += 1
    return tuple(records)


def test_fixed_gap_set_and_repetition_counts() -> None:
    """正式 gap 与 Step 9D 重复次数必须完全冻结。"""
    assert FIXED_GAP_POINTS == (
        0,
        4_096,
        8_192,
        16_384,
        32_768,
        49_152,
        65_536,
        98_304,
        131_072,
    )
    assert WARMUP_REPETITIONS == 2
    assert MEASURED_REPETITIONS == 12
    assert ANCHOR_POS == 131_072


def test_schedule_is_deterministic_cyclic_and_balanced() -> None:
    """固定种子循环顺序不得与 gap 大小绑定。"""
    schedule = build_profile_schedule()
    assert schedule == build_profile_schedule(ORDER_SEED)
    assert len(schedule) == len(FIXED_GAP_POINTS) * 14
    assert len({case.case_id for case in schedule}) == len(schedule)
    assert tuple(
        case.target_gap for case in schedule[: len(FIXED_GAP_POINTS)]
    ) != FIXED_GAP_POINTS
    for gap in FIXED_GAP_POINTS:
        measured_positions = [
            case.gap_order_position
            for case in schedule
            if case.target_gap == gap and not case.is_warmup
        ]
        counts = [
            measured_positions.count(position)
            for position in range(len(FIXED_GAP_POINTS))
        ]
        assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("gap", FIXED_GAP_POINTS)
def test_scenario_changes_only_recurrent_frontier(gap: int) -> None:
    """所有场景必须固定物理前缀并仅改变循环状态前沿。"""
    scenario, selected_ids = build_profile_scenario(gap)
    continuation = scenario.continuations[0]
    candidates = {
        candidate.checkpoint_id: candidate
        for candidate in scenario.candidates
    }
    assert continuation.planning_target == ANCHOR_POS
    assert candidates["PROFILE_DEEP"].token_pos == ANCHOR_POS
    if gap == 0:
        assert selected_ids == ("PROFILE_DEEP",)
    elif gap == ANCHOR_POS:
        assert selected_ids == ()
    else:
        assert selected_ids == ("PROFILE_SHALLOW",)
        assert candidates["PROFILE_SHALLOW"].token_pos == ANCHOR_POS - gap


def test_long_extensions_expose_chunk_boundary_cleanup_states() -> None:
    """长扩展产生的中间循环状态必须进入同一 controller snapshot。"""
    scenario, selected_ids = build_profile_scenario(49_152)
    positions = {
        candidate.checkpoint_id: candidate.token_pos
        for candidate in scenario.candidates
    }
    assert selected_ids == ("PROFILE_SHALLOW",)
    assert positions == {
        "PROFILE_SHALLOW": 81_920,
        "PROFILE_CLEANUP_126976": 126_976,
        "PROFILE_DEEP": 131_072,
    }

    full_gap_scenario, full_gap_selected = build_profile_scenario(131_072)
    assert full_gap_selected == ()
    assert [
        candidate.token_pos for candidate in full_gap_scenario.candidates
    ] == [45_056, 90_112, 131_072]


def test_context_capability_requires_native_and_effective_coverage() -> None:
    """模型、tokenizer 与 SGLang admission 必须同时覆盖正式请求。"""
    result = validate_context_capabilities(
        {"text_config": {"max_position_embeddings": 262_144}},
        {"model_max_length": 262_144},
    )
    assert result["formal_request_input_tokens"] == TARGET_REQUEST_INPUT_TOKENS
    assert result["sglang_effective_max_context"] == EFFECTIVE_CONTEXT_LENGTH
    assert result["admission_limit_changed_from_step9d"] is True
    assert result["model_semantics_changed"] is False
    assert result["cache_policy_changed"] is False
    with pytest.raises(ValueError, match="模型原生上下文不足"):
        validate_context_capabilities(
            {"text_config": {"max_position_embeddings": 65_536}},
            {"model_max_length": 262_144},
        )
    with pytest.raises(ValueError, match="SGLang 有效上下文不足"):
        validate_context_capabilities(
            {"text_config": {"max_position_embeddings": 262_144}},
            {"model_max_length": 262_144},
            effective_context_length=131_072,
        )


def test_feasibility_rejects_truncation_and_silent_clipping() -> None:
    """服务端报告的 prompt token 必须与 128K 输入完全相等。"""
    result = validate_feasibility_response(
        {
            "prompt_tokens": 131_072,
            "completion_tokens": 1,
            "num_retractions": 0,
        }
    )
    assert result["request_completed"] is True
    assert result["truncation"] is False
    assert result["silent_clipping"] is False
    with pytest.raises(RuntimeError, match="截断或静默裁剪"):
        validate_feasibility_response(
            {
                "prompt_tokens": 131_071,
                "completion_tokens": 1,
                "num_retractions": 0,
            }
        )


def test_runtime_gap_requires_exact_h_e_g() -> None:
    """任何 H、E 或 G 偏差都必须使 trial 无效。"""
    assert validate_runtime_gap(
        49_152,
        {
            "physical_fa_hit": 131_072,
            "executable_prefix": 81_920,
            "replay_gap": 49_152,
        },
    ) == {
        "runtime_H": 131_072,
        "runtime_E": 81_920,
        "runtime_G": 49_152,
    }
    with pytest.raises(RuntimeError, match="runtime E 不匹配"):
        validate_runtime_gap(
            49_152,
            {
                "physical_fa_hit": 131_072,
                "executable_prefix": 81_919,
                "replay_gap": 49_153,
            },
        )


def test_baseline_adjustment_excludes_warmup() -> None:
    """warmup 不得进入均值，MeasuredPhi 必须相对 G=0。"""
    summary = summarize_trials(_synthetic_records())
    by_gap = {
        int(row["gap_tokens"]): row for row in summary["gap_rows"]
    }
    assert summary["all_fixed_points_measured"] is True
    assert by_gap[0]["mean_ms"] == pytest.approx(30.55)
    assert by_gap[0]["measured_phi_ms"] == 0.0
    assert by_gap[32_768]["measured_phi_ms"] == pytest.approx(
        0.045 * 32_768
    )
    assert all(
        row["valid_measured_count"] == MEASURED_REPETITIONS
        for row in by_gap.values()
    )


def test_small_negative_measured_phi_is_retained() -> None:
    """噪声导致的负增量必须保留，不能静默截断为零。"""
    records = [dict(record) for record in _synthetic_records()]
    for record in records:
        if record["target_gap"] == 4_096 and not record["is_warmup"]:
            record["ttft_ms"] = 30.50
    summary = summarize_trials(records)
    by_gap = {
        int(row["gap_tokens"]): row for row in summary["gap_rows"]
    }
    assert by_gap[4_096]["measured_phi_ms"] == pytest.approx(-0.05)


def test_old_phi_audit_marks_long_points_as_extrapolation() -> None:
    """32K 以上旧 Phi 只能标记为外推，不能称为已验证预测。"""
    model = RecoveryCostModel()
    measured = {gap: model.estimate(gap) + 10.0 for gap in FIXED_GAP_POINTS}
    measured[0] = 0.0
    audit = old_phi_audit(measured, model)
    points = {row["gap_tokens"]: row for row in audit["points"]}
    assert points[32_768]["validated_prediction"] is True
    assert points[49_152]["validated_prediction"] is False
    assert audit["above_32k_extrapolation"]["mae_ms"] == pytest.approx(10.0)


def test_long_gap_shape_and_diagnostic_fits_are_deterministic() -> None:
    """线性合成曲线必须得到单调与近似线性诊断。"""
    measured = {gap: 0.05 * gap for gap in FIXED_GAP_POINTS}
    shape = audit_long_gap_shape(measured)
    fits = diagnostic_fits(measured)
    assert shape["monotonic"] is True
    assert shape["approximately_linear"] == "YES"
    assert all(
        row["ms_per_ki_token"] == pytest.approx(51.2)
        for row in shape["slopes"]
    )
    assert fits["linear"]["full_128k"]["mae_ms"] == pytest.approx(0.0)
    assert fits["piecewise_linear"]["fixed_knots"] == list(
        PIECEWISE_DIAGNOSTIC_KNOTS
    )
    assert fits["piecewise_linear"]["full_128k"]["mae_ms"] == pytest.approx(0.0)


def test_artifact_writer_never_overwrites_existing_directory(tmp_path: Path) -> None:
    """每次运行必须创建独立目录并保留全部约定文件。"""
    writer = ProfilerArtifactWriter.create(tmp_path, "20260826_000000_000000")
    writer.ensure_required_files()
    assert writer.raw_trials_path.is_file()
    assert (writer.directory / "summary.csv").is_file()
    assert (writer.directory / "gap_audit.csv").is_file()
    assert (writer.directory / "fit_diagnostics.json").is_file()
    with pytest.raises(FileExistsError):
        ProfilerArtifactWriter.create(tmp_path, "20260826_000000_000000")


def test_profiler_has_no_policy_or_tracelab_selection_dependency() -> None:
    """Profiler 源码不得读取任何策略选择或策略表现。"""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "select_global_lru",
        "KVFlowStylePolicy",
        "MarconiStylePolicy",
        "GlobalOptimizer(",
        "tracelab_nontrivial_protocol.json",
        "selected_main_snapshots",
        "policy_performance",
    )
    assert all(value not in source for value in forbidden)


def test_engine_configuration_changes_only_admission_related_limit() -> None:
    """128K 扩展不得改变模型与 Hybrid/Mamba cache 基本语义。"""
    assert ENGINE_CONFIGURATION_128K["model_path"] == "/model"
    assert ENGINE_CONFIGURATION_128K["tp_size"] == 1
    assert ENGINE_CONFIGURATION_128K["context_length"] == 131_200
    assert ENGINE_CONFIGURATION_128K["chunked_prefill_size"] == 45_056
    assert ENGINE_CONFIGURATION_128K["mamba_radix_cache_strategy"] == "extra_buffer"
    assert ENGINE_CONFIGURATION_128K["mamba_track_interval"] == 256
    assert ENGINE_CONFIGURATION_128K["mamba_max_states_per_path"] == -1


def test_formal_phi_policies_workloads_and_protocol_are_unchanged() -> None:
    """正式 Phi、策略、工作负载、TraceLab 协议与 motivation 必须不变。"""
    for relative_path, expected_hash in PROTECTED_HASHES.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_hash
