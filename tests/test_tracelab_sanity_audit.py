from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import shutil

import pytest

from evaluation.public_agent_trace import tracelab_sanity_audit
from evaluation.public_agent_trace.tracelab_sanity_audit import (
    FORMAL_RESULT_DIRECTORY,
    FROZEN_SANITY_AUDIT_DIRECTORY,
    audit_constrained_representative_case,
    benefit_decomposition,
    full_budget_analysis,
    load_formal_results,
    marconi_candidate_scores,
    select_constrained_representative_cases,
    select_representative_cases,
    verify_frozen_results,
    write_constrained_representative_artifacts,
    x4_tight_budget_analysis,
)
from flowstate.recovery_model import RecoveryCostModel


@pytest.fixture(scope="module")
def frozen_data():
    return load_formal_results()


def test_formal_results_are_exact_frozen_artifact() -> None:
    assert FORMAL_RESULT_DIRECTORY.name == (
        "formal_policy_results_20260827_075548_356403"
    )
    assert FORMAL_RESULT_DIRECTORY.is_dir()


def test_formal_results_are_read_only_during_load(frozen_data) -> None:
    before = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in FORMAL_RESULT_DIRECTORY.iterdir()
        if item.is_file()
    }
    load_formal_results()
    after = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in FORMAL_RESULT_DIRECTORY.iterdir()
        if item.is_file()
    }
    assert before == after


def test_snapshot_and_result_counts_are_frozen(frozen_data) -> None:
    main, secondary, rows = frozen_data
    assert len(main) == 105
    assert len(secondary) == 37
    assert len([row for row in rows if row.cohort == "main"]) == 1_680
    assert len([row for row in rows if row.cohort == "secondary_x4"]) == 592


def test_marconi_and_other_frozen_results_recompute_without_mismatch(
    frozen_data,
) -> None:
    main, _, rows = frozen_data
    main_rows = tuple(row for row in rows if row.cohort == "main")
    verification = verify_frozen_results(main, main_rows)
    assert verification == {
        "metric_mismatches": 0,
        "lru_selection_mismatches": 0,
        "kvflow_selection_mismatches": 0,
        "marconi_selection_mismatches": 0,
        "implementation_bug_found": False,
    }


def test_representative_case_selection_has_required_categories(frozen_data) -> None:
    main, _, rows = frozen_data
    main_rows = tuple(row for row in rows if row.cohort == "main")
    selected = select_representative_cases(main, main_rows)
    assert sum(item["category"] == "clear_advantage" for item in selected) == 3
    assert sum(item["category"] == "tie_or_close" for item in selected) == 3
    assert sum(item["category"] == "x4_tight_budget" for item in selected) == 2
    assert len({(item["snapshot_id"], item["budget_ratio"]) for item in selected}) == 8


def test_constrained_representative_cases_are_frozen_and_deterministic(
    frozen_data,
) -> None:
    main, _, rows = frozen_data
    main_rows = tuple(row for row in rows if row.cohort == "main")
    first = select_constrained_representative_cases(main, main_rows)
    second = select_constrained_representative_cases(main, tuple(reversed(main_rows)))
    assert first == second
    assert tuple(item["category"] for item in first) == (
        "25% budget",
        "50% budget",
        "75% budget",
        "X>=4 constrained budget",
    )
    assert tuple(item["budget_ratio"] for item in first) == (
        0.25,
        0.50,
        0.75,
        0.25,
    )
    assert tuple(item["snapshot_id"] for item in first) == (
        "c128-small-claude-round-28601",
        "c128-medium-claude-round-18027",
        "c128-medium-claude-round-67076",
        "c128-medium-claude-round-94304",
    )
    assert len({item["snapshot_id"] for item in first}) == 4
    assert all(item["budget_ratio"] < 1.0 for item in first)
    assert all(item["cost_difference_ms"] > 0.0 for item in first)
    assert first[-1]["x"] >= 4


def test_constrained_case_details_reuse_frozen_policy_rows(frozen_data) -> None:
    main, _, rows = frozen_data
    main_rows = tuple(row for row in rows if row.cohort == "main")
    index = tracelab_sanity_audit._main_result_index(main_rows)
    descriptors = select_constrained_representative_cases(main, main_rows)
    for descriptor in descriptors:
        snapshot = main[descriptor["snapshot_id"]]
        item = audit_constrained_representative_case(
            descriptor,
            snapshot,
            index,
        )
        marconi = index[
            (snapshot.snapshot_id, descriptor["budget_ratio"], "Marconi-style")
        ]
        flow = index[
            (snapshot.snapshot_id, descriptor["budget_ratio"], "FlowState")
        ]
        assert item["marconi_selection"] == marconi.selected_checkpoint_ids
        assert item["flowstate_selection"] == flow.selected_checkpoint_ids
        assert item["marconi_total_recovery_cost_ms"] == (
            marconi.total_formal_recovery_cost_ms
        )
        assert item["flowstate_total_recovery_cost_ms"] == (
            flow.total_formal_recovery_cost_ms
        )
        assert item["policy_rerun"] is False
        assert "结构性描述，不是因果结论" in item["mechanism_explanation"]


def test_constrained_artifact_marks_full_budget_as_sanity_only(
    frozen_data,
    tmp_path: Path,
) -> None:
    del frozen_data
    audit = tmp_path / "audit"
    audit.mkdir()
    for name in (
        "marconi_sanity.json",
        "selected_case_audit.csv",
        "full_budget_analysis.json",
    ):
        shutil.copyfile(FROZEN_SANITY_AUDIT_DIRECTORY / name, audit / name)
    cases = write_constrained_representative_artifacts(
        FORMAL_RESULT_DIRECTORY,
        audit,
    )
    assert len(cases) == 4
    assert (audit / "constrained_representative_cases.csv").is_file()
    report = (audit / "constrained_representative_cases.md").read_text(
        encoding="utf-8"
    )
    assert "demand-sufficient sanity examples" in report
    assert "不作为 main benefit examples" in report
    assert "没有重新运行策略" in report


def test_marconi_utility_is_recency_plus_flop_efficiency(frozen_data) -> None:
    main, _, _ = frozen_data
    snapshot = main[sorted(main)[0]]
    scores = marconi_candidate_scores(snapshot)
    assert scores
    for item in scores:
        assert item["utility"] == pytest.approx(
            item["normalized_recency"]
            + item["normalized_flop_efficiency"]
        )


def test_benefit_decomposition_is_descriptive_and_complete(frozen_data) -> None:
    main, _, rows = frozen_data
    main_rows = tuple(row for row in rows if row.cohort == "main")
    result = benefit_decomposition(main, main_rows, RecoveryCostModel())
    counts = result["counts"]
    assert counts["different_selection_cases"] > 0
    assert (
        counts["flowstate_cost_advantage_cases"]
        + counts["objective_tie_cases"]
        == counts["different_selection_cases"]
    )
    assert result["definitions"]["overlap_allowed"] is True
    assert "因果 ablation" in result["interpretation"]


def test_x4_tight_budget_uses_pre_registered_slice(frozen_data) -> None:
    main, _, rows = frozen_data
    main_rows = tuple(row for row in rows if row.cohort == "main")
    result = x4_tight_budget_analysis(main, main_rows, RecoveryCostModel())
    assert result["snapshot_count"] == 13
    assert tuple(result["budgets"]) == ("25%", "50%", "75%")
    assert result["budgets"]["25%"]["mean_k"] == 1.0
    assert result["budgets"]["25%"]["relative_cost_reduction"] == pytest.approx(
        0.02697007362513316
    )


def test_full_budget_is_demand_sufficient_not_candidate_full(frozen_data) -> None:
    main, _, rows = frozen_data
    main_rows = tuple(row for row in rows if row.cohort == "main")
    result = full_budget_analysis(main, main_rows)
    assert result["condition"] == "K=X"
    assert result["flowstate_exact_parent_set_violations"] == 0
    assert result["flowstate_all_gaps_zero"] is True
    assert result["policies"]["FlowState"]["missing_exact_parent_demands"] == 0
    assert result["policies"]["Marconi-style"]["missing_exact_parent_demands"] > 0
    assert "NOT CORE PERFORMANCE CLAIM" in result["claim_boundary"]


def test_audit_has_no_gpu_database_resampling_or_policy_execution_dependency() -> None:
    source = inspect.getsource(tracelab_sanity_audit)
    assert "GlobalOptimizer" not in source
    assert "select_global_lru" not in source
    assert "KVFlowStylePolicy" not in source
    assert "MarconiStylePolicy" not in source
    assert "open_database_read_only" not in source
    assert "SGLangAdapter" not in source
    assert "import torch" not in source
    assert "import random" not in source


def test_formal_model_and_protocol_paths_are_not_outputs() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    protected = (
        repository_root / "flowstate/recovery_model.py",
        repository_root
        / "evaluation/public_agent_trace/tracelab_nontrivial_protocol.json",
    )
    assert all(path.is_file() for path in protected)
    assert all(FORMAL_RESULT_DIRECTORY not in path.parents for path in protected)
