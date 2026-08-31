from evaluation.openhands_recurrent_only_causal_gate import (
    ENGINE_CONFIGURATION_CAUSAL_GATE,
    POST_EVICTION_SCHEDULE,
    PRE_EVICTION_SCHEDULE,
    SCHEDULE,
    _census_unexpected_mamba_change,
    audit_earlier_nonzero_gaps,
    exact_lcp,
    token_digest,
    validate_eviction_response,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K


def test_schedule_places_intervention_between_rounds_two_and_three() -> None:
    assert PRE_EVICTION_SCHEDULE == (
        ("A", 1),
        ("B", 1),
        ("C", 1),
        ("D", 1),
        ("A", 2),
        ("B", 2),
        ("C", 2),
        ("D", 2),
    )
    assert POST_EVICTION_SCHEDULE == (
        ("A", 3),
        ("B", 3),
        ("C", 3),
        ("D", 3),
    )
    assert len(SCHEDULE) == 12


def test_dedicated_configuration_only_overrides_physical_pool() -> None:
    differences = {
        key
        for key in ENGINE_CONFIGURATION_CAUSAL_GATE
        if ENGINE_CONFIGURATION_CAUSAL_GATE[key]
        != ENGINE_CONFIGURATION_128K.get(key)
    }
    assert differences == {"max_mamba_cache_size"}
    assert ENGINE_CONFIGURATION_CAUSAL_GATE["max_mamba_cache_size"] == 28
    assert ENGINE_CONFIGURATION_128K["max_mamba_cache_size"] == 24


def test_exact_lcp_and_digest_are_deterministic() -> None:
    assert exact_lcp([1, 2, 3], [1, 2, 4, 5]) == 2
    assert exact_lcp([1, 2], [1, 2, 3]) == 2
    assert token_digest([1, 2, 3]) == token_digest((1, 2, 3))


def test_earlier_gap_audit_distinguishes_cross_workflow_prefix() -> None:
    requests = [
        {"workflow_label": "A", "turn": 1, "input_ids": [1] * 2_500},
        {
            "workflow_label": "B",
            "turn": 1,
            "input_ids": [1] * 2_437 + [2],
        },
    ]
    result = audit_earlier_nonzero_gaps(requests)["B1"]
    assert result["max_cross_workflow_lcp"] == 2_437
    assert result["cross_workflow_prefix_possible"] is True
    assert result["causal_source_proven"] is False


def test_expected_recurrent_removal_is_not_native_eviction() -> None:
    census = {
        "removed_mamba_node_ids": [12],
        "changed_existing_mamba_node_ids": [],
    }
    assert _census_unexpected_mamba_change(census, 12) is False
    assert _census_unexpected_mamba_change(census, 13) is True


def test_formal_eviction_proof_requires_fa_and_identity_preservation() -> None:
    response = {
        "before": {
            "path": {
                "target_mamba_present": True,
                "target_full_present": True,
                "path_full_all_present": True,
            }
        },
        "after": {
            "path": {
                "target_mamba_present": False,
                "target_full_present": True,
                "path_full_all_present": True,
            }
        },
        "proof": {
            "same_node": True,
            "fa_unchanged": True,
            "path_unchanged": True,
            "tree_unchanged": True,
            "only_target_mamba_changed": True,
            "sanity_check": True,
            "cascade_called": False,
            "fa_identity_unchanged": True,
        },
    }
    assert validate_eviction_response(response) is True
    response["proof"]["fa_unchanged"] = False
    assert validate_eviction_response(response) is False
