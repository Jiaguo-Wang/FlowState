from evaluation.openhands_recurrent_recovery_paired_latency import (
    ENGINE_CONFIGURATION_PAIRED_LATENCY,
    TRIAL_ORDER,
    evaluate_trial,
    summarize_values,
)
from evaluation.recovery_profiler_128k import ENGINE_CONFIGURATION_128K


def test_trial_order_is_deterministic_and_alternating() -> None:
    assert TRIAL_ORDER == (
        ("CONTROL", 1),
        ("EVICT", 1),
        ("CONTROL", 2),
        ("EVICT", 2),
        ("CONTROL", 3),
        ("EVICT", 3),
        ("CONTROL", 4),
        ("EVICT", 4),
        ("CONTROL", 5),
        ("EVICT", 5),
    )


def test_dedicated_configuration_only_overrides_physical_pool() -> None:
    differences = {
        key
        for key in ENGINE_CONFIGURATION_PAIRED_LATENCY
        if ENGINE_CONFIGURATION_PAIRED_LATENCY[key]
        != ENGINE_CONFIGURATION_128K.get(key)
    }
    assert differences == {"max_mamba_cache_size"}
    assert ENGINE_CONFIGURATION_PAIRED_LATENCY["max_mamba_cache_size"] == 28
    assert ENGINE_CONFIGURATION_128K["max_mamba_cache_size"] == 24


def test_summary_statistics_use_sample_standard_deviation() -> None:
    result = summarize_values([1.0, 2.0, 3.0])
    assert result["count"] == 3
    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0
    assert result["sample_std"] == 1.0


def _trial(condition: str, h_value: int, e_value: int, g_value: int) -> dict:
    return {
        "condition": condition,
        "measured_request": {
            "request_completed": True,
            "token_count_exact": True,
            "runtime_metrics_valid": True,
            "h": h_value,
            "e": e_value,
            "g": g_value,
        },
        "eviction": {"correctness_pass": True},
        "setup_requests_valid": True,
        "native_mamba_capacity_eviction": False,
        "fa_kv_cascade": False,
        "oom": False,
        "truncation_or_clipping": False,
        "error": None,
        "engine_shutdown_error": None,
    }


def test_control_trial_requires_zero_gap() -> None:
    assert evaluate_trial(_trial("CONTROL", 128, 128, 0))["valid"] is True
    assert evaluate_trial(_trial("CONTROL", 128, 64, 64))["valid"] is False


def test_evict_trial_requires_positive_gap() -> None:
    assert evaluate_trial(_trial("EVICT", 128, 64, 64))["valid"] is True
    assert evaluate_trial(_trial("EVICT", 128, 128, 0))["valid"] is False


def test_safety_failure_keeps_trial_but_marks_invalid() -> None:
    trial = _trial("EVICT", 128, 64, 64)
    trial["native_mamba_capacity_eviction"] = True
    result = evaluate_trial(trial)
    assert result["status"] == "INVALID"
    assert result["measured_request"]["g"] == 64
