from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_GATE_PATH = (
    Path(__file__).resolve().parent
    / "runtime"
    / "wp3b_end_to_end_gate.py"
)
_SPEC = spec_from_file_location("_flowstate_wp3b_end_to_end_gate", _GATE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("无法加载 WP3B 端到端 gate 模块")
_GATE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GATE
_PREVIOUS_DONT_WRITE_BYTECODE = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    _SPEC.loader.exec_module(_GATE)
finally:
    sys.dont_write_bytecode = _PREVIOUS_DONT_WRITE_BYTECODE


def test_missing_ttft_is_optional_and_correctness_continues() -> None:
    result = _GATE.validate_sibling_observation(
        "W1",
        {
            "physical_fa_hit": 32_769,
            "executable_prefix": 32_768,
            "replay_gap": 1,
        },
        {"e2e_latency": 0.125},
    )

    assert result["physical_hit"] == 32_769
    assert result["executable_prefix"] == 32_768
    assert result["gap"] == 1
    assert result["ttft_ms"] is None
    assert result["request_e2e_ms"] == 125.0
