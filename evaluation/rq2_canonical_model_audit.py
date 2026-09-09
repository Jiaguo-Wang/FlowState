#!/usr/bin/env python3
"""从冻结证据重算 RQ2 指标并核对下游目标函数身份。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.recovery_model_freeze import (
    fit_candidate_models,
    heldout_metrics,
    load_calibration_points,
    structural_validation,
)
from evaluation.rq3_agentx_formal_policy_evaluation import (
    build_agentx_formal_snapshots,
)
from evaluation.rq3_formal_policy_evaluation import load_eligible_snapshots
from evaluation.rq3_frozen_snapshot_evaluator import (
    _select_policy,
    evaluate_objective,
    recovery_model_identity,
    select_exact_opt,
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FREEZE_ROOT = (
    REPOSITORY_ROOT
    / "evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020"
)
OPENHANDS_ROOT = (
    REPOSITORY_ROOT
    / "evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017"
)
OPENHANDS_EVALUATION_ROOT = (
    REPOSITORY_ROOT
    / "evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011"
)
AGENTX_EVALUATION_ROOT = REPOSITORY_ROOT / "rq3_agentx_policy_eval_20260906_214956"
RQ4_ROOT = REPOSITORY_ROOT / "rq4_runtime_formal_output/rq4_runtime_formal_20260907_221652"
RQ5_ROOT = (
    REPOSITORY_ROOT
    / "evaluation/runtime_artifacts/rq5_frozen_snapshot_ablation_20260908_110726"
)
DEFAULT_OUTPUT_PARENT = REPOSITORY_ROOT / "evaluation/rq2_model_audit_output"


def _read_json(path: Path) -> Any:
    """读取一个 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取一个 UTF-8 CSV 文件。"""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    """流式计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """用稳定键序写入 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _numeric_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """比较两个恢复模型身份的数值与语义字段。"""
    float_fields = ("coefficient_a", "coefficient_b", "coefficient_c")
    exact_fields = (
        "name",
        "gap_unit",
        "target_unit",
        "output_unit",
        "calibration_artifact",
        "minimum_gap_tokens",
        "maximum_target_tokens",
    )
    return all(
        math.isclose(float(left[field]), float(right[field]), rel_tol=0.0, abs_tol=5e-7)
        for field in float_fields
    ) and all(left[field] == right[field] for field in exact_fields)


def _artifact_paths() -> dict[str, Path]:
    """列出审计必须读取的原始证据、执行源码与冻结报告。"""
    return {
        "rq2_calibration_position": REPOSITORY_ROOT
        / "evaluation/runtime_artifacts/recovery_position_audit_20260826_144654_852303/position_matrix_summary.csv",
        "rq2_calibration_long_gap": REPOSITORY_ROOT
        / "evaluation/runtime_artifacts/recovery_profiler_128k_20260826_133712_596813/summary.csv",
        "rq2_candidate_models": FREEZE_ROOT / "candidate_models.json",
        "rq2_model_selection": FREEZE_ROOT / "model_selection.json",
        "rq2_heldout_raw": FREEZE_ROOT / "heldout_raw.csv",
        "rq2_heldout_summary": FREEZE_ROOT / "heldout_summary.csv",
        "rq2_heldout_predictions": FREEZE_ROOT / "heldout_predictions.csv",
        "rq2_structural_validation": FREEZE_ROOT / "structural_validation.json",
        "rq2_fit_script": REPOSITORY_ROOT / "evaluation/recovery_model_freeze.py",
        "canonical_runtime_model": REPOSITORY_ROOT / "flowstate/recovery_model.py",
        "rq3_common_evaluator": REPOSITORY_ROOT
        / "evaluation/rq3_frozen_snapshot_evaluator.py",
        "rq3_formal_runner": REPOSITORY_ROOT
        / "evaluation/rq3_formal_policy_evaluation.py",
        "rq3_openhands_config": OPENHANDS_EVALUATION_ROOT / "EVALUATION_PROTOCOL.json",
        "rq3_openhands_results": OPENHANDS_EVALUATION_ROOT / "aggregate_results.json",
        "rq3_agentx_runner": REPOSITORY_ROOT
        / "evaluation/rq3_agentx_formal_policy_evaluation.py",
        "rq3_agentx_manifest": AGENTX_EVALUATION_ROOT / "manifest.json",
        "rq3_agentx_source_integrity": AGENTX_EVALUATION_ROOT / "source_integrity.json",
        "rq4_runtime_harness": REPOSITORY_ROOT
        / "evaluation/rq4_unified_runtime_harness.py",
        "rq4_frozen_protocol": RQ4_ROOT / "frozen_protocol.json",
        "rq4_source_integrity": RQ4_ROOT / "source_integrity.json",
        "rq5_ablation_runner": REPOSITORY_ROOT
        / "evaluation/rq5_frozen_snapshot_ablation.py",
        "rq5_manifest": RQ5_ROOT / "manifest.json",
        "rq5_source_integrity": RQ5_ROOT / "source_integrity.json",
    }


def _recompute_metrics(models: Mapping[str, Any]) -> dict[str, Any]:
    """从冻结预测逐行重算误差，并重新执行结构网格检查。"""
    prediction_rows = _read_csv(FREEZE_ROOT / "heldout_predictions.csv")
    metrics = heldout_metrics(prediction_rows)
    structural = structural_validation(models)
    raw_rows = _read_csv(FREEZE_ROOT / "heldout_raw.csv")
    measured = [row for row in raw_rows if row["is_warmup"] == "False"]
    successful = [
        row
        for row in measured
        if row["status"] == "PASS" and row["correctness_pass"] == "True"
    ]
    m2_structural = structural["models"]["M2"]
    return {
        "heldout_nonzero_configuration_count": len(prediction_rows),
        "heldout_measured_trial_count": len(measured),
        "heldout_successful_trial_count": len(successful),
        "models": metrics,
        "headline": metrics["M2"]["overall"],
        "structural": structural,
        "structural_headline": {
            "checked_points": m2_structural["checked_points"],
            "passed_points": (
                m2_structural["checked_points"]
                if m2_structural["status"] == "PASS"
                else 0
            ),
            "phi_zero": not m2_structural["phi_zero_failures"],
            "nonnegative": not m2_structural["negative_failures"],
            "fixed_target_monotonic": not m2_structural["monotonic_failures"],
            "status": m2_structural["status"],
        },
    }


def _coefficient_payload(models: Mapping[str, Any]) -> dict[str, Any]:
    """从 authoritative M2 文件生成唯一 canonical 模型记录。"""
    selected = _read_json(FREEZE_ROOT / "model_selection.json")["selected_model"]
    if selected != "M2":
        raise RuntimeError(f"冻结选择不是 M2：{selected}")
    parameters = models[selected]["parameters"]
    return {
        "status": "RQ2_CANONICAL_MODEL_READY",
        "model_name": "position_aware_quadratic_v1",
        "canonical_formula": "Phi(G,T) = theta1*g + theta2*g*tau + theta3*g^2",
        "g_definition": "G / 1024",
        "tau_definition": "T / 1024",
        "theta1": float(parameters["a"]),
        "theta2": float(parameters["b"]),
        "theta3": float(parameters["c"]),
        "valid_domain": {"minimum_gap_tokens": 0, "maximum_target_tokens": 131072, "constraint": "G <= T"},
        "authoritative_source": str(FREEZE_ROOT / "candidate_models.json"),
    }


def _consistency(canonical: Mapping[str, Any]) -> dict[str, Any]:
    """核对 RQ2、RQ3、Exact、RQ4 与 RQ5-A 的实际执行路径。"""
    expected = {
        "name": canonical["model_name"],
        "coefficient_a": canonical["theta1"],
        "coefficient_b": canonical["theta2"],
        "coefficient_c": canonical["theta3"],
        "gap_unit": "tokens",
        "target_unit": "tokens",
        "output_unit": "ms",
        "calibration_artifact": FREEZE_ROOT.name,
        "minimum_gap_tokens": 0,
        "maximum_target_tokens": 131072,
    }
    runtime = asdict(recovery_model_identity())
    openhands = load_eligible_snapshots(OPENHANDS_ROOT)
    openhands_identities = [asdict(item.recovery_model) for item in openhands]
    agentx, agentx_audit = build_agentx_formal_snapshots()
    agentx_identities = [asdict(item.snapshot.recovery_model) for item in agentx]

    exact_source = inspect.getsource(select_exact_opt)
    common_source = inspect.getsource(evaluate_objective)
    dispatcher_source = inspect.getsource(_select_policy)
    exact_same_objective = (
        "objective_function(snapshot, subset)" in exact_source
        and "select_exact_opt(" in dispatcher_source
        and "objective_function=objective_function" in dispatcher_source
    )
    common_uses_canonical = (
        "model = RecoveryCostModel()" in common_source
        and "model.estimate" in common_source
    )
    rq5_source = (REPOSITORY_ROOT / "evaluation/rq5_frozen_snapshot_ablation.py").read_text(
        encoding="utf-8"
    )
    rq5_full_same = (
        "full_order, full_native_cost = _select_greedy(variant, RecoveryCostModel())"
        in rq5_source
        and "full_objective = evaluate_objective(variant, full_order)" in rq5_source
    )
    rq4_source = (REPOSITORY_ROOT / "evaluation/rq4_unified_runtime_harness.py").read_text(
        encoding="utf-8"
    )
    rq4_same = (
        "_select_policy(policy, snapshot, evaluate_objective)" in rq4_source
        and "create_budget_variant(frozen, k)" in rq4_source
    )
    openhands_match = len(openhands) == 168 and all(
        _numeric_equal(identity, expected) for identity in openhands_identities
    )
    agentx_match = len(agentx) == 23 and all(
        _numeric_equal(identity, expected) for identity in agentx_identities
    )
    runtime_match = _numeric_equal(runtime, expected)
    coefficient_deltas = {
        "theta1": float(runtime["coefficient_a"]) - float(canonical["theta1"]),
        "theta2": float(runtime["coefficient_b"]) - float(canonical["theta2"]),
        "theta3": float(runtime["coefficient_c"]) - float(canonical["theta3"]),
    }
    return {
        "canonical_expected": expected,
        "runtime_model": runtime,
        "byte_level_equal_to_full_precision_fit": runtime == expected,
        "coefficient_deltas_runtime_minus_fit": coefficient_deltas,
        "numeric_tolerance": 5e-7,
        "runtime_numeric_match": runtime_match,
        "rq3_openhands": {
            "snapshot_count": len(openhands),
            "unique_model_identities": len(
                {json.dumps(value, sort_keys=True) for value in openhands_identities}
            ),
            "numeric_match": openhands_match,
        },
        "rq3_agentx": {
            "snapshot_count": len(agentx),
            "unique_model_identities": len(
                {json.dumps(value, sort_keys=True) for value in agentx_identities}
            ),
            "numeric_match": agentx_match,
            "assembly_gate": agentx_audit,
        },
        "common_objective_uses_canonical_model": common_uses_canonical,
        "exact_opt_uses_common_objective": exact_same_objective,
        "rq4_uses_rq3_selector_and_objective": rq4_same,
        "rq5_full_uses_canonical_selector_and_common_objective": rq5_full_same,
        "coefficient_consistency": (
            runtime_match and openhands_match and agentx_match
        ),
        "exact_objective_consistency": exact_same_objective and common_uses_canonical,
        "downstream_consistency": rq4_same and rq5_full_same,
    }


def _inventory(paths: Mapping[str, Path]) -> dict[str, Any]:
    """记录证据角色、绝对路径、大小与存在性。"""
    return {
        "priority": [
            "原始冻结 artifact",
            "可执行 evaluator/config",
            "生成报告",
            "Markdown 摘要",
            "论文文字",
        ],
        "authoritative_freeze_root": str(FREEZE_ROOT),
        "artifacts": {
            name: {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            }
            for name, path in paths.items()
        },
    }


def _report(
    canonical: Mapping[str, Any],
    metrics: Mapping[str, Any],
    consistency: Mapping[str, Any],
    ready: bool,
) -> str:
    """生成人工可读但不替代机器证据的最终审计报告。"""
    headline = metrics["headline"]
    structural = metrics["structural_headline"]
    status = "RQ2_CANONICAL_MODEL_READY" if ready else "RQ2_CANONICAL_MODEL_BLOCKED"
    theta1 = float(canonical["theta1"])
    theta2 = float(canonical["theta2"])
    theta3 = float(canonical["theta3"])
    formula = (
        f"{theta1:.6f} g "
        f"{'+' if theta2 >= 0 else '-'} {abs(theta2):.9f} g*tau "
        f"{'+' if theta3 >= 0 else '-'} {abs(theta3):.9f} g^2"
    )
    return "\n".join(
        [
            "# RQ2 冻结恢复模型独立审计",
            "",
            "本报告从冻结系数、held-out 逐行预测和实际下游 evaluator 重算，未使用论文草稿作为真值。",
            "",
            f"canonical formula = {formula}",
            f"theta1 = {canonical['theta1']:.15f}",
            f"theta2 = {canonical['theta2']:.15f}",
            f"theta3 = {canonical['theta3']:.15f}",
            "",
            f"RQ2 metrics reproduction = {'PASS' if metrics['headline_reproduced'] else 'FAIL'}",
            f"RQ2 vs RQ3 coefficient consistency = {'PASS' if consistency['coefficient_consistency'] else 'FAIL'}",
            f"Exact OPT objective consistency = {'PASS' if consistency['exact_objective_consistency'] else 'FAIL'}",
            "historical artifacts modified = NO",
            "",
            "## 重算结果",
            "",
            f"- held-out MAPE：{headline['mape_percent']:.12f}%",
            f"- held-out MAE：{headline['mae_ms']:.12f} ms",
            f"- max relative error：{headline['max_relative_error_percent']:.12f}%",
            f"- max absolute error：{headline['max_absolute_error_ms']:.12f} ms",
            f"- 结构网格：{structural['passed_points']}/{structural['checked_points']}，{structural['status']}",
            f"- Phi(0,T)=0：{'PASS' if structural['phi_zero'] else 'FAIL'}",
            f"- 非负性：{'PASS' if structural['nonnegative'] else 'FAIL'}",
            f"- 固定 T 单调性：{'PASS' if structural['fixed_target_monotonic'] else 'FAIL'}",
            "- 有效域：0 <= G <= T <= 131072",
            "",
            "## 下游一致性",
            "",
            f"- OpenHands 冻结快照：{consistency['rq3_openhands']['snapshot_count']} 个，身份一致：{consistency['rq3_openhands']['numeric_match']}",
            f"- AgentX 正式快照：{consistency['rq3_agentx']['snapshot_count']} 个，身份一致：{consistency['rq3_agentx']['numeric_match']}",
            f"- full-precision fit 与部署序列化逐字节相同：{consistency['byte_level_equal_to_full_precision_fit']}；差异仅为冻结发布精度舍入，数值门限为 {consistency['numeric_tolerance']}",
            f"- RQ4 复用 RQ3 selector/objective：{consistency['rq4_uses_rq3_selector_and_objective']}",
            f"- RQ5-A Full 复用 canonical selector/objective：{consistency['rq5_full_uses_canonical_selector_and_common_objective']}",
            "",
            f"最终状态：{status}",
            "",
        ]
    )


def run_audit(output_root: Path) -> Path:
    """执行只读审计并把新证据写入独立目录。"""
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在：{output_root}")
    paths = _artifact_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"审计证据缺失：{missing}")
    before = {name: _sha256(path) for name, path in paths.items()}
    models = _read_json(FREEZE_ROOT / "candidate_models.json")
    canonical = _coefficient_payload(models)
    metrics = _recompute_metrics(models)
    frozen_headline = _read_json(FREEZE_ROOT / "model_selection.json")[
        "heldout_metrics"
    ]["M2"]["overall"]
    metrics["headline_reproduced"] = all(
        math.isclose(
            float(metrics["headline"][field]),
            float(frozen_headline[field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for field in (
            "mape_percent",
            "mae_ms",
            "max_relative_error_percent",
            "max_absolute_error_ms",
        )
    )
    refitted = fit_candidate_models(load_calibration_points())
    metrics["calibration_refit_audit_only"] = {
        "note": "仅用冻结校准点重算以验证 artifact，未写回或替换任何模型。",
        "m2_parameters": refitted["M2"]["parameters"],
        "matches_frozen": all(
            math.isclose(
                float(refitted["M2"]["parameters"][field]),
                float(models["M2"]["parameters"][field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for field in ("a", "b", "c")
        ),
    }
    consistency = _consistency(canonical)
    after = {name: _sha256(path) for name, path in paths.items()}
    historical_unchanged = before == after
    structure = metrics["structural_headline"]
    ready = all(
        (
            metrics["headline_reproduced"],
            metrics["calibration_refit_audit_only"]["matches_frozen"],
            structure["checked_points"] == 147,
            structure["status"] == "PASS",
            consistency["coefficient_consistency"],
            consistency["exact_objective_consistency"],
            consistency["downstream_consistency"],
            historical_unchanged,
        )
    )
    canonical["status"] = (
        "RQ2_CANONICAL_MODEL_READY" if ready else "RQ2_CANONICAL_MODEL_BLOCKED"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "canonical_model.json", canonical)
    _write_json(output_root / "artifact_inventory.json", _inventory(paths))
    _write_json(
        output_root / "sha256_manifest.json",
        {
            "before": before,
            "after": after,
            "historical_artifacts_modified": not historical_unchanged,
        },
    )
    _write_json(output_root / "recomputed_metrics.json", metrics)
    _write_json(output_root / "rq2_rq3_consistency.json", consistency)
    (output_root / "final_report.md").write_text(
        _report(canonical, metrics, consistency, ready), encoding="utf-8"
    )
    if not ready:
        raise RuntimeError("RQ2 canonical 模型审计未通过")
    return output_root


def main(argv: Iterable[str] | None = None) -> int:
    """解析输出路径并运行审计。"""
    parser = argparse.ArgumentParser(description="RQ2 canonical 模型只读审计")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = args.output_root or (
        DEFAULT_OUTPUT_PARENT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    print(run_audit(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
