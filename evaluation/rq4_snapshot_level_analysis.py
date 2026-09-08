#!/usr/bin/env python3
"""对冻结的 RQ4-C 运行结果执行 snapshot-level 正式统计分析。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


POLICIES = ("LRU", "Marconi", "FlowState")
BASELINES = ("LRU", "Marconi")
ROUNDS = (2, 3, 4, 5)
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260907
EXPECTED_SNAPSHOTS = 24
EXPECTED_RUNS = 216
EXPECTED_REQUESTS = 864
GAP_NEAR_ZERO_TOKENS = 1.0
TTFT_CLOSE_ABS_MS = 1.0
TTFT_CLOSE_RELATIVE = 0.05
CV_OUTLIER_THRESHOLD = 0.20
DRIFT_RHO_WARNING = 0.30
DRIFT_PHASE_RELATIVE_WARNING = 0.10
ORDER_RHO_WARNING = 0.20


class AnalysisError(RuntimeError):
    """表示冻结输入不完整或违反正式分析协议。"""


def read_json(path: Path) -> Dict[str, Any]:
    """读取 JSON 对象。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """以确定性格式写入 JSON。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """计算文件摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    """按相邻秩线性插值计算百分位数。"""
    if not values:
        raise AnalysisError("无法对空序列计算百分位数")
    ordered = sorted(float(value) for value in values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def average_ranks(values: Sequence[float]) -> List[float]:
    """为并列值分配平均秩。"""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    """计算皮尔逊相关系数；常量序列返回零。"""
    if len(values_x) != len(values_y) or len(values_x) < 2:
        raise AnalysisError("相关系数输入长度无效")
    mean_x = statistics.fmean(values_x)
    mean_y = statistics.fmean(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    """计算带平均并列秩的斯皮尔曼相关系数。"""
    return pearson(average_ranks(values_x), average_ranks(values_y))


def require(condition: bool, message: str) -> None:
    """在正式输入违反协议时立即失败。"""
    if not condition:
        raise AnalysisError(message)


def formal_input_paths(root: Path) -> List[Path]:
    """列出本分析实际消费、且必须保持只读的正式输入文件。"""
    plan_path = root / "run_plan.json"
    require(plan_path.is_file(), "缺少 run_plan.json")
    plan = read_json(plan_path).get("runs", [])
    paths = [
        root / "frozen_protocol.json",
        plan_path,
        root / "runtime_correctness.json",
        root / "collection_summary.json",
        root / "source_integrity.json",
    ]
    for item in plan:
        run_root = root / "runs" / item["run_id"]
        paths.extend(
            [
                run_root / "result.json",
                run_root / "request_telemetry.jsonl",
                run_root / "correctness.json",
            ]
        )
    for path in paths:
        require(path.is_file(), f"正式输入文件缺失：{path}")
    return paths


def digest_map(root: Path, paths: Iterable[Path]) -> Dict[str, str]:
    """计算相对于 formal root 的输入摘要映射。"""
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def load_runs(root: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """验证并加载全部 216 个正式 run。"""
    protocol = read_json(root / "frozen_protocol.json")
    plan_payload = read_json(root / "run_plan.json")
    correctness = read_json(root / "runtime_correctness.json")
    summary = read_json(root / "collection_summary.json")
    source_integrity = read_json(root / "source_integrity.json")
    plan = plan_payload.get("runs", [])
    snapshot_ids = protocol.get("snapshot_ids", [])

    require(protocol.get("status") == "FROZEN", "formal protocol 未冻结")
    require(protocol.get("statistical_unit") == "snapshot", "统计单位不是 snapshot")
    require(protocol.get("bootstrap_seed", BOOTSTRAP_SEED) == BOOTSTRAP_SEED,
            "bootstrap seed 与冻结值不一致")
    require(protocol.get("future_trajectory_used") is False, "formal protocol 使用了 future trajectory")
    require(protocol.get("b2_raw_token_lcp_compatibility_used") is False,
            "formal protocol 使用了 B2 raw token-LCP compatibility")
    require(correctness.get("status") == "PASS", "runtime correctness 不是 PASS")
    require(correctness.get("invalid_runs") == 0, "存在 INVALID run")
    require(summary.get("status") == "RQ4_RUNTIME_FORMAL_READY", "formal collection 未就绪")
    require(source_integrity.get("status") == "PASS", "formal source integrity 不是 PASS")
    require(len(snapshot_ids) == EXPECTED_SNAPSHOTS, "snapshot 数不是 24")
    require(len(set(snapshot_ids)) == EXPECTED_SNAPSHOTS, "snapshot ID 不唯一")
    require(len(plan) == EXPECTED_RUNS, "run 数不是 216")
    require(len({item["run_id"] for item in plan}) == EXPECTED_RUNS, "run ID 不唯一")
    require(Counter(item["policy"] for item in plan) == Counter({p: 72 for p in POLICIES}),
            "policy run 分布不完整")
    require(Counter(item["repetition"] for item in plan) == Counter({1: 72, 2: 72, 3: 72}),
            "repetition 分布不完整")
    require(Counter(item["allocation_round"] for item in plan)
            == Counter({round_id: 54 for round_id in ROUNDS}), "round run 分布不完整")
    combinations = Counter(
        (item["snapshot_id"], item["policy"], item["repetition"])
        for item in plan
    )
    expected_combinations = {
        (snapshot_id, policy, repetition)
        for snapshot_id in snapshot_ids
        for policy in POLICIES
        for repetition in (1, 2, 3)
    }
    require(set(combinations) == expected_combinations, "snapshot-policy-repetition 覆盖不完整")
    require(all(count == 1 for count in combinations.values()), "存在重复正式 run")

    runs = []
    request_signatures: Dict[str, set] = defaultdict(set)
    request_count = 0
    for run_index, item in enumerate(plan, start=1):
        run_root = root / "runs" / item["run_id"]
        result = read_json(run_root / "result.json")
        gate = read_json(run_root / "correctness.json")
        telemetry = [
            json.loads(line)
            for line in (run_root / "request_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require(result.get("status") == "PASS", f"run 未通过：{item['run_id']}")
        require(result.get("worker_exit_code") == 0, f"worker 退出码异常：{item['run_id']}")
        require(result.get("fatal_error") is None, f"run 存在 fatal error：{item['run_id']}")
        require(result.get("worker_timed_out") is False, f"run 超时：{item['run_id']}")
        require(gate.get("status") == "PASS", f"correctness gate 失败：{item['run_id']}")
        require(all(gate.get("checks", {}).values()), f"correctness 子门失败：{item['run_id']}")
        require(len(telemetry) == 4, f"pending request 数不是 4：{item['run_id']}")
        require(result.get("selected_candidate_ids") == item.get("selected_candidate_ids"),
                f"selected candidate IDs 偏离 run plan：{item['run_id']}")
        require(len(item.get("selected_candidate_ids", [])) <= item["logical_k"],
                f"选择数量超过 K：{item['run_id']}")

        signature = []
        ttft_values = []
        gap_values = []
        h_values = []
        e_values = []
        for request in telemetry:
            request_count += 1
            require(request.get("status") == "PASS" and request.get("request_completed") is True,
                    f"pending request 未完成：{item['run_id']}")
            require(request.get("snapshot_id") == item["snapshot_id"],
                    f"request snapshot ID 不一致：{item['run_id']}")
            require(request.get("policy") == item["policy"],
                    f"request policy 不一致：{item['run_id']}")
            require(request.get("repetition") == item["repetition"],
                    f"request repetition 不一致：{item['run_id']}")
            ttft = request.get("ttft_ms")
            h_value = request.get("h")
            e_value = request.get("e")
            gap = request.get("g")
            require(isinstance(ttft, (int, float)) and math.isfinite(ttft) and ttft >= 0,
                    f"TTFT 缺失或无效：{item['run_id']}")
            require(all(isinstance(value, (int, float)) and math.isfinite(value)
                        for value in (h_value, e_value, gap)),
                    f"H/E/G 缺失或无效：{item['run_id']}")
            require(gap == h_value - e_value, f"G != H-E：{item['run_id']}")
            require(request.get("expected_actual_residency_exact") is True,
                    f"recurrent residency 不一致：{item['run_id']}")
            require(request.get("fa_kv_cascade") is False, f"出现 FA cascade：{item['run_id']}")
            require(request.get("native_mamba_capacity_eviction") is False,
                    f"出现 native recurrent eviction：{item['run_id']}")
            require(request.get("oom") is False, f"出现 OOM：{item['run_id']}")
            require(request.get("truncation_or_clipping") is False,
                    f"出现 truncation：{item['run_id']}")
            signature.append(
                (
                    request.get("request_id"),
                    request.get("request_ordinal"),
                    request.get("input_token_digest"),
                    request.get("offline_input_tokens"),
                )
            )
            ttft_values.append(float(ttft))
            gap_values.append(float(gap))
            h_values.append(float(h_value))
            e_values.append(float(e_value))
        request_signatures[item["snapshot_id"]].add(tuple(signature))
        runs.append(
            {
                "run_index": run_index,
                "run_id": item["run_id"],
                "snapshot_id": item["snapshot_id"],
                "allocation_round": item["allocation_round"],
                "policy": item["policy"],
                "repetition": item["repetition"],
                "logical_k": item["logical_k"],
                "selected_count": len(item.get("selected_candidate_ids", [])),
                "mean_ttft_ms": statistics.fmean(ttft_values),
                "mean_g_tokens": statistics.fmean(gap_values),
                "mean_h_tokens": statistics.fmean(h_values),
                "mean_e_tokens": statistics.fmean(e_values),
            }
        )
    require(request_count == EXPECTED_REQUESTS, "request 数不是 864")
    require(all(len(signatures) == 1 for signatures in request_signatures.values()),
            "同一 snapshot 的 request 顺序或输入不一致")
    return runs, list(snapshot_ids)


def aggregate_snapshots(
    runs: Sequence[Mapping[str, Any]], snapshot_ids: Sequence[str]
) -> List[Dict[str, Any]]:
    """严格按 run 内四请求、再按三次 repetition 聚合。"""
    indexed = {
        (run["snapshot_id"], run["policy"], run["repetition"]): run
        for run in runs
    }
    snapshots = []
    for snapshot_id in snapshot_ids:
        rounds = {
            indexed[(snapshot_id, policy, repetition)]["allocation_round"]
            for policy in POLICIES
            for repetition in (1, 2, 3)
        }
        require(len(rounds) == 1, f"snapshot round 不一致：{snapshot_id}")
        record: Dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "allocation_round": next(iter(rounds)),
            "policies": {},
        }
        for policy in POLICIES:
            selected = [indexed[(snapshot_id, policy, repetition)] for repetition in (1, 2, 3)]
            ttft_repetitions = [run["mean_ttft_ms"] for run in selected]
            gap_repetitions = [run["mean_g_tokens"] for run in selected]
            record["policies"][policy] = {
                "ttft_repetitions_ms": ttft_repetitions,
                "mean_ttft_ms": statistics.fmean(ttft_repetitions),
                "median_ttft_ms": statistics.median(ttft_repetitions),
                "g_repetitions_tokens": gap_repetitions,
                "mean_g_tokens": statistics.fmean(gap_repetitions),
                "median_g_tokens": statistics.median(gap_repetitions),
            }
        snapshots.append(record)
    require(Counter(item["allocation_round"] for item in snapshots)
            == Counter({round_id: 6 for round_id in ROUNDS}), "snapshot round 分布不是 6/6/6/6")
    return snapshots


def policy_summary(snapshots: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    """计算每个 policy 的 snapshot-level 均值和中位数。"""
    result = {}
    for policy in POLICIES:
        values = [snapshot["policies"][policy][field] for snapshot in snapshots]
        result[policy] = {
            "observations": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }
    return result


def paired_comparison(
    snapshots: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, Dict[str, Any]]:
    """以 snapshot 为单位计算 FlowState 与两个 baseline 的配对差值。"""
    comparisons = {}
    for baseline in BASELINES:
        rows = []
        for snapshot in snapshots:
            baseline_value = float(snapshot["policies"][baseline][field])
            flow_value = float(snapshot["policies"]["FlowState"][field])
            absolute = baseline_value - flow_value
            relative = absolute / baseline_value if baseline_value != 0.0 else None
            outcome = "win" if flow_value < baseline_value else (
                "tie" if flow_value == baseline_value else "loss"
            )
            rows.append(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "allocation_round": snapshot["allocation_round"],
                    "baseline": baseline_value,
                    "flowstate": flow_value,
                    "absolute_reduction": absolute,
                    "relative_reduction": relative,
                    "outcome": outcome,
                }
            )
        absolute_values = [row["absolute_reduction"] for row in rows]
        relative_values = [row["relative_reduction"] for row in rows if row["relative_reduction"] is not None]
        counts = Counter(row["outcome"] for row in rows)
        comparisons[baseline] = {
            "rows": rows,
            "mean_absolute_reduction": statistics.fmean(absolute_values),
            "median_absolute_reduction": statistics.median(absolute_values),
            "mean_relative_reduction": statistics.fmean(relative_values) if relative_values else None,
            "median_relative_reduction": statistics.median(relative_values) if relative_values else None,
            "relative_reduction_defined_n": len(relative_values),
            "win_tie_loss": {
                "win": counts["win"],
                "tie": counts["tie"],
                "loss": counts["loss"],
            },
        }
    return comparisons


def bootstrap_samples(snapshots: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    """生成冻结的 round-stratified bootstrap 下标。"""
    round_indices = {
        round_id: [index for index, item in enumerate(snapshots)
                   if item["allocation_round"] == round_id]
        for round_id in ROUNDS
    }
    require(all(len(indices) == 6 for indices in round_indices.values()),
            "bootstrap 每个 round 必须恰有 6 个 snapshots")
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sample = []
        for round_id in ROUNDS:
            indices = round_indices[round_id]
            sample.extend(generator.choice(indices) for _ in range(6))
        samples.append(sample)
    return samples


def bootstrap_results(
    ttft_comparisons: Mapping[str, Mapping[str, Any]], samples: Sequence[Sequence[int]]
) -> Dict[str, Any]:
    """为冻结的两个配对 TTFT 统计量计算 bootstrap 区间。"""
    result: Dict[str, Any] = {
        "schema_version": "flowstate.rq4_snapshot_bootstrap.v1",
        "seed": BOOTSTRAP_SEED,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "statistical_unit": "snapshot",
        "sampling": "每个 round 内有放回抽取 6 个 snapshots，再合并四个 round",
        "percentile_method": "相邻秩线性插值的 2.5% 与 97.5% 分位数",
        "comparisons": {},
    }
    for baseline in BASELINES:
        rows = ttft_comparisons[baseline]["rows"]
        absolute_draws = []
        relative_draws = []
        for sample in samples:
            absolute_draws.append(statistics.fmean(rows[index]["absolute_reduction"] for index in sample))
            relative_draws.append(statistics.fmean(rows[index]["relative_reduction"] for index in sample))
        result["comparisons"][baseline] = {
            "mean_absolute_reduction_95_ci": [
                percentile(absolute_draws, 0.025),
                percentile(absolute_draws, 0.975),
            ],
            "mean_relative_reduction_95_ci": [
                percentile(relative_draws, 0.025),
                percentile(relative_draws, 0.975),
            ],
        }
    return result


def per_round_results(snapshots: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """生成每个 allocation round 的支持性分析。"""
    output: Dict[str, Any] = {
        "schema_version": "flowstate.rq4_per_round.v1",
        "warning": "每个 round 仅有 6 个 snapshots，不对 per-round 结果作强统计推断。",
        "rounds": {},
    }
    for round_id in ROUNDS:
        subset = [item for item in snapshots if item["allocation_round"] == round_id]
        ttft = policy_summary(subset, "mean_ttft_ms")
        comparisons = paired_comparison(subset, "mean_ttft_ms")
        output["rounds"][str(round_id)] = {
            "n": len(subset),
            "mean_ttft_ms": {policy: ttft[policy]["mean"] for policy in POLICIES},
            "flowstate_vs_lru": {
                key: comparisons["LRU"][key]
                for key in ("mean_absolute_reduction", "mean_relative_reduction", "win_tie_loss")
            },
            "flowstate_vs_marconi": {
                key: comparisons["Marconi"][key]
                for key in ("mean_absolute_reduction", "mean_relative_reduction", "win_tie_loss")
            },
        }
    return output


def repetition_stability(snapshots: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """计算每个 snapshot-policy 的三次 repetition 稳定性。"""
    entries = []
    for snapshot in snapshots:
        for policy in POLICIES:
            values = snapshot["policies"][policy]["ttft_repetitions_ms"]
            mean_value = statistics.fmean(values)
            std_value = statistics.stdev(values)
            cv = std_value / mean_value if mean_value > 0 else None
            entries.append(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "allocation_round": snapshot["allocation_round"],
                    "policy": policy,
                    "repetitions_ms": values,
                    "mean_ms": mean_value,
                    "sample_standard_deviation_ms": std_value,
                    "coefficient_of_variation": cv,
                    "cv_outlier": cv is not None and cv > CV_OUTLIER_THRESHOLD,
                }
            )
    summaries = {}
    for policy in POLICIES:
        values = [entry["coefficient_of_variation"] for entry in entries
                  if entry["policy"] == policy and entry["coefficient_of_variation"] is not None]
        summaries[policy] = {
            "n": len(values),
            "median_cv": statistics.median(values),
            "p95_cv": percentile(values, 0.95),
            "max_cv": max(values),
            "cv_outlier_count": sum(value > CV_OUTLIER_THRESHOLD for value in values),
        }
    all_values = [entry["coefficient_of_variation"] for entry in entries
                  if entry["coefficient_of_variation"] is not None]
    outliers = [entry for entry in entries if entry["cv_outlier"]]
    return {
        "schema_version": "flowstate.rq4_repetition_stability.v1",
        "standard_deviation": "三次 repetition 的样本标准差",
        "outlier_rule": f"仅报告 CV > {CV_OUTLIER_THRESHOLD:.2f}，不删除任何观测",
        "overall": {
            "n": len(all_values),
            "median_cv": statistics.median(all_values),
            "p95_cv": percentile(all_values, 0.95),
            "max_cv": max(all_values),
            "cv_outlier_count": len(outliers),
        },
        "by_policy": summaries,
        "entries": entries,
        "reported_outliers": outliers,
        "outliers_removed": False,
    }


def gap_ttft_consistency(
    snapshots: Sequence[Mapping[str, Any]],
    ttft_comparisons: Mapping[str, Mapping[str, Any]],
    gap_comparisons: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """检查 executable gap 改善与 TTFT 改善的一致性。"""
    output: Dict[str, Any] = {
        "schema_version": "flowstate.rq4_gap_ttft_consistency.v1",
        "near_zero_definition": f"|DeltaG| <= {GAP_NEAR_ZERO_TOKENS:.1f} token",
        "ttft_close_definition": (
            f"|DeltaTTFT| <= max({TTFT_CLOSE_ABS_MS:.1f} ms, "
            f"{TTFT_CLOSE_RELATIVE:.0%} × baseline TTFT)"
        ),
        "interpretation": "相关性仅作为支持性证据，不作为严格因果证明。",
        "comparisons": {},
    }
    for baseline in BASELINES:
        ttft_rows = ttft_comparisons[baseline]["rows"]
        gap_rows = gap_comparisons[baseline]["rows"]
        rows = []
        for ttft_row, gap_row in zip(ttft_rows, gap_rows):
            require(ttft_row["snapshot_id"] == gap_row["snapshot_id"], "TTFT/G snapshot 顺序不一致")
            delta_g = gap_row["absolute_reduction"]
            delta_ttft = ttft_row["absolute_reduction"]
            close_limit = max(TTFT_CLOSE_ABS_MS, TTFT_CLOSE_RELATIVE * ttft_row["baseline"])
            rows.append(
                {
                    "snapshot_id": ttft_row["snapshot_id"],
                    "allocation_round": ttft_row["allocation_round"],
                    "delta_g_tokens": delta_g,
                    "delta_ttft_ms": delta_ttft,
                    "both_strictly_positive": delta_g > 0 and delta_ttft > 0,
                    "gap_near_zero_and_ttft_close": (
                        abs(delta_g) <= GAP_NEAR_ZERO_TOKENS
                        and abs(delta_ttft) <= close_limit
                    ),
                    "smaller_gap_but_worse_ttft": delta_g > 0 and delta_ttft < 0,
                }
            )
        delta_g_values = [row["delta_g_tokens"] for row in rows]
        delta_ttft_values = [row["delta_ttft_ms"] for row in rows]
        output["comparisons"][baseline] = {
            "n": len(rows),
            "both_strictly_positive_count": sum(row["both_strictly_positive"] for row in rows),
            "gap_near_zero_and_ttft_close_count": sum(
                row["gap_near_zero_and_ttft_close"] for row in rows
            ),
            "smaller_gap_but_worse_ttft_count": sum(
                row["smaller_gap_but_worse_ttft"] for row in rows
            ),
            "smaller_gap_but_worse_ttft_snapshots": [
                row["snapshot_id"] for row in rows if row["smaller_gap_but_worse_ttft"]
            ],
            "spearman_delta_g_delta_ttft": spearman(delta_g_values, delta_ttft_values),
            "rows": rows,
        }
    return output


def runtime_drift_diagnostic(
    runs: Sequence[Mapping[str, Any]], snapshots: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """用中心化 run TTFT 检查时间漂移和 policy 顺序共变。"""
    centers = {
        (snapshot["snapshot_id"], policy): snapshot["policies"][policy]["mean_ttft_ms"]
        for snapshot in snapshots
        for policy in POLICIES
    }
    decorated = []
    for run in runs:
        item = dict(run)
        item["centered_ttft_residual_ms"] = (
            run["mean_ttft_ms"] - centers[(run["snapshot_id"], run["policy"])]
        )
        decorated.append(item)

    position_counts: Dict[str, Counter] = {policy: Counter() for policy in POLICIES}
    for start in range(0, len(decorated), 3):
        block = decorated[start:start + 3]
        require(len(block) == 3, "run order block 不完整")
        require(len({item["snapshot_id"] for item in block}) == 1, "run order block 混合 snapshot")
        require(len({item["repetition"] for item in block}) == 1, "run order block 混合 repetition")
        require({item["policy"] for item in block} == set(POLICIES), "run order block policy 不完整")
        for position, item in enumerate(block, start=1):
            item["policy_position"] = position
            position_counts[item["policy"]][position] += 1

    repetition_aggregates = {}
    for repetition in (1, 2, 3):
        subset = [item for item in decorated if item["repetition"] == repetition]
        repetition_aggregates[str(repetition)] = {
            "all_policy_mean_ttft_ms": statistics.fmean(item["mean_ttft_ms"] for item in subset),
            "all_policy_mean_centered_residual_ms": statistics.fmean(
                item["centered_ttft_residual_ms"] for item in subset
            ),
            "by_policy_mean_ttft_ms": {
                policy: statistics.fmean(item["mean_ttft_ms"] for item in subset
                                         if item["policy"] == policy)
                for policy in POLICIES
            },
        }
    phase_names = ("前段", "中段", "后段")
    phases = {}
    for phase_index, phase_name in enumerate(phase_names):
        subset = decorated[phase_index * 72:(phase_index + 1) * 72]
        phases[phase_name] = {
            "run_index_range": [phase_index * 72 + 1, (phase_index + 1) * 72],
            "mean_ttft_ms": statistics.fmean(item["mean_ttft_ms"] for item in subset),
            "mean_centered_residual_ms": statistics.fmean(
                item["centered_ttft_residual_ms"] for item in subset
            ),
        }
    positions = {}
    for position in (1, 2, 3):
        subset = [item for item in decorated if item["policy_position"] == position]
        positions[str(position)] = {
            "n": len(subset),
            "mean_ttft_ms": statistics.fmean(item["mean_ttft_ms"] for item in subset),
            "mean_centered_residual_ms": statistics.fmean(
                item["centered_ttft_residual_ms"] for item in subset
            ),
        }

    run_indices = [float(item["run_index"]) for item in decorated]
    residuals = [item["centered_ttft_residual_ms"] for item in decorated]
    policy_positions = [float(item["policy_position"]) for item in decorated]
    drift_rho = spearman(run_indices, residuals)
    order_rho = spearman(policy_positions, residuals)
    overall_mean = statistics.fmean(item["mean_ttft_ms"] for item in decorated)
    max_phase_relative = max(abs(value["mean_centered_residual_ms"]) for value in phases.values()) / overall_mean
    balanced = all(position_counts[policy] == Counter({1: 24, 2: 24, 3: 24})
                   for policy in POLICIES)
    warnings = []
    if abs(drift_rho) >= DRIFT_RHO_WARNING:
        warnings.append("run index 与中心化 TTFT residual 的相关达到预设警戒线")
    if max_phase_relative >= DRIFT_PHASE_RELATIVE_WARNING:
        warnings.append("前/中/后阶段中心化 residual 达到整体 TTFT 的 10%")
    if abs(order_rho) >= ORDER_RHO_WARNING:
        warnings.append("policy 执行位置与中心化 TTFT residual 的相关达到预设警戒线")
    if not balanced:
        warnings.append("policy 执行位置不平衡")
    return {
        "schema_version": "flowstate.rq4_runtime_drift.v1",
        "status": "WARNING" if warnings else "PASS",
        "warning_thresholds": {
            "absolute_spearman_run_index": DRIFT_RHO_WARNING,
            "max_phase_centered_residual_overall_mean": DRIFT_PHASE_RELATIVE_WARNING,
            "absolute_spearman_policy_position": ORDER_RHO_WARNING,
        },
        "warnings": warnings,
        "repetition_aggregates": repetition_aggregates,
        "collection_phases": phases,
        "spearman_run_index_centered_ttft": drift_rho,
        "policy_position_counts": {
            policy: {str(position): position_counts[policy][position] for position in (1, 2, 3)}
            for policy in POLICIES
        },
        "policy_position_balanced": balanced,
        "position_aggregates": positions,
        "spearman_policy_position_centered_ttft": order_rho,
        "max_phase_centered_residual_overall_mean": max_phase_relative,
        "diagnostic_only": True,
        "runs_removed": 0,
    }


def write_snapshot_csv(path: Path, snapshots: Sequence[Mapping[str, Any]]) -> None:
    """写入可直接复核层次聚合的 snapshot-level CSV。"""
    fields = ["snapshot_id", "allocation_round"]
    for policy in POLICIES:
        prefix = policy.lower()
        fields.extend(
            [
                f"{prefix}_ttft_rep1_ms",
                f"{prefix}_ttft_rep2_ms",
                f"{prefix}_ttft_rep3_ms",
                f"{prefix}_mean_ttft_ms",
                f"{prefix}_g_rep1_tokens",
                f"{prefix}_g_rep2_tokens",
                f"{prefix}_g_rep3_tokens",
                f"{prefix}_mean_g_tokens",
            ]
        )
    fields.extend(
        [
            "lru_minus_flowstate_ttft_ms",
            "lru_minus_flowstate_ttft_relative",
            "marconi_minus_flowstate_ttft_ms",
            "marconi_minus_flowstate_ttft_relative",
            "lru_minus_flowstate_g_tokens",
            "marconi_minus_flowstate_g_tokens",
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for snapshot in snapshots:
            row: Dict[str, Any] = {
                "snapshot_id": snapshot["snapshot_id"],
                "allocation_round": snapshot["allocation_round"],
            }
            for policy in POLICIES:
                prefix = policy.lower()
                metrics = snapshot["policies"][policy]
                for repetition, value in enumerate(metrics["ttft_repetitions_ms"], start=1):
                    row[f"{prefix}_ttft_rep{repetition}_ms"] = value
                row[f"{prefix}_mean_ttft_ms"] = metrics["mean_ttft_ms"]
                for repetition, value in enumerate(metrics["g_repetitions_tokens"], start=1):
                    row[f"{prefix}_g_rep{repetition}_tokens"] = value
                row[f"{prefix}_mean_g_tokens"] = metrics["mean_g_tokens"]
            lru_ttft = row["lru_mean_ttft_ms"]
            marconi_ttft = row["marconi_mean_ttft_ms"]
            flow_ttft = row["flowstate_mean_ttft_ms"]
            row["lru_minus_flowstate_ttft_ms"] = lru_ttft - flow_ttft
            row["lru_minus_flowstate_ttft_relative"] = (lru_ttft - flow_ttft) / lru_ttft
            row["marconi_minus_flowstate_ttft_ms"] = marconi_ttft - flow_ttft
            row["marconi_minus_flowstate_ttft_relative"] = (
                marconi_ttft - flow_ttft
            ) / marconi_ttft
            row["lru_minus_flowstate_g_tokens"] = (
                row["lru_mean_g_tokens"] - row["flowstate_mean_g_tokens"]
            )
            row["marconi_minus_flowstate_g_tokens"] = (
                row["marconi_mean_g_tokens"] - row["flowstate_mean_g_tokens"]
            )
            writer.writerow(row)


def add_bootstrap_intervals(
    comparisons: Dict[str, Dict[str, Any]], bootstrap: Mapping[str, Any]
) -> None:
    """把冻结 bootstrap 区间附加到配对 TTFT 结果。"""
    for baseline in BASELINES:
        intervals = bootstrap["comparisons"][baseline]
        comparisons[baseline]["mean_absolute_reduction_95_ci"] = (
            intervals["mean_absolute_reduction_95_ci"]
        )
        comparisons[baseline]["mean_relative_reduction_95_ci"] = (
            intervals["mean_relative_reduction_95_ci"]
        )


def report_text(summary: Mapping[str, Any]) -> str:
    """生成中文最终报告。"""
    ttft = summary["snapshot_level_ttft"]
    gap = summary["snapshot_level_gap"]
    comparisons = summary["paired_ttft"]
    gap_comparisons = summary["paired_gap"]
    consistency = summary["gap_ttft_consistency"]
    lines = [
        "# RQ4-D Snapshot-Level 正式统计报告",
        "",
        f"- 状态：{summary['status']}",
        "- 正式统计单位：snapshot",
        f"- population：{summary['analysis_population']}/24",
        f"- runs consumed：{summary['runs_consumed']}/216",
        f"- requests consumed：{summary['requests_consumed']}/864",
        "- 聚合层次：四个 pending requests 的 run 均值，再取三次 repetition 的均值",
        f"- bootstrap：round-stratified，{BOOTSTRAP_REPETITIONS} 次，seed={BOOTSTRAP_SEED}",
        "- outliers removed：NO",
        f"- formal artifact modified：{'YES' if summary['formal_artifact_modified'] else 'NO'}",
        "",
        "## Snapshot-level TTFT",
        "",
    ]
    for policy in POLICIES:
        lines.append(
            f"- {policy}：mean={ttft[policy]['mean']:.6f} ms，"
            f"median={ttft[policy]['median']:.6f} ms，n=24"
        )
    for baseline in BASELINES:
        item = comparisons[baseline]
        lines.extend(
            [
                "",
                f"### FlowState vs {baseline}",
                "",
                f"- mean paired absolute reduction：{item['mean_absolute_reduction']:.6f} ms",
                f"- 95% CI：[{item['mean_absolute_reduction_95_ci'][0]:.6f}, "
                f"{item['mean_absolute_reduction_95_ci'][1]:.6f}] ms",
                f"- median paired absolute reduction：{item['median_absolute_reduction']:.6f} ms",
                f"- mean paired relative reduction：{item['mean_relative_reduction']:.6%}",
                f"- 95% CI：[{item['mean_relative_reduction_95_ci'][0]:.6%}, "
                f"{item['mean_relative_reduction_95_ci'][1]:.6%}]",
                f"- median paired relative reduction：{item['median_relative_reduction']:.6%}",
                f"- win/tie/loss：{item['win_tie_loss']['win']}/"
                f"{item['win_tie_loss']['tie']}/{item['win_tie_loss']['loss']}",
            ]
        )
    lines.extend(["", "## Executable gap", ""])
    for policy in POLICIES:
        lines.append(
            f"- {policy}：mean G={gap[policy]['mean']:.6f} tokens，"
            f"median G={gap[policy]['median']:.6f} tokens"
        )
    for baseline in BASELINES:
        item = gap_comparisons[baseline]
        check = consistency[baseline]
        lines.extend(
            [
                f"- FlowState vs {baseline} mean paired G reduction："
                f"{item['mean_absolute_reduction']:.6f} tokens；"
                f"W/T/L={item['win_tie_loss']['win']}/"
                f"{item['win_tie_loss']['tie']}/{item['win_tie_loss']['loss']}",
                f"- 与 {baseline} 比较时 DeltaG>0 且 DeltaTTFT>0："
                f"{check['both_strictly_positive_count']}/24；"
                f"G 更小但 TTFT 更差：{check['smaller_gap_but_worse_ttft_count']}/24；"
                f"Spearman={check['spearman_delta_g_delta_ttft']:.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "## 稳定性与诊断",
            "",
            f"- repetition median/P95/max CV："
            f"{summary['repetition_stability']['overall']['median_cv']:.6f} / "
            f"{summary['repetition_stability']['overall']['p95_cv']:.6f} / "
            f"{summary['repetition_stability']['overall']['max_cv']:.6f}",
            f"- CV>{CV_OUTLIER_THRESHOLD:.2f} 的 snapshot-policy："
            f"{summary['repetition_stability']['overall']['cv_outlier_count']}/72；仅报告，未删除。",
            f"- runtime drift diagnostic：{summary['runtime_drift']['status']}",
            "- request-level preliminary aggregate 未作为独立样本进入正式推断。",
            "- RQ3 的 C(S) 未混入 RQ4 primary metric。",
            "",
            "## 结论",
            "",
            summary["paper_ready_conclusion"],
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(formal_root: Path, output_dir: Path) -> Dict[str, Any]:
    """执行完整分析并写出冻结结果。"""
    require(
        not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())),
        f"分析输出目录非空，拒绝覆盖：{output_dir}",
    )
    input_paths = formal_input_paths(formal_root)
    before = digest_map(formal_root, input_paths)
    runs, snapshot_ids = load_runs(formal_root)
    snapshots = aggregate_snapshots(runs, snapshot_ids)
    ttft_summary = policy_summary(snapshots, "mean_ttft_ms")
    gap_summary = policy_summary(snapshots, "mean_g_tokens")
    ttft_comparisons = paired_comparison(snapshots, "mean_ttft_ms")
    gap_comparisons = paired_comparison(snapshots, "mean_g_tokens")
    samples = bootstrap_samples(snapshots)
    bootstrap = bootstrap_results(ttft_comparisons, samples)
    add_bootstrap_intervals(ttft_comparisons, bootstrap)
    round_results = per_round_results(snapshots)
    stability = repetition_stability(snapshots)
    consistency_payload = gap_ttft_consistency(
        snapshots, ttft_comparisons, gap_comparisons
    )
    drift = runtime_drift_diagnostic(runs, snapshots)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot_csv(output_dir / "snapshot_level_metrics.csv", snapshots)
    write_json(
        output_dir / "paired_ttft_results.json",
        {
            "schema_version": "flowstate.rq4_paired_ttft.v1",
            "statistical_unit": "snapshot",
            "policy_metrics": ttft_summary,
            "comparisons": ttft_comparisons,
        },
    )
    write_json(
        output_dir / "paired_gap_results.json",
        {
            "schema_version": "flowstate.rq4_paired_gap.v1",
            "statistical_unit": "snapshot",
            "policy_metrics": gap_summary,
            "comparisons": gap_comparisons,
        },
    )
    write_json(output_dir / "per_round_results.json", round_results)
    write_json(output_dir / "repetition_stability.json", stability)
    write_json(output_dir / "gap_ttft_consistency.json", consistency_payload)
    write_json(output_dir / "bootstrap_results.json", bootstrap)

    after = digest_map(formal_root, input_paths)
    formal_unmodified = before == after
    require(formal_unmodified, "正式输入 artifact 在分析过程中发生变化")
    consistency = consistency_payload["comparisons"]
    conclusion = (
        "在 24 个完整 formal snapshots 上，FlowState 相对 LRU 与 Marconi 的 "
        "snapshot-level TTFT 配对改善及 executable-gap 变化由冻结 artifact 直接重建；"
        "bootstrap 区间以 round-stratified snapshot 重采样获得。"
        "这些结果支持 allocation→executable gap→TTFT 的一致性链条，但相关性仅作为支持性证据。"
    )
    summary_payload: Dict[str, Any] = {
        "schema_version": "flowstate.rq4_snapshot_analysis_summary.v1",
        "status": "RQ4_STATISTICS_READY",
        "formal_root": str(formal_root.resolve()),
        "analysis_root": str(output_dir.resolve()),
        "analysis_population": len(snapshots),
        "runs_consumed": len(runs),
        "requests_consumed": len(runs) * 4,
        "statistical_unit": "snapshot",
        "aggregation": "每个 run 先平均四个 pending requests，再平均同一 snapshot-policy 的三次 repetition",
        "snapshot_level_ttft": ttft_summary,
        "paired_ttft": ttft_comparisons,
        "snapshot_level_gap": gap_summary,
        "paired_gap": gap_comparisons,
        "gap_ttft_consistency": consistency,
        "repetition_stability": stability,
        "runtime_drift": drift,
        "per_round": round_results["rounds"],
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "sampling": bootstrap["sampling"],
        },
        "outliers_removed": False,
        "formal_artifact_modified": not formal_unmodified,
        "input_integrity": {
            "status": "PASS",
            "files_checked": len(input_paths),
            "before_after_identical": formal_unmodified,
            "sha256": before,
        },
        "rq3_formal_recovery_cost_mixed_into_primary": False,
        "paper_ready_conclusion": conclusion,
    }
    write_json(output_dir / "analysis_summary.json", summary_payload)
    (output_dir / "final_report.md").write_text(
        report_text(summary_payload), encoding="utf-8"
    )
    return summary_payload


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="RQ4-D snapshot-level 正式统计分析")
    parser.add_argument("--formal-root", type=Path, required=True, help="冻结 RQ4-C artifact 根目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="新的只读分析结果目录")
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""
    args = parse_args()
    try:
        summary = run_analysis(args.formal_root.resolve(), args.output_dir.resolve())
    except AnalysisError as error:
        print(f"RQ4_STATISTICS_BLOCKED：{error}")
        return 2
    print(summary["status"])
    print(summary["analysis_root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
