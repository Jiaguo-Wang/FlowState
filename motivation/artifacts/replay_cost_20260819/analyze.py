"""Parse WP2 runtime evidence, validate it, fit replay cost, and plot it."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EVENTS = HERE / "driver_events.jsonl"
SERVER_LOG = HERE / "server_full.log"
FORMAL_LOG = HERE / "server_formal.log"
RAW_CSV = HERE / "raw_runs.csv"
GATE_CSV = HERE / "gate_validation.csv"
SUMMARY_CSV = HERE / "replay_cost.csv"
FIT_JSON = HERE / "fit_metrics.json"
PNG = HERE / "plot_replay_latency.png"
PDF = HERE / "plot_replay_latency.pdf"

GROUPS = (0, 1_024, 4_096, 8_192, 16_384, 32_768)

MATCH_RE = re.compile(
    r"\[FSVAL\] match_end req=(\S+) full_kv_hit=(\d+) "
    r"exec_prefix=(\d+) mamba_boundary=(\d+|None) "
    r"branching=(\d+|None) gap=(-?\d+)"
)
EXTEND_RE = re.compile(
    r"\[FSVAL\] extend req=(\S+) fill_ids=(\d+) prefix_len=(\d+) "
    r"extend_start=(\d+|None) extend_len=(\d+|None) extend_end=(\d+|None)"
)
TIMING_RE = re.compile(
    r"\[FSWP2\] request_timing req=(\S+) "
    r"first_token_latency_ms=([0-9.]+) e2e_latency_ms=([0-9.]+)"
)


def optional_int(value: str) -> int | None:
    return None if value == "None" else int(value)


def read_log() -> tuple[dict, dict, dict, dict, list[str]]:
    all_lines = SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    # The same long-lived server was used for an explicitly excluded pilot.
    # Anchor on the last gate build, then walk back to its successful flush, so
    # formal evidence cannot silently mix with earlier same-RID pilot records.
    gate_anchors = [
        index
        for index, line in enumerate(all_lines)
        if "[FSVAL] match_end req=gate_e8192_r0_S1 " in line
    ]
    if not gate_anchors:
        raise RuntimeError("formal gate anchor not found in server log")
    log_start = gate_anchors[-1]
    while (
        log_start > 0
        and "Cache flushed successfully!" not in all_lines[log_start]
    ):
        log_start -= 1
    if "Cache flushed successfully!" not in all_lines[log_start]:
        raise RuntimeError("successful flush before formal gate not found")
    lines = all_lines[log_start:]
    FORMAL_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    matches: dict[str, dict] = {}
    extends: dict[str, dict] = {}
    timings: dict[str, dict] = {}
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"match": 0, "extend": 0, "timing": 0}
    )
    for line_number, line in enumerate(lines, start=1):
        if match := MATCH_RE.search(line):
            rid, full, executable, boundary, branching, gap = match.groups()
            matches[rid] = {
                "line": line_number,
                "absolute_line": log_start + line_number,
                "fa_kv_hit_tokens": int(full),
                "executable_prefix_tokens": int(executable),
                "mamba_boundary_tokens": optional_int(boundary),
                "branching_tokens": optional_int(branching),
                "fsval_gap_tokens": int(gap),
            }
            counts[rid]["match"] += 1
        if extend := EXTEND_RE.search(line):
            rid, fill, prefix, extend_start, length, end = extend.groups()
            extends[rid] = {
                "line": line_number,
                "fill_ids_tokens": int(fill),
                "scheduled_prefix_tokens": int(prefix),
                "extend_start_tokens": optional_int(extend_start),
                "actual_extend_tokens": optional_int(length),
                "extend_end_tokens": optional_int(end),
            }
            counts[rid]["extend"] += 1
        if timing := TIMING_RE.search(line):
            rid, ttft_ms, e2e_ms = timing.groups()
            timings[rid] = {
                "line": line_number,
                "logged_recovery_latency_ms": float(ttft_ms),
                "logged_e2e_latency_ms": float(e2e_ms),
            }
            counts[rid]["timing"] += 1
    return matches, extends, timings, counts, lines


def read_events() -> tuple[list[dict], dict[str, dict]]:
    events = [
        json.loads(line)
        for line in EVENTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    flushes = {
        event["case_id"]: event for event in events if event.get("event") == "flush"
    }
    measured = [
        event
        for event in events
        if event.get("event") == "generate" and event.get("role") == "measure"
    ]
    block_positions: dict[int, int] = defaultdict(int)
    for event in measured:
        if event.get("phase") == "sweep":
            repetition = int(event["repetition"])
            event["_order_position"] = block_positions[repetition]
            block_positions[repetition] += 1
    return measured, flushes


def build_rows() -> tuple[list[dict], list[dict]]:
    matches, extends, timings, counts, lines = read_log()
    measured, flushes = read_events()
    rows: list[dict] = []
    for event in measured:
        rid = event["rid"]
        if rid not in matches or rid not in extends or rid not in timings:
            raise RuntimeError(f"missing runtime probe record for {rid}")
        match = matches[rid]
        extend = extends[rid]
        timing = timings[rid]
        expected = int(event["expected_replay_tokens"])
        case_id = rid[: -len("_MEASURE")]
        first_rid_suffix = "_R1" if expected == 32_768 else "_S1"
        first_rid = case_id + first_rid_suffix
        deep_rid = case_id + "_EXT"
        start_line = matches.get(first_rid, {}).get("line", match["line"])
        end_line = match["line"]
        evictions = sum(
            "[FSVAL] mamba_evict" in line
            for line in lines[start_line - 1 : end_line - 1]
        )
        error_lines = sum(
            bool(re.search(r"\bERROR\b|Traceback|out of memory", line, re.I))
            for line in lines[start_line - 1 : timing["line"]]
        )
        batch_gt_one_lines = sum(
            bool(re.search(r"#new-seq:\s*([2-9]|[1-9][0-9]+)", line))
            for line in lines[start_line - 1 : timing["line"]]
        )

        replay_start = extend["extend_start_tokens"]
        replay_end = min(extend["extend_end_tokens"], match["fa_kv_hit_tokens"])
        actual_replay = max(0, replay_end - replay_start)
        meta = event.get("meta", {})
        recovery_ms = float(meta["first_token_latency"]) * 1_000
        e2e_ms = float(meta["e2e_latency"]) * 1_000
        invalid: list[str] = []
        checks = {
            "flush": case_id in flushes and flushes[case_id].get("status") == 200,
            "physical_hit": match["fa_kv_hit_tokens"] == 32_768,
            "executable_prefix": match["executable_prefix_tokens"]
            == 32_768 - expected,
            "replay": actual_replay == expected,
            "fsval_gap": match["fsval_gap_tokens"] == expected,
            "extend": extend["actual_extend_tokens"] == expected + 1,
            "extend_start": extend["extend_start_tokens"]
            == match["executable_prefix_tokens"],
            "scheduled_prefix": extend["scheduled_prefix_tokens"]
            == match["executable_prefix_tokens"],
            "extend_end": extend["extend_end_tokens"] == 32_769,
            "fill_ids": extend["fill_ids_tokens"] == 32_769,
            "prompt": event["prompt_tokens"] == 32_769,
            "mamba_boundary": match["mamba_boundary_tokens"]
            == match["executable_prefix_tokens"],
            "cached_meta": meta.get("cached_tokens")
            == match["executable_prefix_tokens"],
            "one_output": meta.get("completion_tokens") == 1,
            "one_output_id": len(event.get("output_ids", [])) == 1,
            "length_finish": meta.get("finish_reason")
            == {"type": "length", "length": 1},
            "no_retraction": meta.get("num_retractions", 0) == 0,
            "deep_checkpoint_primed": matches.get(deep_rid, {}).get(
                "executable_prefix_tokens"
            )
            == 32_768,
            "eviction_pressure": evictions > 0,
            "timing_match": abs(
                recovery_ms - timing["logged_recovery_latency_ms"]
            )
            < 0.1,
            "timing_order": 0 < recovery_ms <= e2e_ms <= event["client_wall_ms"],
            "unique_probe_records": counts[rid]
            == {"match": 1, "extend": 1, "timing": 1},
            "no_runtime_error": error_lines == 0,
            "single_request_batches": batch_gt_one_lines == 0,
        }
        invalid.extend(name for name, passed in checks.items() if not passed)
        rows.append(
            {
                "phase": event["phase"],
                "repetition": int(event["repetition"]),
                "order_position": event.get("_order_position"),
                "rid": rid,
                "expected_replay_tokens": expected,
                "actual_replay_tokens": actual_replay,
                "actual_extend_tokens": extend["actual_extend_tokens"],
                "fa_kv_hit_tokens": match["fa_kv_hit_tokens"],
                "executable_prefix_tokens": match["executable_prefix_tokens"],
                "mamba_boundary_tokens": match["mamba_boundary_tokens"],
                "branching_tokens": match["branching_tokens"],
                "extend_start_tokens": extend["extend_start_tokens"],
                "extend_end_tokens": extend["extend_end_tokens"],
                "recovery_latency_ms": recovery_ms,
                "request_e2e_latency_ms": e2e_ms,
                "client_wall_ms": float(event["client_wall_ms"]),
                "cached_tokens_reported": meta.get("cached_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "num_retractions": meta.get("num_retractions", 0),
                "mamba_evictions_before_measure": evictions,
                "runtime_match_records": counts[rid]["match"],
                "runtime_extend_records": counts[rid]["extend"],
                "runtime_timing_records": counts[rid]["timing"],
                "runtime_error_lines": error_lines,
                "batch_gt_one_lines": batch_gt_one_lines,
                "server_log_absolute_line": match["absolute_line"],
                "valid": not invalid,
                "invalid_reason": ";".join(invalid),
            }
        )
    return (
        [row for row in rows if row["phase"] == "sweep"],
        [row for row in rows if row["phase"] == "gate"],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = intercept + slope * x
    residual = y - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else math.nan
    return {
        "intercept_ms": float(intercept),
        "slope_ms_per_token": float(slope),
        "slope_ms_per_1k_tokens": float(slope * 1_024),
        "r_squared": r_squared,
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "n": int(len(x)),
    }


def summarize(rows: list[dict]) -> tuple[list[dict], dict]:
    valid = [row for row in rows if row["valid"]]
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in valid:
        grouped[row["expected_replay_tokens"]].append(row)
    if set(grouped) != set(GROUPS):
        raise RuntimeError(f"missing replay groups: got {sorted(grouped)}")
    if any(len(grouped[group]) < 5 for group in GROUPS):
        raise RuntimeError(
            "fewer than five valid runs: "
            + repr({group: len(grouped[group]) for group in GROUPS})
        )

    baseline = float(
        np.median([row["recovery_latency_ms"] for row in grouped[0]])
    )
    summary: list[dict] = []
    for group in GROUPS:
        group_rows = grouped[group]
        replay = np.asarray([row["actual_replay_tokens"] for row in group_rows])
        latency = np.asarray([row["recovery_latency_ms"] for row in group_rows])
        e2e = np.asarray([row["request_e2e_latency_ms"] for row in group_rows])
        median = float(np.median(latency))
        summary.append(
            {
                "expected_replay_tokens": group,
                "actual_replay_tokens_min": int(np.min(replay)),
                "actual_replay_tokens_max": int(np.max(replay)),
                "n_valid": len(group_rows),
                "recovery_latency_mean_ms": float(np.mean(latency)),
                "recovery_latency_std_ms": float(np.std(latency, ddof=1)),
                "recovery_latency_median_ms": median,
                "recovery_latency_p25_ms": float(np.percentile(latency, 25)),
                "recovery_latency_p75_ms": float(np.percentile(latency, 75)),
                "recovery_latency_min_ms": float(np.min(latency)),
                "recovery_latency_max_ms": float(np.max(latency)),
                "median_increase_vs_0_ms": median - baseline,
                "median_ratio_vs_0": median / baseline,
                "request_e2e_latency_median_ms": float(np.median(e2e)),
            }
        )

    x = np.asarray([row["actual_replay_tokens"] for row in valid], dtype=float)
    y = np.asarray([row["recovery_latency_ms"] for row in valid], dtype=float)
    fit = {"raw_run_ols": fit_ols(x, y)}
    median_x = np.asarray(
        [entry["actual_replay_tokens_min"] for entry in summary], dtype=float
    )
    median_y = np.asarray(
        [entry["recovery_latency_median_ms"] for entry in summary], dtype=float
    )
    fit["group_median_ols"] = fit_ols(median_x, median_y)
    repetitions = np.asarray([row["repetition"] for row in valid], dtype=float)
    order_positions = np.asarray(
        [row["order_position"] for row in valid], dtype=float
    )
    adjusted_design = np.column_stack(
        (np.ones_like(x), x, repetitions, order_positions)
    )
    adjusted_coefficients = np.linalg.lstsq(adjusted_design, y, rcond=None)[0]
    adjusted_prediction = adjusted_design @ adjusted_coefficients
    adjusted_ss_res = float(np.sum((y - adjusted_prediction) ** 2))
    adjusted_ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    fit["repetition_and_order_adjusted_ols"] = {
        "intercept_ms": float(adjusted_coefficients[0]),
        "slope_ms_per_token": float(adjusted_coefficients[1]),
        "slope_ms_per_1k_tokens": float(adjusted_coefficients[1] * 1_024),
        "repetition_coefficient_ms": float(adjusted_coefficients[2]),
        "order_position_coefficient_ms": float(adjusted_coefficients[3]),
        "r_squared": 1.0 - adjusted_ss_res / adjusted_ss_tot,
        "n": int(len(x)),
    }
    # Randomized complete-block sensitivity: remove each repetition's mean
    # before fitting.  This controls arbitrary between-block intercept drift.
    x_within = x.copy()
    y_within = y.copy()
    for repetition in np.unique(repetitions):
        mask = repetitions == repetition
        x_within[mask] -= np.mean(x[mask])
        y_within[mask] -= np.mean(y[mask])
    within_slope = float(np.dot(x_within, y_within) / np.dot(x_within, x_within))
    within_residual = y_within - within_slope * x_within
    fit["repetition_fixed_effect_sensitivity"] = {
        "slope_ms_per_token": within_slope,
        "slope_ms_per_1k_tokens": within_slope * 1_024,
        "within_r_squared": 1.0
        - float(np.sum(within_residual**2)) / float(np.sum(y_within**2)),
        "n": int(len(x)),
    }
    return summary, fit


def plot(rows: list[dict], summary: list[dict], fit: dict) -> None:
    valid = [row for row in rows if row["valid"]]
    x = np.asarray([row["actual_replay_tokens"] for row in valid], dtype=float)
    y = np.asarray([row["recovery_latency_ms"] for row in valid], dtype=float)
    sx = np.asarray([entry["actual_replay_tokens_min"] for entry in summary])
    sy = np.asarray([entry["recovery_latency_median_ms"] for entry in summary])
    low = sy - np.asarray([entry["recovery_latency_p25_ms"] for entry in summary])
    high = np.asarray([entry["recovery_latency_p75_ms"] for entry in summary]) - sy
    raw_fit = fit["raw_run_ols"]
    line_x = np.linspace(0, 32_768, 300)
    line_y = raw_fit["intercept_ms"] + raw_fit["slope_ms_per_token"] * line_x

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#4B5563",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.scatter(
        x / 1_024,
        y,
        s=32,
        alpha=0.48,
        color="#2F6B9A",
        edgecolors="none",
        label="Measured runs (n=30)",
        zorder=2,
    )
    ax.errorbar(
        sx / 1_024,
        sy,
        yerr=np.vstack((low, high)),
        fmt="o",
        markersize=6,
        capsize=4,
        linewidth=1.6,
        color="#C07A1C",
        label="Group median and IQR",
        zorder=4,
    )
    ax.plot(
        line_x / 1_024,
        line_y,
        color="#243B53",
        linewidth=2,
        label="OLS fit (all runs)",
        zorder=3,
    )
    ax.set_title("Recovery latency rises with actual replay tokens", loc="left", pad=16)
    ax.text(
        0,
        1.015,
        "Physical FA-KV fixed at 32K; executable recurrent prefix varied",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#52606D",
        va="bottom",
    )
    ax.set_xlabel("Actual replay tokens (Ki tokens; 1 Ki = 1024)")
    ax.set_ylabel("Recovery latency / TTFT (ms)")
    ax.set_xticks([0, 1, 4, 8, 16, 32])
    ax.set_xlim(-0.8, 33.2)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9E2EC", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02,
        0.97,
        f"slope = {raw_fit['slope_ms_per_1k_tokens']:.2f} ms/Ki token\n"
        f"R² = {raw_fit['r_squared']:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#243B53",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#BCCCDC"},
    )
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(PNG, dpi=300)
    fig.savefig(PDF)
    plt.close(fig)


def main() -> None:
    sweep_rows, gate_rows = build_rows()
    write_csv(RAW_CSV, sweep_rows)
    write_csv(GATE_CSV, gate_rows)
    if invalid := [row for row in sweep_rows if not row["valid"]]:
        raise RuntimeError(f"invalid sweep runs: {invalid}")
    if len(sweep_rows) != 30:
        raise RuntimeError(f"expected 30 sweep runs, got {len(sweep_rows)}")
    summary, fit = summarize(sweep_rows)
    write_csv(SUMMARY_CSV, summary)
    FIT_JSON.write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
    plot(sweep_rows, summary, fit)
    print(json.dumps({"summary": summary, "fit": fit}, indent=2))


if __name__ == "__main__":
    main()
