#!/usr/bin/env python3

import csv
import math
import re
import statistics
import sys
from pathlib import Path


LOG_RE = re.compile(
    r"r(?P<rep>\d+)_o(?P<order>\d+)_(?P<policy>prompt_lru|workflow_k)\.log$"
)

MATCH_RE = re.compile(
    r"\[FSVAL\] match_end req=.*?"
    r"full_kv_hit=(?P<full>\d+) "
    r"exec_prefix=(?P<exec>\d+) "
    r"mamba_boundary=(?P<mamba>\d+) "
    r"branching=(?P<branching>\S+) "
    r"gap=(?P<gap>\d+)"
)

TIMING_RE = re.compile(
    r"\[FSWP2\] request_timing req=.*?"
    r"first_token_latency_ms=(?P<ttft>[0-9.]+) "
    r"e2e_latency_ms=(?P<e2e>[0-9.]+)"
)

REQ_RE = re.compile(
    r"ReqTimeStats\(rid=.*?"
    r"input_len=(?P<input>\d+), "
    r"cached_input_len=(?P<cached>\d+), "
    r"output_len=(?P<output>\d+).*?"
    r"queue_duration=(?P<queue>[0-9.]+)ms, "
    r"forward_duration=(?P<forward>[0-9.]+)ms"
)

PRE_RE = re.compile(
    r"\[WP3B-GOG\] PRE_MEASURE "
    r"policy=(?P<policy>\S+) "
    r"count=(?P<count>\d+) "
    r"parent_mamba=(?P<parent_mamba>True|False) "
    r"parent_fa=(?P<parent_fa>True|False) "
    r"child_mamba=(?P<child_mamba>True|False) "
    r"child_fa=(?P<child_fa>True|False)"
)


def as_bool(x):
    return x == "True"


def mean_ci95(values):
    """
    Paired n=5 formal experiment.
    Use Student-t 95% CI for the mean.
    t_(0.975, df=4) = 2.776445...
    """
    n = len(values)
    mean = statistics.mean(values)

    if n <= 1:
        return mean, float("nan"), float("nan")

    sd = statistics.stdev(values)

    # Our formal design is n=5 pairs.
    # Keep a small table in case we later rerun with nearby n.
    tcrit = {
        2: 12.706205,
        3: 4.302653,
        4: 3.182446,
        5: 2.776445,
        6: 2.570582,
        7: 2.446912,
        8: 2.364624,
        9: 2.306004,
        10: 2.262157,
    }.get(n)

    if tcrit is None:
        # Normal approximation only for unexpected larger n.
        tcrit = 1.96

    half = tcrit * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def fmt(x, digits=3):
    if isinstance(x, float) and math.isnan(x):
        return "NA"
    return f"{x:.{digits}f}"


def parse_one(path):
    text = path.read_text(errors="replace")

    m_name = LOG_RE.search(path.name)
    if not m_name:
        raise RuntimeError(f"unexpected log filename: {path.name}")

    rep = int(m_name.group("rep"))
    order = int(m_name.group("order"))
    policy = m_name.group("policy")

    matches = MATCH_RE.findall(text)
    timings = TIMING_RE.findall(text)
    reqs = REQ_RE.findall(text)
    pres = PRE_RE.findall(text)

    # Filter to measured Child-B lines by taking the last matching record.
    if not matches:
        raise RuntimeError(f"{path}: missing FSVAL match_end")
    if not timings:
        raise RuntimeError(f"{path}: missing FSWP2 timing")
    if not pres:
        raise RuntimeError(f"{path}: missing PRE_MEASURE")

    full, exec_prefix, mamba_boundary, branching, gap = matches[-1]
    ttft, e2e = timings[-1]

    pre = pres[-1]
    (
        pre_policy,
        count,
        parent_mamba,
        parent_fa,
        child_mamba,
        child_fa,
    ) = pre

    if pre_policy != policy:
        raise RuntimeError(
            f"{path}: filename policy={policy}, log policy={pre_policy}"
        )

    status_complete = "[WP3B-GOG] STATUS=complete" in text

    state_gate_valid = (
        int(count) == 4
        and as_bool(parent_fa)
        and as_bool(child_fa)
    )

    if policy == "prompt_lru":
        state_gate_valid = (
            state_gate_valid
            and not as_bool(parent_mamba)
            and as_bool(child_mamba)
        )
    else:
        state_gate_valid = (
            state_gate_valid
            and as_bool(parent_mamba)
            and not as_bool(child_mamba)
        )

    # Do NOT use this to discard data.
    # It is only a diagnostic that the expected causal path occurred.
    if policy == "prompt_lru":
        expected_path = (
            int(exec_prefix) == 0
            and int(gap) >= 32768
        )
    else:
        expected_path = (
            int(exec_prefix) == 32768
            and int(gap) <= 1
        )

    row = {
        "rep": rep,
        "order": order,
        "policy": policy,
        "status_complete": status_complete,
        "state_gate_valid": state_gate_valid,
        "expected_path": expected_path,
        "full_kv_hit": int(full),
        "exec_prefix": int(exec_prefix),
        "mamba_boundary": int(mamba_boundary),
        "gap_tokens": int(gap),
        "branching": branching,
        "ttft_ms": float(ttft),
        "e2e_ms": float(e2e),
        "log_file": path.name,
    }

    if reqs:
        inp, cached, out, queue, forward = reqs[-1]
        row.update(
            {
                "input_len": int(inp),
                "cached_input_len": int(cached),
                "output_len": int(out),
                "queue_ms": float(queue),
                "forward_ms": float(forward),
            }
        )
    else:
        row.update(
            {
                "input_len": "",
                "cached_input_len": "",
                "output_len": "",
                "queue_ms": "",
                "forward_ms": "",
            }
        )

    return row


def main():
    if len(sys.argv) != 2:
        print(
            "usage: parse_formal_k4.py <formal_result_dir>",
            file=sys.stderr,
        )
        return 2

    root = Path(sys.argv[1]).resolve()
    logs = sorted(root.glob("r*_o*_*.log"))

    if len(logs) != 10:
        raise RuntimeError(
            f"expected 10 arm logs, found {len(logs)} in {root}"
        )

    rows = [parse_one(p) for p in logs]
    rows.sort(key=lambda x: (x["rep"], x["order"]))

    csv_path = root / "formal_k4_results.csv"

    fieldnames = [
        "rep",
        "order",
        "policy",
        "status_complete",
        "state_gate_valid",
        "expected_path",
        "full_kv_hit",
        "exec_prefix",
        "mamba_boundary",
        "gap_tokens",
        "input_len",
        "cached_input_len",
        "output_len",
        "queue_ms",
        "forward_ms",
        "ttft_ms",
        "e2e_ms",
        "branching",
        "log_file",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------
    # Hard validity report.
    # We preserve all parsed rows in CSV even if one fails.
    # ------------------------------------------------------------
    invalid = [
        r
        for r in rows
        if not r["status_complete"] or not r["state_gate_valid"]
    ]

    path_deviations = [
        r for r in rows if not r["expected_path"]
    ]

    by_rep = {}

    for r in rows:
        by_rep.setdefault(r["rep"], {})[r["policy"]] = r

    if sorted(by_rep) != [1, 2, 3, 4, 5]:
        raise RuntimeError(
            f"expected reps 1..5, found {sorted(by_rep)}"
        )

    paired = []

    for rep in range(1, 6):
        p = by_rep[rep].get("prompt_lru")
        w = by_rep[rep].get("workflow_k")

        if p is None or w is None:
            raise RuntimeError(
                f"rep {rep}: missing one policy arm"
            )

        paired.append(
            {
                "rep": rep,
                "prompt_ttft_ms": p["ttft_ms"],
                "workflow_ttft_ms": w["ttft_ms"],
                "delta_ttft_ms": p["ttft_ms"] - w["ttft_ms"],
                "ttft_reduction_pct": (
                    (p["ttft_ms"] - w["ttft_ms"])
                    / p["ttft_ms"]
                    * 100.0
                ),
                "speedup_x": p["ttft_ms"] / w["ttft_ms"],
                "prompt_gap": p["gap_tokens"],
                "workflow_gap": w["gap_tokens"],
                "delta_gap": p["gap_tokens"] - w["gap_tokens"],
                "prompt_full_hit": p["full_kv_hit"],
                "workflow_full_hit": w["full_kv_hit"],
                "prompt_exec": p["exec_prefix"],
                "workflow_exec": w["exec_prefix"],
            }
        )

    pair_csv = root / "formal_k4_paired.csv"

    with pair_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(paired[0].keys()),
        )
        writer.writeheader()
        writer.writerows(paired)

    prompt = [r for r in rows if r["policy"] == "prompt_lru"]
    workflow = [r for r in rows if r["policy"] == "workflow_k"]

    prompt_ttft = [r["ttft_ms"] for r in prompt]
    workflow_ttft = [r["ttft_ms"] for r in workflow]

    delta_ttft = [r["delta_ttft_ms"] for r in paired]
    delta_gap = [r["delta_gap"] for r in paired]
    reduction = [r["ttft_reduction_pct"] for r in paired]
    speedup = [r["speedup_x"] for r in paired]

    mean_delta, ci_lo, ci_hi = mean_ci95(delta_ttft)

    wp2_predicted_32k_ms = 1504.474
    measured_vs_predicted = mean_delta / wp2_predicted_32k_ms

    summary = []

    summary.append("# WP3B Formal K=4 Paired Result")
    summary.append("")
    summary.append("## Experimental design")
    summary.append("")
    summary.append("- Recurrent-state budget: K=4 checkpoints")
    summary.append("- Checkpoint size: 49.125 MiB")
    summary.append("- Logical recurrent-state budget: 196.5 MiB")
    summary.append("- Paired repetitions: n=5")
    summary.append("- Policy order alternates across repetitions")
    summary.append("- Each arm uses a fresh engine/cache rebuild")
    summary.append("")

    summary.append("## Validity")
    summary.append("")
    summary.append(
        f"- Complete/state-valid arms: {10 - len(invalid)}/10"
    )
    summary.append(
        f"- Expected-path arms: {10 - len(path_deviations)}/10"
    )
    summary.append("")

    if invalid:
        summary.append("Invalid arms:")
        for r in invalid:
            summary.append(
                f"- rep={r['rep']} policy={r['policy']} "
                f"complete={r['status_complete']} "
                f"state_gate={r['state_gate_valid']}"
            )
        summary.append("")

    if path_deviations:
        summary.append("Path deviations (not automatically excluded):")
        for r in path_deviations:
            summary.append(
                f"- rep={r['rep']} policy={r['policy']} "
                f"full={r['full_kv_hit']} "
                f"exec={r['exec_prefix']} gap={r['gap_tokens']}"
            )
        summary.append("")

    summary.append("## Policy-level result")
    summary.append("")
    summary.append(
        f"- Prompt-LRU TTFT median: "
        f"{fmt(statistics.median(prompt_ttft))} ms"
    )
    summary.append(
        f"- Prompt-LRU TTFT mean: "
        f"{fmt(statistics.mean(prompt_ttft))} ms"
    )
    summary.append(
        f"- Workflow-K TTFT median: "
        f"{fmt(statistics.median(workflow_ttft))} ms"
    )
    summary.append(
        f"- Workflow-K TTFT mean: "
        f"{fmt(statistics.mean(workflow_ttft))} ms"
    )
    summary.append("")

    summary.append("## Paired effect")
    summary.append("")
    summary.append(
        f"- Mean paired TTFT reduction: {fmt(mean_delta)} ms"
    )
    summary.append(
        f"- Median paired TTFT reduction: "
        f"{fmt(statistics.median(delta_ttft))} ms"
    )
    summary.append(
        f"- 95% t-CI for mean paired TTFT reduction: "
        f"[{fmt(ci_lo)}, {fmt(ci_hi)}] ms"
    )
    summary.append(
        f"- Mean relative TTFT reduction: "
        f"{fmt(statistics.mean(reduction), 2)}%"
    )
    summary.append(
        f"- Median speedup: "
        f"{fmt(statistics.median(speedup), 2)}x"
    )
    summary.append(
        f"- Mean replay/gap reduction: "
        f"{fmt(statistics.mean(delta_gap), 1)} tokens"
    )
    summary.append("")

    summary.append("## Cross-check against WP2")
    summary.append("")
    summary.append(
        f"- WP2 predicted saving for 32K recovery: "
        f"{wp2_predicted_32k_ms:.3f} ms"
    )
    summary.append(
        f"- Formal mean measured paired TTFT saving / prediction: "
        f"{measured_vs_predicted:.3f}x"
    )
    summary.append("")

    summary.append("## Per-repetition")
    summary.append("")
    summary.append(
        "| Rep | Prompt TTFT ms | Workflow TTFT ms | "
        "Delta ms | Reduction | Speedup | "
        "Prompt gap | Workflow gap | Delta gap |"
    )
    summary.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for r in paired:
        summary.append(
            f"| {r['rep']} "
            f"| {r['prompt_ttft_ms']:.3f} "
            f"| {r['workflow_ttft_ms']:.3f} "
            f"| {r['delta_ttft_ms']:.3f} "
            f"| {r['ttft_reduction_pct']:.2f}% "
            f"| {r['speedup_x']:.2f}x "
            f"| {r['prompt_gap']} "
            f"| {r['workflow_gap']} "
            f"| {r['delta_gap']} |"
        )

    summary.append("")
    summary.append(
        "> Interpretation should only be finalized after checking all "
        "10 raw logs and validity gates."
    )

    summary_path = root / "formal_k4_summary.md"
    summary_path.write_text("\n".join(summary) + "\n")

    print(f"[FORMAL-K4] arms_csv={csv_path}")
    print(f"[FORMAL-K4] paired_csv={pair_csv}")
    print(f"[FORMAL-K4] summary={summary_path}")
    print(
        f"[FORMAL-K4] valid_arms={10-len(invalid)}/10 "
        f"expected_paths={10-len(path_deviations)}/10"
    )
    print(
        f"[FORMAL-K4] prompt_median_ttft_ms="
        f"{statistics.median(prompt_ttft):.3f}"
    )
    print(
        f"[FORMAL-K4] workflow_median_ttft_ms="
        f"{statistics.median(workflow_ttft):.3f}"
    )
    print(
        f"[FORMAL-K4] paired_mean_delta_ttft_ms="
        f"{mean_delta:.3f}"
    )
    print(
        f"[FORMAL-K4] paired_95ci_ms="
        f"[{ci_lo:.3f}, {ci_hi:.3f}]"
    )
    print(
        f"[FORMAL-K4] paired_median_speedup_x="
        f"{statistics.median(speedup):.3f}"
    )

    # Return nonzero only for hard experimental validity failure.
    if invalid:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
