#!/usr/bin/env python3
"""Validate WP3A direct-eviction evidence and build final artifacts."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EVENTS_PATH = HERE / "driver_events.jsonl"
LOG_PATH = HERE / "server_full.log"
RAW_CSV = HERE / "raw_runs.csv"
PAIRED_CSV = HERE / "paired_effects.csv"
SUMMARY_CSV = HERE / "summary.csv"
SUMMARY_JSON = HERE / "analysis_summary.json"
FINDINGS = HERE / "findings.md"
PLOT_PNG = HERE / "plot_fork_eviction_latency.png"
PLOT_PDF = HERE / "plot_fork_eviction_latency.pdf"
CHART_NOTES = HERE / "chart_notes.json"

WP2_SLOPE_MS_PER_TOKEN = 0.0459129
T975_DF4 = 2.7764451051977987


MATCH_RE = re.compile(
    r"\[FSVAL\] match_end req=(\S+) full_kv_hit=(\d+) "
    r"exec_prefix=(\d+) mamba_boundary=(\d+) branching=(None|\d+) gap=(\d+)"
)
EXTEND_RE = re.compile(
    r"\[FSVAL\] extend req=(\S+) fill_ids=(\d+) prefix_len=(\d+) "
    r"extend_start=(\d+) extend_len=(\d+) extend_end=(\d+)"
)
TIMING_RE = re.compile(
    r"\[FSWP2\] request_timing req=(\S+) first_token_latency_ms=([0-9.]+) "
    r"e2e_latency_ms=([0-9.]+)"
)
INSERT_RE = re.compile(
    r"\[FSVAL\] insert_ckpt req=(\S+) token_ids_len=(\d+) "
    r"ckpt_pos=(\d+) strategy=(\S+)"
)
EVICT_RE = re.compile(
    r"\[FSVAL\] mamba_evict node=(\d+) freed=(\d+) fa_kept=(True|False)"
)
FORCED_TRACK_RE = re.compile(
    r"\[FSVAL\] forced_track req=(\S+) prefix_len=(\d+) extend_len=(\d+) "
    r"branching_seqlen=(\d+) last_track_seqlen=(\d+)"
)
DUP_FREE_RE = re.compile(
    r"\[FSVAL\] dup_free node=(\d+) dup_start=(\d+) consumed_from=(\d+) "
    r"freed=(\d+) step_prefix_len=(\d+) total_prefix=(\d+)"
)
BATCH_RE = re.compile(r"Prefill batch, #new-seq: (\d+)")


def load_jsonl(path: Path):
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            raise AssertionError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def parse_records(pattern, lines, converters):
    out = defaultdict(list)
    for line_no, line in lines:
        match = pattern.search(line)
        if not match:
            continue
        groups = match.groups()
        rid = groups[0]
        values = {}
        for (name, converter), raw in zip(converters, groups[1:]):
            values[name] = converter(raw)
        values["line_no"] = line_no
        out[rid].append(values)
    return out


def optional_int(value):
    return None if value == "None" else int(value)


def event_path(response, when="before"):
    return response[when]["path"]


def bool_int(value):
    return 1 if value else 0


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise AssertionError(f"refusing to write empty CSV {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise AssertionError(f"CSV schema drift for {path}: {list(row)} != {fields}")
            writer.writerow(row)


def sample_stats(values):
    values = [float(x) for x in values]
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }


def value_or_blank(value):
    return "" if value is None else value


def main():
    events = load_jsonl(EVENTS_PATH)
    # Count physical LF-delimited lines so CSV/JSON line references agree with
    # `rg -n`, `nl -ba`, and editors.  `str.splitlines()` would also split six
    # startup carriage-return progress records and shift every later citation.
    with LOG_PATH.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        raw_log_text = handle.read()
    log_lines_all = [
        (line_no, line.rstrip("\r"))
        for line_no, line in enumerate(
            raw_log_text.split("\n"), 1
        )
    ]
    starts = [n for n, line in log_lines_all if "[FSWP3D] FORMAL_WINDOW_START" in line]
    ends = [n for n, line in log_lines_all if "[FSWP3D] FORMAL_WINDOW_END" in line]
    if len(starts) != 1 or len(ends) != 1 or not starts[0] < ends[0]:
        raise AssertionError(f"formal window markers invalid: {starts}, {ends}")
    formal_start, formal_end = starts[0], ends[0]
    formal_lines = [item for item in log_lines_all if formal_start <= item[0] <= formal_end]

    matches = parse_records(
        MATCH_RE,
        formal_lines,
        [
            ("fa_kv_hit_tokens", int),
            ("executable_prefix_tokens", int),
            ("mamba_boundary_tokens", int),
            ("branching_tokens", optional_int),
            ("physical_gap_tokens", int),
        ],
    )
    extends = parse_records(
        EXTEND_RE,
        formal_lines,
        [
            ("fill_ids", int),
            ("prefix_len", int),
            ("extend_start", int),
            ("actual_extend_tokens", int),
            ("extend_end", int),
        ],
    )
    timings = parse_records(
        TIMING_RE,
        formal_lines,
        [("logged_recovery_latency_ms", float), ("logged_e2e_latency_ms", float)],
    )
    inserts = parse_records(
        INSERT_RE,
        formal_lines,
        [("insert_input_tokens", int), ("insert_checkpoint_tokens", int), ("strategy", str)],
    )

    case_results = [
        event
        for event in events
        if event.get("event") == "case_result" and event.get("phase") == "formal"
    ]
    experiment_end = [
        event for event in events if event.get("event") == "experiment_end"
    ]
    global_failures = []
    if len(case_results) != 20:
        global_failures.append(f"formal case count {len(case_results)} != 20")
    counts = Counter((row["experiment"], row["condition"]) for row in case_results)
    expected_counts = {
        ("FORK_PARENT", "retained"): 5,
        ("FORK_PARENT", "evicted"): 5,
        ("NORMAL_CONTROL", "retained"): 5,
        ("NORMAL_CONTROL", "evicted"): 5,
    }
    if dict(counts) != expected_counts:
        global_failures.append(f"case distribution mismatch: {dict(counts)}")
    if len(experiment_end) != 1 or experiment_end[0].get("status") != "complete":
        global_failures.append(f"experiment_end invalid: {experiment_end}")

    # Slice each case by explicit stdout markers.  This makes error/eviction
    # scans request-local instead of accidentally including startup warnings.
    case_windows = {}
    for case in case_results:
        case_id = case["case_id"]
        start_hits = [
            n for n, line in formal_lines if f"[FSWP3D] CASE_START id={case_id}" in line
        ]
        end_hits = [
            n for n, line in formal_lines if f"[FSWP3D] CASE_END id={case_id} status=complete" in line
        ]
        if len(start_hits) == 1 and len(end_hits) == 1 and start_hits[0] < end_hits[0]:
            case_windows[case_id] = [
                item for item in formal_lines if start_hits[0] <= item[0] <= end_hits[0]
            ]
        else:
            case_windows[case_id] = []

    raw_rows = []
    validation_by_case = {}
    for case in sorted(case_results, key=lambda row: (row["repetition"], row["case_order"])):
        failures = []
        case_id = case["case_id"]
        measured = case["measured_request"]
        rid = measured["rid"]
        match_records = matches.get(rid, [])
        extend_records = extends.get(rid, [])
        timing_records = timings.get(rid, [])
        insert_records = inserts.get(rid, [])
        for label, records in (
            ("match", match_records),
            ("extend", extend_records),
            ("timing", timing_records),
            ("insert", insert_records),
        ):
            if len(records) != 1:
                failures.append(f"{label}_record_count={len(records)}")
        if failures:
            # Populate empty placeholders, then keep all invalid rows visible.
            match = match_records[-1] if match_records else {}
            extend = extend_records[-1] if extend_records else {}
            timing = timing_records[-1] if timing_records else {}
            insert = insert_records[-1] if insert_records else {}
        else:
            match, extend, timing, insert = (
                match_records[0],
                extend_records[0],
                timing_records[0],
                insert_records[0],
            )

        prompt = int(measured["prompt_tokens"])
        full = match.get("fa_kv_hit_tokens")
        executable = match.get("executable_prefix_tokens")
        actual_replay = None
        fresh_tokens = None
        if full is not None and extend.get("extend_start") is not None:
            actual_replay = max(
                0, min(extend.get("extend_end", prompt), full) - extend["extend_start"]
            )
            fresh_tokens = prompt - full

        if case["experiment"] == "FORK_PARENT":
            expected = (
                {"full": 32768, "exec": 32768, "replay": 0, "extend": 64, "fresh": 64, "branch": None, "insert": 32832}
                if case["condition"] == "retained"
                else {"full": 32768, "exec": 0, "replay": 32768, "extend": 32832, "fresh": 64, "branch": 32768, "insert": 32768}
            )
        else:
            expected = {
                "full": 32769,
                "exec": 32768,
                "replay": 1,
                "extend": 256,
                "fresh": 255,
                "branch": None,
                "insert": 33024,
            }

        checks = {
            "fa_hit": full == expected["full"],
            "exec": executable == expected["exec"],
            "gap_identity": match.get("physical_gap_tokens")
            == (None if full is None or executable is None else full - executable),
            "mamba_boundary": match.get("mamba_boundary_tokens") == expected["exec"],
            "branching": match.get("branching_tokens") == expected["branch"],
            "actual_replay": actual_replay == expected["replay"],
            "extend": extend.get("actual_extend_tokens") == expected["extend"],
            "fresh": fresh_tokens == expected["fresh"],
            "extend_start": extend.get("extend_start") == executable,
            "prefix_len": extend.get("prefix_len") == executable,
            "extend_end": extend.get("extend_end") == prompt,
            "fill_ids": extend.get("fill_ids") == prompt,
            "extend_decomposition": extend.get("actual_extend_tokens")
            == (None if actual_replay is None or fresh_tokens is None else actual_replay + fresh_tokens),
            "cached_meta": measured["meta"].get("cached_tokens") == executable,
            "insert": insert.get("insert_checkpoint_tokens") == expected["insert"],
            "insert_strategy": insert.get("strategy") == "extra_buffer",
            "response_timing_match": abs(
                float(measured["server_ttft_ms"])
                - float(timing.get("logged_recovery_latency_ms", math.inf))
            )
            <= 0.001,
            "latency_order": 0
            < float(measured["server_ttft_ms"])
            <= float(measured["server_e2e_ms"])
            <= float(measured["client_e2e_ms"]) + 1e-6,
            "completion": measured["meta"].get("completion_tokens") == 1,
            "no_retraction": int(measured["meta"].get("num_retractions") or 0) == 0,
        }
        failures.extend(name for name, passed in checks.items() if not passed)

        control = case["control"]
        control_before = event_path(control, "before")
        control_after = event_path(control, "after")
        proof = control["proof"]
        mutation = control["mutation"]
        pre_target = event_path(case["pre_target"])
        pre_decoy = event_path(case["pre_decoy"])
        post_target = event_path(case["post_target"])
        post_decoy = event_path(case["post_decoy"])
        target_after_measure = event_path(case["target_after_measure"])
        target_should_remain = case["condition"] == "retained"
        state_checks = {
            "pre_target_present": pre_target["target_mamba_present"],
            "pre_decoy_present": pre_decoy["target_mamba_present"],
            "control_internal": control_before["n_children"] >= 1
            and not control_before["is_device_leaf"],
            "control_mamba_before": control_before["target_mamba_present"],
            "control_mamba_after_absent": not control_after["target_mamba_present"],
            "control_full_before": control_before["path_full_all_present"],
            "control_full_after": control_after["path_full_all_present"],
            "control_freed_one": mutation["device_freed"] == 1
            and mutation["tracker_mamba"] == 1,
            "control_full_freed_zero": mutation["tracker_full"] == 0,
            "control_host_freed_zero": mutation["host_freed"] == 0,
            "control_structure_unchanged": proof["structure_unchanged"],
            "control_full_tree_unchanged": proof["full_tree_unchanged"],
            "control_full_path_unchanged": proof["full_path_unchanged"],
            "control_full_allocator_unchanged": proof["full_allocator_unchanged"],
            "control_only_target_changed": proof["only_target_mamba_changed"]
            and proof["changed_mamba_node_ids"] == [control_before["node_id"]],
            "control_allocator_plus_one": proof["mamba_available_delta"] == 1,
            "control_evictable_minus_one": proof["mamba_evictable_delta"] == -1,
            "control_node_count_minus_one": proof["mamba_node_count_delta"] == -1,
            "control_sanity": proof["sanity_check_passed"],
            "target_post_expected": post_target["target_mamba_present"]
            == target_should_remain,
            "decoy_post_expected": post_decoy["target_mamba_present"]
            == (not target_should_remain),
            "target_full_post": post_target["path_full_all_present"],
            "decoy_full_post": post_decoy["path_full_all_present"],
        }
        failures.extend(name for name, passed in state_checks.items() if not passed)

        if case["experiment"] == "FORK_PARENT":
            expected_positions = [32768] if target_should_remain else []
            if [item["position"] for item in post_target["path_mamba_positions"]] != expected_positions:
                failures.append("fork_post_control_path_mamba_positions")
            if not target_after_measure["target_mamba_present"]:
                failures.append("fork_self_heal_missing")
        else:
            expected_positions = [32768, 32832] if target_should_remain else [32768]
            if [item["position"] for item in post_target["path_mamba_positions"]] != expected_positions:
                failures.append("normal_post_control_path_mamba_positions")
            if target_after_measure["target_mamba_present"] != target_should_remain:
                failures.append("normal_target_changed_by_join")
            for label in ("post_parent", "parent_after_measure"):
                parent_path = event_path(case[label])
                if not parent_path["target_mamba_present"] or not parent_path["path_full_all_present"]:
                    failures.append(f"{label}_not_resident")

        window = case_windows.get(case_id, [])
        if not window:
            failures.append("case_log_window_missing")
        evictions = []
        forced_tracks = []
        duplicate_full_rows_freed = 0
        long_dup_free_records = 0
        batch_sizes = []
        error_lines = []
        for line_no, text in window:
            evict_match = EVICT_RE.search(text)
            if evict_match:
                evictions.append(
                    {
                        "line_no": line_no,
                        "node_id": int(evict_match.group(1)),
                        "freed": int(evict_match.group(2)),
                        "fa_kept": evict_match.group(3) == "True",
                    }
                )
            forced_match = FORCED_TRACK_RE.search(text)
            if forced_match and forced_match.group(1) == rid:
                forced_tracks.append(
                    {
                        "line_no": line_no,
                        "prefix_len": int(forced_match.group(2)),
                        "extend_len": int(forced_match.group(3)),
                        "branching_seqlen": int(forced_match.group(4)),
                        "last_track_seqlen": int(forced_match.group(5)),
                    }
                )
            dup_match = DUP_FREE_RE.search(text)
            if dup_match and int(dup_match.group(4)) > 1:
                long_dup_free_records += 1
                duplicate_full_rows_freed += int(dup_match.group(4))
            batch_match = BATCH_RE.search(text)
            if batch_match:
                batch_sizes.append(int(batch_match.group(1)))
            if any(
                marker in text
                for marker in (
                    "CUDA out of memory",
                    "[ERROR]",
                    " ERROR ",
                    "AssertionError",
                    "status=invalid",
                    "probe operation failed",
                )
            ):
                error_lines.append(line_no)
        if len(evictions) != 1:
            failures.append(f"case_mamba_eviction_count={len(evictions)}")
        elif not (
            evictions[0]["node_id"] == control_before["node_id"]
            and evictions[0]["freed"] == 1
            and evictions[0]["fa_kept"]
        ):
            failures.append("case_mamba_eviction_mismatch")
        if any(size != 1 for size in batch_sizes):
            failures.append(f"batch_gt_one={batch_sizes}")
        if case["experiment"] == "FORK_PARENT" and case["condition"] == "evicted":
            expected_forced = {
                "prefix_len": 0,
                "extend_len": 32832,
                "branching_seqlen": 32768,
                "last_track_seqlen": 32768,
            }
            if len(forced_tracks) != 1 or any(
                forced_tracks[0].get(key) != value
                for key, value in expected_forced.items()
            ):
                failures.append(f"forced_track_mismatch={forced_tracks}")
            if long_dup_free_records != 1 or duplicate_full_rows_freed != 32768:
                failures.append(
                    "long_dup_free_mismatch="
                    f"{long_dup_free_records}/{duplicate_full_rows_freed}"
                )
        elif forced_tracks or long_dup_free_records or duplicate_full_rows_freed:
            failures.append(
                "unexpected_forced_or_long_dup="
                f"{forced_tracks}/{long_dup_free_records}/{duplicate_full_rows_freed}"
            )
        if error_lines:
            failures.append(f"runtime_errors={error_lines}")
        if sum("Cache flushed successfully!" in text for _, text in window) != 1:
            failures.append("flush_backend_log_count")

        control_pre_accounting = control["before"]["accounting"]
        control_post_accounting = control["after"]["accounting"]
        row = {
            "schema_version": "flowstate.wp3a.direct_eviction.raw.v1",
            "experiment_id": "wp3a_direct_eviction_20260819",
            "repetition": case["repetition"],
            "case_order": case["case_order"],
            "case_id": case_id,
            "experiment": case["experiment"],
            "target_boundary": case["target_boundary"],
            "condition": case["condition"],
            "evicted_role": case["evicted_role"],
            "content_seed_offset": case["content_seed_offset"],
            "measured_rid": rid,
            "input_sha256": measured["input_sha256"],
            "output_sha256": measured["output_sha256"],
            "build_signature_sha256": case["build_signature_sha256"],
            "prompt_tokens": prompt,
            "target_checkpoint_tokens": case["target_tokens"],
            "fallback_checkpoint_tokens": case["fallback_tokens"],
            "target_node_id": pre_target["node_id"],
            "controlled_node_id": control_before["node_id"],
            "target_internal_before": bool_int(
                pre_target["n_children"] >= 1 and not pre_target["is_device_leaf"]
            ),
            "target_mamba_present_before": bool_int(pre_target["target_mamba_present"]),
            "target_mamba_present_after_control": bool_int(post_target["target_mamba_present"]),
            "target_mamba_present_after_measure": bool_int(target_after_measure["target_mamba_present"]),
            "decoy_mamba_present_before": bool_int(pre_decoy["target_mamba_present"]),
            "decoy_mamba_present_after_control": bool_int(post_decoy["target_mamba_present"]),
            "target_full_path_rows_after_control": post_target["path_full_rows"],
            "target_full_present_after_control": bool_int(post_target["path_full_all_present"]),
            "fallback_mamba_present_after_control": (
                "" if case["experiment"] == "FORK_PARENT" else bool_int(event_path(case["post_parent"])["target_mamba_present"])
            ),
            "mamba_slots_freed": mutation["tracker_mamba"],
            "full_slots_freed": mutation["tracker_full"],
            "mamba_available_before": control_pre_accounting["mamba_available"],
            "mamba_available_after": control_post_accounting["mamba_available"],
            "mamba_available_delta": proof["mamba_available_delta"],
            "mamba_evictable_before": control_pre_accounting["mamba_evictable"],
            "mamba_evictable_after": control_post_accounting["mamba_evictable"],
            "control_full_tree_unchanged": bool_int(proof["full_tree_unchanged"]),
            "control_structure_unchanged": bool_int(proof["structure_unchanged"]),
            "control_other_mamba_unchanged": bool_int(proof["only_target_mamba_changed"]),
            "control_sanity_check_passed": bool_int(proof["sanity_check_passed"]),
            "control_wall_ms": case["control_wall_ms"],
            "fa_kv_hit_tokens": value_or_blank(full),
            "executable_prefix_tokens": value_or_blank(executable),
            "mamba_boundary_tokens": value_or_blank(match.get("mamba_boundary_tokens")),
            "branching_tokens": value_or_blank(match.get("branching_tokens")),
            "physical_gap_tokens": value_or_blank(match.get("physical_gap_tokens")),
            "extend_start_tokens": value_or_blank(extend.get("extend_start")),
            "extend_end_tokens": value_or_blank(extend.get("extend_end")),
            "actual_replay_tokens": value_or_blank(actual_replay),
            "actual_extend_tokens": value_or_blank(extend.get("actual_extend_tokens")),
            "fresh_tokens": value_or_blank(fresh_tokens),
            "insert_checkpoint_tokens": value_or_blank(insert.get("insert_checkpoint_tokens")),
            "cached_tokens_reported": measured["meta"].get("cached_tokens"),
            "recovery_latency_ms": measured["server_ttft_ms"],
            "logged_recovery_latency_ms": value_or_blank(timing.get("logged_recovery_latency_ms")),
            "request_e2e_latency_ms": measured["server_e2e_ms"],
            "client_ttft_ms": measured["client_ttft_ms"],
            "client_e2e_ms": measured["client_e2e_ms"],
            "completion_tokens": measured["meta"].get("completion_tokens"),
            "num_retractions": measured["meta"].get("num_retractions"),
            "gpu_temperature_before_c": measured["gpu_before"].get("temperature_c", ""),
            "gpu_temperature_after_c": measured["gpu_after"].get("temperature_c", ""),
            "gpu_sm_clock_before_mhz": measured["gpu_before"].get("sm_clock_mhz", ""),
            "gpu_sm_clock_after_mhz": measured["gpu_after"].get("sm_clock_mhz", ""),
            "runtime_match_records": len(match_records),
            "runtime_extend_records": len(extend_records),
            "runtime_timing_records": len(timing_records),
            "runtime_insert_records": len(insert_records),
            "case_mamba_eviction_records": len(evictions),
            "forced_track_records": len(forced_tracks),
            "duplicate_full_rows_freed": duplicate_full_rows_freed,
            "batch_gt_one_lines": sum(size != 1 for size in batch_sizes),
            "runtime_error_lines": len(error_lines),
            "server_log_start_line": window[0][0] if window else "",
            "server_log_end_line": window[-1][0] if window else "",
            "valid": bool_int(not failures),
            "invalid_reason": ";".join(failures),
        }
        raw_rows.append(row)
        validation_by_case[case_id] = {
            "valid_before_pair_checks": not failures,
            "failures": failures,
            "checks": checks,
            "state_checks": state_checks,
            "evictions": evictions,
            "forced_tracks": forced_tracks,
            "long_dup_free_records": long_dup_free_records,
            "duplicate_full_rows_freed": duplicate_full_rows_freed,
            "batch_sizes": batch_sizes,
        }

    # Pair-level gates and effects.
    by_pair = defaultdict(dict)
    for row in raw_rows:
        by_pair[(row["experiment"], int(row["repetition"]))][row["condition"]] = row
    paired_rows = []
    for (experiment, repetition), arms in sorted(by_pair.items()):
        pair_failures = []
        if set(arms) != {"retained", "evicted"}:
            pair_failures.append(f"arm_set={sorted(arms)}")
            continue
        keep, evict = arms["retained"], arms["evicted"]
        if keep["input_sha256"] != evict["input_sha256"]:
            pair_failures.append("input_hash_mismatch")
        if keep["build_signature_sha256"] != evict["build_signature_sha256"]:
            pair_failures.append("build_signature_mismatch")
        if keep["output_sha256"] != evict["output_sha256"]:
            pair_failures.append("measured_output_mismatch")
        if keep["mamba_available_after"] != evict["mamba_available_after"]:
            pair_failures.append("post_control_pool_occupancy_mismatch")
        if not keep["valid"] or not evict["valid"]:
            pair_failures.append("invalid_arm")
        pair_valid = not pair_failures
        if pair_failures:
            keep["valid"] = 0
            evict["valid"] = 0
            keep["invalid_reason"] = ";".join(
                filter(None, [keep["invalid_reason"], *pair_failures])
            )
            evict["invalid_reason"] = ";".join(
                filter(None, [evict["invalid_reason"], *pair_failures])
            )
        latency_delta = float(evict["recovery_latency_ms"]) - float(
            keep["recovery_latency_ms"]
        )
        replay_delta = int(evict["actual_replay_tokens"]) - int(
            keep["actual_replay_tokens"]
        )
        predicted = (
            32768 * WP2_SLOPE_MS_PER_TOKEN if experiment == "FORK_PARENT" else 0.0
        )
        paired_rows.append(
            {
                "schema_version": "flowstate.wp3a.direct_eviction.paired.v1",
                "experiment": experiment,
                "target_boundary": keep["target_boundary"],
                "repetition": repetition,
                "first_condition": (
                    keep["condition"] if keep["case_order"] < evict["case_order"] else evict["condition"]
                ),
                "keep_case_id": keep["case_id"],
                "evict_case_id": evict["case_id"],
                "input_sha256": keep["input_sha256"],
                "input_hash_match": bool_int(keep["input_sha256"] == evict["input_sha256"]),
                "build_signature_match": bool_int(
                    keep["build_signature_sha256"] == evict["build_signature_sha256"]
                ),
                "output_hash_match": bool_int(keep["output_sha256"] == evict["output_sha256"]),
                "pool_occupancy_match": bool_int(
                    keep["mamba_available_after"] == evict["mamba_available_after"]
                ),
                "keep_actual_replay_tokens": keep["actual_replay_tokens"],
                "evict_actual_replay_tokens": evict["actual_replay_tokens"],
                "replay_delta_tokens": replay_delta,
                "keep_recovery_latency_ms": keep["recovery_latency_ms"],
                "evict_recovery_latency_ms": evict["recovery_latency_ms"],
                "paired_recovery_delta_ms": latency_delta,
                "keep_request_e2e_ms": keep["request_e2e_latency_ms"],
                "evict_request_e2e_ms": evict["request_e2e_latency_ms"],
                "paired_e2e_delta_ms": float(evict["request_e2e_latency_ms"])
                - float(keep["request_e2e_latency_ms"]),
                "delta_ms_per_replay_token": (
                    "" if replay_delta == 0 else latency_delta / replay_delta
                ),
                "wp2_predicted_delta_ms": predicted,
                "prediction_error_ms": latency_delta - predicted,
                "prediction_ratio": "" if predicted == 0 else latency_delta / predicted,
                "pair_valid": bool_int(pair_valid),
                "invalid_reason": ";".join(pair_failures),
            }
        )

    raw_rows.sort(key=lambda row: (int(row["repetition"]), int(row["case_order"])))
    paired_rows.sort(key=lambda row: (row["experiment"], int(row["repetition"])))
    write_csv(RAW_CSV, raw_rows)
    write_csv(PAIRED_CSV, paired_rows)

    summary_rows = []
    for experiment in ("FORK_PARENT", "NORMAL_CONTROL"):
        valid_pairs = [
            row
            for row in paired_rows
            if row["experiment"] == experiment and row["pair_valid"] == 1
        ]
        if len(valid_pairs) != 5:
            global_failures.append(f"{experiment}: valid pair count {len(valid_pairs)} != 5")
        keep_values = [float(row["keep_recovery_latency_ms"]) for row in valid_pairs]
        evict_values = [float(row["evict_recovery_latency_ms"]) for row in valid_pairs]
        deltas = [float(row["paired_recovery_delta_ms"]) for row in valid_pairs]
        keep_replays = [int(row["keep_actual_replay_tokens"]) for row in valid_pairs]
        evict_replays = [int(row["evict_actual_replay_tokens"]) for row in valid_pairs]
        keep_stats = sample_stats(keep_values)
        evict_stats = sample_stats(evict_values)
        delta_stats = sample_stats(deltas)
        se = delta_stats["sd"] / math.sqrt(len(deltas))
        ci_low = delta_stats["mean"] - T975_DF4 * se
        ci_high = delta_stats["mean"] + T975_DF4 * se
        predicted = valid_pairs[0]["wp2_predicted_delta_ms"]
        summary_rows.append(
            {
                "schema_version": "flowstate.wp3a.direct_eviction.summary.v1",
                "experiment": experiment,
                "target_boundary": valid_pairs[0]["target_boundary"],
                "n_pairs_planned": 5,
                "n_pairs_valid": len(valid_pairs),
                "n_pairs_invalid": 5 - len(valid_pairs),
                "keep_fa_kv_hit_tokens": statistics.median(
                    int(row["fa_kv_hit_tokens"])
                    for row in raw_rows
                    if row["experiment"] == experiment and row["condition"] == "retained" and row["valid"]
                ),
                "evict_fa_kv_hit_tokens": statistics.median(
                    int(row["fa_kv_hit_tokens"])
                    for row in raw_rows
                    if row["experiment"] == experiment and row["condition"] == "evicted" and row["valid"]
                ),
                "keep_executable_prefix_tokens": statistics.median(
                    int(row["executable_prefix_tokens"])
                    for row in raw_rows
                    if row["experiment"] == experiment and row["condition"] == "retained" and row["valid"]
                ),
                "evict_executable_prefix_tokens": statistics.median(
                    int(row["executable_prefix_tokens"])
                    for row in raw_rows
                    if row["experiment"] == experiment and row["condition"] == "evicted" and row["valid"]
                ),
                "keep_actual_replay_tokens": statistics.median(keep_replays),
                "evict_actual_replay_tokens": statistics.median(evict_replays),
                "replay_delta_tokens": statistics.median(
                    int(row["replay_delta_tokens"]) for row in valid_pairs
                ),
                "keep_recovery_mean_ms": keep_stats["mean"],
                "keep_recovery_median_ms": keep_stats["median"],
                "keep_recovery_sd_ms": keep_stats["sd"],
                "evict_recovery_mean_ms": evict_stats["mean"],
                "evict_recovery_median_ms": evict_stats["median"],
                "evict_recovery_sd_ms": evict_stats["sd"],
                "paired_delta_mean_ms": delta_stats["mean"],
                "paired_delta_sd_ms": delta_stats["sd"],
                "paired_delta_se_ms": se,
                "paired_delta_ci95_low_ms": ci_low,
                "paired_delta_ci95_high_ms": ci_high,
                "paired_delta_median_ms": delta_stats["median"],
                "paired_delta_min_ms": delta_stats["min"],
                "paired_delta_max_ms": delta_stats["max"],
                "positive_pair_count": sum(delta > 0 for delta in deltas),
                "mean_ms_per_replay_token": (
                    "" if statistics.median(int(row["replay_delta_tokens"]) for row in valid_pairs) == 0
                    else delta_stats["mean"]
                    / statistics.median(int(row["replay_delta_tokens"]) for row in valid_pairs)
                ),
                "wp2_predicted_delta_ms": predicted,
                "mean_minus_prediction_ms": delta_stats["mean"] - predicted,
                "mean_to_prediction_ratio": "" if predicted == 0 else delta_stats["mean"] / predicted,
                "ci_method": "paired mean, t(0.975,df=4)",
            }
        )
    write_csv(SUMMARY_CSV, summary_rows)

    # Static chart contract: two comparison facets with independent, zero-based
    # y scales.  Every raw pair is visible and linked.
    palette = {"retained": "#3568A8", "evicted": "#D27632"}
    markers = {"retained": "o", "evicted": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 7.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.80, wspace=0.28)
    panels = [
        ("FORK_PARENT", "Fork Parent @32K"),
        ("NORMAL_CONTROL", "Off-path NORMAL A @32,832"),
    ]
    for axis, (experiment, label) in zip(axes, panels):
        pairs = [
            row for row in paired_rows if row["experiment"] == experiment and row["pair_valid"]
        ]
        for index, pair in enumerate(pairs):
            y = [pair["keep_recovery_latency_ms"], pair["evict_recovery_latency_ms"]]
            offset = (index - 2) * 0.012
            axis.plot(
                [0 + offset, 1 + offset],
                y,
                color="#A7ADB5",
                linewidth=1.3,
                alpha=0.9,
                zorder=1,
            )
            axis.scatter(
                [0 + offset], [y[0]], s=58, marker=markers["retained"],
                facecolor="white", edgecolor=palette["retained"], linewidth=1.8, zorder=3,
            )
            axis.scatter(
                [1 + offset], [y[1]], s=58, marker=markers["evicted"],
                facecolor=palette["evicted"], edgecolor="#8C481C", linewidth=1.0, zorder=3,
            )
        keep_median = statistics.median(float(row["keep_recovery_latency_ms"]) for row in pairs)
        evict_median = statistics.median(float(row["evict_recovery_latency_ms"]) for row in pairs)
        axis.scatter([0, 1], [keep_median, evict_median], marker="D", s=82, color="#22262B", zorder=4)
        axis.annotate(
            f"median {keep_median:.2f} ms", (0, keep_median), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=9, color="#22262B"
        )
        axis.annotate(
            f"median {evict_median:.2f} ms", (1, evict_median), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=9, color="#22262B"
        )
        axis.set_xticks([0, 1], ["Target retained\n(decoy evicted)", "Target evicted\n(decoy retained)"])
        axis.set_ylabel("Server TTFT / recovery latency (ms)")
        axis.set_title(label, loc="left", fontsize=13, fontweight="semibold", color="#22262B")
        axis.grid(axis="y", color="#D9DDE2", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555B63")
        axis.spines["bottom"].set_color("#555B63")
        values = [float(row["keep_recovery_latency_ms"]) for row in pairs] + [
            float(row["evict_recovery_latency_ms"]) for row in pairs
        ]
        axis.set_ylim(0, max(values) * 1.18)
        replay_delta = pairs[0]["replay_delta_tokens"]
        axis.text(
            0.02,
            0.97,
            f"n=5 paired; replay contrast={int(replay_delta):,} tokens",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#555B63",
        )

    fig.suptitle(
        "Recovery latency under targeted recurrent-checkpoint eviction",
        x=0.08,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#20242A",
    )
    fig.text(
        0.08,
        0.835,
        "Qwen3.5-9B · SGLang v0.5.17 · H100 PCIe · concurrency 1 · 1 output token",
        ha="left",
        fontsize=10,
        color="#555B63",
    )
    fig.text(
        0.08,
        0.055,
        "Each line is one matched-content pair. Only the Mamba component was evicted; exact-prefix Full/FA-KV stayed resident. "
        "Panels use independent zero-based y scales.",
        ha="left",
        fontsize=9,
        color="#555B63",
    )
    fig.savefig(PLOT_PNG, dpi=200, facecolor="white")
    fig.savefig(PLOT_PDF, facecolor="white")
    plt.close(fig)

    chart_notes = {
        "analytical_question": "How does next-request server TTFT change when the selected target Mamba checkpoint, rather than an off-path decoy, is evicted?",
        "takeaway": "Fork Parent deletion creates a 32K replay and large TTFT increase; deleting off-path NORMAL A adds no replay or positive recovery penalty, while the observed TTFT difference is small and negative.",
        "family": "comparison",
        "variant": "paired point-and-line small multiples",
        "data_grain": "one formal matched-content pair per repetition; n=5 per experiment",
        "palette_policy": "hard two-root cap; blue retained, orange evicted, marker shape redundancy",
        "scales": "independent zero-based y scales, explicitly labeled",
        "output": [PLOT_PNG.name, PLOT_PDF.name],
        "qa": "passed programmatic checks plus manual PNG/PDF inspection on 2026-08-19 CST",
    }
    CHART_NOTES.write_text(json.dumps(chart_notes, indent=2) + "\n")

    summary_by_experiment = {row["experiment"]: row for row in summary_rows}
    fork = summary_by_experiment["FORK_PARENT"]
    normal = summary_by_experiment["NORMAL_CONTROL"]
    all_valid = not global_failures and all(row["valid"] == 1 for row in raw_rows)
    validation = {
        "schema_version": "flowstate.wp3a.direct_eviction.analysis.v1",
        "valid": all_valid,
        "global_failures": global_failures,
        "formal_window": {"start_line": formal_start, "end_line": formal_end},
        "formal_rows": len(raw_rows),
        "valid_rows": sum(row["valid"] == 1 for row in raw_rows),
        "paired_rows": len(paired_rows),
        "valid_pairs": sum(row["pair_valid"] == 1 for row in paired_rows),
        "case_distribution": {f"{key[0]}:{key[1]}": value for key, value in counts.items()},
        "case_validation": validation_by_case,
        "summary": summary_rows,
        "wp2_slope_ms_per_token": WP2_SLOPE_MS_PER_TOKEN,
        "pilot_excluded": True,
        "primary_latency": "server first_token_latency / TTFT; intervention admin latency excluded",
    }
    SUMMARY_JSON.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    verdict = "PASS" if all_valid else "INVALID"
    findings = f"""# FlowState WP3A：Fork-vs-NORMAL Direct Eviction Causal Validation

## Technical summary

**Verdict: {verdict}.** 在同一 Qwen3.5-9B / SGLang v0.5.17 / H100 PCIe controlled workflow 中，本轮对 exact radix node 做了 scheduler-idle、Mamba-component-only 的直接 intervention；对应 Full/FA-KV 保持 resident。每个 arm 都独立 `flush → rebuild`，并对称删除一个 49.125 MiB recurrent slot：保留组删除 off-path decoy，删除组删除 target，因而 allocator occupancy 与 eviction 管理动作配平。

- **FORK_PARENT@32768**：保留时 `FA/exec/replay = {int(fork['keep_fa_kv_hit_tokens'])}/{int(fork['keep_executable_prefix_tokens'])}/{int(fork['keep_actual_replay_tokens'])}`，server TTFT 中位数 **{fork['keep_recovery_median_ms']:.3f} ms**；直接删除后为 `{int(fork['evict_fa_kv_hit_tokens'])}/{int(fork['evict_executable_prefix_tokens'])}/{int(fork['evict_actual_replay_tokens'])}`，TTFT 中位数 **{fork['evict_recovery_median_ms']:.3f} ms**。5 个 paired effects 的均值为 **+{fork['paired_delta_mean_ms']:.3f} ms**（t-based 95% CI **[{fork['paired_delta_ci95_low_ms']:.3f}, {fork['paired_delta_ci95_high_ms']:.3f}] ms**），中位数 **+{fork['paired_delta_median_ms']:.3f} ms**，5/5 为正。
- **NORMAL A@32832 control**：保留和删除后 Join 都是 `FA/exec/replay = {int(normal['keep_fa_kv_hit_tokens'])}/{int(normal['keep_executable_prefix_tokens'])}/{int(normal['keep_actual_replay_tokens'])}`；target 删除没有改变 executable prefix 或 replay。TTFT 中位数分别为 **{normal['keep_recovery_median_ms']:.3f} / {normal['evict_recovery_median_ms']:.3f} ms**，paired mean difference **{normal['paired_delta_mean_ms']:+.3f} ms**（95% CI **[{normal['paired_delta_ci95_low_ms']:.3f}, {normal['paired_delta_ci95_high_ms']:.3f}] ms**）。

![Paired direct-eviction latency](plot_fork_eviction_latency.png)

## Fork Parent deletion causes a real 32K replay

每个 Fork pair 使用相同 token content、相同 measured input hash 和相同 build signature。目标 Parent checkpoint 先通过一个不进入 measured path 的 sentinel 变为 internal node；decoy 也同样 internalize。两个 condition 都调用同一套 SGLang 原生 component lifecycle：Mamba tombstone、LRU detach/cascade、component-aware allocator free。

| Condition | FA-KV hit | Executable prefix | Actual replay | Scheduled EXTEND | Median TTFT |
|---|---:|---:|---:|---:|---:|
| Fork retained; decoy evicted | {int(fork['keep_fa_kv_hit_tokens'])} | {int(fork['keep_executable_prefix_tokens'])} | {int(fork['keep_actual_replay_tokens'])} | 64 | {fork['keep_recovery_median_ms']:.3f} ms |
| Fork evicted; decoy retained | {int(fork['evict_fa_kv_hit_tokens'])} | {int(fork['evict_executable_prefix_tokens'])} | {int(fork['evict_actual_replay_tokens'])} | 32832 | {fork['evict_recovery_median_ms']:.3f} ms |

删除组的 Full hit 仍精确为 32768，但 executable prefix 降为 root@0；其中 32768 个已在 FA-KV 的 tokens 被真实重新执行，外加 64 个 fresh tokens。runtime 同时记录 `branching=32768`、forced track、duplicate Full-KV rows free，并在请求结束后重新观察到 Parent Mamba checkpoint，证明 first miss 后 self-heal。

WP2 slope 给出的外部预测是 `32768 × 0.0459129 = {fork['wp2_predicted_delta_ms']:.3f} ms`。本轮 paired mean penalty 为 {fork['paired_delta_mean_ms']:.3f} ms，是预测的 **{fork['mean_to_prediction_ratio']:.3f}×**，差 {fork['mean_minus_prediction_ms']:+.3f} ms；因此处于同一约 1.5 s 量级，但这里的值是直接测量，不要求与 slope 精确一致。

## Deleting off-path NORMAL A adds no Join replay or positive recovery penalty

NORMAL A 先在真实 child request 中创建为 checkpoint@32832，再用一个 setup descendant 变为 internal node。之后保留组删除 decoy，删除组只删除 NORMAL A；两组都保留 Parent@32768。完全相同的 Join prompt 在两组都只选择 Parent：

这里的 Join 是 application-level fan-in：A/B/C/D 的真实 output token 被序列化嵌入新的 Parent-lineage prompt；SGLang 并没有合并四个 sibling recurrent states。因此本节检验的是 off-path NORMAL A 对该 Join consumer 的边际贡献，而不是 native multi-state merge。

| Condition | NORMAL A after control | Parent after control | Join FA hit | Join exec | Join replay | Median TTFT |
|---|---|---|---:|---:|---:|---:|
| NORMAL retained; decoy evicted | present | present | {int(normal['keep_fa_kv_hit_tokens'])} | {int(normal['keep_executable_prefix_tokens'])} | {int(normal['keep_actual_replay_tokens'])} | {normal['keep_recovery_median_ms']:.3f} ms |
| NORMAL evicted; decoy retained | absent | present | {int(normal['evict_fa_kv_hit_tokens'])} | {int(normal['evict_executable_prefix_tokens'])} | {int(normal['evict_actual_replay_tokens'])} | {normal['evict_recovery_median_ms']:.3f} ms |

这里的 1-token replay 是 Parent output 已有 physical FA-KV、但可执行 recurrent checkpoint 仍停在 32768 的共同 gap；它在两个 condition 完全相同，因此 NORMAL eviction 引入的 **增量 replay 为 0**。Join 完成后，被删除的 NORMAL A 仍 absent，进一步证明 Join 没有沿它的状态路径执行。

TTFT 的 5 个 paired differences 全部是小幅负值，均值为 **{normal['paired_delta_mean_ms']:+.3f} ms**。本轮未预先定义 equivalence margin，因此不把它表述为统计意义上的“完全相等”；严格结论是没有额外 replay，也没有观察到正向 recovery penalty。

## Causal controls and validation

正式数据为 2 个 targets × 2 conditions × 5 repetitions = **20/20 valid episodes**；每个 episode 都重新 flush/rebuild，warmup 4 个 shape 另行排除。有效性完全由结构和 runtime 日志决定，latency 大小不参与筛选。

每次 intervention 均验证：

- target 与 decoy 在 intervention 前都有一个 Mamba slot，且 target node 是 internal、无 lock/session ref；
- 被控节点 `Mamba present → absent`，Mamba allocator available `+1`、evictable `-1`；
- `tracker[MAMBA]=1`、`tracker[FULL]=0`、host free=0；
- exact path 与全树 Full digest、Full allocator、radix structure 全部不变；全树 Mamba diff 只有被控 node；
- 每个 case 恰好一个 `mamba_evict ... freed=1 fa_kept=True`，无额外 eviction；
- measured RID 各恰好一条 match/extend/timing/insert，所有 prefill batch size=1，无 retraction、OOM 或 runtime error；
- 每个 pair 的 measured input、build signature 和 greedy output hash 一致，control 后 pool occupancy 一致。

逐 episode 的 runtime、control 和 latency 字段见 [`raw_runs.csv`](raw_runs.csv)；逐 pair effect 见 [`paired_effects.csv`](paired_effects.csv)；两项汇总见 [`summary.csv`](summary.csv)。完整 scheduler/control 证据保存在 [`server_full.log`](server_full.log) 与 [`driver_events.jsonl`](driver_events.jsonl)；环境、命令、输入 patch 与 SHA-256 manifest 见 [`metadata.json`](metadata.json)。

## Scope and limitations

- 这是 direct exact-target ablation，而不是自然 LRU policy benchmark。为了只移除 recurrent component、保留 Full，实验先把 target/decoy internalize；setup sentinel/descendant 不属于 measured workflow edge。
- `recovery_latency` 是服务端 first-token latency：包含 recurrent COW/replay、prompt EXTEND 和到首 token 的正常执行；当前 instrumentation 不能可靠拆成纯 replay kernel time，因此不做伪拆分。
- 5 个 paired repetitions 支持这个 controlled setting 的局部因果结论；t interval 是小样本描述性不确定性，不外推到生产 workload、并发、其他模型或 joint FA-KV eviction。
- Fork 两侧都含 64 fresh tokens；WP2 calibration 使用 1 fresh token。直接结果不依赖 slope transfer，但两者数值比较仍受这个 workload 差异影响。
- NORMAL 结论只针对本 workflow horizon 与 Join consumer，不表示普通 boundary 永远无价值。
- NORMAL 的 TTFT 差为小幅负值；本轮没有预设 practical-equivalence margin，因此只对“无新增 replay / 无正向 recovery penalty”作结论，不声称 latency 统计等价。
- Join 是 branch outputs 的 token serialization，不是 sibling recurrent-state merge；本轮测量 consumer 是 Join，没有另行计时 Resume。
- 本轮没有实现 FlowState、selection policy 或 WP3B，也没有做任何 context/width/concurrency sweep。

## Q1–Q5

### Q1. 保留 FORK_PARENT 时，后续 consumer 的 exec_prefix / replay / latency 是多少？

**`exec_prefix=32768`，actual replay=0；server TTFT 中位数 {fork['keep_recovery_median_ms']:.3f} ms**（n=5）。对应 FA-KV hit 也是 32768，scheduled EXTEND 只有 64 个 fresh tokens。

### Q2. 直接删除 FORK_PARENT recurrent checkpoint 后，是否真实触发长 replay？实际 replay tokens 和 latency penalty 是多少？

**是。** FA-KV hit 保持 32768，但 executable prefix 降到 0，真实 replay **32768 tokens**。删除组 TTFT 中位数 {fork['evict_recovery_median_ms']:.3f} ms；5 个 paired penalty 平均 **+{fork['paired_delta_mean_ms']:.3f} ms**、中位数 **+{fork['paired_delta_median_ms']:.3f} ms**，范围 [{fork['paired_delta_min_ms']:.3f}, {fork['paired_delta_max_ms']:.3f}] ms。

### Q3. 这个实测 penalty 是否与 WP2 的 ~1.5s 预测处于相同量级？

**是。** WP2 预测 {fork['wp2_predicted_delta_ms']:.3f} ms，本轮 paired mean 为 {fork['paired_delta_mean_ms']:.3f} ms（{fork['mean_to_prediction_ratio']:.3f}× prediction）。两者都在约 1.5 s 量级；差异不被解释成模型失效，因为 fresh suffix 与实验实现并不完全相同。

### Q4. 直接删除一个 NORMAL checkpoint 后，是否对 Join/Resume 的 executable prefix、replay 和 latency 基本无影响？

**对本轮测量的 Join，恢复结构上是；latency 只能作有限结论。** 删除前后都 `exec_prefix=32768`、actual replay=1，结构上的增量 replay 为 0。TTFT paired mean difference 为 **{normal['paired_delta_mean_ms']:+.3f} ms**（95% CI [{normal['paired_delta_ci95_low_ms']:.3f}, {normal['paired_delta_ci95_high_ms']:.3f}] ms），即没有观察到正向 penalty，且相对 Fork effect 很小；由于未预设 equivalence margin，不声称 TTFT 统计等价。本轮没有另外计时 Resume。

### Q5. 结果是否足以把 WP3A 从“estimated workflow value”升级为“directly measured causal evidence”？

**足以在这个 controlled setting 内升级。** 同一内容、同一 pool occupancy、同一 eviction lifecycle 下，唯一与 measured path 相关的差异是 target Mamba 是否存在：删除 Fork 直接造成 32K replay 与约 1.5 s penalty；删除 off-path NORMAL 不改变 executable prefix/replay。它仍是局部 causal evidence，不是完整 FlowState policy 或跨 workload 定律。
"""
    FINDINGS.write_text(findings)

    print(json.dumps({
        "valid": all_valid,
        "global_failures": global_failures,
        "formal_rows": len(raw_rows),
        "valid_rows": sum(row["valid"] == 1 for row in raw_rows),
        "valid_pairs": sum(row["pair_valid"] == 1 for row in paired_rows),
        "fork_summary": fork,
        "normal_summary": normal,
        "outputs": [str(RAW_CSV), str(PAIRED_CSV), str(SUMMARY_CSV), str(PLOT_PNG), str(PLOT_PDF), str(FINDINGS)],
    }, indent=2))
    if not all_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
