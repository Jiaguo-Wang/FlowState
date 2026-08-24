"""Validate the WP3A runtime trace and build boundary-value artifacts."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import driver as workload


HERE = Path(__file__).resolve().parent
DRIVER_EVENTS = HERE / "driver_events.jsonl"
SERVER_LOG = HERE / "server_full.log"
TRACE = HERE / "workflow_trace.jsonl"
BOUNDARIES_CSV = HERE / "boundary_events.csv"
SUMMARY_JSON = HERE / "analysis_summary.json"
PNG = HERE / "plot_boundary_value.png"
PDF = HERE / "plot_boundary_value.pdf"

WP2_SLOPE_MS_PER_TOKEN = 0.0459129
COUNTERFACTUAL_SCOPE = {
    "removed_component": "mamba_recurrent_checkpoint_only",
    "exact_prefix_fa_kv_resident": True,
    "nearest_shallower_mamba_checkpoint_resident": True,
    "joint_fa_kv_eviction_out_of_scope": True,
    "wp2_calibration_fresh_tokens": 1,
    "wp3a_counterfactual_fresh_tokens": 64,
    "slope_transfer_assumption": (
        "marginal replay cost is additive with the 64-token fresh suffix and "
        "has no material interaction"
    ),
}

MATCH_RE = re.compile(
    r"\[FSVAL\] match_end req=(\S+) full_kv_hit=(\d+) "
    r"exec_prefix=(\d+) mamba_boundary=(\d+|None) "
    r"branching=(\d+|None) gap=(-?\d+)"
)
EXTEND_RE = re.compile(
    r"\[FSVAL\] extend req=(\S+) fill_ids=(\d+) prefix_len=(\d+) "
    r"extend_start=(\d+|None) extend_len=(\d+|None) extend_end=(\d+|None)"
)
INSERT_RE = re.compile(
    r"\[FSVAL\] insert_ckpt req=(\S+) token_ids_len=(\d+) "
    r"ckpt_pos=(\d+|None) strategy=(\S+)"
)

REQUEST_ORDER = (
    "wp3a_parent",
    "wp3a_child_a",
    "wp3a_child_b",
    "wp3a_child_c",
    "wp3a_child_d",
    "wp3a_join",
    "wp3a_resume",
)

EXPECTED_RUNTIME = {
    "wp3a_parent": {
        "prompt": 32_768,
        "full": 0,
        "exec": 0,
        "gap": 0,
        "extend": 32_768,
        "insert": 32_768,
    },
    "wp3a_child_a": {
        "prompt": 32_832,
        "full": 32_768,
        "exec": 32_768,
        "gap": 0,
        "extend": 64,
        "insert": 32_832,
    },
    "wp3a_child_b": {
        "prompt": 32_832,
        "full": 32_769,
        "exec": 32_768,
        "gap": 1,
        "extend": 64,
        "insert": 32_832,
    },
    "wp3a_child_c": {
        "prompt": 32_832,
        "full": 32_769,
        "exec": 32_768,
        "gap": 1,
        "extend": 64,
        "insert": 32_832,
    },
    "wp3a_child_d": {
        "prompt": 32_832,
        "full": 32_769,
        "exec": 32_768,
        "gap": 1,
        "extend": 64,
        "insert": 32_832,
    },
    "wp3a_join": {
        "prompt": 33_024,
        "full": 32_769,
        "exec": 32_768,
        "gap": 1,
        "extend": 256,
        "insert": 33_024,
    },
    "wp3a_resume": {
        "prompt": 33_088,
        "full": 33_024,
        "exec": 33_024,
        "gap": 0,
        "extend": 64,
        "insert": 33_088,
    },
}


def optional_int(value: str) -> int | None:
    return None if value == "None" else int(value)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_runtime() -> tuple[dict[str, dict], dict[str, Counter], list[str]]:
    # Split only on newline so carriage-return progress bars do not skew the
    # physical line numbers shown by nl/rg/editors.
    with SERVER_LOG.open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        lines = handle.read().split("\n")
    records: dict[str, dict] = defaultdict(dict)
    counts: dict[str, Counter] = defaultdict(Counter)
    for line_number, line in enumerate(lines, start=1):
        if match := MATCH_RE.search(line):
            rid, full, executable, boundary, branching, gap = match.groups()
            if rid in REQUEST_ORDER:
                records[rid]["match"] = {
                    "server_log_line": line_number,
                    "fa_kv_hit_tokens": int(full),
                    "executable_prefix_tokens": int(executable),
                    "mamba_boundary_tokens": optional_int(boundary),
                    "branching_tokens": optional_int(branching),
                    "physical_only_gap_tokens": int(gap),
                }
                counts[rid]["match"] += 1
        if extend := EXTEND_RE.search(line):
            rid, fill, prefix, start, length, end = extend.groups()
            if rid in REQUEST_ORDER:
                records[rid]["extend"] = {
                    "server_log_line": line_number,
                    "fill_ids_tokens": int(fill),
                    "scheduled_prefix_tokens": int(prefix),
                    "extend_start_tokens": optional_int(start),
                    "actual_extend_tokens": optional_int(length),
                    "extend_end_tokens": optional_int(end),
                }
                counts[rid]["extend"] += 1
        if insert := INSERT_RE.search(line):
            rid, token_ids_len, checkpoint, strategy = insert.groups()
            if rid in REQUEST_ORDER:
                records[rid]["insert"] = {
                    "server_log_line": line_number,
                    "token_ids_len": int(token_ids_len),
                    "checkpoint_position_tokens": optional_int(checkpoint),
                    "strategy": strategy,
                }
                counts[rid]["insert"] += 1
    return records, counts, lines


def validate_input_dependencies(
    request_events: dict[str, dict], events: list[dict]
) -> tuple[dict, dict[str, list[int]]]:
    parent = workload.toks(51_001, workload.PARENT_LEN)
    parent_output = int(request_events["wp3a_parent"]["output_ids"][0])
    forbidden_first: set[int] = set()
    child_outputs: dict[str, int] = {}
    reconstructed: dict[str, list[int]] = {"wp3a_parent": parent}
    for index, branch_id in enumerate(("A", "B", "C", "D")):
        suffix = workload.distinct_tail(
            80_001 + index * 20_003,
            workload.BRANCH_SUFFIX_LEN,
            forbidden_first,
        )
        rid = f"wp3a_child_{branch_id.lower()}"
        reconstructed[rid] = parent + [parent_output] + suffix
        child_outputs[branch_id] = int(request_events[rid]["output_ids"][0])

    join_payload = workload.distinct_tail(
        180_001, workload.JOIN_SUFFIX_LEN, forbidden_first
    )
    positions = {"A": 31, "B": 95, "C": 159, "D": 223}
    for branch_id, position in positions.items():
        join_payload[position] = child_outputs[branch_id]
    join_prompt = parent + [parent_output] + join_payload
    reconstructed["wp3a_join"] = join_prompt
    join_output = int(request_events["wp3a_join"]["output_ids"][0])
    reconstructed["wp3a_resume"] = (
        join_prompt
        + [join_output]
        + workload.toks(230_001, workload.RESUME_SUFFIX_LEN)
    )

    digest_checks = {
        rid: workload.token_digest(ids) == request_events[rid]["input_sha256"]
        for rid, ids in reconstructed.items()
    }
    bound_event = next(event for event in events if event["event"] == "join_inputs_bound")
    join_binding_check = (
        bound_event["child_output_ids"] == child_outputs
        and bound_event["join_payload_positions"] == positions
        and bound_event["join_payload_sha256"] == workload.token_digest(join_payload)
    )
    resume_event = next(
        event for event in events if event["event"] == "resume_input_bound"
    )
    resume_binding_check = (
        int(resume_event["join_output_id"]) == join_output
        and int(resume_event["join_output_position"]) == len(join_prompt)
        and int(resume_event["resume_suffix_tokens"])
        == workload.RESUME_SUFFIX_LEN
    )
    source_to_boundary = {
        "wp3a_parent": "b_fork_parent",
        "wp3a_child_a": "b_child_a_normal",
        "wp3a_child_b": "b_child_b_normal",
        "wp3a_child_c": "b_child_c_normal",
        "wp3a_child_d": "b_child_d_normal",
        "wp3a_join": "b_join",
        "wp3a_resume": "b_resume",
    }
    raw_prefix_matches: dict[str, int] = {}
    for source_index, source_rid in enumerate(REQUEST_ORDER):
        source_prompt = reconstructed[source_rid]
        raw_prefix_matches[source_to_boundary[source_rid]] = sum(
            reconstructed[future_rid][: len(source_prompt)] == source_prompt
            for future_rid in REQUEST_ORDER[source_index + 1 :]
        )
    return (
        {
            "all_input_digests_match": all(digest_checks.values()),
            "input_digest_checks": digest_checks,
            "join_embeds_all_four_actual_child_outputs": join_binding_check,
            "resume_embeds_actual_join_output": resume_binding_check,
            "raw_future_prefix_match_count_by_boundary": raw_prefix_matches,
        },
        reconstructed,
    )


def validate() -> tuple[list[dict], dict[str, dict], dict, dict[str, list[int]]]:
    events = read_jsonl(DRIVER_EVENTS)
    records, counts, server_lines = parse_runtime()
    request_event_list = [
        event for event in events if event.get("event") == "request_completed"
    ]
    request_events = {
        event["rid"]: event
        for event in request_event_list
    }
    boundaries = [
        event for event in events if event.get("event") == "boundary_declared"
    ]

    failures: list[str] = []
    sequences = [int(event["sequence"]) for event in events]
    if sequences != list(range(1, len(events) + 1)):
        failures.append("driver event sequence is not unique, contiguous, and ordered")
    if len(request_event_list) != len(request_events):
        failures.append("duplicate request RID in driver events")
    boundary_ids = [boundary["boundary_id"] for boundary in boundaries]
    if len(boundary_ids) != len(set(boundary_ids)):
        failures.append("duplicate boundary ID in driver events")
    if tuple(request_events) != REQUEST_ORDER:
        failures.append(f"request order mismatch: {tuple(request_events)}")
    if len(boundaries) != 7:
        failures.append(f"expected 7 boundaries, found {len(boundaries)}")
    flushes = [event for event in events if event.get("event") == "cache_flush"]
    if len(flushes) != 1 or flushes[0].get("http_status") != 200:
        failures.append("exactly one successful cache flush was not observed")

    for rid, expected in EXPECTED_RUNTIME.items():
        if counts[rid] != Counter({"match": 1, "extend": 1, "insert": 1}):
            failures.append(f"{rid}: runtime record counts {dict(counts[rid])}")
            continue
        event = request_events[rid]
        match = records[rid]["match"]
        extend = records[rid]["extend"]
        insert = records[rid]["insert"]
        checks = {
            "prompt": int(event["prompt_tokens"]) == expected["prompt"],
            "full_hit": match["fa_kv_hit_tokens"] == expected["full"],
            "exec_prefix": match["executable_prefix_tokens"] == expected["exec"],
            "boundary": match["mamba_boundary_tokens"] == expected["exec"],
            "gap": match["physical_only_gap_tokens"] == expected["gap"],
            "gap_identity": match["physical_only_gap_tokens"]
            == match["fa_kv_hit_tokens"] - match["executable_prefix_tokens"],
            "cached_meta": event["meta"].get("cached_tokens") == expected["exec"],
            "extend_start": extend["extend_start_tokens"] == expected["exec"],
            "scheduled_prefix": extend["scheduled_prefix_tokens"] == expected["exec"],
            "extend_len": extend["actual_extend_tokens"] == expected["extend"],
            "extend_identity": extend["actual_extend_tokens"]
            == expected["prompt"] - expected["exec"],
            "extend_partition_identity": extend["actual_extend_tokens"]
            == match["physical_only_gap_tokens"]
            + expected["prompt"]
            - match["fa_kv_hit_tokens"],
            "extend_end": extend["extend_end_tokens"] == expected["prompt"],
            "fill_ids": extend["fill_ids_tokens"] == expected["prompt"],
            "insert_position": insert["checkpoint_position_tokens"]
            == expected["insert"],
            "insert_strategy": insert["strategy"] == "extra_buffer",
            "one_output": len(event.get("output_ids", [])) == 1,
            "one_completion": event["meta"].get("completion_tokens") == 1,
            "no_retraction": event["meta"].get("num_retractions", 0) == 0,
            "length_finish": event["meta"].get("finish_reason")
            == {"type": "length", "length": 1},
        }
        failures.extend(f"{rid}: {name}" for name, passed in checks.items() if not passed)

    dependency_checks, reconstructed = validate_input_dependencies(
        request_events, events
    )
    failures.extend(
        f"dependency check failed: {name}"
        for name, passed in dependency_checks.items()
        if name
        not in {"input_digest_checks", "raw_future_prefix_match_count_by_boundary"}
        and not passed
    )
    if not all(dependency_checks["input_digest_checks"].values()):
        failures.append("one or more request input digests failed reconstruction")

    parent_match_line = records[REQUEST_ORDER[0]]["match"]["server_log_line"]
    backend_flush_lines = [
        line_number
        for line_number in range(1, parent_match_line + 1)
        if "Cache flushed successfully!" in server_lines[line_number - 1]
    ]
    if not backend_flush_lines:
        failures.append("backend did not log a successful flush before parent")
    first_line = backend_flush_lines[-1] if backend_flush_lines else parent_match_line
    last_insert_line = records[REQUEST_ORDER[-1]]["insert"]["server_log_line"]
    last_line = next(
        (
            line_number
            for line_number in range(last_insert_line + 1, len(server_lines) + 1)
            if "Prefill batch" in server_lines[line_number - 1]
        ),
        last_insert_line,
    )
    measured_window = server_lines[first_line - 1 : last_line]
    eviction_lines = [line for line in measured_window if "[FSVAL] mamba_evict" in line]
    error_lines = [
        line
        for line in measured_window
        if re.search(r"\bERROR\b|Traceback|out of memory", line, re.I)
    ]
    batch_sizes = [
        int(match.group(1))
        for line in measured_window
        if (match := re.search(r"#new-seq:\s*(\d+)", line))
    ]
    if eviction_lines:
        failures.append(f"unexpected recurrent eviction count: {len(eviction_lines)}")
    if error_lines:
        failures.append(f"runtime errors in measured window: {len(error_lines)}")
    if not batch_sizes or any(size != 1 for size in batch_sizes):
        failures.append(f"non-serial batch sizes: {batch_sizes}")

    runtime_summary = {
        "valid": not failures,
        "failures": failures,
        "request_count": len(request_events),
        "boundary_count": len(boundaries),
        "cache_flush_http_status": flushes[0]["http_status"] if flushes else None,
        "backend_cache_flush_observed": bool(backend_flush_lines),
        "backend_cache_flush_server_log_line": (
            backend_flush_lines[-1] if backend_flush_lines else None
        ),
        "mamba_evictions_in_measured_window": len(eviction_lines),
        "runtime_errors_in_measured_window": len(error_lines),
        "observed_prefill_batch_sizes": batch_sizes,
        "concurrency": 1,
        "counterfactual_scope": COUNTERFACTUAL_SCOPE,
        "dependency_checks": dependency_checks,
        "server_log_window": {"first_line": first_line, "last_line": last_line},
    }
    if failures:
        raise RuntimeError("WP3A validation failed:\n- " + "\n- ".join(failures))
    return events, records, runtime_summary, reconstructed


def derive_boundary_rows(
    events: list[dict],
    records: dict[str, dict],
    raw_prefix_matches: dict[str, int],
    reconstructed: dict[str, list[int]],
) -> tuple[list[dict], list[dict]]:
    requests = [
        event for event in events if event.get("event") == "request_completed"
    ]
    boundaries = [
        event for event in events if event.get("event") == "boundary_declared"
    ]
    driver_sequence = {event["rid"]: int(event["sequence"]) for event in requests}
    boundary_by_id = {event["boundary_id"]: event for event in boundaries}

    reuse_events: list[dict] = []
    consumers: dict[str, list[str]] = defaultdict(list)
    credit_details: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        rid = request["rid"]
        executable = records[rid]["match"]["executable_prefix_tokens"]
        if executable == 0:
            continue
        eligible = [
            boundary
            for boundary in boundaries
            if int(boundary["sequence"]) < driver_sequence[rid]
            and int(boundary["token_position"]) == executable
            and records[boundary["source_rid"]]["insert"][
                "checkpoint_position_tokens"
            ]
            == int(boundary["token_position"])
            and reconstructed[rid][:executable]
            == reconstructed[boundary["source_rid"]][:executable]
        ]
        if len(eligible) != 1:
            raise RuntimeError(
                f"{rid}: expected one prior candidate at exec={executable}, "
                f"found {[item['boundary_id'] for item in eligible]}"
            )
        boundary = eligible[0]
        consumers[boundary["boundary_id"]].append(rid)
        shallower = [
            candidate
            for candidate in boundaries
            if int(candidate["sequence"]) < driver_sequence[rid]
            and int(candidate["token_position"]) < executable
            and records[candidate["source_rid"]]["insert"][
                "checkpoint_position_tokens"
            ]
            == int(candidate["token_position"])
            and reconstructed[rid][: int(candidate["token_position"])]
            == reconstructed[candidate["source_rid"]][
                : int(candidate["token_position"])
            ]
        ]
        fallback = max(
            (int(candidate["token_position"]) for candidate in shallower),
            default=0,
        )
        distance = executable - fallback
        prefix_sha256 = workload.token_digest(reconstructed[rid][:executable])
        credit_details[boundary["boundary_id"]].append(
            {
                "consumer_rid": rid,
                "fallback_token_position": fallback,
                "incremental_checkpoint_depth_tokens": distance,
                "candidate_prefix_sha256": prefix_sha256,
            }
        )
        reuse_events.append(
            {
                "event": "runtime_boundary_reused",
                "boundary_id": boundary["boundary_id"],
                "boundary_type": boundary["boundary_type"],
                "consumer_rid": rid,
                "selected_exec_prefix_tokens": executable,
                "fa_kv_hit_tokens": records[rid]["match"]["fa_kv_hit_tokens"],
                "physical_only_gap_tokens": records[rid]["match"][
                    "physical_only_gap_tokens"
                ],
                "candidate_prefix_sha256": prefix_sha256,
                "exact_token_prefix_match": True,
                "counterfactual_fallback_token_position": fallback,
                "incremental_checkpoint_depth_tokens": distance,
                "credit_rule": (
                    "unique ready prior candidate at actual exec_prefix with "
                    "an exact token-prefix match"
                ),
            }
        )

    # Primary value is one checkpoint-miss episode under this sequential
    # workflow.  WP2 established that the first replay self-heals the state,
    # so reuse_count must not be multiplied into the one-miss saving.  The
    # gross independent-miss score is retained only as an explicit upper bound.
    rows: list[dict] = []
    for boundary in boundaries:
        boundary_id = boundary["boundary_id"]
        observed_reuse_count = len(consumers[boundary_id])
        right_censored = boundary_id == "b_resume"
        reuse_count = None if right_censored else observed_reuse_count
        future_reused = (
            None if right_censored else int(observed_reuse_count > 0)
        )
        details = credit_details[boundary_id]
        fallback_positions = {
            detail["fallback_token_position"] for detail in details
        }
        distances = {
            detail["incremental_checkpoint_depth_tokens"] for detail in details
        }
        if len(fallback_positions) > 1 or len(distances) > 1:
            raise RuntimeError(
                f"{boundary_id}: heterogeneous fallback paths need an event-level "
                "value table: {details}"
            )
        fallback = next(iter(fallback_positions)) if fallback_positions else None
        observed_replay_distance = next(iter(distances)) if distances else 0
        observed_avoided = observed_replay_distance if future_reused else 0
        observed_gross = sum(
            detail["incremental_checkpoint_depth_tokens"] for detail in details
        )
        replay_distance = None if right_censored else observed_replay_distance
        avoided = None if right_censored else observed_avoided
        gross = None if right_censored else observed_gross
        insert = records[boundary["source_rid"]]["insert"]
        rows.append(
            {
                "workflow_id": boundary["workflow_id"],
                "boundary_id": boundary_id,
                "boundary_type": boundary["boundary_type"],
                "token_position": int(boundary["token_position"]),
                "future_reused": future_reused,
                "reuse_count": reuse_count,
                "observed_reuse_count_to_horizon": observed_reuse_count,
                "raw_future_prefix_match_count": raw_prefix_matches[boundary_id],
                "replay_distance_tokens": replay_distance,
                "avoided_replay_tokens": avoided,
                "incremental_checkpoint_depth_tokens": replay_distance,
                "single_miss_avoided_replay_tokens": avoided,
                "estimated_recovery_saving_ms": (
                    None
                    if avoided is None
                    else round(WP2_SLOPE_MS_PER_TOKEN * avoided, 6)
                ),
                "source_rid": boundary["source_rid"],
                "semantic_roles": "|".join(boundary["semantic_roles"]),
                "checkpoint_insert_observed": 1,
                "insert_checkpoint_position": insert[
                    "checkpoint_position_tokens"
                ],
                "future_consumer_rids": "|".join(consumers[boundary_id]),
                "observation_status": (
                    "right_censored_terminal"
                    if boundary_id == "b_resume"
                    else "complete_within_workflow"
                ),
                "counterfactual_fallback_token_position": (
                    "" if fallback is None else fallback
                ),
                "gross_if_independently_missing_each_reuse_tokens": gross,
                "gross_if_independently_missing_each_reuse_ms": (
                    None
                    if gross is None
                    else round(WP2_SLOPE_MS_PER_TOKEN * gross, 6)
                ),
                "notes": boundary["notes"],
            }
        )

    # Exact expected credit assignments are a hard semantic/runtime gate.
    expected_consumers = {
        "b_fork_parent": [
            "wp3a_child_a",
            "wp3a_child_b",
            "wp3a_child_c",
            "wp3a_child_d",
            "wp3a_join",
        ],
        "b_child_a_normal": [],
        "b_child_b_normal": [],
        "b_child_c_normal": [],
        "b_child_d_normal": [],
        "b_join": ["wp3a_resume"],
        "b_resume": [],
    }
    actual_consumers = {
        boundary_id: consumers[boundary_id] for boundary_id in boundary_by_id
    }
    if actual_consumers != expected_consumers:
        raise RuntimeError(
            f"unexpected boundary credit assignment: {actual_consumers}"
        )
    return rows, reuse_events


def write_csv(rows: list[dict]) -> None:
    with BOUNDARIES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_trace(
    events: list[dict],
    records: dict[str, dict],
    reuse_events: list[dict],
    runtime_summary: dict,
) -> None:
    reuse_by_consumer = {
        event["consumer_rid"]: event for event in reuse_events
    }
    trace_events: list[dict] = []
    for event in events:
        enriched = {"source": "driver", **event}
        if event.get("event") == "request_completed":
            rid = event["rid"]
            enriched["runtime"] = records[rid]
            enriched["runtime_valid"] = True
            enriched["executable_state_boundary_id"] = (
                reuse_by_consumer[rid]["boundary_id"]
                if rid in reuse_by_consumer
                else "ROOT"
            )
            if rid == "wp3a_join":
                enriched["data_dependency_boundary_ids"] = [
                    "b_child_a_normal",
                    "b_child_b_normal",
                    "b_child_c_normal",
                    "b_child_d_normal",
                ]
                enriched["native_recurrent_state_merge"] = False
            elif rid == "wp3a_resume":
                enriched["data_dependency_boundary_ids"] = ["b_join"]
                enriched["native_recurrent_state_merge"] = False
        trace_events.append(enriched)
        if event.get("event") == "request_completed" and event["rid"] in reuse_by_consumer:
            trace_events.append(
                {
                    "source": "derived_from_runtime",
                    "schema_version": "flowstate.wp3a.trace.v1",
                    "workflow_id": workload.WORKFLOW_ID,
                    **reuse_by_consumer[event["rid"]],
                }
            )
    trace_events.append(
        {
            "source": "analysis",
            "schema_version": "flowstate.wp3a.trace.v1",
            "workflow_id": workload.WORKFLOW_ID,
            "event": "validation_summary",
            **runtime_summary,
        }
    )
    with TRACE.open("w", encoding="utf-8") as handle:
        for index, event in enumerate(trace_events, start=1):
            handle.write(
                json.dumps({"trace_sequence": index, **event}, sort_keys=True)
                + "\n"
            )


def plot(rows: list[dict]) -> None:
    labels = [
        "Fork parent",
        "Normal A",
        "Normal B",
        "Normal C",
        "Normal D",
        "Join",
        "Resume*",
    ]
    censored = [row["estimated_recovery_saving_ms"] is None for row in rows]
    values = [
        0.0 if is_censored else float(row["estimated_recovery_saving_ms"])
        for row, is_censored in zip(rows, censored)
    ]
    colors = [
        "#6C5CE7",
        "#AAB2BD",
        "#AAB2BD",
        "#AAB2BD",
        "#AAB2BD",
        "#E67E22",
        "#2E86AB",
    ]
    y = list(range(len(rows)))
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.5, 6.4),
        gridspec_kw={"width_ratios": [2.2, 1.15]},
    )
    left.barh(y, values, color=colors, height=0.63)
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlim(0, 1_650)
    left.set_xlabel("Estimated one-miss recovery saving (ms)")
    left.set_title("All boundaries")
    left.grid(axis="x", alpha=0.22)
    for index, (value, row) in enumerate(zip(values, rows)):
        if censored[index]:
            left.scatter(8, index, marker="x", color="#2E86AB", s=45, zorder=3)
            left.text(
                22,
                index,
                "N/A (right-censored)",
                va="center",
                fontsize=9,
                color="#2E86AB",
            )
            continue
        inside = value > 250
        x = value - 18 if inside else (value + 18 if value else 8)
        left.text(
            x,
            index,
            f"{value:,.3f} ms  (reuse={row['reuse_count']})",
            va="center",
            ha="right" if inside else "left",
            color="white" if inside else "black",
            fontweight="bold" if inside else "normal",
            fontsize=9,
        )

    zoom_labels = labels[1:]
    zoom_values = values[1:]
    zoom_censored = censored[1:]
    zoom_colors = colors[1:]
    zy = list(range(len(zoom_values)))
    right.barh(zy, zoom_values, color=zoom_colors, height=0.63)
    right.set_yticks(zy, zoom_labels)
    right.invert_yaxis()
    right.set_xlim(0, 13.2)
    right.set_xlabel("Estimated saving (ms)")
    right.set_title("Non-fork zoom")
    right.grid(axis="x", alpha=0.22)
    for index, value in enumerate(zoom_values):
        if zoom_censored[index]:
            right.scatter(0.12, index, marker="x", color="#2E86AB", s=45, zorder=3)
            right.text(
                0.35,
                index,
                "N/A",
                va="center",
                fontsize=9,
                color="#2E86AB",
            )
            continue
        right.text(
            value + 0.18 if value else 0.12,
            index,
            f"{value:,.3f}",
            va="center",
            fontsize=9,
        )

    fig.suptitle(
        "Future recurrent-state recovery value differs by workflow boundary",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Single sequential 32K-parent workflow; WP2 marginal slope × avoided replay tokens",
        ha="center",
        color="#4A4A4A",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.018,
        "One-miss holds exact-prefix FA-KV resident; first replay self-heals. "
        "Slope transfer assumes no replay × 64-fresh-token interaction.\n"
        "*Resume is terminal/right-censored. Values are not direct latency measurements.",
        ha="center",
        color="#555555",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.15, wspace=0.48)
    fig.savefig(PNG, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    events, records, runtime_summary, reconstructed = validate()
    rows, reuse_events = derive_boundary_rows(
        events,
        records,
        runtime_summary["dependency_checks"][
            "raw_future_prefix_match_count_by_boundary"
        ],
        reconstructed,
    )
    write_csv(rows)
    write_trace(events, records, reuse_events, runtime_summary)
    plot(rows)
    summary = {
        **runtime_summary,
        "wp2_slope_ms_per_token": WP2_SLOPE_MS_PER_TOKEN,
        "primary_value_semantics": (
            "one Mamba-only missing-checkpoint episode with exact-prefix FA-KV "
            "and nearest shallower Mamba fallback resident; first replay "
            "self-heals, so reuse_count is not multiplied"
        ),
        "aggregate_candidate_scores": {
            "primary_single_miss_avoided_replay_tokens": sum(
                int(row["avoided_replay_tokens"])
                for row in rows
                if row["avoided_replay_tokens"] is not None
            ),
            "primary_single_miss_estimated_saving_ms": round(
                sum(
                    float(row["estimated_recovery_saving_ms"])
                    for row in rows
                    if row["estimated_recovery_saving_ms"] is not None
                ),
                6,
            ),
            "gross_independent_miss_avoided_replay_tokens": sum(
                int(row["gross_if_independently_missing_each_reuse_tokens"])
                for row in rows
                if row["gross_if_independently_missing_each_reuse_tokens"]
                is not None
            ),
            "gross_independent_miss_estimated_saving_ms": round(
                sum(
                    float(row["gross_if_independently_missing_each_reuse_ms"])
                    for row in rows
                    if row["gross_if_independently_missing_each_reuse_ms"]
                    is not None
                ),
                6,
            ),
        },
        "boundary_rows": rows,
        "runtime_reuse_events": reuse_events,
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
