#!/usr/bin/env python3
"""Run the paired direct-eviction experiment for FlowState WP3A.

Every measured episode is independent:

  flush -> rebuild target and an internal decoy -> evict one Mamba component
        -> verify exact tree state -> issue one measured request

KEEP_TARGET evicts the off-path decoy.  EVICT_TARGET evicts the target.  Both
arms therefore execute the same allocator-aware component eviction and finish
with the same number of retained Mamba slots.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from array import array
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eviction_engine import DirectEvictionEngine
from targeted_probe import ControlClient


VOCAB_SIZE = 248_320
PARENT_LEN = 32_768
BRANCH_SUFFIX_LEN = 63
JOIN_SUFFIX_LEN = 255
DECOY_LEN = 64
INTERNALIZER_LEN = 64
OUTPUT_TOKENS = 1
CONTROL_PORT = int(os.environ.get("WP3D_CTRL_PORT", "49935"))

ENGINE_CONFIG = {
    "model_path": "/model",
    "context_length": 45_056,
    "mem_fraction_static": 0.40,
    "tp_size": 1,
    "chunked_prefill_size": 45_056,
    "max_mamba_cache_size": 16,
    "mamba_radix_cache_strategy": "extra_buffer",
    "disable_cuda_graph": True,
    "disable_overlap_schedule": True,
    "enable_request_time_stats_logging": True,
    "stream_interval": 1,
    "log_level": "info",
}

SAMPLING_PARAMS = {
    "max_new_tokens": OUTPUT_TOKENS,
    "temperature": 0,
    "ignore_eos": True,
}

FORMAL_ORDERS = [
    ["fork_retained", "fork_evicted", "normal_retained", "normal_evicted"],
    ["fork_evicted", "fork_retained", "normal_evicted", "normal_retained"],
    ["normal_retained", "normal_evicted", "fork_evicted", "fork_retained"],
    ["normal_evicted", "normal_retained", "fork_retained", "fork_evicted"],
    ["fork_evicted", "normal_retained", "normal_evicted", "fork_retained"],
]


def toks(seed: int, count: int) -> list[int]:
    return [(seed + index * 7_919) % VOCAB_SIZE for index in range(count)]


def token_sha256(ids) -> str:
    return hashlib.sha256(array("q", [int(x) for x in ids]).tobytes()).hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_first_distinct(ids: list[int], forbidden: set[int]) -> list[int]:
    while ids[0] in forbidden:
        ids[0] = (ids[0] + 1) % VOCAB_SIZE
    forbidden.add(ids[0])
    return ids


def gpu_telemetry() -> dict:
    query = (
        "temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,"
        "utilization.gpu"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
        names = [
            "temperature_c",
            "sm_clock_mhz",
            "memory_clock_mhz",
            "power_w",
            "memory_used_mib",
            "utilization_pct",
        ]
        return {name: float(value) for name, value in zip(names, values)}
    except Exception as exc:
        return {"error": repr(exc)}


def simplified_meta(meta: dict) -> dict:
    keys = (
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "num_retractions",
        "finish_reason",
        "first_token_latency",
        "e2e_latency",
    )
    return {key: meta.get(key) for key in keys}


def validate_generation(result: dict, rid: str) -> tuple[list[int], dict]:
    if not isinstance(result, dict):
        raise AssertionError(f"{rid}: response is not a dict")
    output_ids = [int(x) for x in (result.get("output_ids") or [])]
    meta = result.get("meta_info") or {}
    if len(output_ids) != OUTPUT_TOKENS:
        raise AssertionError(f"{rid}: expected one output token, got {output_ids}")
    if int(meta.get("completion_tokens", OUTPUT_TOKENS)) != OUTPUT_TOKENS:
        raise AssertionError(f"{rid}: bad completion token count: {meta}")
    if int(meta.get("num_retractions", 0) or 0) != 0:
        raise AssertionError(f"{rid}: request retracted: {meta}")
    finish = meta.get("finish_reason")
    finish_type = finish.get("type") if isinstance(finish, dict) else finish
    if finish_type != "length":
        raise AssertionError(f"{rid}: unexpected finish reason: {finish}")
    return output_ids, meta


class EventWriter:
    def __init__(self, path: Path):
        self.path = path
        self.sequence = 0

    def emit(self, event: dict) -> dict:
        self.sequence += 1
        record = {
            "schema_version": "flowstate.wp3a.direct_eviction.trace.v1",
            "sequence": self.sequence,
            **event,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record


class Experiment:
    def __init__(self, engine, client: ControlClient, writer: EventWriter):
        self.engine = engine
        self.client = client
        self.writer = writer
        self.case_counter = 0

    def nonce(self, case_id: str, label: str) -> str:
        return f"{case_id}:{label}"

    def idle_barrier(self, case_id: str, label: str) -> dict:
        return self.client.census(self.nonce(case_id, f"census:{label}"))

    def flush(self, case_id: str) -> dict:
        started = time.perf_counter()
        last_error = None
        for _ in range(2):
            try:
                self.engine.flush_cache()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)
        if last_error is not None:
            raise last_error
        census = self.idle_barrier(case_id, "after_flush")
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if census["tree"]["mamba_node_count"] != 0:
            raise AssertionError(f"{case_id}: flush left Mamba nodes: {census}")
        return self.writer.emit(
            {
                "event": "cache_flush",
                "case_id": case_id,
                "client_wall_ms": elapsed_ms,
                "post_flush_census": census,
            }
        )

    def generate_blocking(
        self,
        *,
        case_id: str,
        rid: str,
        role: str,
        ids: list[int],
    ) -> dict:
        started = time.perf_counter()
        result = self.engine.generate(
            input_ids=ids,
            sampling_params=SAMPLING_PARAMS,
            rid=rid,
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        output_ids, meta = validate_generation(result, rid)
        return self.writer.emit(
            {
                "event": "request_completed",
                "case_id": case_id,
                "rid": rid,
                "role": role,
                "measured": False,
                "prompt_tokens": len(ids),
                "input_sha256": token_sha256(ids),
                "output_ids": output_ids,
                "output_sha256": token_sha256(output_ids),
                "client_wall_ms": elapsed_ms,
                "meta": simplified_meta(meta),
            }
        )

    def generate_measured(
        self,
        *,
        case_id: str,
        rid: str,
        role: str,
        ids: list[int],
    ) -> dict:
        loop = getattr(self.engine, "loop", None) or asyncio.new_event_loop()

        async def run():
            submitted = time.perf_counter()
            generator = await self.engine.async_generate(
                input_ids=ids,
                sampling_params=SAMPLING_PARAMS,
                stream=True,
                rid=rid,
            )
            first = None
            last = None
            output_ids = []
            async for chunk in generator:
                last = chunk
                current = chunk.get("output_ids") if isinstance(chunk, dict) else []
                output_ids = [int(x) for x in (current or [])]
                if output_ids and first is None:
                    first = time.perf_counter()
            finished = time.perf_counter()
            if first is None:
                first = finished
            return last, (first - submitted) * 1_000.0, (finished - submitted) * 1_000.0

        gpu_before = gpu_telemetry()
        last, client_ttft_ms, client_e2e_ms = loop.run_until_complete(run())
        gpu_after = gpu_telemetry()
        result = last if isinstance(last, dict) else {}
        output_ids, meta = validate_generation(result, rid)
        if meta.get("first_token_latency") is None or meta.get("e2e_latency") is None:
            raise AssertionError(f"{rid}: server timing fields are missing: {meta}")
        server_ttft_ms = float(meta["first_token_latency"]) * 1_000.0
        server_e2e_ms = float(meta["e2e_latency"]) * 1_000.0
        if not (0 < server_ttft_ms <= server_e2e_ms <= client_e2e_ms + 1e-6):
            raise AssertionError(
                f"{rid}: invalid latency ordering: "
                f"server_ttft={server_ttft_ms}, server_e2e={server_e2e_ms}, "
                f"client_e2e={client_e2e_ms}"
            )
        return self.writer.emit(
            {
                "event": "request_completed",
                "case_id": case_id,
                "rid": rid,
                "role": role,
                "measured": True,
                "prompt_tokens": len(ids),
                "input_sha256": token_sha256(ids),
                "output_ids": output_ids,
                "output_sha256": token_sha256(output_ids),
                "client_ttft_ms": client_ttft_ms,
                "client_e2e_ms": client_e2e_ms,
                "server_ttft_ms": server_ttft_ms,
                "server_e2e_ms": server_e2e_ms,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "meta": simplified_meta(meta),
            }
        )

    def inspect(self, case_id: str, label: str, ids: list[int]) -> dict:
        return self.client.checkpoint_control(
            nonce=self.nonce(case_id, f"inspect:{label}"),
            label=f"{case_id}:{label}",
            action="inspect",
            token_ids=ids,
        )

    def evict(self, case_id: str, label: str, ids: list[int]) -> tuple[dict, float]:
        started = time.perf_counter()
        response = self.client.checkpoint_control(
            nonce=self.nonce(case_id, f"evict:{label}"),
            label=f"{case_id}:{label}",
            action="evict_mamba",
            token_ids=ids,
        )
        return response, (time.perf_counter() - started) * 1_000.0

    def build_decoy(self, case_id: str, seed_offset: int, parent_first: int):
        forbidden = {int(parent_first)}
        decoy = make_first_distinct(toks(241_001 + seed_offset, DECOY_LEN), forbidden)
        extension = make_first_distinct(
            toks(211_003 + seed_offset, INTERNALIZER_LEN), forbidden
        )
        decoy_event = self.generate_blocking(
            case_id=case_id,
            rid=f"{case_id}_decoy",
            role="DECOY_BUILD",
            ids=decoy,
        )
        self.generate_blocking(
            case_id=case_id,
            rid=f"{case_id}_decoy_desc",
            role="DECOY_INTERNALIZER",
            ids=decoy + extension,
        )
        self.idle_barrier(case_id, "after_decoy")
        return decoy, decoy_event

    def run_fork_case(
        self,
        *,
        phase: str,
        repetition: int,
        condition: str,
        case_order: int,
        seed_offset: int,
    ) -> dict:
        case_id = f"{phase}_r{repetition:02d}_o{case_order:02d}_fork_{condition}"
        self.writer.emit(
            {
                "event": "case_start",
                "case_id": case_id,
                "phase": phase,
                "repetition": repetition,
                "case_order": case_order,
                "experiment": "FORK_PARENT",
                "condition": condition,
                "content_seed_offset": seed_offset,
            }
        )
        print(f"[FSWP3D] CASE_START id={case_id}", flush=True)
        self.flush(case_id)

        parent = toks(51_001 + seed_offset, PARENT_LEN)
        parent_event = self.generate_blocking(
            case_id=case_id,
            rid=f"{case_id}_parent",
            role="PARENT_BUILD",
            ids=parent,
        )
        parent_output = int(parent_event["output_ids"][0])

        # Make Parent@32768 internal without caching the next token used by the
        # measured child.  The measured path therefore still has a 32768 Full hit.
        internalizer = toks(71_003 + seed_offset, INTERNALIZER_LEN)
        if internalizer[0] == parent_output:
            internalizer[0] = (internalizer[0] + 1) % VOCAB_SIZE
        self.generate_blocking(
            case_id=case_id,
            rid=f"{case_id}_parent_internalizer",
            role="PARENT_INTERNALIZER",
            ids=parent + internalizer,
        )

        decoy, _ = self.build_decoy(case_id, seed_offset, parent[0])
        self.idle_barrier(case_id, "before_control")
        pre_target = self.inspect(case_id, "target_pre", parent)
        pre_decoy = self.inspect(case_id, "decoy_pre", decoy)

        if condition == "retained":
            evicted_role = "DECOY"
            control, control_ms = self.evict(case_id, "decoy", decoy)
        elif condition == "evicted":
            evicted_role = "TARGET"
            control, control_ms = self.evict(case_id, "fork_parent", parent)
        else:
            raise AssertionError(condition)

        post_target = self.inspect(case_id, "target_post_control", parent)
        post_decoy = self.inspect(case_id, "decoy_post_control", decoy)
        pre_measure_census = self.idle_barrier(case_id, "pre_measure")

        suffix = toks(81_001 + seed_offset, BRANCH_SUFFIX_LEN)
        measured_input = parent + [parent_output] + suffix
        measured = self.generate_measured(
            case_id=case_id,
            rid=f"{case_id}_measure",
            role="FORK_CHILD_MEASURE",
            ids=measured_input,
        )
        self.idle_barrier(case_id, "post_measure")
        target_after_measure = self.inspect(case_id, "target_post_measure", parent)
        post_measure_census = self.idle_barrier(case_id, "case_end")

        build_signature = {
            "parent_sha256": token_sha256(parent),
            "parent_output": parent_output,
            "internalizer_sha256": token_sha256(internalizer),
            "decoy_sha256": token_sha256(decoy),
            "measured_input_sha256": token_sha256(measured_input),
        }
        result = self.writer.emit(
            {
                "event": "case_result",
                "case_id": case_id,
                "phase": phase,
                "repetition": repetition,
                "case_order": case_order,
                "experiment": "FORK_PARENT",
                "target_boundary": "FORK_PARENT@32768",
                "target_tokens": PARENT_LEN,
                "fallback_tokens": 0,
                "condition": condition,
                "evicted_role": evicted_role,
                "content_seed_offset": seed_offset,
                "build_signature": build_signature,
                "build_signature_sha256": json_sha256(build_signature),
                "measured_request": measured,
                "pre_target": pre_target,
                "pre_decoy": pre_decoy,
                "control": control,
                "control_wall_ms": control_ms,
                "post_target": post_target,
                "post_decoy": post_decoy,
                "target_after_measure": target_after_measure,
                "pre_measure_census": pre_measure_census,
                "post_measure_census": post_measure_census,
            }
        )
        print(f"[FSWP3D] CASE_END id={case_id} status=complete", flush=True)
        return result

    def run_normal_case(
        self,
        *,
        phase: str,
        repetition: int,
        condition: str,
        case_order: int,
        seed_offset: int,
    ) -> dict:
        case_id = f"{phase}_r{repetition:02d}_o{case_order:02d}_normal_{condition}"
        self.writer.emit(
            {
                "event": "case_start",
                "case_id": case_id,
                "phase": phase,
                "repetition": repetition,
                "case_order": case_order,
                "experiment": "NORMAL_CONTROL",
                "condition": condition,
                "content_seed_offset": seed_offset,
            }
        )
        print(f"[FSWP3D] CASE_START id={case_id}", flush=True)
        self.flush(case_id)

        parent = toks(51_001 + seed_offset, PARENT_LEN)
        parent_event = self.generate_blocking(
            case_id=case_id,
            rid=f"{case_id}_parent",
            role="PARENT_BUILD",
            ids=parent,
        )
        parent_output = int(parent_event["output_ids"][0])

        forbidden_first = set()
        child_inputs = {}
        child_outputs = {}
        for index, branch in enumerate(("A", "B", "C", "D")):
            suffix = make_first_distinct(
                toks(80_001 + index * 20_003 + seed_offset, BRANCH_SUFFIX_LEN),
                forbidden_first,
            )
            child_input = parent + [parent_output] + suffix
            child_inputs[branch] = child_input
            event = self.generate_blocking(
                case_id=case_id,
                rid=f"{case_id}_child_{branch.lower()}",
                role=f"CHILD_{branch}_BUILD",
                ids=child_input,
            )
            child_outputs[branch] = int(event["output_ids"][0])

        # Make NORMAL A@32832 internal.  This is an out-of-workflow setup
        # request; the downstream measured workflow edge remains the same Join.
        continuation = toks(161_003 + seed_offset, BRANCH_SUFFIX_LEN)
        normal_a_descendant = (
            child_inputs["A"] + [child_outputs["A"]] + continuation
        )
        if len(normal_a_descendant) != 32_896:
            raise AssertionError(len(normal_a_descendant))
        self.generate_blocking(
            case_id=case_id,
            rid=f"{case_id}_normal_a_internalizer",
            role="NORMAL_A_INTERNALIZER",
            ids=normal_a_descendant,
        )

        decoy, _ = self.build_decoy(case_id, seed_offset, parent[0])

        join_payload = make_first_distinct(
            toks(180_001 + seed_offset, JOIN_SUFFIX_LEN), forbidden_first
        )
        embedding_positions = {"A": 31, "B": 95, "C": 159, "D": 223}
        for branch, position in embedding_positions.items():
            join_payload[position] = child_outputs[branch]
        join_input = parent + [parent_output] + join_payload
        if len(join_input) != 33_024:
            raise AssertionError(len(join_input))

        self.idle_barrier(case_id, "before_control")
        pre_target = self.inspect(case_id, "target_pre", child_inputs["A"])
        pre_decoy = self.inspect(case_id, "decoy_pre", decoy)
        pre_parent = self.inspect(case_id, "parent_pre", parent)

        if condition == "retained":
            evicted_role = "DECOY"
            control, control_ms = self.evict(case_id, "decoy", decoy)
        elif condition == "evicted":
            evicted_role = "TARGET"
            control, control_ms = self.evict(
                case_id, "normal_a", child_inputs["A"]
            )
        else:
            raise AssertionError(condition)

        post_target = self.inspect(
            case_id, "target_post_control", child_inputs["A"]
        )
        post_decoy = self.inspect(case_id, "decoy_post_control", decoy)
        post_parent = self.inspect(case_id, "parent_post_control", parent)
        pre_measure_census = self.idle_barrier(case_id, "pre_measure")

        measured = self.generate_measured(
            case_id=case_id,
            rid=f"{case_id}_measure",
            role="JOIN_MEASURE",
            ids=join_input,
        )
        self.idle_barrier(case_id, "post_measure")
        target_after_measure = self.inspect(
            case_id, "target_post_measure", child_inputs["A"]
        )
        parent_after_measure = self.inspect(
            case_id, "parent_post_measure", parent
        )
        post_measure_census = self.idle_barrier(case_id, "case_end")

        build_signature = {
            "parent_sha256": token_sha256(parent),
            "parent_output": parent_output,
            "child_input_sha256": {
                branch: token_sha256(ids) for branch, ids in child_inputs.items()
            },
            "child_outputs": child_outputs,
            "normal_a_descendant_sha256": token_sha256(normal_a_descendant),
            "decoy_sha256": token_sha256(decoy),
            "join_input_sha256": token_sha256(join_input),
            "join_embedding_positions": embedding_positions,
        }
        result = self.writer.emit(
            {
                "event": "case_result",
                "case_id": case_id,
                "phase": phase,
                "repetition": repetition,
                "case_order": case_order,
                "experiment": "NORMAL_CONTROL",
                "target_boundary": "NORMAL_A@32832",
                "target_tokens": 32_832,
                "fallback_tokens": PARENT_LEN,
                "condition": condition,
                "evicted_role": evicted_role,
                "content_seed_offset": seed_offset,
                "build_signature": build_signature,
                "build_signature_sha256": json_sha256(build_signature),
                "measured_request": measured,
                "join_embeds_outputs": child_outputs,
                "pre_target": pre_target,
                "pre_decoy": pre_decoy,
                "pre_parent": pre_parent,
                "control": control,
                "control_wall_ms": control_ms,
                "post_target": post_target,
                "post_decoy": post_decoy,
                "post_parent": post_parent,
                "target_after_measure": target_after_measure,
                "parent_after_measure": parent_after_measure,
                "pre_measure_census": pre_measure_census,
                "post_measure_census": post_measure_census,
            }
        )
        print(f"[FSWP3D] CASE_END id={case_id} status=complete", flush=True)
        return result

    def run_case(
        self,
        *,
        case_name: str,
        phase: str,
        repetition: int,
        case_order: int,
        seed_offset: int,
    ):
        experiment, condition = case_name.split("_", 1)
        if experiment == "fork":
            return self.run_fork_case(
                phase=phase,
                repetition=repetition,
                condition=condition,
                case_order=case_order,
                seed_offset=seed_offset,
            )
        if experiment == "normal":
            return self.run_normal_case(
                phase=phase,
                repetition=repetition,
                condition=condition,
                case_order=case_order,
                seed_offset=seed_offset,
            )
        raise AssertionError(case_name)


def wait_for_probe(client: ControlClient, timeout_s: float = 300.0):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            response = client.ping()
            if response.get("ok"):
                return response
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"control probe did not start: {last_error!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "formal"), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.events.parent.mkdir(parents=True, exist_ok=True)
    if args.events.exists() and args.events.stat().st_size:
        if not args.overwrite:
            raise RuntimeError(f"refusing to append to {args.events}")
        args.events.unlink()
    writer = EventWriter(args.events)
    writer.emit(
        {
            "event": "experiment_start",
            "stage": args.stage,
            "engine_config": ENGINE_CONFIG,
            "sampling_params": SAMPLING_PARAMS,
            "formal_order_manifest": FORMAL_ORDERS,
            "planned_formal_repetitions": 5,
            "concurrency": 1,
        }
    )

    engine = None
    status = "invalid"
    error = None
    try:
        engine = DirectEvictionEngine(**ENGINE_CONFIG)
        client = ControlClient(CONTROL_PORT)
        ping = wait_for_probe(client)
        writer.emit({"event": "probe_ready", "stage": args.stage, "response": ping})
        experiment = Experiment(engine, client, writer)

        if args.stage == "pilot":
            print("[FSWP3D] PILOT_WINDOW_START", flush=True)
            order = [
                "fork_retained",
                "fork_evicted",
                "normal_retained",
                "normal_evicted",
            ]
            for case_order, case_name in enumerate(order):
                experiment.run_case(
                    case_name=case_name,
                    phase="pilot",
                    repetition=0,
                    case_order=case_order,
                    seed_offset=900_000,
                )
            print("[FSWP3D] PILOT_WINDOW_END", flush=True)
        else:
            # Excluded multi-shape warmup.  It covers both 32K replay and the
            # short Join path using the exact intervention mechanics.
            print("[FSWP3D] WARMUP_WINDOW_START", flush=True)
            warm_order = [
                "fork_retained",
                "fork_evicted",
                "normal_retained",
                "normal_evicted",
            ]
            for case_order, case_name in enumerate(warm_order):
                experiment.run_case(
                    case_name=case_name,
                    phase="warmup",
                    repetition=-1,
                    case_order=case_order,
                    seed_offset=800_000,
                )
            print("[FSWP3D] WARMUP_WINDOW_END", flush=True)

            print("[FSWP3D] FORMAL_WINDOW_START", flush=True)
            for repetition, order in enumerate(FORMAL_ORDERS):
                seed_offset = repetition * 1_009
                for case_order, case_name in enumerate(order):
                    experiment.run_case(
                        case_name=case_name,
                        phase="formal",
                        repetition=repetition,
                        case_order=case_order,
                        seed_offset=seed_offset,
                    )
            print("[FSWP3D] FORMAL_WINDOW_END", flush=True)

        status = "complete"
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        writer.emit(
            {
                "event": "experiment_end",
                "stage": args.stage,
                "status": status,
                "error": error,
            }
        )
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                pass

    print(json.dumps({"stage": args.stage, "status": status, "events": str(args.events)}))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

