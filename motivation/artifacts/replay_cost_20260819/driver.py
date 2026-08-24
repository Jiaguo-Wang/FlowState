"""FlowState WP2 replay-cost runtime driver.

This reuses the validated gap-replay recipe from
runtime_validation_gap_replay_20260819.  Every measured request is preceded by
an idle cache reset and a freshly keyed setup, because the measured replay
self-heals the recurrent checkpoint.

Gate:
    python3 driver.py --mode gate
Sweep:
    python3 driver.py --mode sweep --repetitions 5
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


VOCAB = 248_320
TARGET_LEN = 32_768
TAIL_LEN = 2_048
BASE32_LEN = 16_384
N_FILLERS = 24
GROUPS = (0, 1_024, 4_096, 8_192, 16_384, 32_768)


def toks(seed: int, n: int) -> list[int]:
    """Deterministic token IDs; case seeds make every cache key fresh."""
    return [(seed + i * 7_919) % VOCAB for i in range(n)]


class Driver:
    def __init__(self, base_url: str, event_path: Path) -> None:
        self.base_url = base_url.rstrip("/")
        self.event_path = event_path
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        # The host has HTTP proxy variables; localhost experiment traffic must
        # never leave the machine.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def emit(self, event: dict) -> None:
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def flush(self, case_id: str) -> None:
        req = urllib.request.Request(
            f"{self.base_url}/flush_cache?timeout=60", data=b"", method="POST"
        )
        t0 = time.perf_counter()
        try:
            with self.opener.open(req, timeout=60) as response:
                body = response.read().decode("utf-8", errors="replace")
                status = response.status
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"flush failed ({exc.code}): {body}") from exc
        wall_ms = (time.perf_counter() - t0) * 1_000
        if status != 200 or "Cache flushed" not in body:
            raise RuntimeError(f"flush failed ({status}): {body}")
        self.emit(
            {
                "event": "flush",
                "case_id": case_id,
                "status": status,
                "wall_ms": wall_ms,
                "body": body.strip(),
            }
        )

    def generate(
        self,
        *,
        rid: str,
        ids: list[int],
        role: str,
        phase: str,
        expected_replay_tokens: int,
        repetition: int,
    ) -> dict:
        payload = {
            "input_ids": ids,
            "extra_key": f"{phase}-e{expected_replay_tokens}-r{repetition}",
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0,
                "ignore_eos": True,
            },
            "rid": rid,
        }
        request = urllib.request.Request(
            f"{self.base_url}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with self.opener.open(request, timeout=900) as response:
            out = json.loads(response.read())
        wall_ms = (time.perf_counter() - t0) * 1_000
        meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
        event = {
            "event": "generate",
            "phase": phase,
            "role": role,
            "rid": rid,
            "expected_replay_tokens": expected_replay_tokens,
            "repetition": repetition,
            "prompt_tokens": len(ids),
            "client_wall_ms": wall_ms,
            "output_ids": out.get("output_ids", []) if isinstance(out, dict) else [],
            "meta": meta,
        }
        self.emit(event)
        print(
            f"{rid}: role={role} prompt={len(ids)} wall={wall_ms:.1f}ms "
            f"ttft={float(meta.get('first_token_latency', float('nan'))) * 1000:.1f}ms",
            flush=True,
        )
        if meta.get("num_retractions", 0) != 0:
            raise RuntimeError(f"{rid}: unexpected retractions: {meta}")
        if role == "measure" and "first_token_latency" not in meta:
            raise RuntimeError(f"{rid}: first_token_latency missing from response")
        return event

    def run_case(self, expected: int, repetition: int, phase: str) -> None:
        if expected not in GROUPS:
            raise ValueError(f"unsupported replay target: {expected}")
        case_id = f"{phase}_e{expected}_r{repetition}"
        # Keep token IDs comfortably inside the model vocabulary and make the
        # entire 32K target unique per run, so no self-healed state can leak in.
        # A repetition is a randomized block: all six groups use the same
        # target content so token identity / routing cannot confound gap size.
        case_seed = 10_000 + repetition * 31_337
        target = toks(case_seed, TARGET_LEN)
        target_tail = toks(case_seed + 700_001, TAIL_LEN)
        # The measured request is a cached 32K prefix followed by one new
        # token.  Radix matching is capped at prompt_len - 1, so this yields an
        # exact 32768-token physical hit and avoids the misleading 32767/32704
        # end-of-prompt alignment tail of the earlier validation request.
        occupied_next_tokens = {(case_seed + 700_001) % VOCAB}
        occupied_next_tokens.update(
            (case_seed + 1_100_001 + index * 10_007) % VOCAB
            for index in range(N_FILLERS)
        )
        query_token = (case_seed + 2_000_003) % VOCAB
        while query_token in occupied_next_tokens:
            query_token = (query_token + 1) % VOCAB
        measured_prompt = target + [query_token]
        self.flush(case_id)

        common = dict(
            phase=phase,
            expected_replay_tokens=expected,
            repetition=repetition,
        )

        if expected == TARGET_LEN:
            # No target-path shallow checkpoint.  A separate 16K filler base
            # keeps KV usage modest while its 24 branches evict target-path
            # recurrent checkpoints only; component eviction retains FA-KV.
            target_build = self.generate(
                rid=f"{case_id}_R1", ids=target, role="target_build", **common
            )
            if target_build["output_ids"][:1] == target_tail[:1]:
                raise RuntimeError(f"{case_id}: generated token collided with EXT branch")
            self.generate(
                rid=f"{case_id}_EXT",
                ids=target + target_tail,
                role="target_internalize",
                **common,
            )
            filler_base = toks(case_seed + 900_001, BASE32_LEN)
            self.generate(
                rid=f"{case_id}_FB",
                ids=filler_base,
                role="filler_base",
                **common,
            )
            for index in range(N_FILLERS):
                self.generate(
                    rid=f"{case_id}_F{index:02d}",
                    ids=filler_base
                    + toks(case_seed + 1_100_001 + index * 10_007, TAIL_LEN),
                    role="pressure",
                    **common,
                )
        else:
            keep_len = TARGET_LEN - expected if expected else TARGET_LEN
            keep = target[:keep_len]
            # S1 is the recurrent checkpoint that must survive.  For the 0K
            # baseline, it is the exact 32K target-prefix checkpoint.
            checkpoint_keep = self.generate(
                rid=f"{case_id}_S1", ids=keep, role="checkpoint_keep", **common
            )
            intended_next = target[keep_len] if keep_len < TARGET_LEN else target_tail[0]
            if checkpoint_keep["output_ids"][:1] == [intended_next]:
                raise RuntimeError(f"{case_id}: generated token collided at keep boundary")
            if keep_len < TARGET_LEN:
                target_build = self.generate(
                    rid=f"{case_id}_R1", ids=target, role="target_build", **common
                )
                if target_build["output_ids"][:1] == target_tail[:1]:
                    raise RuntimeError(
                        f"{case_id}: generated token collided with EXT branch"
                    )
            self.generate(
                rid=f"{case_id}_EXT",
                ids=target + target_tail,
                role="target_internalize",
                **common,
            )
            for index in range(N_FILLERS):
                self.generate(
                    rid=f"{case_id}_F{index:02d}",
                    ids=keep
                    + toks(case_seed + 1_100_001 + index * 10_007, TAIL_LEN),
                    role="pressure",
                    **common,
                )

        self.generate(
            rid=f"{case_id}_MEASURE",
            ids=measured_prompt,
            role="measure",
            **common,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate", "sweep"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:49932")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--events",
        type=Path,
        default=Path(__file__).with_name("driver_events.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    driver = Driver(args.base_url, args.events)
    if args.mode == "gate":
        driver.run_case(expected=8_192, repetition=0, phase="gate")
        return

    # Five cyclic rotations of one randomized base order form near-balanced
    # blocks: every replay group occupies five different within-block slots.
    rng = random.Random(20_260_819)
    base_order = list(GROUPS)
    rng.shuffle(base_order)
    for repetition in range(args.repetitions):
        offset = repetition % len(base_order)
        order = base_order[offset:] + base_order[:offset]
        print(f"sweep repetition {repetition}: {order}", flush=True)
        for expected in order:
            driver.run_case(expected=expected, repetition=repetition, phase="sweep")


if __name__ == "__main__":
    main()
