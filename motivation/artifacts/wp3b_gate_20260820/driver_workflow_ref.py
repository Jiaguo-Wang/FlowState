"""Run the single controlled workflow used by FlowState Motivation WP3A.

The workload intentionally uses exact token IDs.  This keeps every boundary
position deterministic while still making JOIN depend on all four child
outputs and RESUME depend on the JOIN output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from array import array
from pathlib import Path


VOCAB_SIZE = 248_320
PARENT_LEN = 32_768
BRANCH_SUFFIX_LEN = 63
JOIN_SUFFIX_LEN = 255
RESUME_SUFFIX_LEN = 63
WORKFLOW_ID = "wp3a_controlled_w0"
EXTRA_KEY = "wp3a-control-20260819-w0"


def toks(seed: int, count: int) -> list[int]:
    """Return a deterministic, collision-resistant token sequence."""
    return [(seed + index * 7_919) % VOCAB_SIZE for index in range(count)]


def token_digest(ids: list[int]) -> str:
    return hashlib.sha256(array("I", ids).tobytes()).hexdigest()


def distinct_tail(seed: int, count: int, forbidden_first: set[int]) -> list[int]:
    ids = toks(seed, count)
    while ids[0] in forbidden_first:
        ids[0] = (ids[0] + 1) % VOCAB_SIZE
    forbidden_first.add(ids[0])
    return ids


class WorkflowDriver:
    def __init__(self, base_url: str, event_path: Path) -> None:
        self.base_url = base_url.rstrip("/")
        self.event_path = event_path
        self.sequence = 0
        self.request_ids: list[str] = []
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def emit(self, event: dict) -> None:
        self.sequence += 1
        event = {
            "schema_version": "flowstate.wp3a.trace.v1",
            "sequence": self.sequence,
            "workflow_id": WORKFLOW_ID,
            **event,
        }
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def flush(self) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/flush_cache?timeout=60", data=b"", method="POST"
        )
        started = time.perf_counter()
        try:
            with self.opener.open(request, timeout=60) as response:
                body = response.read().decode("utf-8", errors="replace")
                status = response.status
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"flush failed ({exc.code}): {body}") from exc
        wall_ms = (time.perf_counter() - started) * 1_000
        if status != 200 or "Cache flushed" not in body:
            raise RuntimeError(f"flush failed ({status}): {body}")
        self.emit(
            {
                "event": "cache_flush",
                "http_status": status,
                "client_wall_ms": wall_ms,
                "response": body.strip(),
            }
        )

    def generate(
        self,
        *,
        rid: str,
        role: str,
        ids: list[int],
        dependencies: list[str],
        branch_id: str | None = None,
    ) -> dict:
        payload = {
            "input_ids": ids,
            "extra_key": EXTRA_KEY,
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
        started = time.perf_counter()
        with self.opener.open(request, timeout=900) as response:
            result = json.loads(response.read())
        wall_ms = (time.perf_counter() - started) * 1_000
        if not isinstance(result, dict):
            raise RuntimeError(f"{rid}: response is not an object")
        output_ids = result.get("output_ids", [])
        meta = result.get("meta_info", {})
        if len(output_ids) != 1:
            raise RuntimeError(f"{rid}: expected exactly one output token: {output_ids}")
        if meta.get("completion_tokens") != 1:
            raise RuntimeError(f"{rid}: completion token count is not one: {meta}")
        if meta.get("num_retractions", 0) != 0:
            raise RuntimeError(f"{rid}: unexpected retraction: {meta}")
        if meta.get("finish_reason") != {"type": "length", "length": 1}:
            raise RuntimeError(f"{rid}: unexpected finish reason: {meta}")
        event = {
            "event": "request_completed",
            "rid": rid,
            "role": role,
            "branch_id": branch_id,
            "declared_dependencies": dependencies,
            "prompt_tokens": len(ids),
            "input_sha256": token_digest(ids),
            "output_ids": output_ids,
            "client_wall_ms": wall_ms,
            "meta": meta,
        }
        self.emit(event)
        self.request_ids.append(rid)
        print(
            f"{rid}: role={role} prompt={len(ids)} "
            f"cached={meta.get('cached_tokens')} wall={wall_ms:.1f}ms",
            flush=True,
        )
        return event

    def boundary(
        self,
        *,
        boundary_id: str,
        boundary_type: str,
        token_position: int,
        source_rid: str,
        semantic_roles: list[str],
        notes: str,
    ) -> None:
        self.emit(
            {
                "event": "boundary_declared",
                "boundary_id": boundary_id,
                "boundary_type": boundary_type,
                "token_position": token_position,
                "source_rid": source_rid,
                "semantic_roles": semantic_roles,
                "notes": notes,
            }
        )

    def run(self) -> None:
        self.emit(
            {
                "event": "workflow_start",
                "model": "Qwen3.5-9B",
                "runtime": "SGLang v0.5.17",
                "parent_tokens": PARENT_LEN,
                "branch_width": 4,
                "branch_depth": 1,
                "concurrency": 1,
                "max_new_tokens": 1,
                "extra_key": EXTRA_KEY,
            }
        )
        self.flush()

        parent = toks(51_001, PARENT_LEN)
        parent_event = self.generate(
            rid="wp3a_parent",
            role="PARENT",
            ids=parent,
            dependencies=[],
        )
        parent_output = int(parent_event["output_ids"][0])
        self.boundary(
            boundary_id="b_fork_parent",
            boundary_type="FORK_PARENT",
            token_position=PARENT_LEN,
            source_rid="wp3a_parent",
            semantic_roles=["FORK_PARENT", "PENDING_RESUME"],
            notes=(
                "The 32K parent prompt endpoint. All four children and JOIN "
                "branch directly from this recurrent checkpoint."
            ),
        )

        # The sampled parent token is part of every downstream history.  The
        # 63-token child suffix makes each ordinary endpoint exactly 32832.
        forbidden_first: set[int] = set()
        child_outputs: dict[str, int] = {}
        child_first_tokens: dict[str, int] = {}
        for index, branch_id in enumerate(("A", "B", "C", "D")):
            suffix = distinct_tail(
                80_001 + index * 20_003, BRANCH_SUFFIX_LEN, forbidden_first
            )
            child_first_tokens[branch_id] = suffix[0]
            event = self.generate(
                rid=f"wp3a_child_{branch_id.lower()}",
                role="FORK_CHILD",
                branch_id=branch_id,
                ids=parent + [parent_output] + suffix,
                dependencies=["b_fork_parent"],
            )
            child_outputs[branch_id] = int(event["output_ids"][0])
            self.boundary(
                boundary_id=f"b_child_{branch_id.lower()}_normal",
                boundary_type="NORMAL",
                token_position=PARENT_LEN + 1 + BRANCH_SUFFIX_LEN,
                source_rid=f"wp3a_child_{branch_id.lower()}",
                semantic_roles=["ORDINARY_PROMPT_END"],
                notes=(
                    f"Ordinary endpoint of child {branch_id}; its output is "
                    "consumed as JOIN data, but its recurrent state is not on "
                    "the JOIN prefix path."
                ),
            )

        join_payload = distinct_tail(180_001, JOIN_SUFFIX_LEN, forbidden_first)
        embedding_positions = {"A": 31, "B": 95, "C": 159, "D": 223}
        for branch_id, position in embedding_positions.items():
            join_payload[position] = child_outputs[branch_id]
        self.emit(
            {
                "event": "join_inputs_bound",
                "child_output_ids": child_outputs,
                "join_payload_positions": embedding_positions,
                "join_payload_sha256": token_digest(join_payload),
                "branch_first_tokens": child_first_tokens,
                "join_first_token": join_payload[0],
            }
        )
        join_prompt = parent + [parent_output] + join_payload
        join_event = self.generate(
            rid="wp3a_join",
            role="JOIN",
            ids=join_prompt,
            dependencies=[
                "b_fork_parent",
                "b_child_a_normal",
                "b_child_b_normal",
                "b_child_c_normal",
                "b_child_d_normal",
            ],
        )
        join_output = int(join_event["output_ids"][0])
        self.boundary(
            boundary_id="b_join",
            boundary_type="JOIN",
            token_position=PARENT_LEN + 1 + JOIN_SUFFIX_LEN,
            source_rid="wp3a_join",
            semantic_roles=["JOIN", "PENDING_RESUME"],
            notes=(
                "JOIN prompt embeds all four child outputs; the next request "
                "resumes directly from this checkpoint."
            ),
        )

        resume_suffix = toks(230_001, RESUME_SUFFIX_LEN)
        resume_prompt = join_prompt + [join_output] + resume_suffix
        self.emit(
            {
                "event": "resume_input_bound",
                "join_boundary_id": "b_join",
                "join_output_id": join_output,
                "join_output_position": len(join_prompt),
                "resume_suffix_tokens": RESUME_SUFFIX_LEN,
            }
        )
        self.generate(
            rid="wp3a_resume",
            role="PARENT_RESUME",
            ids=resume_prompt,
            dependencies=["b_join"],
        )
        self.boundary(
            boundary_id="b_resume",
            boundary_type="RESUME",
            token_position=len(resume_prompt),
            source_rid="wp3a_resume",
            semantic_roles=["RESUME", "TERMINAL_IN_OBSERVATION_HORIZON"],
            notes="Parent resume endpoint; no later consumer exists in this workflow.",
        )
        self.emit(
            {
                "event": "workflow_end",
                "status": "driver_complete",
                "request_ids": self.request_ids,
                "declared_boundary_count": 7,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:49933")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path(__file__).with_name("driver_events.jsonl"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing driver event file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.events.parent.mkdir(parents=True, exist_ok=True)
    if args.events.exists() and args.events.stat().st_size:
        if not args.overwrite:
            raise RuntimeError(f"refusing to append to existing {args.events}")
        args.events.unlink()
    WorkflowDriver(args.base_url, args.events).run()


if __name__ == "__main__":
    main()
