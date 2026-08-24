#!/usr/bin/env python3

import json
import traceback

import driver_eviction_ref as ref


PARENT_SEEDS = {
    "P1": 51_001,
    "P2": 91_003,
    "P3": 131_009,
    "P4": 171_017,
}


def inspect(client, label, ids):
    return client.checkpoint_control(
        nonce=f"wp3b_state_gate:inspect:{label}",
        label=f"wp3b_state_gate:{label}",
        action="inspect",
        token_ids=ids,
    )


def main():
    engine = None

    try:
        print("[WP3B-GATE] starting engine", flush=True)

        engine = ref.DirectEvictionEngine(**ref.ENGINE_CONFIG)

        client = ref.ControlClient(ref.CONTROL_PORT)
        probe = ref.wait_for_probe(client)

        print(
            "[WP3B-GATE] probe ready:",
            json.dumps(probe, sort_keys=True),
            flush=True,
        )

        # ------------------------------------------------------------
        # 1. Flush everything
        # ------------------------------------------------------------
        print("[WP3B-GATE] flushing cache", flush=True)

        engine.flush_cache()

        census_after_flush = client.census(
            "wp3b_state_gate:census:after_flush"
        )

        print(
            "[WP3B-GATE] census_after_flush:",
            json.dumps(census_after_flush, sort_keys=True),
            flush=True,
        )

        # ------------------------------------------------------------
        # 2. Build four independent 32K parents
        # ------------------------------------------------------------
        parents = {}
        parent_outputs = {}

        first_tokens = set()

        for name, seed in PARENT_SEEDS.items():
            parent = ref.toks(seed, ref.PARENT_LEN)

            # Simple protection against accidental common first token.
            if parent[0] in first_tokens:
                raise RuntimeError(
                    f"{name}: duplicate first token {parent[0]}"
                )

            first_tokens.add(parent[0])
            parents[name] = parent

            rid = f"wp3b_state_gate_{name.lower()}_parent"

            print(
                f"[WP3B-GATE] building {name}: "
                f"len={len(parent)} first_token={parent[0]}",
                flush=True,
            )

            result = engine.generate(
                input_ids=parent,
                sampling_params=ref.SAMPLING_PARAMS,
                rid=rid,
            )

            output_ids, meta = ref.validate_generation(result, rid)

            parent_outputs[name] = int(output_ids[0])

            print(
                f"[WP3B-GATE] {name} built: "
                f"output={parent_outputs[name]} "
                f"cached={meta.get('cached_tokens')}",
                flush=True,
            )

            # Inspect immediately after creation.
            state = inspect(
                client,
                f"{name.lower()}_after_build",
                parent,
            )

            print(
                f"[WP3B-GATE] {name}_after_build:",
                json.dumps(state, sort_keys=True),
                flush=True,
            )

        # ------------------------------------------------------------
        # 3. Inspect all four again after all Parents exist.
        # ------------------------------------------------------------
        print(
            "[WP3B-GATE] re-inspecting all parents after four builds",
            flush=True,
        )

        final_inspects = {}

        for name, parent in parents.items():
            state = inspect(
                client,
                f"{name.lower()}_final",
                parent,
            )

            final_inspects[name] = state

            print(
                f"[WP3B-GATE] {name}_FINAL:",
                json.dumps(state, sort_keys=True),
                flush=True,
            )

        # ------------------------------------------------------------
        # 4. Global cache census
        # ------------------------------------------------------------
        census_final = client.census(
            "wp3b_state_gate:census:final"
        )

        print(
            "[WP3B-GATE] census_final:",
            json.dumps(census_final, sort_keys=True),
            flush=True,
        )

        summary = {
            "status": "complete",
            "parent_len": ref.PARENT_LEN,
            "parent_first_tokens": {
                name: parents[name][0]
                for name in parents
            },
            "parent_outputs": parent_outputs,
            "final_inspects": final_inspects,
            "census_final": census_final,
        }

        print(
            "[WP3B-GATE] SUMMARY="
            + json.dumps(summary, sort_keys=True),
            flush=True,
        )

        return 0

    except Exception:
        traceback.print_exc()
        return 1

    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
