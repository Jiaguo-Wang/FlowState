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

CHILD_SEEDS = {
    "P1": 201_001,
    "P2": 211_003,
    "P3": 221_009,
    "P4": 231_017,
}


def inspect(client, label, ids):
    return client.checkpoint_control(
        nonce=f"wp3b_child_gate:inspect:{label}",
        label=f"wp3b_child_gate:{label}",
        action="inspect",
        token_ids=ids,
    )


def main():
    engine = None

    try:
        print("[WP3B-CHILD-GATE] starting engine", flush=True)

        engine = ref.DirectEvictionEngine(**ref.ENGINE_CONFIG)
        client = ref.ControlClient(ref.CONTROL_PORT)

        probe = ref.wait_for_probe(client)
        print(
            "[WP3B-CHILD-GATE] probe ready:",
            json.dumps(probe, sort_keys=True),
            flush=True,
        )

        # ------------------------------------------------------------
        # 1. Flush
        # ------------------------------------------------------------
        engine.flush_cache()

        census = client.census("wp3b_child_gate:census:after_flush")

        print(
            "[WP3B-CHILD-GATE] census_after_flush:",
            json.dumps(census, sort_keys=True),
            flush=True,
        )

        parents = {}
        parent_outputs = {}
        children = {}

        # ------------------------------------------------------------
        # 2. Build P1/P2/P3/P4 Parent@32768
        # ------------------------------------------------------------
        for name, seed in PARENT_SEEDS.items():
            parent = ref.toks(seed, ref.PARENT_LEN)
            parents[name] = parent

            rid = f"wp3b_child_gate_{name.lower()}_parent"

            print(
                f"[WP3B-CHILD-GATE] building {name} Parent",
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
                f"[WP3B-CHILD-GATE] {name}_PARENT_BUILT "
                f"cached={meta.get('cached_tokens')} "
                f"output={parent_outputs[name]}",
                flush=True,
            )

        # Verify all four parents exist before children.
        print(
            "[WP3B-CHILD-GATE] checking four Parents before Child-A",
            flush=True,
        )

        for name, parent in parents.items():
            state = inspect(client, f"{name.lower()}_parent_pre_child", parent)

            print(
                f"[WP3B-CHILD-GATE] {name}_PARENT_PRE_CHILD:",
                json.dumps(state, sort_keys=True),
                flush=True,
            )

        # ------------------------------------------------------------
        # 3. Build one Child-A for each workflow
        #
        # Child endpoint:
        #   32768 Parent
        # +     1 Parent output
        # +    63 branch suffix
        # = 32832 tokens
        # ------------------------------------------------------------
        for name in ("P1", "P2", "P3", "P4"):
            suffix = ref.toks(
                CHILD_SEEDS[name],
                ref.BRANCH_SUFFIX_LEN,
            )

            # Do not let the first fresh branch token accidentally equal
            # the sampled Parent output.
            if suffix[0] == parent_outputs[name]:
                suffix[0] = (suffix[0] + 1) % ref.VOCAB_SIZE

            child = (
                parents[name]
                + [parent_outputs[name]]
                + suffix
            )

            if len(child) != 32_832:
                raise RuntimeError(
                    f"{name}: unexpected Child-A length {len(child)}"
                )

            children[name] = child

            rid = f"wp3b_child_gate_{name.lower()}_child_a"

            print(
                f"[WP3B-CHILD-GATE] building {name} Child-A "
                f"len={len(child)}",
                flush=True,
            )

            result = engine.generate(
                input_ids=child,
                sampling_params=ref.SAMPLING_PARAMS,
                rid=rid,
            )

            output_ids, meta = ref.validate_generation(result, rid)

            print(
                f"[WP3B-CHILD-GATE] {name}_CHILD_A_BUILT "
                f"cached={meta.get('cached_tokens')} "
                f"output={int(output_ids[0])}",
                flush=True,
            )

            child_state = inspect(
                client,
                f"{name.lower()}_child_a_after_build",
                child,
            )

            print(
                f"[WP3B-CHILD-GATE] {name}_CHILD_A_AFTER_BUILD:",
                json.dumps(child_state, sort_keys=True),
                flush=True,
            )

        # ------------------------------------------------------------
        # 4. Re-inspect all Parents after all Child-A requests.
        # ------------------------------------------------------------
        print(
            "[WP3B-CHILD-GATE] final Parent/Child inspection",
            flush=True,
        )

        for name in ("P1", "P2", "P3", "P4"):
            parent_state = inspect(
                client,
                f"{name.lower()}_parent_final",
                parents[name],
            )

            child_state = inspect(
                client,
                f"{name.lower()}_child_a_final",
                children[name],
            )

            print(
                f"[WP3B-CHILD-GATE] {name}_PARENT_FINAL:",
                json.dumps(parent_state, sort_keys=True),
                flush=True,
            )

            print(
                f"[WP3B-CHILD-GATE] {name}_CHILD_A_FINAL:",
                json.dumps(child_state, sort_keys=True),
                flush=True,
            )

        census_final = client.census(
            "wp3b_child_gate:census:final"
        )

        print(
            "[WP3B-CHILD-GATE] census_final:",
            json.dumps(census_final, sort_keys=True),
            flush=True,
        )

        print(
            "[WP3B-CHILD-GATE] STATUS=complete",
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
