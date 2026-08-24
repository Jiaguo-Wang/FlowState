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
        nonce=f"wp3b_prompt_lru:inspect:{label}",
        label=f"wp3b_prompt_lru:{label}",
        action="inspect",
        token_ids=ids,
    )


def evict_mamba(client, label, ids):
    return client.checkpoint_control(
        nonce=f"wp3b_prompt_lru:evict:{label}",
        label=f"wp3b_prompt_lru:{label}",
        action="evict_mamba",
        token_ids=ids,
    )


def path_state(response):
    return response["after"]["path"]


def main():
    engine = None

    try:
        print("[WP3B-PROMPT-LRU] starting engine", flush=True)

        engine = ref.DirectEvictionEngine(**ref.ENGINE_CONFIG)
        client = ref.ControlClient(ref.CONTROL_PORT)

        ref.wait_for_probe(client)

        # ------------------------------------------------------------
        # 1. Start from a clean cache.
        # ------------------------------------------------------------
        engine.flush_cache()

        parents = {}
        parent_outputs = {}
        children = {}

        # ------------------------------------------------------------
        # 2. Build four 32K Parents.
        # ------------------------------------------------------------
        for name, seed in PARENT_SEEDS.items():
            parent = ref.toks(seed, ref.PARENT_LEN)
            parents[name] = parent

            rid = f"wp3b_prompt_lru_{name.lower()}_parent"

            result = engine.generate(
                input_ids=parent,
                sampling_params=ref.SAMPLING_PARAMS,
                rid=rid,
            )

            output_ids, _ = ref.validate_generation(result, rid)
            parent_outputs[name] = int(output_ids[0])

            print(
                f"[WP3B-PROMPT-LRU] {name}_PARENT_BUILT",
                flush=True,
            )

        # ------------------------------------------------------------
        # 3. Build Child-A for every workflow.
        # ------------------------------------------------------------
        for name in ("P1", "P2", "P3", "P4"):
            suffix = ref.toks(
                CHILD_SEEDS[name],
                ref.BRANCH_SUFFIX_LEN,
            )

            if suffix[0] == parent_outputs[name]:
                suffix[0] = (suffix[0] + 1) % ref.VOCAB_SIZE

            child = (
                parents[name]
                + [parent_outputs[name]]
                + suffix
            )

            assert len(child) == 32_832
            children[name] = child

            rid = f"wp3b_prompt_lru_{name.lower()}_child_a"

            result = engine.generate(
                input_ids=child,
                sampling_params=ref.SAMPLING_PARAMS,
                rid=rid,
            )

            ref.validate_generation(result, rid)

            print(
                f"[WP3B-PROMPT-LRU] {name}_CHILD_A_BUILT",
                flush=True,
            )

        # ------------------------------------------------------------
        # 4. Hard pre-policy gate: exactly eight Mamba CPs.
        # ------------------------------------------------------------
        census_pre = client.census(
            "wp3b_prompt_lru:census:pre_policy"
        )

        pre_count = census_pre["tree"]["mamba_node_count"]

        print(
            f"[WP3B-PROMPT-LRU] PRE_POLICY_MAMBA_COUNT={pre_count}",
            flush=True,
        )

        if pre_count != 8:
            raise RuntimeError(
                f"expected 8 Mamba checkpoints before policy, got {pre_count}"
            )

        # Verify every Parent and Child currently has both states.
        for name in ("P1", "P2", "P3", "P4"):
            p = path_state(
                inspect(
                    client,
                    f"{name.lower()}_parent_pre",
                    parents[name],
                )
            )

            c = path_state(
                inspect(
                    client,
                    f"{name.lower()}_child_pre",
                    children[name],
                )
            )

            if not p["target_mamba_present"]:
                raise RuntimeError(f"{name} Parent Mamba missing before policy")

            if not c["target_mamba_present"]:
                raise RuntimeError(f"{name} Child Mamba missing before policy")

            if not p["target_full_present"] or not c["target_full_present"]:
                raise RuntimeError(f"{name}: FA-KV missing before policy")

        # ------------------------------------------------------------
        # 5. Prompt-LRU K=4
        #
        # Request order:
        #   P1 P2 P3 P4 C1 C2 C3 C4
        #
        # Latest four request-end checkpoints are therefore:
        #   C1 C2 C3 C4
        #
        # Evict the four oldest checkpoints:
        #   P1 P2 P3 P4
        # ------------------------------------------------------------
        print(
            "[WP3B-PROMPT-LRU] applying K=4: evict P1/P2/P3/P4 Parents",
            flush=True,
        )

        for name in ("P1", "P2", "P3", "P4"):
            result = evict_mamba(
                client,
                f"{name.lower()}_parent",
                parents[name],
            )

            print(
                f"[WP3B-PROMPT-LRU] EVICT_{name}_PARENT "
                f"ok={result.get('ok')}",
                flush=True,
            )

        # ------------------------------------------------------------
        # 6. Hard post-policy gate.
        # ------------------------------------------------------------
        for name in ("P1", "P2", "P3", "P4"):
            p_response = inspect(
                client,
                f"{name.lower()}_parent_post",
                parents[name],
            )

            c_response = inspect(
                client,
                f"{name.lower()}_child_post",
                children[name],
            )

            p = path_state(p_response)
            c = path_state(c_response)

            print(
                f"[WP3B-PROMPT-LRU] {name}_POST "
                f"parent_mamba={p['target_mamba_present']} "
                f"parent_fa={p['target_full_present']} "
                f"child_mamba={c['target_mamba_present']} "
                f"child_fa={c['target_full_present']}",
                flush=True,
            )

            # Parent recurrent checkpoint must be gone.
            if p["target_mamba_present"]:
                raise RuntimeError(
                    f"{name}: Parent Mamba still present after Prompt-LRU"
                )

            # But Parent FA-KV must remain.
            if not p["target_full_present"]:
                raise RuntimeError(
                    f"{name}: Parent FA-KV was accidentally evicted"
                )

            # Child recurrent checkpoint must remain.
            if not c["target_mamba_present"]:
                raise RuntimeError(
                    f"{name}: Child Mamba unexpectedly missing"
                )

            if not c["target_full_present"]:
                raise RuntimeError(
                    f"{name}: Child FA-KV unexpectedly missing"
                )

        census_post = client.census(
            "wp3b_prompt_lru:census:post_policy"
        )

        post_count = census_post["tree"]["mamba_node_count"]

        print(
            f"[WP3B-PROMPT-LRU] POST_POLICY_MAMBA_COUNT={post_count}",
            flush=True,
        )

        print(
            "[WP3B-PROMPT-LRU] MAMBA_ROWS="
            + json.dumps(
                census_post["tree"]["mamba_rows"],
                sort_keys=True,
            ),
            flush=True,
        )

        if post_count != 4:
            raise RuntimeError(
                f"expected 4 Mamba checkpoints after policy, got {post_count}"
            )

        print(
            "[WP3B-PROMPT-LRU] STATUS=complete",
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
