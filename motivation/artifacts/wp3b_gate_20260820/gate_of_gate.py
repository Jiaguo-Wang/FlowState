#!/usr/bin/env python3

import argparse
import json
import traceback

import driver_eviction_ref as ref


PARENT_SEEDS = {
    "P1": 51_001,
    "P2": 91_003,
    "P3": 131_009,
    "P4": 171_017,
}

CHILD_A_SEEDS = {
    "P1": 201_001,
    "P2": 211_003,
    "P3": 221_009,
    "P4": 231_017,
}

CHILD_B_SEED = 241_019


def inspect(client, policy, label, ids):
    return client.checkpoint_control(
        nonce=f"wp3b_gog:{policy}:inspect:{label}",
        label=f"wp3b_gog:{policy}:{label}",
        action="inspect",
        token_ids=ids,
    )


def evict(client, policy, label, ids, action):
    return client.checkpoint_control(
        nonce=f"wp3b_gog:{policy}:evict:{label}",
        label=f"wp3b_gog:{policy}:{label}",
        action=action,
        token_ids=ids,
    )


def generate(engine, rid, ids):
    result = engine.generate(
        input_ids=ids,
        sampling_params=ref.SAMPLING_PARAMS,
        rid=rid,
    )
    output_ids, _ = ref.validate_generation(result, rid)
    return output_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        required=True,
        choices=("prompt_lru", "workflow_k"),
    )
    args = parser.parse_args()
    policy = args.policy

    engine = None

    try:
        print(
            f"[WP3B-GOG] POLICY={policy}",
            flush=True,
        )

        engine = ref.DirectEvictionEngine(**ref.ENGINE_CONFIG)
        client = ref.ControlClient(ref.CONTROL_PORT)
        ref.wait_for_probe(client)

        engine.flush_cache()

        parents = {}
        parent_outputs = {}
        child_as = {}

        # ------------------------------------------------------------
        # 1. Build four disjoint 32K Parents
        # ------------------------------------------------------------
        for name in ("P1", "P2", "P3", "P4"):
            parent = ref.toks(
                PARENT_SEEDS[name],
                ref.PARENT_LEN,
            )
            parents[name] = parent

            rid = f"wp3b_gog_{policy}_{name.lower()}_parent"

            output_ids = generate(
                engine,
                rid,
                parent,
            )

            parent_outputs[name] = int(output_ids[0])

            print(
                f"[WP3B-GOG] {name}_PARENT_BUILT",
                flush=True,
            )

        # ------------------------------------------------------------
        # 2. Build one Child-A for every Parent.
        #
        # Input:
        # Parent@32768 + parent_output + 63-token suffix
        # = 32832 tokens
        # ------------------------------------------------------------
        for name in ("P1", "P2", "P3", "P4"):
            suffix = ref.toks(
                CHILD_A_SEEDS[name],
                ref.BRANCH_SUFFIX_LEN,
            )

            child_a = (
                parents[name]
                + [parent_outputs[name]]
                + suffix
            )

            assert len(child_a) == 32_832
            child_as[name] = child_a

            rid = f"wp3b_gog_{policy}_{name.lower()}_child_a"

            generate(
                engine,
                rid,
                child_a,
            )

            print(
                f"[WP3B-GOG] {name}_CHILD_A_BUILT",
                flush=True,
            )

        census_pre = client.census(
            f"wp3b_gog:{policy}:census:pre_policy"
        )

        pre_count = census_pre["tree"]["mamba_node_count"]

        print(
            f"[WP3B-GOG] PRE_POLICY_MAMBA_COUNT={pre_count}",
            flush=True,
        )

        if pre_count != 8:
            raise RuntimeError(
                f"expected 8 Mamba checkpoints before policy, got {pre_count}"
            )

        # ------------------------------------------------------------
        # 3. Enforce equal K=4 recurrent-state budget.
        #
        # Prompt-LRU:
        #   recent Child-A CPs win -> evict Parents.
        #
        # Workflow-K:
        #   pending Fork Parents win -> evict Child-A CPs.
        # ------------------------------------------------------------
        if policy == "prompt_lru":
            for name in ("P1", "P2", "P3", "P4"):
                result = evict(
                    client,
                    policy,
                    f"{name.lower()}_parent",
                    parents[name],
                    action="evict_mamba",
                )

                print(
                    f"[WP3B-GOG] EVICT_{name}_PARENT "
                    f"ok={result['ok']}",
                    flush=True,
                )

        elif policy == "workflow_k":
            for name in ("P1", "P2", "P3", "P4"):
                result = evict(
                    client,
                    policy,
                    f"{name.lower()}_child_a",
                    child_as[name],
                    action="evict_mamba_only",
                )

                print(
                    f"[WP3B-GOG] EVICT_{name}_CHILD_A "
                    f"ok={result['ok']}",
                    flush=True,
                )

        # ------------------------------------------------------------
        # 4. PRE-MEASURE HARD GATE
        # ------------------------------------------------------------
        census_post = client.census(
            f"wp3b_gog:{policy}:census:post_policy"
        )

        post_count = census_post["tree"]["mamba_node_count"]

        if post_count != 4:
            raise RuntimeError(
                f"expected exactly 4 Mamba checkpoints, got {post_count}"
            )

        p1_parent = inspect(
            client,
            policy,
            "p1_parent_pre_measure",
            parents["P1"],
        )["after"]["path"]

        p1_child = inspect(
            client,
            policy,
            "p1_child_a_pre_measure",
            child_as["P1"],
        )["after"]["path"]

        print(
            "[WP3B-GOG] PRE_MEASURE "
            f"policy={policy} "
            f"count={post_count} "
            f"parent_mamba={p1_parent['target_mamba_present']} "
            f"parent_fa={p1_parent['target_full_present']} "
            f"child_mamba={p1_child['target_mamba_present']} "
            f"child_fa={p1_child['target_full_present']}",
            flush=True,
        )

        if not p1_parent["target_full_present"]:
            raise RuntimeError("P1 Parent FA-KV disappeared")

        if not p1_child["target_full_present"]:
            raise RuntimeError("P1 Child-A FA-KV disappeared")

        if policy == "prompt_lru":
            if p1_parent["target_mamba_present"]:
                raise RuntimeError(
                    "Prompt-LRU unexpectedly retained P1 Parent Mamba"
                )
            if not p1_child["target_mamba_present"]:
                raise RuntimeError(
                    "Prompt-LRU unexpectedly evicted P1 Child-A Mamba"
                )

        else:
            if not p1_parent["target_mamba_present"]:
                raise RuntimeError(
                    "Workflow-K unexpectedly evicted P1 Parent Mamba"
                )
            if p1_child["target_mamba_present"]:
                raise RuntimeError(
                    "Workflow-K unexpectedly retained P1 Child-A Mamba"
                )

        print(
            "[WP3B-GOG] MAMBA_ROWS_PRE_MEASURE="
            + json.dumps(
                census_post["tree"]["mamba_rows"],
                sort_keys=True,
            ),
            flush=True,
        )

        # ------------------------------------------------------------
        # 5. Construct a sibling Child-B of P1.
        #
        # Child-A and Child-B intentionally share:
        #   Parent + parent_output
        #
        # but diverge at the first suffix token.
        # ------------------------------------------------------------
        child_a_suffix_first = child_as["P1"][ref.PARENT_LEN + 1]

        child_b_suffix = ref.toks(
            CHILD_B_SEED,
            ref.BRANCH_SUFFIX_LEN,
        )

        forbidden = {
            int(child_a_suffix_first),
        }

        if child_b_suffix[0] in forbidden:
            child_b_suffix[0] = (
                child_b_suffix[0] + 1
            ) % ref.VOCAB_SIZE

        if child_b_suffix[0] == child_a_suffix_first:
            raise RuntimeError(
                "Child-B does not diverge from Child-A"
            )

        child_b = (
            parents["P1"]
            + [parent_outputs["P1"]]
            + child_b_suffix
        )

        assert len(child_b) == 32_832

        print(
            "[WP3B-GOG] MEASURE_START "
            f"policy={policy} "
            f"child_b_len={len(child_b)}",
            flush=True,
        )

        # IMPORTANT:
        # exactly one measured request after the policy intervention.
        # Do not repeat it, because the request can self-heal/cache state.
        rid = f"wp3b_gog_{policy}_p1_child_b"

        output_ids = generate(
            engine,
            rid,
            child_b,
        )

        print(
            "[WP3B-GOG] MEASURE_DONE "
            f"policy={policy} "
            f"rid={rid} "
            f"output={output_ids[0]}",
            flush=True,
        )

        print(
            "[WP3B-GOG] STATUS=complete",
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
