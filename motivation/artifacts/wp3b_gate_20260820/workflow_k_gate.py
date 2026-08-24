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
        nonce=f"wp3b_workflow_k:inspect:{label}",
        label=f"wp3b_workflow_k:{label}",
        action="inspect",
        token_ids=ids,
    )


def evict_mamba(client, label, ids):
    return client.checkpoint_control(
        nonce=f"wp3b_workflow_k:evict:{label}",
        label=f"wp3b_workflow_k:{label}",
        action="evict_mamba_only",
        token_ids=ids,
    )


def path_state(response):
    return response["after"]["path"]


def main():
    engine = None

    try:
        print("[WP3B-WORKFLOW-K] starting engine", flush=True)

        engine = ref.DirectEvictionEngine(**ref.ENGINE_CONFIG)
        client = ref.ControlClient(ref.CONTROL_PORT)

        ref.wait_for_probe(client)

        # ------------------------------------------------------------
        # 1. Clean cache
        # ------------------------------------------------------------
        engine.flush_cache()

        parents = {}
        parent_outputs = {}
        children = {}

        # ------------------------------------------------------------
        # 2. Build four pending Fork Parents
        # ------------------------------------------------------------
        for name, seed in PARENT_SEEDS.items():
            parent = ref.toks(seed, ref.PARENT_LEN)
            parents[name] = parent

            rid = f"wp3b_workflow_k_{name.lower()}_parent"

            result = engine.generate(
                input_ids=parent,
                sampling_params=ref.SAMPLING_PARAMS,
                rid=rid,
            )

            output_ids, _ = ref.validate_generation(result, rid)
            parent_outputs[name] = int(output_ids[0])

            print(
                f"[WP3B-WORKFLOW-K] {name}_PARENT_BUILT",
                flush=True,
            )

        # ------------------------------------------------------------
        # 3. Build Child-A for every workflow
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

            rid = f"wp3b_workflow_k_{name.lower()}_child_a"

            result = engine.generate(
                input_ids=child,
                sampling_params=ref.SAMPLING_PARAMS,
                rid=rid,
            )

            ref.validate_generation(result, rid)

            print(
                f"[WP3B-WORKFLOW-K] {name}_CHILD_A_BUILT",
                flush=True,
            )

        # ------------------------------------------------------------
        # 4. Hard pre-policy gate
        # ------------------------------------------------------------
        census_pre = client.census(
            "wp3b_workflow_k:census:pre_policy"
        )

        pre_count = census_pre["tree"]["mamba_node_count"]

        print(
            f"[WP3B-WORKFLOW-K] PRE_POLICY_MAMBA_COUNT={pre_count}",
            flush=True,
        )

        if pre_count != 8:
            raise RuntimeError(
                f"expected 8 Mamba checkpoints before policy, got {pre_count}"
            )

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
                raise RuntimeError(
                    f"{name}: Parent Mamba missing before policy"
                )

            if not c["target_mamba_present"]:
                raise RuntimeError(
                    f"{name}: Child Mamba missing before policy"
                )

            if not p["target_full_present"] or not c["target_full_present"]:
                raise RuntimeError(
                    f"{name}: FA-KV missing before policy"
                )

        # ------------------------------------------------------------
        # 5. Workflow-K, K=4
        #
        # Workflow knowledge:
        #   Parent = pending FORK boundary
        #   Child-A = completed branch endpoint
        #
        # Keep:
        #   P1 P2 P3 P4
        #
        # Evict:
        #   C1 C2 C3 C4
        # ------------------------------------------------------------
        print(
            "[WP3B-WORKFLOW-K] applying K=4: "
            "evict P1/P2/P3/P4 Child-A checkpoints",
            flush=True,
        )

        for name in ("P1", "P2", "P3", "P4"):
            result = evict_mamba(
                client,
                f"{name.lower()}_child_a",
                children[name],
            )

            print(
                f"[WP3B-WORKFLOW-K] EVICT_{name}_CHILD_A "
                f"ok={result.get('ok')}",
                flush=True,
            )

        # ------------------------------------------------------------
        # 6. Hard post-policy gate
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
                f"[WP3B-WORKFLOW-K] {name}_POST "
                f"parent_mamba={p['target_mamba_present']} "
                f"parent_fa={p['target_full_present']} "
                f"child_mamba={c['target_mamba_present']} "
                f"child_fa={c['target_full_present']}",
                flush=True,
            )

            # Workflow-important Parent must remain executable.
            if not p["target_mamba_present"]:
                raise RuntimeError(
                    f"{name}: Parent Mamba unexpectedly missing"
                )

            if not p["target_full_present"]:
                raise RuntimeError(
                    f"{name}: Parent FA-KV unexpectedly missing"
                )

            # Completed Child-A recurrent checkpoint must be gone.
            if c["target_mamba_present"]:
                raise RuntimeError(
                    f"{name}: Child Mamba still present after Workflow-K"
                )

            # FA-KV must still remain.
            if not c["target_full_present"]:
                raise RuntimeError(
                    f"{name}: Child FA-KV was accidentally evicted"
                )

        census_post = client.census(
            "wp3b_workflow_k:census:post_policy"
        )

        post_count = census_post["tree"]["mamba_node_count"]

        print(
            f"[WP3B-WORKFLOW-K] POST_POLICY_MAMBA_COUNT={post_count}",
            flush=True,
        )

        print(
            "[WP3B-WORKFLOW-K] MAMBA_ROWS="
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
            "[WP3B-WORKFLOW-K] STATUS=complete",
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
