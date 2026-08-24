#!/usr/bin/env python3

import json
import traceback

import driver_eviction_ref as ref


PARENT_SEED = 51_001
CHILD_SEED = 201_001


def inspect(client, label, ids):
    return client.checkpoint_control(
        nonce=f"wp3b_leaf_safety:inspect:{label}",
        label=f"wp3b_leaf_safety:{label}",
        action="inspect",
        token_ids=ids,
    )


def evict_mamba_only(client, label, ids):
    return client.checkpoint_control(
        nonce=f"wp3b_leaf_safety:evict:{label}",
        label=f"wp3b_leaf_safety:{label}",
        action="evict_mamba_only",
        token_ids=ids,
    )


def main():
    engine = None

    try:
        print("[WP3B-LEAF-SAFETY] starting engine", flush=True)

        engine = ref.DirectEvictionEngine(**ref.ENGINE_CONFIG)
        client = ref.ControlClient(ref.CONTROL_PORT)
        ref.wait_for_probe(client)

        engine.flush_cache()

        # ------------------------------------------------------------
        # 1. Build Parent@32768
        # ------------------------------------------------------------
        parent = ref.toks(PARENT_SEED, ref.PARENT_LEN)

        rid = "wp3b_leaf_safety_parent"
        result = engine.generate(
            input_ids=parent,
            sampling_params=ref.SAMPLING_PARAMS,
            rid=rid,
        )

        output_ids, _ = ref.validate_generation(result, rid)
        parent_output = int(output_ids[0])

        print("[WP3B-LEAF-SAFETY] PARENT_BUILT", flush=True)

        # ------------------------------------------------------------
        # 2. Build Child-A@32832
        # ------------------------------------------------------------
        suffix = ref.toks(CHILD_SEED, ref.BRANCH_SUFFIX_LEN)

        if suffix[0] == parent_output:
            suffix[0] = (suffix[0] + 1) % ref.VOCAB_SIZE

        child = parent + [parent_output] + suffix
        assert len(child) == 32_832

        rid = "wp3b_leaf_safety_child_a"
        result = engine.generate(
            input_ids=child,
            sampling_params=ref.SAMPLING_PARAMS,
            rid=rid,
        )
        ref.validate_generation(result, rid)

        print("[WP3B-LEAF-SAFETY] CHILD_BUILT", flush=True)

        # ------------------------------------------------------------
        # 3. Hard pre-eviction checks
        # ------------------------------------------------------------
        parent_pre = inspect(client, "parent_pre", parent)
        child_pre = inspect(client, "child_pre", child)
        census_pre = client.census(
            "wp3b_leaf_safety:census:pre"
        )

        pp = parent_pre["after"]["path"]
        cp = child_pre["after"]["path"]

        pre_count = census_pre["tree"]["mamba_node_count"]

        print(
            "[WP3B-LEAF-SAFETY] PRE "
            f"count={pre_count} "
            f"parent_mamba={pp['target_mamba_present']} "
            f"parent_fa={pp['target_full_present']} "
            f"child_leaf={cp['is_device_leaf']} "
            f"child_children={cp['n_children']} "
            f"child_mamba={cp['target_mamba_present']} "
            f"child_fa={cp['target_full_present']} "
            f"child_node={cp['node_id']}",
            flush=True,
        )

        if pre_count != 2:
            raise RuntimeError(
                f"expected exactly 2 Mamba checkpoints, got {pre_count}"
            )

        if not pp["target_mamba_present"] or not pp["target_full_present"]:
            raise RuntimeError("Parent state missing before test")

        if not cp["is_device_leaf"] or cp["n_children"] != 0:
            raise RuntimeError(
                f"Child-A is not a leaf before test: {cp}"
            )

        if not cp["target_mamba_present"] or not cp["target_full_present"]:
            raise RuntimeError("Child state missing before test")

        child_node_before = cp["node_id"]
        full_tree_before = child_pre["after"]["tree"]["full_tree_sha256"]
        structure_before = child_pre["after"]["tree"]["structure_sha256"]
        full_path_before = cp["path_full_sha256"]
        full_allocator_before = child_pre["after"]["accounting"]["full_allocator"]

        # ------------------------------------------------------------
        # 4. Experimental leaf Mamba-only eviction
        # ------------------------------------------------------------
        print(
            "[WP3B-LEAF-SAFETY] invoking evict_mamba_only on Child-A",
            flush=True,
        )

        eviction = evict_mamba_only(
            client,
            "child_a",
            child,
        )

        print(
            "[WP3B-LEAF-SAFETY] EVICTION_PROOF="
            + json.dumps(eviction["proof"], sort_keys=True),
            flush=True,
        )

        print(
            "[WP3B-LEAF-SAFETY] EVICTION_MUTATION="
            + json.dumps(eviction["mutation"], sort_keys=True),
            flush=True,
        )

        # ------------------------------------------------------------
        # 5. Hard post-eviction invariants
        # ------------------------------------------------------------
        parent_post = inspect(client, "parent_post", parent)
        child_post = inspect(client, "child_post", child)
        census_post = client.census(
            "wp3b_leaf_safety:census:post"
        )

        pp2 = parent_post["after"]["path"]
        cp2 = child_post["after"]["path"]

        post_count = census_post["tree"]["mamba_node_count"]

        print(
            "[WP3B-LEAF-SAFETY] POST "
            f"count={post_count} "
            f"parent_mamba={pp2['target_mamba_present']} "
            f"parent_fa={pp2['target_full_present']} "
            f"child_leaf={cp2['is_device_leaf']} "
            f"child_children={cp2['n_children']} "
            f"child_mamba={cp2['target_mamba_present']} "
            f"child_fa={cp2['target_full_present']} "
            f"child_node={cp2['node_id']}",
            flush=True,
        )

        # Child node itself must survive.
        if cp2["node_id"] != child_node_before:
            raise RuntimeError(
                f"Child node changed: {child_node_before} -> {cp2['node_id']}"
            )

        if not cp2["is_device_leaf"] or cp2["n_children"] != 0:
            raise RuntimeError("Child radix structure changed")

        # Only Mamba should disappear.
        if cp2["target_mamba_present"]:
            raise RuntimeError("Child Mamba still present")

        if not cp2["target_full_present"]:
            raise RuntimeError("Child FA-KV was accidentally evicted")

        # Parent must remain intact.
        if not pp2["target_mamba_present"]:
            raise RuntimeError("Parent Mamba was accidentally modified")

        if not pp2["target_full_present"]:
            raise RuntimeError("Parent FA-KV was accidentally modified")

        # Exactly one recurrent checkpoint should disappear.
        if post_count != 1:
            raise RuntimeError(
                f"expected 1 Mamba checkpoint after eviction, got {post_count}"
            )

        # Full / radix state must be byte-for-byte logically unchanged.
        if child_post["after"]["tree"]["structure_sha256"] != structure_before:
            raise RuntimeError("radix structure changed")

        if child_post["after"]["tree"]["full_tree_sha256"] != full_tree_before:
            raise RuntimeError("Full KV tree changed")

        if cp2["path_full_sha256"] != full_path_before:
            raise RuntimeError("Child Full KV path changed")

        if (
            child_post["after"]["accounting"]["full_allocator"]
            != full_allocator_before
        ):
            raise RuntimeError("Full allocator changed")

        # Also require the probe's own causal proof.
        proof = eviction["proof"]

        required_true = (
            "same_node_id",
            "structure_unchanged",
            "full_tree_unchanged",
            "full_path_unchanged",
            "full_allocator_unchanged",
            "only_target_mamba_changed",
            "sanity_check_passed",
        )

        for key in required_true:
            if not proof.get(key):
                raise RuntimeError(
                    f"probe proof failed: {key}={proof.get(key)}"
                )

        changed = proof.get("changed_mamba_node_ids")
        if changed != [child_node_before]:
            raise RuntimeError(
                f"unexpected Mamba changes: {changed}"
            )

        print(
            "[WP3B-LEAF-SAFETY] MAMBA_ROWS_POST="
            + json.dumps(
                census_post["tree"]["mamba_rows"],
                sort_keys=True,
            ),
            flush=True,
        )

        print(
            "[WP3B-LEAF-SAFETY] STATUS=complete",
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
