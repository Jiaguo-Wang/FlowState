from evaluation.openhands_4workflow_occupancy_calibration import (
    ADDITIONAL_POOL_BYTES,
    EXPLICIT_SPARE_SLOTS,
    MAMBA_POOL_BYTES,
    MAMBA_SLOT_BYTES,
    PHYSICAL_MAX_MAMBA_CACHE_SIZE,
    SCHEDULE,
    build_census_attribution,
    reconstruct_node_positions,
)


def test_schedule_is_strict_round_robin() -> None:
    assert len(SCHEDULE) == 20
    assert SCHEDULE[:4] == (("A", 1), ("B", 1), ("C", 1), ("D", 1))
    assert SCHEDULE[-4:] == (("A", 5), ("B", 5), ("C", 5), ("D", 5))


def test_physical_pool_formula_has_explicit_headroom() -> None:
    assert PHYSICAL_MAX_MAMBA_CACHE_SIZE == 28
    assert MAMBA_SLOT_BYTES == 51_511_296
    assert MAMBA_POOL_BYTES == 1_493_827_584
    assert ADDITIONAL_POOL_BYTES == 206_045_184
    assert EXPLICIT_SPARE_SLOTS == 4


def test_reconstruct_node_positions_uses_parent_segments() -> None:
    positions = reconstruct_node_positions(
        [[0, None, [1], 0], [1, 0, [2], 64], [2, 1, [], 128]]
    )
    assert positions == {0: 0, 1: 64, 2: 192}


def test_census_delta_attribution_keeps_final_resident_nodes() -> None:
    baseline = {
        "workflow": None,
        "turn": None,
        "request_ordinal": 0,
        "added_mamba_node_ids": [],
        "resident_mamba_nodes": [],
    }
    first = {
        "workflow": "A",
        "turn": 1,
        "request_ordinal": 1,
        "added_mamba_node_ids": [11],
        "resident_mamba_nodes": [
            {"node_id": 11, "token_position": 64, "slots": [3]}
        ],
    }
    second = {
        "workflow": "B",
        "turn": 1,
        "request_ordinal": 2,
        "added_mamba_node_ids": [12],
        "resident_mamba_nodes": [
            {"node_id": 11, "token_position": 64, "slots": [3]},
            {"node_id": 12, "token_position": 128, "slots": [6]},
        ],
    }
    result = build_census_attribution([baseline, first, second])
    assert result["resident_checkpoint_counts"]["A"] == 1
    assert result["resident_checkpoint_counts"]["B"] == 1
    assert result["checkpoints"]["B"][0]["node_id"] == 12
