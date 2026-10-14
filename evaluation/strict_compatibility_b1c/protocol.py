"""隔离方案与跨条件比较所用的纯 CPU 检查。"""

import array
import hashlib


def token_digest(tokens):
    return hashlib.sha256(array.array("q", map(int, tokens)).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def isolation_plan(candidates, nodes, pending_tokens, target_id):
    """覆盖全部物理驻留节点后，选择必须安全移除的干扰状态。"""
    resident = {row["node_id"]: row for row in nodes if row["mamba_slots"]}
    mapped = {row["path"]["node_id"]: row for row in candidates}
    require(len(mapped) == len(candidates), "逻辑候选映射到相同节点，停止隔离")
    require(set(resident) == set(mapped), "物理驻留节点与候选清单不一致，停止隔离")
    slots = [slot for row in resident.values() for slot in row["mamba_slots"]]
    require(len(slots) == len(set(slots)), "发现物理槽位别名，停止隔离")
    target = next(row for row in candidates if row["checkpoint_id"] == target_id)
    length = len(target["token_ids"])
    require(target["token_ids"] == pending_tokens[:length], "目标前缀不兼容")
    distractors = []
    for row in candidates:
        node = resident[row["path"]["node_id"]]
        require(node["token_ids"] == row["token_ids"], "树节点实际 Token 与候选不一致")
        if row["checkpoint_id"] != target_id and len(row["token_ids"]) >= length:
            if row["token_ids"] == pending_tokens[:len(row["token_ids"])]:
                distractors.append(row)
    return sorted(distractors, key=lambda row: (-len(row["token_ids"]), row["checkpoint_id"]))


def semantic_nodes(snapshot):
    """跨 lifecycle 比较 Token 身份和驻留，不依赖重用的节点或槽位编号。"""
    return sorted((row["prefix_digest"], row["position"], row["full_present"],
                   row["full_count"], bool(row["mamba_slots"])) for row in snapshot["nodes"])
