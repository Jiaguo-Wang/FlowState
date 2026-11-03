"""G2 Greedy 对照臂的最小测试适配：在满足 Exact 门禁的 Epoch 上复用生产 Marginal Greedy。

不修改 flowstate 核心。plan_strict 的超域分支（flowstate/optimizer.py）在生产中于规模超门禁时
执行 Marginal Greedy；B5-C4（7候选/4Pending/K=2）满足门禁，生产入口会自动选 Exact。
G2 实验设计要求同一真实 Epoch 上的 Greedy 对照——本适配逐行复用 plan_strict 超域分支的同一内部
路径（同一 frontiers 评分闭包、同一 _select_candidates 生产循环、同一 StrictAllocationPlan 构造），
仅跳过 Exact 枚举本身；不篡改候选数量、门禁或预算（control 侧另行断言域满足门禁）。
"""
from flowstate.optimizer import GlobalOptimizer
from flowstate.strict_epoch import PendingPlan, StrictAllocationPlan


def plan_greedy_strict(optimizer: GlobalOptimizer, epoch) -> StrictAllocationPlan:
    """按 plan_strict 超域分支的同一生产路径在门禁内 Epoch 上求解 Marginal Greedy。"""
    epoch.validate()
    candidates = tuple(c.candidate for c in epoch.candidates)

    def score_ids(selected_ids):
        frontiers = epoch.frontiers(selected_ids)
        return sum(optimizer._recovery_cost_model.estimate(p.target - e, p.target)
                   for p, e in zip(epoch.pendings, frontiers))

    size = candidates[0].memory_bytes if candidates else 1
    allocation = optimizer._select_candidates(
        candidates, epoch.K * size,
        lambda selected: score_ids(tuple(c.checkpoint_id for c in selected)))
    ids = tuple(c.checkpoint_id for c in allocation.selected)
    cost_after_ms = allocation.recovery_cost_after_ms
    total_benefit_ms = allocation.total_benefit_ms
    frontiers = epoch.frontiers(ids)
    pending = tuple(PendingPlan(p.pending_id, p.target, e, p.target - e,
                     optimizer._recovery_cost_model.estimate(p.target - e, p.target))
                    for p, e in zip(epoch.pendings, frontiers))
    return StrictAllocationPlan(
        epoch.epoch_id, epoch.snapshot_sha256, epoch.content_digest, epoch.K,
        ids, len(ids), pending, cost_after_ms, total_benefit_ms,
        tuple(sorted(c.checkpoint_id for c in candidates if c.checkpoint_id not in ids)),
    )
