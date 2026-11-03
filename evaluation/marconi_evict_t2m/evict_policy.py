"""T2-Mini：Marconi Adaptation 的最小机制补强（纯 CPU 逻辑层，不操作 GPU）。

在既有 MarconiStylePolicy 之上仅做两项修改，其余全部复用现有实现（不重写评分代码）：
1. 增量收益：有父候选的收益代理 = L(自身) − L(最近保留祖先)；无父候选用 L(自身)。
   内存分母使用实测 Recurrent Checkpoint 大小（T2-R 由 Census component_slices 核验的
   51,511,296 bytes，等大小候选，min-max 归一化下与逻辑单位同选集——已在 T2-R 测试证明）。
2. 动态驱逐：参考官方 evict_v2 的逐轮合同——只有叶/单孩子节点可驱逐；每轮在当前可驱逐
   集合内用既有 _build_marconi_metrics（归一化 Recency + α·归一化增量效率，α=1）重评分，
   驱逐最低 utility（同分按 checkpoint_id 升序，确定性适配）；驱逐后子节点父指针等价重接
   到最近保留祖先（逻辑层合并，同官方 _evict_intermediate_node 语义）；重复直到满足 K。

明示适配边界（不声称完整复现 Marconi）：Recency 沿用声明访问次序（非官方 1/(t−ts)）；
收益仍为 Token 代理，非 Qwen3.5 GDN 真实 FLOPs；未实现 Radix 物理 KV 合并、Admission、
Prefetch 与 Serving Engine；本模块不产生任何物理操作凭据。
"""
from collections import Counter
from dataclasses import dataclass
from time import perf_counter_ns

from evaluation.sota_policies import _build_marconi_metrics, _validate_budget_k, _validate_finite_number
from flowstate.state_catalog import CheckpointCandidate

# 实测等大小 Recurrent Checkpoint 字节数（证据：baseline_reuse_t2r/historical_physical_mapping.json）
REAL_CHECKPOINT_BYTES = 51_511_296


@dataclass(frozen=True)
class EvictionSelection:
    """一次驱逐式选择的确定性结果与完整轨迹。"""

    selected_checkpoint_ids: tuple[str, ...]
    evicted_checkpoint_ids: tuple[str, ...]  # 按驱逐先后
    rounds: tuple[dict, ...]  # 每轮：可驱逐集、增量代理、utility、受害者
    call_ms: float


class MarconiEvictPolicy:
    """官方 evict_v2 风格的最小增强 Marconi Adaptation（逻辑树逐轮驱逐）。"""

    policy_name = "Marconi Adaptation (evict)"

    def select(self, candidates, budget_k, last_access_by_checkpoint, parent_by_checkpoint, alpha=1.0):
        """从全体候选出发逐轮驱逐至预算 K；评分复用既有 _build_marconi_metrics。"""
        k = _validate_budget_k(budget_k)
        ids = [c.checkpoint_id for c in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError('候选 checkpoint_id 必须唯一')
        pos = {c.checkpoint_id: c.token_pos for c in candidates}
        recency, parent = {}, {}
        for c in candidates:
            cid = c.checkpoint_id
            recency[cid] = _validate_finite_number(last_access_by_checkpoint[cid], '检查点 %s 的 last_access' % cid)
            if c.token_pos < 0 or (c.memory_bytes != REAL_CHECKPOINT_BYTES):
                raise ValueError('候选 %s 的 token_pos/memory_bytes 违反等大小合同' % cid)
            p = parent_by_checkpoint[cid]
            if p is not None and (p not in pos or pos[p] >= pos[cid]):
                raise ValueError('候选 %s 的父关系非法（缺失或父不更浅）' % cid)
            parent[cid] = p
        _validate_finite_number(alpha, 'alpha', nonnegative=True)
        # 祖先链环检测
        for cid in ids:
            seen, cur = set(), cid
            while cur is not None:
                if cur in seen:
                    raise ValueError('父关系存在环：%s' % cid)
                seen.add(cur)
                cur = parent[cur]

        core = {c.checkpoint_id: c for c in candidates}
        retained = set(ids)
        rounds = []
        start = perf_counter_ns()
        while len(retained) > k:
            # 有效父 = 沿父链跳过已驱逐节点的最近保留祖先（等价于官方中间节点驱逐后的合并）
            eff_parent = {c: self._nearest_retained(parent[c], retained, parent) for c in retained}
            child_count = Counter(p for p in eff_parent.values() if p is not None)
            evictable = sorted(c for c in retained if child_count.get(c, 0) <= 1)
            if not evictable:
                raise ValueError('无可驱逐候选：树结构异常')
            # 增量收益代理：相对最近保留祖先；无祖先用自身 L
            proxy = {c: (pos[c] - pos[eff_parent[c]] if eff_parent[c] is not None else pos[c]) for c in evictable}
            metrics = _build_marconi_metrics(tuple(core[c] for c in evictable), recency, proxy, alpha)
            victim = min(evictable, key=lambda c: (metrics[c], c))
            rounds.append({'retained_count': len(retained), 'evictable': evictable,
                           'incremental_proxy': proxy, 'utility': metrics, 'victim': victim})
            retained.remove(victim)
        call_ms = (perf_counter_ns() - start) / 1e6
        evicted = tuple(r['victim'] for r in rounds)
        selected = tuple(sorted(retained))
        # 输出合同：选集/驱逐序/保留集 划分全体候选且无重复
        assert len(selected) <= k and not (set(selected) & set(evicted)) and set(selected) | set(evicted) == set(ids)
        return EvictionSelection(selected, evicted, tuple(rounds), call_ms)

    @staticmethod
    def _nearest_retained(node, retained, parent):
        """沿父链跳过已驱逐节点，返回最近保留祖先；到根返回 None。"""
        cur = node
        while cur is not None and cur not in retained:
            cur = parent[cur]
        return cur


def plan_case(case, k, alpha=1.0):
    """在冻结声明输入上运行驱逐式 Marconi，并用既有共同模型给出 E/G 与 J。

    输出形状对齐 baseline_reuse_t2r/interface.plan_declared 的 Marconi 分支；不修改输入，
    不产生物理执行授权。耗时为单次调用诊断，不是正式微基准。
    """
    from copy import deepcopy
    from evaluation.baseline_reuse_t2r.interface import validate_declared_case
    from evaluation.kvflow_multiworkflow_cpu_main import run_cpu as legacy

    original = deepcopy(case)
    validate_declared_case(case, k)  # 复用 T2-R 的冻结输入验证（含 Token 前缀父证据）
    cs = case['candidates']
    core = tuple(CheckpointCandidate(c['checkpoint_id'], c['checkpoint_id'], (), c['token_position_L'],
                                     REAL_CHECKPOINT_BYTES, True) for c in cs)
    recency = {c['checkpoint_id']: c['last_access_order'] for c in cs}
    parent = {c['checkpoint_id']: c['parent_candidate'] for c in cs}
    result = MarconiEvictPolicy().select(core, k, recency, parent, alpha)
    ep = legacy.epoch(case, k)
    score = legacy.common.score(ep, list(result.selected_checkpoint_ids))
    if case != original:
        raise ValueError('纯规划修改了输入')
    return {'policy': MarconiEvictPolicy.policy_name, 'epoch_id': case['epoch_id'], 'K': k,
            'selected': list(result.selected_checkpoint_ids), 'evicted': list(result.evicted_checkpoint_ids),
            'rounds': [{'retained_count': r['retained_count'], 'evictable': r['evictable'],
                        'victim': r['victim']} for r in result.rounds],
            'per_pending': score['per_pending'], 'J_ms': score['cost_ms'],
            'evidence_level': 'DECLARED_SYNTHETIC_CPU_ONLY', 'timing': {'policy_call_ms': result.call_ms}}
