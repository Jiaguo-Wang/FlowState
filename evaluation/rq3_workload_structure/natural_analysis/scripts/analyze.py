"""从冻结逐行输入独立统计自然来源工作负载；不调用任何策略求解器。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[1]
EPS = 1e-9
SIMILARITY = 0.05
POLICIES = ('FlowState', 'PCB', 'LDR', 'WFC')
P1 = ROOT / 'evaluation/p1_strong_baseline_validation'
OH = ROOT / 'evaluation/runtime_artifacts/rq3_formal_policy_eval_20260904_110011/per_snapshot_results.jsonl'
AX = ROOT / 'rq3_agentx_policy_eval_20260906_214956/per_case_results.jsonl'
AX_RAW = Path('/home/wjg/data/agentx/cc-traces-weka-062126/traces.jsonl')
AX_RUNTIME = Path('/home/wjg/data/agentx/audits/agentx_runtime_formal_20260906_023649/runtime_snapshots.jsonl')
OH_RAW = Path('/home/wjg/data/agentic_coding_trajectories/sessions.parquet')


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= EPS,
            f'数值不一致：{a} 与 {b}')


def phi(gap, target):
    """独立复现冻结公式，不导入当前生产模型。"""
    require(0 <= gap <= target <= 131072, '恢复位置超出冻结数学定义域')
    g, t = gap / 1024, target / 1024
    return 37.828150 * g + 0.345974143 * g * t - 0.156201917 * g * g


class Sources:
    """读取前后核对源文件，记录路径、内容摘要及实际使用记录。"""
    def __init__(self):
        self.entries = {}

    def add(self, path, role):
        p = Path(path).resolve()
        if str(p) not in self.entries:
            require(p.is_file(), '必要来源缺失：' + str(p))
            self.entries[str(p)] = {'path': str(p), 'role': role, 'bytes': p.stat().st_size,
                                    'sha256_before': sha(p)}
        return p

    def obj(self, path, role):
        return json.loads(self.add(path, role).read_text())

    def lines(self, path, role):
        p = self.add(path, role)
        values = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        self.entries[str(p)]['records'] = len(values)
        return values

    def finish(self):
        for item in self.entries.values():
            item['sha256_after'] = sha(item['path'])
            require(item['sha256_before'] == item['sha256_after'], '审计期间来源内容发生变化')
        return list(self.entries.values())


def cover_certificate(positions, compatible, targets):
    """对最深前沿覆盖做小Pending域位掩码动态规划；同时计数最小覆盖集合。"""
    best = [max([l for l, s in zip(positions, compatible) if p in s] + [0])
            for p in range(len(targets))]
    required = [p for p, v in enumerate(best) if v > 0]
    bit = {p: i for i, p in enumerate(required)}
    masks = [sum(1 << bit[p] for p in required if p in s and l == best[p])
             for l, s in zip(positions, compatible)]
    dp = {0: (0, 1, ())}
    for i, mask in enumerate(masks):
        if not mask:
            continue
        old = dict(dp)
        for covered, (size, ways, witness) in old.items():
            nxt = covered | mask
            if nxt == covered:
                continue
            option = (size + 1, ways, witness + (i,))
            if nxt not in dp or option[0] < dp[nxt][0]:
                dp[nxt] = option
            elif option[0] == dp[nxt][0]:
                present = dp[nxt]
                dp[nxt] = (present[0], present[1] + ways, min(present[2], option[2]))
    full = (1 << len(required)) - 1
    size, ways, witness = dp[full]
    realizers = [[i for i, (l, s) in enumerate(zip(positions, compatible)) if p in s and l == best[p]]
                 for p in required]
    forced = set().union(*(set(v) for v in realizers)) if realizers else set()
    unique = all(len(v) == 1 for v in realizers)
    if unique:
        require(size == len(forced), '位掩码证书与唯一前沿实现者证书不一致')
    require(all(max([positions[i] for i in witness if p in compatible[i]] + [0]) == best[p]
                for p in range(len(targets))), '最小覆盖见证没有达到完整前沿')
    return best, size, ways, witness, unique, masks


def inspect_raw_sources(sources, oh_sessions, trace_ids):
    """只从原始记录提取可直接观察的字段，不把调用容器等同原生Fork。"""
    import pyarrow.parquet as pq
    sources.add(OH_RAW, '原始OpenHands及其他来源消息归档')
    original = {}
    for batch in pq.ParquetFile(OH_RAW).iter_batches(
            batch_size=128, columns=['session_id', 'source_dataset', 'messages_json', 'ground_truth_meta_json']):
        for row in batch.to_pylist():
            if row['session_id'] not in oh_sessions:
                continue
            messages = json.loads(row['messages_json'])
            original[row['session_id']] = {
                'source_dataset': row['source_dataset'], 'message_count': len(messages),
                'role_counts': dict(Counter(m.get('role', 'UNKNOWN') for m in messages)),
                'message_content_sha256': hashlib.sha256(encode(messages).encode()).hexdigest(),
                'native_fork_event': 'UNKNOWN', 'decision_ready_timestamp': 'UNKNOWN'}
    require(set(original) == oh_sessions, 'OpenHands原始Session联接不完整')
    sources.add(AX_RAW, '原始AgentX调用与局部块身份归档')
    traces = {}
    with AX_RAW.open('rb') as f:
        for line_no, line in enumerate(f, 1):
            match = re.search(rb'"id"\s*:\s*"([^"]+)"', line[:1024])
            if not match or match.group(1).decode() not in trace_ids:
                continue
            trace = json.loads(line)
            counts = Counter()
            depth = [0]
            missing = {'prompt', 'messages', 'input_token_ids', 'token_ids'}
            found = set()

            def walk(value, level=0):
                if isinstance(value, dict):
                    found.update(set(value) & missing)
                    if 'type' in value:
                        counts[str(value['type'])] += 1
                    if 'hash_ids' in value and 'in' in value:
                        counts['calls_with_block_ids'] += 1
                    for key, child in value.items():
                        if isinstance(child, (list, dict)):
                            next_level = level + int(key == 'requests')
                            depth[0] = max(depth[0], next_level)
                            walk(child, next_level)
                elif isinstance(value, list):
                    for child in value:
                        walk(child, level)

            walk(trace)
            traces[trace['id']] = {'source_line': line_no, 'record_sha256': hashlib.sha256(line.rstrip(b'\r\n')).hexdigest(),
                                  'block_size': trace.get('block_size'), 'hash_id_scope': trace.get('hash_id_scope'),
                                  'raw_type_counts': dict(counts), 'raw_requests_container_depth': depth[0],
                                  'full_input_fields_found': sorted(found),
                                  'native_fork_event': 'UNKNOWN', 'decision_ready_timestamp': 'UNKNOWN'}
    require(set(traces) == trace_ids, 'AgentX原始trace联接不完整')
    return original, traces


def analyze(sources):
    rows = sources.lines(P1 / 'per_case_results.jsonl', '四种策略逐条件冻结结果及原始行定位')
    source_rows = {str(p): sources.lines(p, '原始逐预算评估输出') for p in (OH, AX)}
    locator = sources.lines(ROOT / 'evaluation/p0_allocation_mechanism_audit/all_cases.jsonl', '只用于定位原始快照文件')
    loc = {(r['population'], r['snapshot_id'], r['budget_ratio']): r['metadata']['source'] for r in locator}
    auxiliary = {r['instance_id']: r for r in sources.lines(
        ROOT / 'evaluation/gate1_step1_allocation_equivalence/matrices.jsonl', '逐格收益及预算交叉核对，非统计输入')}
    manifest = sources.obj(ROOT / 'rq3_agentx_policy_eval_20260906_214956/formal_snapshot_manifest.json', 'AgentX正式兼容关系定义')
    ax_manifest = {r['snapshot_id']: r for r in manifest}
    ax_runtime = {r['trace_id']: r for r in sources.lines(AX_RUNTIME, 'AgentX旧实际物化与Raw前缀谓词')}
    popmanifest = sources.obj(ROOT / 'evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017/population_manifest.json', 'OpenHands受控分组与原始数据来源')
    for name, expected in sources.obj(P1 / 'implementation_lock.json', '历史简单策略实现锁').items():
        p = sources.add(P1 / name, '历史评分及简单策略实现，只读不导入')
        require(sha(p) == expected, '历史策略源码与锁不一致')
    sources.add(ROOT / 'flowstate/recovery_model.py', '冻结公式实现身份；统计独立复现公式')
    result, agreements, snapshots, detail = [], [], {}, []
    max_diff = 0.0
    for line_no, saved in enumerate(rows, 1):
        population, sid, k, ratio = (saved[x] for x in ('population', 'snapshot_id', 'k', 'budget_ratio'))
        iid = f'{sid}::budget={ratio}'
        require(iid not in {r['condition_id'] for r in result}, '条件标识重复')
        raw = source_rows[saved['source_file']][saved['source_line'] - 1]
        require((raw['snapshot_id'], raw['k'], raw['budget_ratio'], raw['snapshot_digest']) ==
                (sid, k, ratio, saved['snapshot_digest']), '原始评估行身份不一致')
        md = raw['mechanism_diagnostics']
        pending_rows = raw['policies']['FlowState']['per_continuation']
        pids = [p['continuation_id'] for p in pending_rows]
        targets = [p['target_tokens'] for p in pending_rows]
        cids = sorted(md['standalone_benefits'])
        positions = [md['standalone_benefits'][c]['token_pos'] for c in cids]
        compatible = [{i for i, pid in enumerate(pids) if md['compatibility_matrix'][c][pid]} for c in cids]
        n, m = len(cids), len(pids)
        require(n == raw['candidate_count'] and m == raw['pending_count'], '候选或Pending数量不一致')
        require(len(set(cids)) == n and len(set(pids)) == m, '候选或Pending身份冲突')
        for l, support in zip(positions, compatible):
            require(all(0 <= l <= targets[p] for p in support), '准入位置非法')
        b = [[phi(targets[p], targets[p]) - phi(targets[p] - l, targets[p]) if p in support else 0.0
              for p in range(m)] for l, support in zip(positions, compatible)]
        for i, c in enumerate(cids):
            recorded = {v['continuation_id']: v for v in md['standalone_benefits'][c]['per_pending']}
            for p, pid in enumerate(pids):
                close(b[i][p], recorded[pid]['benefit_ms'])
                close(b[i][p], saved['benefit_matrix_ms'][c][p])
        matrix = auxiliary[iid]
        for i, c in enumerate(cids):
            j = matrix['candidate_ids'].index(c)
            require(positions[i] == matrix['positions'][j], '交叉矩阵位置不一致')
            for p, pid in enumerate(pids):
                q = matrix['pending_ids'].index(pid)
                require((p in compatible[i]) == bool(matrix['compatibility'][q][j]), '交叉矩阵关系不一致')
                close(b[i][p], matrix['benefit_ms'][q][j])
        physical_source = loc[(population, sid, ratio)]
        source_sessions = []
        relation_diffs = {'formal_only': 0, 'raw_only': 0}
        if population == 'OpenHands':
            package = sources.obj(physical_source, 'OpenHands原始冻结物理快照')
            canonical = json.loads(package['canonical_snapshot'])
            original_c = {c['checkpoint_id']: c for c in canonical['eligible_candidates']}
            require(set(original_c) == set(cids), '原始物理候选域不同')
            source_sessions = package['session_ids']
            original_p = {p['continuation_id']: p for p in canonical['pending_continuations']}
            for i, c in enumerate(cids):
                cc = original_c[c]
                require(cc['token_pos'] == positions[i] and cc['recurrent_resident'], '原始候选位置或驻留不一致')
                for p, pid in enumerate(pids):
                    pp = original_p[pid]
                    t = min(pp['anchor_pos'], pp['resident_fa_frontier'])
                    lineage = cc['lineage_path']
                    yes = cc['workflow_id'] == pp['workflow_id'] and pp['lineage_path'][:len(lineage)] == lineage and cc['token_pos'] <= t
                    require(t == targets[p] and yes == (p in compatible[i]), '原始Legacy谓词与矩阵不一致')
            node_map = {v['checkpoint_id']: v['node_id'] for v in canonical['runtime_evidence']}
            sizes = [original_c[c]['memory_bytes'] for c in cids]
            size_status = 'RECORDED'
            trace_id = ''
        else:
            fm = ax_manifest[sid]
            require(all(set(fm['objective_usable_pending_ids'][c]) == {pids[p] for p in compatible[i]}
                        for i, c in enumerate(cids)), 'AgentX正式定义与冻结矩阵不同')
            trace_id = raw['trace_id']
            runtime = ax_runtime[trace_id]
            rec = {c['checkpoint_id']: c for c in runtime['checkpoints']}
            require(set(rec) == set(cids), 'AgentX物理候选域不同')
            node_map = {c: rec[c]['node_id'] for c in cids}
            for i, c in enumerate(cids):
                require(rec[c]['recurrent_prefix_length'] == positions[i], 'AgentX物理位置不一致')
                actual = set(rec[c]['compatible_pending_ids'])
                formal = {pids[p] for p in compatible[i]}
                relation_diffs['formal_only'] += len(formal - actual)
                relation_diffs['raw_only'] += len((actual & set(pids)) - formal)
            sizes = matrix['checkpoint_bytes']
            size_status = 'CODE_DERIVED_NOT_PHYSICAL_BYTE_CERTIFICATION'
        require(len(set(sizes)) == 1 and all(v > 0 for v in sizes), '非等大小预算')
        require(k == matrix['budget_bytes'] // sizes[0], '预算单位不一致')
        support = [{p for p, v in enumerate(vv) if v > EPS} for vv in b]
        positive = [i for i, s in enumerate(support) if s]
        shared = [i for i in positive if len(support[i]) > 1]
        counts = [sum(p in s for s in support) for p in range(m)]
        overlap_pairs, similar_pairs, dominant, equal_pairs, jaccard_sum, cross = 0, 0, set(), 0, 0.0, 0
        redundant_benefit = 0.0
        for i, j in itertools.combinations(positive, 2):
            inter, union = support[i] & support[j], support[i] | support[j]
            if inter:
                overlap_pairs += 1
                jaccard_sum += len(inter) / len(union)
                redundant_benefit += sum(min(x, y) for x, y in zip(b[i], b[j]))
                similar_pairs += any(abs(b[i][p] - b[j][p]) / max(b[i][p], b[j][p]) <= SIMILARITY for p in inter)
                cross += not (support[i] <= support[j] or support[j] <= support[i])
            if all(abs(x - y) <= EPS for x, y in zip(b[i], b[j])):
                equal_pairs += 1
            for a, z in [(i, j), (j, i)]:
                if all(x >= y - EPS for x, y in zip(b[a], b[z])) and any(x > y + EPS for x, y in zip(b[a], b[z])):
                    dominant.add(z)
        best, kmin, cover_count, witness, unique_frontier, masks = cover_certificate(positions, compatible, targets)
        cindex = {c: i for i, c in enumerate(cids)}

        def evaluate(ids):
            require(len(ids) == len(set(ids)) and len(ids) <= k and set(ids) <= set(cids), '冻结选集非法')
            es = [max([positions[cindex[c]] for c in ids if p in compatible[cindex[c]]] + [0]) for p in range(m)]
            gaps = [t - e for t, e in zip(targets, es)]
            return sum(phi(g, t) for g, t in zip(gaps, targets)), es, gaps

        j0, _, _ = evaluate([])
        jall = sum(phi(t - e, t) for t, e in zip(targets, best))
        cert = 'UNKNOWN'
        optimal_count = None
        if all(len(s) <= 1 for s in support):
            q = [max([b[i][p] for i in positive] + [0.0]) for p in range(m)]
            opt_benefit = sum(sorted(q, reverse=True)[:k])
            optimal_count = 0
            min_size = min(k, sum(v > EPS for v in q))
            for group in itertools.combinations([p for p in range(m) if q[p] > EPS], min_size):
                if abs(sum(q[p] for p in group) - opt_benefit) <= EPS:
                    optimal_count += math.prod(sum(abs(b[i][p] - q[p]) <= EPS for i in positive if p in support[i]) for p in group)
            optimum = j0 - opt_benefit
            cert = 'SEPARABLE_TOP_K_BOUND'
        elif k >= kmin:
            optimum, optimal_count = jall, cover_count
            cert = 'FULL_FRONTIER_LOWER_BOUND'
        else:
            optimum = None
        policy_data = {}
        for policy in POLICIES:
            recorded = saved['policies'][policy]
            selected = recorded['selected']
            j, es, gs = evaluate(selected)
            max_diff = max(max_diff, abs(j - recorded['cost_ms']))
            close(j, recorded['cost_ms'])
            require([(v['T'], v['E'], v['G']) for v in recorded['per_pending']] == list(zip(targets, es, gs)), '逐Pending前沿不一致')
            if policy == 'FlowState':
                require(set(selected) == set(raw['policies'][policy]['selected_checkpoint_ids']), '原始FlowState选集不同')
                close(j, raw['policies'][policy]['total_recovery_cost_ms'])
            if optimum is not None:
                close(j, optimum)
            policy_data[policy] = {'selected': selected, 'J': j, 'E': es, 'G': gs}
        fs = policy_data['FlowState']
        trace = md['flowstate_greedy_trace']
        require(set(trace['selection_order']) == set(fs['selected']), '原始逐步轨迹终集不一致')
        u = [0.0] * m
        selected = []
        suppressed, reversal, top_ties, local_tie_changed, redundant_selected = False, False, 0, False, 0
        step_rows = []
        for step, chosen in enumerate(trace['selection_order']):
            rem = [i for i in range(n) if cids[i] not in selected]
            marginal = {i: sum(max(0.0, v - z) for v, z in zip(b[i], u)) for i in rem}
            gain = marginal[cindex[chosen]]
            close(gain, trace['delta_sequence'][step])
            require(max(marginal.values()) - gain <= EPS, '已存Greedy轨迹不是当前最大边际')
            ties = [cids[i] for i in rem if abs(marginal[i] - max(marginal.values())) <= EPS]
            top_ties += len(ties) > 1
            suppressed |= any(sum(b[i]) - marginal[i] > EPS for i in rem)
            active = [i for i in rem if support[i]]
            reversal |= any((sum(b[i]) - sum(b[j])) * (marginal[i] - marginal[j]) < -EPS
                            and abs(sum(b[i]) - sum(b[j])) > EPS and abs(marginal[i] - marginal[j]) > EPS
                            for i, j in itertools.combinations(active, 2))
            local_tie_changed |= len(ties) > 1 and any(abs(b[cindex[c]][p] - b[cindex[chosen]][p]) > EPS for c in ties for p in range(m))
            step_rows.append({'step': step + 1, 'prefix': list(selected), 'chosen': chosen, 'marginal_ms': gain,
                              'top_tied_ids': sorted(ties), 'candidate_marginals_ms': {cids[i]: v for i, v in marginal.items()}})
            selected.append(chosen)
            u = [max(v, z) for v, z in zip(u, b[cindex[chosen]])]
        terminal = {cids[i]: sum(max(0.0, v - z) for v, z in zip(b[i], u)) for i in range(n) if cids[i] not in selected}
        for cid in selected:
            without = [v for v in selected if v != cid]
            redundant_selected += abs(evaluate(without)[0] - fs['J']) <= EPS
        unused_positive = [cids[i] for i in positive if cids[i] not in selected]
        alternatives = []
        if len(selected) < k:
            alternatives = [sorted(selected + [cid]) for cid, v in terminal.items() if v <= EPS][:1]
        same_size_witness = []
        for removed in selected:
            for added in terminal:
                alternative = [v for v in selected if v != removed] + [added]
                if abs(evaluate(alternative)[0] - fs['J']) <= EPS:
                    same_size_witness = sorted(alternative)
                    break
            if same_size_witness:
                break
        set_equal = len({tuple(sorted(v['selected'])) for v in policy_data.values()}) == 1
        order_equal = len({tuple(v['selected']) for v in policy_data.values()}) == 1
        j_equal = max(v['J'] for v in policy_data.values()) - min(v['J'] for v in policy_data.values()) <= EPS
        eg_equal = len({tuple(v['E']) for v in policy_data.values()}) == 1
        invariant = {'ids': cids, 'positions': positions, 'pending_ids': pids, 'targets': targets,
                     'compatibility': [sorted(s) for s in compatible], 'benefits': b, 'sizes': sizes}
        input_sha = hashlib.sha256(encode(invariant).encode()).hexdigest()
        if sid in snapshots:
            require(snapshots[sid]['input_sha'] == input_sha, '同快照不同预算的输入不同')
        else:
            snapshots[sid] = {'input_sha': input_sha, 'population': population, 'source_sessions': source_sessions,
                              'trace_id': trace_id, 'record': invariant}
        row = {'condition_id': iid, 'population': population, 'snapshot_id': sid, 'budget_ratio': ratio,
               'K': k, 'candidate_count': n, 'pending_count': m, 'positive_candidate_count': len(positive),
               'shared_positive_candidate_count': len(shared), 'shared_fraction_of_positive': len(shared) / len(positive) if positive else None,
               'shared_support_size_histogram': dict(Counter(len(support[i]) for i in shared)),
               'positive_support_size_histogram': dict(Counter(len(s) for s in support)),
               'max_positive_support_size': max(map(len, support), default=0), 'positive_candidate_counts_per_pending': dict(zip(pids, counts)),
               'overlapping_positive_pairs': overlap_pairs, 'positive_candidate_pairs': len(positive) * (len(positive) - 1) // 2,
               'overlap_jaccard_mean_on_overlapping_pairs': jaccard_sum / overlap_pairs if overlap_pairs else None,
               'overlap_redundant_benefit_sum_ms': redundant_benefit, 'similar_benefit_pairs_5pct': similar_pairs,
               'strictly_dominated_positive_candidates': len(dominant), 'equal_benefit_candidate_pairs': equal_pairs,
               'nonlaminar_positive_support_pairs': cross, 'K_min_frontier': kmin,
               'K_below_K_min': k < kmin, 'K_below_positive_candidates': k < len(positive),
               'K_covers_all_candidates': k >= n, 'unselected_positive_candidates': len(unused_positive),
               'unselected_positive_marginal_candidates': sum(v > EPS for v in terminal.values()),
               'selected_count': len(selected), 'budget_underfilled': len(selected) < k,
               'full_frontier_reached': fs['E'] == best, 'all_candidate_frontiers': dict(zip(pids, best)),
               'J_empty_ms': j0, 'J_all_candidates_ms': jall, 'J_FlowState_ms': fs['J'], 'formal_optimality_certificate': cert,
               'minimum_cardinality_optimum_count': optimal_count, 'frontier_realizers_unique': unique_frontier,
               'K_min_witness_ids': [cids[i] for i in witness], 'marginal_suppression_observed': suppressed,
               'strict_singleton_to_marginal_rank_reversal': reversal, 'greedy_top_tie_steps': top_ties,
               'greedy_tie_changes_benefit_vector': local_tie_changed, 'selected_zero_removal_contribution_count': redundant_selected,
               'same_size_equal_J_alternative_witness': same_size_witness, 'slack_equal_J_alternative_witness': alternatives,
               'all_four_same_set': set_equal, 'all_four_same_order': order_equal, 'all_four_same_J': j_equal,
               'all_four_same_E_G': eg_equal, 'physical_node_alias_excess': n - len(set(node_map.values())),
               'formal_only_vs_old_raw_edges': relation_diffs['formal_only'] if population == 'AgentX' else None,
               'old_raw_only_edges': relation_diffs['raw_only'] if population == 'AgentX' else None,
               'checkpoint_byte_evidence': size_status, 'input_without_budget_sha256': input_sha,
               'original_branch_count': 'UNKNOWN', 'original_common_history_tokens': 'UNKNOWN',
               'original_common_history_fraction': 'UNKNOWN', 'original_branch_depths': 'UNKNOWN',
               'original_pending_dependency_relation': 'UNKNOWN', 'physical_resume_certification': 'NOT_STRICT_CERTIFIED',
               'source_file': saved['source_file'], 'source_line': saved['source_line'], 'policy_record_line': line_no,
               'source_snapshot_file': physical_source, 'source_session_ids': source_sessions, 'trace_id': trace_id,
               'snapshot_digest': saved['snapshot_digest'], 'candidate_ids': cids, 'positions': positions,
               'pending_ids': pids, 'targets': targets, 'positive_supports': {cids[i]: [pids[p] for p in sorted(s)] for i, s in enumerate(support)},
               'candidate_node_map': node_map, 'selection_by_policy': {p: v['selected'] for p, v in policy_data.items()}}
        result.append(row)
        detail.append({'condition_id': iid, 'independent_benefit_matrix_ms': {cids[i]: b[i] for i in range(n)},
                       'greedy_recorded_path_revaluation': step_rows, 'terminal_marginals_ms': terminal,
                       'frontier_cover_masks': masks, 'policy_results': policy_data})
        for a, z in itertools.combinations(POLICIES, 2):
            va, vz = policy_data[a], policy_data[z]
            agreements.append({'condition_id': iid, 'population': population, 'snapshot_id': sid, 'K': k,
                               'policy_a': a, 'policy_b': z, 'selected_a': sorted(va['selected']), 'selected_b': sorted(vz['selected']),
                               'same_set': set(va['selected']) == set(vz['selected']), 'same_order': va['selected'] == vz['selected'],
                               'J_a_ms': va['J'], 'J_b_ms': vz['J'], 'J_a_minus_b_ms': va['J'] - vz['J'],
                               'same_J': abs(va['J'] - vz['J']) <= EPS, 'same_E_G': va['E'] == vz['E'],
                               'different_set_equal_J': set(va['selected']) != set(vz['selected']) and abs(va['J'] - vz['J']) <= EPS,
                               'E_a': va['E'], 'E_b': vz['E'], 'G_a': va['G'], 'G_b': vz['G'],
                               'source_line': saved['source_line'], 'policy_record_line': line_no})
    require(Counter(r['population'] for r in result) == {'OpenHands': 504, 'AgentX': 69}, '原始条件域不完整')
    return result, agreements, detail, snapshots, max_diff, popmanifest


def write_csv(path, rows):
    require(rows, '不写空结果CSV')
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: encode(v) if isinstance(v, (list, dict, tuple)) else 'UNKNOWN' if v is None else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description='从原始冻结记录重算结构统计，不运行策略或GPU')
    parser.add_argument('--out', type=Path, default=BASE / 'results')
    args = parser.parse_args()
    out = args.out.resolve()
    require(out == BASE or BASE in out.parents, '输出必须位于本次独立目录')
    if out == BASE:
        require(not (out / 'natural_workload_per_condition.csv').exists(), '拒绝覆盖已有分析结果')
    else:
        out.mkdir(parents=True, exist_ok=False)
    sources = Sources()
    sources.add(Path(__file__), '本轮独立统计脚本')
    result, agreements, detail, snapshots, max_diff, popmanifest = analyze(sources)
    original, traces = inspect_raw_sources(sources,
        {sid for v in snapshots.values() for sid in v['source_sessions']},
        {v['trace_id'] for v in snapshots.values() if v['trace_id']})
    replacements = []
    grouped = defaultdict(list)
    for r in result:
        grouped[(r['population'], r['snapshot_id'])].append(r)
    for (pop, sid), values in grouped.items():
        values.sort(key=lambda r: (r['K'], r['budget_ratio']))
        for a, b in itertools.combinations(values, 2):
            same_input = a['input_without_budget_sha256'] == b['input_without_budget_sha256']
            for policy in POLICIES:
                sa, sb = set(a['selection_by_policy'][policy]), set(b['selection_by_policy'][policy])
                replacements.append({'population': pop, 'snapshot_id': sid, 'policy': policy,
                                     'condition_low': a['condition_id'], 'condition_high': b['condition_id'],
                                     'K_low': a['K'], 'K_high': b['K'], 'same_input_except_budget': same_input,
                                     'strictly_increased_K': a['K'] < b['K'], 'selected_low': sorted(sa), 'selected_high': sorted(sb),
                                     'nested': sa <= sb if same_input else 'UNKNOWN', 'removed': sorted(sa - sb) if same_input else [],
                                     'added': sorted(sb - sa) if same_input else [], 'same_set': sa == sb,
                                     'shared_snapshot': a['shared_positive_candidate_count'] > 0})
    metrics = []
    for population in ['OpenHands', 'AgentX', 'ALL']:
        ss = [r for r in result if population == 'ALL' or r['population'] == population]
        rr = [r for r in replacements if population == 'ALL' or r['population'] == population]
        snap = {r['snapshot_id']: r for r in ss}
        boolean_fields = ['all_four_same_set', 'all_four_same_order', 'all_four_same_J', 'all_four_same_E_G',
                          'K_below_K_min', 'K_below_positive_candidates', 'budget_underfilled', 'full_frontier_reached',
                          'marginal_suppression_observed', 'strict_singleton_to_marginal_rank_reversal']
        entries = {'conditions': len(ss), 'snapshots': len(snap),
                   'conditions_with_shared_positive_candidates': sum(r['shared_positive_candidate_count'] > 0 for r in ss),
                   'conditions_with_overlap': sum(r['overlapping_positive_pairs'] > 0 for r in ss),
                   'conditions_with_dominance': sum(r['strictly_dominated_positive_candidates'] > 0 for r in ss),
                   'conditions_with_similar_benefits_5pct': sum(r['similar_benefit_pairs_5pct'] > 0 for r in ss),
                   'conditions_with_equal_candidate_vectors': sum(r['equal_benefit_candidate_pairs'] > 0 for r in ss),
                   'conditions_with_top_ties': sum(r['greedy_top_tie_steps'] > 0 for r in ss),
                   'conditions_with_unique_minimum_optimum': sum(r['minimum_cardinality_optimum_count'] == 1 for r in ss),
                   'conditions_with_slack_equal_J_alternative': sum(bool(r['slack_equal_J_alternative_witness']) for r in ss),
                   'conditions_with_same_size_equal_J_alternative': sum(bool(r['same_size_equal_J_alternative_witness']) for r in ss),
                   'conditions_with_unselected_positive': sum(r['unselected_positive_candidates'] > 0 for r in ss),
                   'conditions_with_unselected_positive_marginal': sum(r['unselected_positive_marginal_candidates'] > 0 for r in ss),
                   'snapshots_with_node_alias': sum(r['physical_node_alias_excess'] > 0 for r in snap.values()),
                   'budget_policy_pairs': len(rr), 'nonnested_budget_policy_pairs': sum(r['nested'] is False for r in rr),
                   'snapshots_with_any_replacement': len({r['snapshot_id'] for r in rr if r['nested'] is False}),
                   'budget_insensitive_snapshots': sum(len({tuple(sorted(v['selection_by_policy']['FlowState'])) for v in ss if v['snapshot_id'] == sid}) == 1 for sid in snap)}
        entries.update({field + '_conditions': sum(bool(r[field]) for r in ss) for field in boolean_fields})
        for key, value in entries.items():
            metrics.append({'population': population, 'metric': key, 'value': value,
                            'unit': '快照数' if 'snapshots' in key or key == 'snapshots' else '比较行数' if 'pairs' in key else '条件数',
                            'denominator_conditions': len(ss), 'denominator_snapshots': len(snap),
                            'evidence_level': 'FROZEN_FORMAL_NOT_STRICT_RUNTIME'})
    write_csv(out / 'natural_workload_per_condition.csv', result)
    write_csv(out / 'natural_workload_structure_summary.csv', metrics)
    write_csv(out / 'strategy_agreement_analysis.csv', agreements)
    write_csv(out / 'budget_replacement_analysis.csv', replacements)
    with (out / 'calculation_evidence.jsonl').open('x') as f:
        for row in detail:
            f.write(encode(row) + '\n')
    evidence = {'parameters': {'epsilon_ms': EPS, 'similar_relative_threshold': SIMILARITY,
                              'policies': POLICIES, 'source_selection': '原573条件全量，不按结果筛选'},
                'original_openhands_sessions': original, 'original_agentx_traces': traces,
                'snapshot_input_fingerprints': {k: v['input_sha'] for k, v in snapshots.items()},
                'max_J_recalculation_error_ms': max_diff, 'source_files': sources.finish(),
                'row_counts': {'conditions': len(result), 'agreements': len(agreements), 'budget_pairs': len(replacements)},
                'status': 'PASS_FORMAL_STATISTICS_BRANCH_EVIDENCE_INCOMPLETE'}
    with (out / 'source_manifest.json').open('x') as f:
        f.write(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(encode({'output': str(out), 'rows': evidence['row_counts'], 'sources': len(evidence['source_files']),
                  'original_sessions': len(original), 'original_traces': len(traces), 'max_J_error_ms': max_diff}))


if __name__ == '__main__':
    main()
