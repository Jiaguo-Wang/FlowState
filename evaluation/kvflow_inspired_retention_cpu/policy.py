"""独立的KVFlow启发式；只消费声明的调用图与固定前缀，不读取恢复收益。"""
from math import inf

class IncompleteEvidence(ValueError):
    """证据不足时禁止生成可比较的选集。"""


def index_unique(rows, key):
    result = {}
    for row in rows:
        name = row[key]
        if not isinstance(name, str) or not name or name in result:
            raise ValueError('身份为空或重复')
        result[name] = row
    return result


def compute_ste(nodes, epoch_id):
    """未知依赖或未解循环保持未知，不把离线未来事件当成当前声明。"""
    indexed = index_unique(nodes, 'invocation_id')
    values = {name: None for name in indexed}
    for name, row in indexed.items():
        if row.get('epoch_id') != epoch_id or not row.get('evidence_path') or row.get('known_at_epoch') is not True:
            continue
        if row.get('ready') is True:
            values[name] = 0
    for _ in indexed:
        changed = False
        for name, row in indexed.items():
            if values[name] is not None or row.get('epoch_id') != epoch_id or row.get('known_at_epoch') is not True or not row.get('evidence_path'):
                continue
            parents = row.get('predecessors')
            if not parents or len(set(parents)) != len(parents) or row.get('aggregation') not in ('AND', 'OR'):
                continue
            upstream = [values.get(p) for p in parents]
            if any(v is None for v in upstream):
                continue
            values[name] = (max(upstream) if row['aggregation'] == 'AND' else min(upstream)) + 1
            changed = True
        if not changed:
            break
    return values


def prefix(a, b):
    return len(a) <= len(b) and list(a) == list(b[:len(a)])


def priorities(candidates, graph, declarations, epoch_id, configuration_digest):
    """固定末端到祖先的传播等价为逐Token前缀包含；动态分类必须有声明。"""
    indexed = index_unique(candidates, 'checkpoint_id')
    ste = compute_ste(graph, epoch_id)
    declared = index_unique(declarations, 'invocation_id')
    if set(declared) != set(ste):
        raise IncompleteEvidence('调用图与固定前缀声明域不完整')
    for name, d in declared.items():
        if d.get('epoch_id') != epoch_id or d.get('configuration_digest') != configuration_digest or d.get('known_at_epoch') is not True or not d.get('evidence_path'):
            raise IncompleteEvidence('固定前缀身份或当时可见性缺失')
        fixed, full = d.get('fixed_token_ids'), d.get('input_token_ids')
        if not fixed or not full or not prefix(fixed, full):
            raise IncompleteEvidence('固定前缀不是实际输入前缀')
    output = []
    for name, c in indexed.items():
        tokens = c.get('token_ids')
        if not tokens or len(tokens) != c.get('token_position_L') or c.get('barrier_id') != epoch_id or c.get('configuration_digest') != configuration_digest:
            raise IncompleteEvidence('候选Token、位置或Epoch身份不一致')
        if any(type(t) is not int or t < 0 for t in tokens):
            raise IncompleteEvidence('候选Token身份非法')
        owners = [a for a, d in declared.items() if prefix(tokens, d['fixed_token_ids'])]
        dynamic = [a for a, d in declared.items() if d.get('dynamic_suffix_confirmed') is True and len(tokens) > len(d['fixed_token_ids']) and prefix(d['fixed_token_ids'], tokens) and prefix(tokens, d['input_token_ids'])]
        if owners:
            vals = [ste[a] for a in owners]
            value = None if any(v is None for v in vals) else min(vals)
            reason = '固定前缀祖先；关联调用STE取最小值'
        elif dynamic:
            value = inf
            reason = '已声明动态后缀；最先淘汰'
        else:
            value = None
            reason = '缺少固定末端或动态后缀映射证据'
        output.append({'checkpoint_id': name, 'agent_ste': {a: ste[a] for a in owners + dynamic}, 'priority': value, 'reason': reason})
    return output


def select(candidates, graph, declarations, epoch_id, configuration_digest, k):
    """同分按Checkpoint ID升序；这是显式适配规则，不冒称官方同分实现。"""
    if type(k) is not int or not 0 <= k <= len(candidates):
        raise ValueError('预算必须为候选数量范围内的整数')
    rows = priorities(candidates, graph, declarations, epoch_id, configuration_digest)
    if any(r['priority'] is None for r in rows):
        raise IncompleteEvidence('存在UNKNOWN优先级，不能以ID排序补齐合法选集')
    return tuple(r['checkpoint_id'] for r in sorted(rows, key=lambda r: (r['priority'], r['checkpoint_id']))[:k]), rows
