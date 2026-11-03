"""真实A/B截断输入、准备边界及只读物理证据合同，不提供Runtime执行能力。"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / 'evaluation/multishare_cpu_input_gate_b5c2'
OUT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def causal_cut(messages, target_turn=3):
    """仅以角色定位请求，不读取目标Assistant内容或之后消息。"""
    require(type(target_turn) is int and target_turn > 0, '目标轮次非法')
    history = []
    turn = 0
    for index, message in enumerate(messages):
        role = message.get('role')
        require(role in {'system', 'user', 'assistant', 'tool'}, '消息角色不支持')
        if role == 'assistant':
            turn += 1
            if turn == target_turn:
                return history, index
        history.append(copy.deepcopy(message))
    raise ValueError('固定来源缺少目标请求，不允许补位')


def tool_causality(history):
    """工具结果必须关联同一截断范围内已经出现的唯一调用。"""
    seen = {}
    completed = set()
    links = []
    for i, message in enumerate(history):
        calls = message.get('tool_calls_json')
        if calls:
            calls = json.loads(calls) if isinstance(calls, str) else calls
            require(message['role'] == 'assistant' and isinstance(calls, list), '工具调用位置非法')
            for call in calls:
                cid = call.get('id')
                require(cid and cid not in seen, '工具调用ID缺失或重复')
                seen[cid] = i
        if message['role'] == 'tool':
            cid = message.get('tool_call_id')
            require(cid in seen and seen[cid] < i and cid not in completed, '工具结果因果关系非法')
            completed.add(cid)
            links.append({'tool_call_id': cid, 'call_index': seen[cid], 'result_index': i})
    require(set(seen) == completed, '输入仍有未完成工具依赖')
    return links


def input_histories(history):
    """构造前两次调用与第三次Pending的完整输入消息，不含其待生成输出。"""
    result = []
    for i, message in enumerate(history):
        if message['role'] == 'assistant':
            result.append(copy.deepcopy(history[:i]))
    result.append(copy.deepcopy(history))
    require(len(result) == 3, '固定原型必须恰有两次历史调用和一个Pending')
    return result


def frozen_normalizer(config):
    """只加载已锁定的原函数，避免导入Runtime依赖。"""
    path = ROOT / 'evaluation/openhands_single_workflow_smoke.py'
    require(sha(path) == config['files'][str(path)], '规范化源码与冻结身份不一致')
    tree = ast.parse(path.read_text())
    names = {'_parse_tool_calls', 'normalize_message'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == 2, '规范化函数缺失')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    scope = {'json': json, 'Mapping': Mapping}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope['normalize_message']


@dataclass(frozen=True)
class PreparedInput:
    """不可变的当前完整输入；不含GPU物理句柄或未来轨迹。"""
    request_id: str
    workflow_id: str
    token_ids: tuple[int, ...]
    token_sha256: str
    configuration_digest: str

    def validate(self):
        require(isinstance(self.request_id, str) and self.request_id and isinstance(self.workflow_id, str) and self.workflow_id, '请求身份缺失')
        require(type(self.token_ids) is tuple and bool(self.token_ids) and all(type(t) is int and t >= 0 for t in self.token_ids), 'Token输入非法')
        require(hashlib.sha256(json.dumps(list(self.token_ids), separators=(',', ':')).encode()).hexdigest() == self.token_sha256, 'Token顺序或摘要不符')
        require(isinstance(self.configuration_digest, str) and len(self.configuration_digest) == 64, '配置摘要缺失')


class PreparedPendingGate:
    """仅管理输入与发送许可；回调阶段不等同已执行GPU操作。"""
    def __init__(self, requests, journal=None, mode='CPU_PREPARATION'):
        require(mode in {'CPU_PREPARATION', 'CPU_CONTRACT_TEST', 'RUNTIME'}, '执行语义非法')
        self.requests = tuple(requests)
        require(len({r.request_id for r in self.requests}) == len(self.requests) and len(self.requests) == 2, 'Pending域不完整或重复')
        for request in self.requests:
            request.validate()
        require(len({r.configuration_digest for r in self.requests}) == 1, 'Pending配置混用')
        self.stage = 'UNPREPARED'
        self.sent = set()
        self.events = []
        self.mode = mode
        self.journal = Path(journal) if journal else None
        if self.journal:
            with self.journal.open('x'):
                pass

    def _event(self, event, **details):
        row = {'event': event, 'execution_mode': self.mode, 'monotonic_ns': time.monotonic_ns(), 'unix_ns': time.time_ns(), **details}
        if self.journal:
            with self.journal.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.events.append(row)

    def prepare(self):
        require(self.stage == 'UNPREPARED', '不得重复准备')
        for request in self.requests:
            request.validate()
            self._event('INPUT_PREPARED', request_id=request.request_id, token_sha256=request.token_sha256, token_count=len(request.token_ids))
        self.stage = 'PREPARED_NOT_SENT'
        self._event('PENDING_SET_FROZEN', pending_ids=[r.request_id for r in self.requests], pending_digest=digest([asdict(r) for r in self.requests]), dispatch_count=0, agent_native_ready=False)

    def record_allocation(self, epoch, plan, runtime_scope):
        require(self.stage == 'PREPARED_NOT_SENT' and self.mode != 'CPU_PREPARATION', '当前阶段不允许记录已执行分配')
        from flowstate.strict_control import validate_plan
        validate_plan(epoch, plan)
        expected = {r.request_id: r for r in self.requests}
        require(set(expected) == {p.pending_id for p in epoch.pendings}, '分配Pending域与准备集合不符')
        for p in epoch.pendings:
            require(p.token_ids == expected[p.pending_id].token_ids and p.configuration_digest == expected[p.pending_id].configuration_digest, '分配输入或配置不符')
        require(runtime_scope.get('epoch_id') == epoch.epoch_id and runtime_scope.get('live') is True and runtime_scope.get('scope') == 'AB_ONLY', '分配缺少当前A/B状态证据')
        from flowstate.strict_control import bind, preflight
        view = runtime_scope.get('physical_view')
        require(view is not None, '不能仅凭live标记认证物理状态')
        binding = bind(epoch, plan, view, view.lease_id)
        preflight(epoch, plan, binding, view, view.lease_id)
        self.epoch, self.plan, self.before, self.binding = epoch, plan, view, binding
        self.epoch_digest = epoch.content_digest
        self.plan_digest = digest(asdict(plan))
        self.stage = 'ALLOCATED'
        self._event('ALLOCATION_RECORDED', epoch_id=epoch.epoch_id, epoch_digest=self.epoch_digest, plan_digest=self.plan_digest, runtime_scope={k: v for k, v in runtime_scope.items() if k != 'physical_view'})

    def record_reconciliation(self, receipt, after=None):
        require(self.stage == 'ALLOCATED', '状态协调阶段非法')
        require(receipt.get('status') == 'PASS' and receipt.get('phase') == 'complete' and receipt.get('epoch_digest') == self.epoch_digest and receipt.get('plan_digest') == self.plan_digest and receipt.get('postcheck') == 'PASS', '协调证据缺失、失败或身份不符')
        from flowstate.strict_control import postcheck
        require(after is not None, '缺少实际协调后物理观察')
        require(tuple(receipt.get('completed_evictions', ())) == self.plan.logical_eviction_ids, '已完成驱逐集合不符')
        postcheck(self.epoch, self.plan, self.binding, self.before, after, self.plan.logical_eviction_ids, self.before.lease_id)
        self.after = after
        self.stage = 'RECONCILED'
        self._event('RECONCILIATION_RECORDED', receipt=receipt)

    def submit(self, request_id, transport, admission_check=None):
        require(self.stage == 'RECONCILED' and self.mode != 'CPU_PREPARATION', '协调完成前禁止提交')
        require(request_id not in self.sent and not self.sent, '本原型仅允许一个正式Pending，其他Pending须新生命周期')
        request = next((r for r in self.requests if r.request_id == request_id), None)
        require(request is not None, '域外Pending')
        request.validate()
        require(callable(admission_check), '缺少Admission前新鲜度检查')
        current = admission_check(request)
        require(current == self.after, 'Admission前物理状态发生变化')
        self.sent.add(request_id)
        self._event('DISPATCH_BEGIN', request_id=request_id, token_sha256=request.token_sha256)
        try:
            result = transport(request)
        except BaseException as exc:
            self.stage = 'FAILED'
            self._event('DISPATCH_FAILED', request_id=request_id, error=repr(exc))
            raise
        self.stage = 'SUBMITTED'
        self._event('TRANSPORT_RETURNED', request_id=request_id, native_resume_validated=False)
        return result


def classify_aliases(logical_to_node):
    """保留逻辑来源列表；一个物理Node只计一个预算单位。"""
    require(all(isinstance(cid, str) and type(node) is int and node >= 0 for cid, node in logical_to_node.items()), '映射身份非法')
    groups = {}
    for cid, node in logical_to_node.items():
        groups.setdefault(node, []).append(cid)
    return {'logical_count': len(logical_to_node), 'physical_count': len(groups), 'alias_groups': {str(n): sorted(ids) for n, ids in groups.items() if len(ids) > 1}}


def strict_budget(physical_nodes, K):
    require(type(K) is int and K >= 0, '预算非法')
    require(len(set(physical_nodes)) == len(physical_nodes), '物理候选重复，不能按逻辑请求计数')
    require(len(physical_nodes) >= 2 and 0 < K < len(physical_nodes), '没有非平凡保留选择')
    return {'K': K, 'unique_physical_candidates': len(physical_nodes), 'eviction_count': len(physical_nodes) - K}


def verify_replay_scope(recorded_history, expected_history, *, live):
    require(live is True, '历史状态不能作为当前物理绑定')
    require(tuple(recorded_history) == tuple(expected_history), '历史回放域或顺序不符')
