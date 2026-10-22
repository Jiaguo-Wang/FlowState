"""不可变严格分配快照与纯规划结果；不提供任何物理执行能力。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import array
import hashlib
import json

from .state_catalog import CheckpointCandidate


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _tokens_digest(tokens: tuple[int, ...]) -> str:
    return hashlib.sha256(array.array('q', tokens).tobytes()).hexdigest()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class StrictCandidate:
    """带原始令牌身份和边界身份的逻辑候选。"""
    candidate: CheckpointCandidate
    token_ids: tuple[int, ...]
    token_digest: str
    epoch_id: str
    configuration_digest: str


@dataclass(frozen=True)
class StrictPending:
    """当前边界已准备的目标输入，不包含未来执行结果。"""
    pending_id: str
    workflow_id: str
    target: int
    token_ids: tuple[int, ...]
    token_digest: str
    epoch_id: str
    configuration_digest: str


@dataclass(frozen=True)
class StrictEdge:
    """保存准入三态及证据条件；未知状态在分配入口被拒绝。"""
    candidate_id: str
    pending_id: str
    status: str
    epoch_id: str
    conditions: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class PendingPlan:
    """仅记录模型前沿，不表示实际请求命中或恢复成功。"""
    pending_id: str
    T: int
    E: int
    G: int
    cost_ms: float


@dataclass(frozen=True)
class StrictAllocationPlan:
    """纯逻辑计划，故意不包含 Runtime 句柄或执行方法。"""
    epoch_id: str
    snapshot_sha256: str
    epoch_digest: str
    K: int
    selected_ids: tuple[str, ...]
    selected_count: int
    pending: tuple[PendingPlan, ...]
    recovery_cost_ms: float
    benefit_ms: float
    logical_eviction_ids: tuple[str, ...]


@dataclass(frozen=True)
class StrictAllocationEpoch:
    """消费已验证来源的完整严格矩阵，不自动认证物理状态。"""
    epoch_id: str
    snapshot_sha256: str
    configuration_digest: str
    candidates: tuple[StrictCandidate, ...]
    pendings: tuple[StrictPending, ...]
    edges: tuple[StrictEdge, ...]
    K: int
    content_digest: str

    def __post_init__(self):
        self.validate()

    def validate(self):
        """所有非法、缺失、混域或未知输入均在选择前拒绝。"""
        _require(isinstance(self.epoch_id, str) and bool(self.epoch_id), '缺少 Epoch 身份')
        _require(_hash(self.snapshot_sha256) and _hash(self.configuration_digest), '来源或配置摘要非法')
        _require(type(self.K) is int and self.K >= 0, '逻辑预算非法')
        _require(type(self.candidates) is tuple and type(self.pendings) is tuple and type(self.edges) is tuple, '严格输入必须不可变')
        _require(all(type(x) is StrictCandidate for x in self.candidates) and all(type(x) is StrictPending for x in self.pendings) and all(type(x) is StrictEdge for x in self.edges), '不得混用 Legacy 与 Strict 对象')
        ids=[]
        for c in self.candidates:
            _require(type(c.candidate) is CheckpointCandidate, '候选类型非法')
            _require(type(c.candidate.lineage_path) is tuple and all(isinstance(v, str) for v in c.candidate.lineage_path), '候选谱系必须不可变')
            _require(isinstance(c.candidate.workflow_id, str) and bool(c.candidate.workflow_id), '候选 Workflow 身份缺失')
            ids.append(c.candidate.checkpoint_id)
        pids=[p.pending_id for p in self.pendings]
        _require(all(isinstance(x,str) and x for x in ids+pids), '候选或 Pending ID 为空')
        _require(len(set(ids))==len(ids), 'Candidate ID 冲突')
        _require(len(set(pids))==len(pids), 'Pending ID 冲突')
        for x in self.candidates+self.pendings:
            _require(x.epoch_id==self.epoch_id, 'Snapshot/Epoch 身份不一致')
            _require(x.configuration_digest==self.configuration_digest, '配置身份不一致')
            _require(type(x.token_ids) is tuple and all(type(t) is int and 0<=t<2**63 for t in x.token_ids), '原始 Token 非法')
            _require(_tokens_digest(x.token_ids)==x.token_digest, 'Token 输入摘要不一致')
        for c in self.candidates:
            _require(type(c.candidate.token_pos) is int and c.candidate.token_pos==len(c.token_ids), '候选位置非法')
            _require(c.candidate.recurrent_resident is True and c.candidate.fa_resident is True, '候选状态不完整')
            _require(type(c.candidate.memory_bytes) is int and c.candidate.memory_bytes>0, '候选大小非法')
        _require(len({c.candidate.memory_bytes for c in self.candidates})<=1, '严格模式仅支持等大小候选')
        for p in self.pendings:
            _require(isinstance(p.workflow_id, str) and bool(p.workflow_id), 'Pending Workflow 身份缺失')
            _require(type(p.target) is int and 0<=p.target<=len(p.token_ids), 'Planning Target 位置非法')
        cd={c.candidate.checkpoint_id:c for c in self.candidates};pd={p.pending_id:p for p in self.pendings};seen=set()
        for e in self.edges:
            _require(e.epoch_id==self.epoch_id, '关系 Epoch 身份不一致')
            _require(e.candidate_id in cd and e.pending_id in pd, '关系超出当前域')
            pair=(e.candidate_id,e.pending_id)
            _require(pair not in seen, '配对重复或冲突');seen.add(pair)
            _require(e.status in ('EPOCH_ADMISSIBLE','REJECTED'), 'UNKNOWN 或非法配对状态')
            _require(type(e.conditions) is tuple and all(type(v) is tuple and len(v)==2 and isinstance(v[0],str) and type(v[1]) is bool for v in e.conditions), '准入条件不完整')
            checks=dict(e.conditions)
            _require(len(checks)==len(e.conditions) and set(checks)==set(_CONDITIONS), '准入证据条件缺失或重复')
            c=cd[e.candidate_id];p=pd[e.pending_id];l=c.candidate.token_pos
            _require(checks['L_le_T']==(l<=p.target) and checks['exact_token_prefix']==(c.token_ids==p.token_ids[:l]), '准入位置或 Token 条件冲突')
            _require(e.status==('EPOCH_ADMISSIBLE' if all(checks.values()) else 'REJECTED'), '准入状态与证据冲突')
        _require(len(seen)==len(ids)*len(pids), '缺失配对')
        payload=asdict(self);payload.pop('content_digest')
        _require(self.content_digest==_digest(payload), 'Epoch 内容摘要不一致')

    @classmethod
    def from_snapshot_bytes(cls, data: bytes, *, expected_sha256: str, K: int):
        """读取 B4 冻结格式；调用方必须提供可信来源清单中的摘要。"""
        _require(type(data) is bytes and _hash(expected_sha256), '必须提供原始字节与可信摘要')
        _require(hashlib.sha256(data).hexdigest()==expected_sha256, 'Snapshot 输入摘要不一致')
        def unique(pairs):
            result={}
            for k,v in pairs:
                _require(k not in result, 'JSON 键重复');result[k]=v
            return result
        s=json.loads(data,object_pairs_hook=unique)
        eid=s['barrier_id'];cfg=_digest(s['configuration'])
        cs=[];ps=[];es=[]
        for c in s['candidates']:
            md=c['metadata'];identity=c['state_identity']
            core=CheckpointCandidate(c['checkpoint_id'],c['workflow_id'],tuple(md['lineage_path']),c['token_position_L'],md['memory_bytes'],identity['target_mamba_present'],identity['target_full_present'] and identity['path_full_all_present'])
            cs.append(StrictCandidate(core,tuple(c['token_ids']),c['token_prefix_sha256'],c['barrier_id'],c['configuration_digest']))
        for p in s['pendings']:
            _require(p['controlled_input_ready'] is True, 'Pending 当前输入未准备')
            ps.append(StrictPending(p['continuation_id'],p['workflow_id'],p['planning_target_T'],tuple(p['token_ids']),p['token_prefix_sha256'],p['barrier_id'],p['configuration_digest']))
        for e in s['edges']:
            _require(e.get('barrier_id', eid)==eid, '关系 Snapshot/Epoch 身份不一致')
            es.append(StrictEdge(e['checkpoint_id'],e['continuation_id'],e['status'],eid,tuple(sorted(e['conditions'].items()))))
        payload=dict(epoch_id=eid,snapshot_sha256=expected_sha256,configuration_digest=cfg,candidates=tuple(cs),pendings=tuple(ps),edges=tuple(es),K=K)
        canonical={k:tuple(asdict(x) for x in v) if k in ('candidates','pendings','edges') else v for k,v in payload.items()}
        return cls(**payload,content_digest=_digest(canonical))

    def frontiers(self, selected_ids):
        """只使用显式准入边，完全不调用 Workflow 或 Lineage 谓词。"""
        selected=tuple(selected_ids);positions={c.candidate.checkpoint_id:c.candidate.token_pos for c in self.candidates}
        _require(len(set(selected))==len(selected) and set(selected)<=set(positions), '选择包含重复或域外候选')
        edges={(e.candidate_id,e.pending_id) for e in self.edges if e.status=='EPOCH_ADMISSIBLE'}
        return tuple(max([positions[c] for c in selected if (c,p.pending_id) in edges]+[0]) for p in self.pendings)


_CONDITIONS=('candidate_token_identity','pending_token_identity','L_le_T','exact_token_prefix','same_configuration','same_boundary','known_pending','full_path_resident','target_full_resident','recurrent_resident','complete_components','references_clear','no_slot_alias','physical_identity')
