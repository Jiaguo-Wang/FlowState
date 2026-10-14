"""固定少量解析前缀树，调用冻结选择器并穷举所有可行集合。"""
import hashlib
import itertools
import json
import sys
from pathlib import Path
from dataclasses import asdict, replace
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'evaluation/p1_strong_baseline_validation'
sys.path.insert(0,str(BASE))
import common
import wfc
from flowstate.optimizer import GlobalOptimizer
from flowstate.state_catalog import CheckpointCandidate
from flowstate.recovery_model import RecoveryCostModel
MODEL=RecoveryCostModel()
EPS=1e-9

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

class MatrixOptimizer(GlobalOptimizer):
    """仅替换成本接口，完整复用原贪心循环及同分规则。"""
    def __init__(self,epoch):
        super().__init__(MODEL);self.epoch=epoch
    def _recovery_cost(self,continuations,selected):
        return common.score(self.epoch,tuple(c.checkpoint_id for c in selected))['cost_ms']

def construct(spec,k):
    """先建立实际整数Token串，随后逐项前缀比较生成兼容集合。"""
    shared=spec['shared']
    pending={p:[10]*shared+[20+i]*(t+1-shared) for i,(p,t) in enumerate(spec['targets'].items())}
    tokens={cid:([10]*length if owner=='root' else pending[owner][:length]) for cid,(owner,length) in spec['checkpoints'].items()}
    cs=[]
    for cid,seq in tokens.items():
        compatible=tuple(p for p,x in pending.items() if len(seq)<=spec['targets'][p] and x[:len(seq)]==seq)
        cs.append(common.Candidate(cid,len(seq),compatible))
    epoch=common.Epoch(tuple(cs),tuple(common.Pending(p,t) for p,t in spec['targets'].items()),k)
    return epoch,{'pending_token_ids':pending,'checkpoint_token_ids':tokens,
        '状态语义':'每个候选代表该Token前缀执行完毕后的完整conv+temporal状态，独立槽位；同模型、位置与配置；无其他可恢复状态。均为解析假设，未实际物化。',
        'allocation_boundary':'所有历史前缀已完成、全部当前输入已知、FA覆盖各T、无活动请求；分析同一固定边界，非真实事件认证'}

def subsets(ids,k):return [s for n in range(k+1) for s in itertools.combinations(sorted(ids),n)]
def analyze(spec,k):
    epoch,tokens=construct(spec,k);bm=common.benefit_matrix(epoch)
    ids=[c.identifier for c in epoch.candidates]
    j0=common.score(epoch,())['cost_ms']
    def f(s):return sum(max([bm[c][i] for c in s]+[0]) for i in range(len(epoch.pending)))
    cs=[CheckpointCandidate(c.identifier,'解析来源'+c.identifier,(),c.position,1) for c in epoch.candidates]
    selected=MatrixOptimizer(epoch).select((),cs,k).selected
    greedy=tuple(c.checkpoint_id for c in selected)
    static=tuple(sorted(sorted([c for c in ids if sum(bm[c])>EPS],key=lambda c:(-sum(bm[c]),c))[:k]))
    wf,wt=wfc.select(epoch)
    allsets=subsets(ids,k);cost={s:common.score(epoch,s)['cost_ms'] for s in allsets}
    minimum=min(cost.values());ties=[s for s,v in cost.items() if abs(v-minimum)<=EPS]
    opt=min(ties,key=lambda s:(len(s),s))
    policies={}
    for name,s in [('FlowState',greedy),('StandaloneTopK',static),('WFC',wf),('Exact OPT',opt)]:
        scored=common.score(epoch,s)
        assert len(set(s))==len(s)<=k
        policies[name]={**scored,'F_ms':f(s),'gap_to_opt_ms':scored['cost_ms']-minimum}
    traces=[];prefix=[]
    for chosen in greedy:
        gain={c:f(prefix+[c])-f(prefix) for c in ids if c not in prefix}
        assert max(gain.values())-gain[chosen]<=EPS
        traces.append({'selected':list(prefix),'gains_ms':gain,'chosen':chosen});prefix.append(chosen)
    for s in subsets(ids,len(ids)):
        assert abs(common.score(replace(epoch,k=len(ids)),s)['cost_ms']+f(s)-j0)<EPS
        for c in ids:
            if c not in s:
                explicit=sum(max(0,bm[c][i]-max([bm[x][i] for x in s]+[0])) for i in range(len(epoch.pending)))
                assert abs(explicit-(f(s+(c,))-f(s)))<EPS
    return {'id':spec['id'],'K':k,'epoch':asdict(epoch),'benefit_matrix_ms':bm,'policies':policies,'flowstate_steps':traces,'wfc_steps':wt,'opt_ties':ties,'checked_subsets':len(allsets)},tokens

def main():
    lock=json.loads((BASE/'implementation_lock.json').read_text())
    for name in ('common.py','wfc.py'):assert sha(BASE/name)==lock[name]
    sources=[OUT/'protocol.json',BASE/'common.py',BASE/'wfc.py',BASE/'implementation_lock.json',ROOT/'flowstate/optimizer.py',ROOT/'flowstate/recovery_model.py',ROOT/'evaluation/rq3_sanity_structure_audit.py']
    sources += [p for p in (ROOT/'evaluation/strict_compatibility_b2').iterdir() if p.is_file()]
    before={str(p):sha(p) for p in sources}
    spec=json.loads((OUT/'protocol.json').read_text());results=[];token_evidence={}
    for case in spec['cases']:
        for k in case['budgets']:
            result,tokens=analyze(case,k);results.append(result);token_evidence[case['id']]=tokens
    write('results.json',{'classification':'ANALYTICAL EXAMPLE','cases':results,'model':asdict(MODEL.metadata)})
    write('analytical_token_trees.json',token_evidence)
    assert before=={str(p):sha(p) for p in sources}
    write('validation.json',{'status':'PASS','case_budget_count':len(results),'checks':'实际Token前缀生成兼容关系；全子集F/J等价与边际一致；穷举Exact；预算；生产贪心逐轮最大边际；冻结实现哈希','source_sha256':before,'gpu_used':False})
    for r in results:print(r['id'],r['K'],[(n,v['selected'],round(v['cost_ms'],6)) for n,v in r['policies'].items()])

if __name__=='__main__':main()
