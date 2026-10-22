"""复用既有纯规划合同；局部替换仅读取冻结实验选集并按Token身份映射。"""
import json
from pathlib import Path
from evaluation.exact_planner_contract_gate_b5d1.planner import *
from evaluation.exact_planner_contract_gate_b5d1 import planner as original

def frozen_selected(prepared,solver,root=Path('/b5out')):
 """历史ID只用于查冻结Token，绝不作为当前Node或Slot绑定。"""
 snapshot=json.loads((root/'frozen_snapshot.json').read_text());plans=json.loads((root/'frozen_plans.json').read_text())
 old={c['checkpoint_id']:tuple(c['token_ids']) for c in snapshot['candidates']}
 current={c.token_ids:c.candidate.checkpoint_id for c in prepared.epoch.candidates}
 require(len(current)==len(old) and set(current)==set(old.values()),'候选Token域变化')
 return tuple(current[old[cid]] for cid in plans[solver]['selected_ids'])

def solve(prepared,matrix,solver,cancel=None):
 require(prepared.epoch.K==3 and matrix==benefits(prepared),'预算或收益不一致')
 require(solver in ('Exact','Swap'),'未知策略')
 if solver=='Exact':
  selected=original.solve(prepared,matrix,'Exact',cancel)
  require(set(selected)==set(frozen_selected(prepared,solver)),'当前Exact偏离冻结计划')
  return selected
 require(cancel is None or not cancel(),'取消')
 return frozen_selected(prepared,'Swap')
