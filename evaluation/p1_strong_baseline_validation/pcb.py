"""严格实现 P1 Section B1，不读取其他策略的选择结果。"""
from common import Epoch
from flowstate.recovery_model import RecoveryCostModel

def select(epoch:Epoch):
    """独立推荐、静态收益排序、按唯一 checkpoint ID 计费。"""
    model=RecoveryCostModel();recommendations=[]
    for p in epoch.pending:
        choices=[]
        for c in epoch.candidates:
            if p.identifier in c.compatible:
                value=model.estimate(p.target,p.target)-model.estimate(p.target-c.position,p.target)
                if value>0:choices.append((-value,c.identifier))
        if choices:
            negative,cid=min(choices)
            recommendations.append(dict(pending_id=p.identifier,checkpoint_id=cid,score=-negative))
    recommendations.sort(key=lambda r:(-r['score'],r['checkpoint_id'],r['pending_id']))
    selected=[]
    for r in recommendations:
        if len(selected)==epoch.k:break
        if r['checkpoint_id'] not in selected:selected.append(r['checkpoint_id'])
    return tuple(selected),dict(recommendations=recommendations)
