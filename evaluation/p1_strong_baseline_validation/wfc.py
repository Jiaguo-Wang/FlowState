"""严格实现 P1 Section B3，按前沿改善数、token 推进和 ID 选择。"""
from common import Epoch

def select(epoch:Epoch):
    """动态更新 E；不读取或计算 Phi、恢复成本或其他策略输出。"""
    frontiers={p.identifier:0 for p in epoch.pending};selected=[];steps=[]
    while len(selected)<epoch.k:
        choices=[]
        for c in epoch.candidates:
            if c.identifier in selected:continue
            advances={p.identifier:max(0,c.position-frontiers[p.identifier]) if p.identifier in c.compatible else 0 for p in epoch.pending}
            q=sum(v>0 for v in advances.values());r=sum(advances.values())
            choices.append(dict(checkpoint_id=c.identifier,q=q,r=r,advances=advances))
        choices.sort(key=lambda x:(-x['q'],-x['r'],x['checkpoint_id']))
        if not choices or choices[0]['q']==0:break
        winner=choices[0]
        steps.append(dict(prefix=list(selected),E_before=dict(frontiers),ranked_candidates=choices,chosen=winner['checkpoint_id'],q=winner['q'],r=winner['r']))
        selected.append(winner['checkpoint_id'])
        for pid,advance in winner['advances'].items():frontiers[pid]+=advance
    return tuple(selected),dict(steps=steps,final_frontiers=frontiers)
