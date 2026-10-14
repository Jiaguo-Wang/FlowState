"""严格实现 P1 Section B2，仅按最深位置静态推荐。"""
from common import Epoch

def select(epoch:Epoch):
    """每个请求推荐最深合法项，以位置、候选 ID、请求 ID 排序并去重。"""
    recommendations=[]
    for p in epoch.pending:
        choices=[(-c.position,c.identifier) for c in epoch.candidates if p.identifier in c.compatible and c.position>0]
        if choices:
            negative,cid=min(choices)
            recommendations.append(dict(pending_id=p.identifier,checkpoint_id=cid,score=-negative))
    recommendations.sort(key=lambda r:(-r['score'],r['checkpoint_id'],r['pending_id']))
    selected=[]
    for r in recommendations:
        if len(selected)==epoch.k:break
        if r['checkpoint_id'] not in selected:selected.append(r['checkpoint_id'])
    return tuple(selected),dict(recommendations=recommendations)
