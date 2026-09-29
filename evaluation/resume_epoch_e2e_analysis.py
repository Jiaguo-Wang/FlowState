"""以快照为配对单位汇总本轮恢复 epoch，不混入旧实验测量。"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import fmean, median, stdev

METRICS = ['control_inclusive_epoch_latency_ms', 'mean_barrier_to_completion_ms',
           'mean_request_execution_ms', 'total_control_latency_ms',
           'common_observation_ms', 'policy_observation_ms', 'input_construction_ms',
           'allocation_ms', 'reconciliation_ms', 'between_request_validation_and_dispatch_ms']
POLICIES = ['LRU','Marconi','FlowState']
TOLERANCE = 1e-6


def percentile(values, p):
    """计算线性插值百分位数。"""
    values = sorted(values)
    index = (len(values)-1)*p
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def paired(rows, baseline, metric, seed=20260928):
    """按 round 分层重采样快照，正数表示 FlowState 改善。"""
    strata = defaultdict(list)
    cases = []
    for sid,item in sorted(rows.items()):
        delta = item[baseline][metric]-item['FlowState'][metric]
        strata[item['round']].append(delta)
        cases.append({'snapshot_id':sid,'round':item['round'],'baseline_minus_flowstate_ms':delta})
    rng = random.Random(seed)
    samples = []
    for _ in range(10000):
        sampled = [rng.choice(group) for group in strata.values() for _ in group]
        samples.append(fmean(sampled))
    return {'mean_difference_ms':fmean(c['baseline_minus_flowstate_ms'] for c in cases),
            'median_difference_ms':median(c['baseline_minus_flowstate_ms'] for c in cases),
            'bootstrap_95_ci_ms':[percentile(samples,.025),percentile(samples,.975)],
            'win':sum(c['baseline_minus_flowstate_ms']>TOLERANCE for c in cases),
            'tie':sum(abs(c['baseline_minus_flowstate_ms'])<=TOLERANCE for c in cases),
            'loss':sum(c['baseline_minus_flowstate_ms'] < -TOLERANCE for c in cases),
            'non_winning_cases':[c for c in cases if c['baseline_minus_flowstate_ms']<=TOLERANCE],
            'cases':cases}


def break_even(item, baseline):
    """以本轮控制差和单请求节省给出恒定成本假设下的盈亏平衡外推。"""
    fs,base = item['FlowState'],item[baseline]
    control = fs['total_control_latency_ms']-base['total_control_latency_ms']
    specific_keys = ['policy_observation_ms','input_construction_ms','allocation_ms']
    specific = sum(fs[k]-base[k] for k in specific_keys)
    saving = base['mean_request_execution_ms']-fs['mean_request_execution_ms']
    residual = fs['between_request_validation_and_dispatch_ms']-base['between_request_validation_and_dispatch_ms']
    def threshold(cost, gain):
        if gain > 0:
            return max(1,math.floor(cost/gain)+1)
        return None
    return {'total_control_difference_ms':control,'policy_specific_control_difference_ms':specific,
            'common_observation_difference_ms':fs['common_observation_ms']-base['common_observation_ms'],
            'reconciliation_difference_ms':fs['reconciliation_ms']-base['reconciliation_ms'],
            'request_saving_ms':saving,'validation_dispatch_difference_ms':residual,
            'policy_specific_break_even_count':threshold(specific,saving),
            'total_control_break_even_count':threshold(control,saving),
            'validation_adjusted_break_even_count':threshold(control,saving-residual/4),
            '说明':'仅外推：控制成本固定、每请求节省及验证差可重复；空值表示正节省条件不满足，非实测新请求数。'}


def analyze(output):
    """要求正式216次全部有效，否则拒绝生成通过结论。"""
    plan = json.loads((output/'plan.json').read_text())
    assert len(plan)==216 and len({p['snapshot_id'] for p in plan})==24
    grouped = defaultdict(list)
    records=[]
    for p in plan:
        r=json.loads((output/'runs'/p['run_id']/'record.json').read_text())
        assert r['status']=='PASS' and all(r['correctness'].values()), p['run_id']
        assert len(r['requests'])==4
        records.append(r)
        grouped[(r['snapshot_id'],r['policy'])].append(r)
    rows={}
    cvs=[]
    for (sid,policy), group in grouped.items():
        assert len(group)==3 and {r['repetition'] for r in group}=={1,2,3}
        rows.setdefault(sid,{'round':group[0]['allocation_round']})[policy]={k:fmean(r[k] for r in group) for k in METRICS}
        for metric in METRICS[:4]:
            values=[r[metric] for r in group]
            cvs.append({'snapshot_id':sid,'policy':policy,'metric':metric,'cv':stdev(values)/fmean(values)})
    primary={p:{k:fmean(item[p][k] for item in rows.values()) for k in METRICS} for p in POLICIES}
    comparisons={b:{k:paired(rows,b,k) for k in METRICS[:4]} for b in POLICIES[:2]}
    rounds={str(round_id):{'means':{p:{k:fmean(item[p][k] for item in rows.values() if item['round']==round_id) for k in METRICS} for p in POLICIES},
              'paired':{b:paired({sid:item for sid,item in rows.items() if item['round']==round_id},b,METRICS[0]) for b in POLICIES[:2]}}
            for round_id in [2,3,4,5]}
    cv_summary={p:{'median':median(c['cv'] for c in cvs if c['policy']==p and c['metric']==METRICS[0]),
                  'p95':percentile([c['cv'] for c in cvs if c['policy']==p and c['metric']==METRICS[0]],.95),
                  'max':max(c['cv'] for c in cvs if c['policy']==p and c['metric']==METRICS[0])} for p in POLICIES}
    breakeven={b:{'overall':break_even(primary,b),'by_snapshot':{sid:break_even(item,b) for sid,item in rows.items()}}
               for b in POLICIES[:2] if comparisons[b][METRICS[0]]['mean_difference_ms']<=0 or comparisons[b][METRICS[0]]['loss']>0}
    result={'status':'PASS','runs':216,'requests':864,'statistical_unit':'snapshot','snapshots':24,
            'means':primary,'paired':comparisons,'rounds':rounds,'cv_summary':cv_summary,
            'cv_cases':cvs,'break_even':breakeven,'snapshot_means':rows,
            '说明':'仅为四个串行单令牌请求的 control-inclusive resume-epoch latency；正配对差表示基线减FlowState。'}
    (output/'analysis').mkdir(exist_ok=True)
    (output/'analysis/summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 包含控制路径的单令牌恢复 epoch 实验','', '状态：PASS。216/216 独立运行与864/864请求通过正确性及清理门禁。',
           '', '统计单位为24个快照；每种策略先对三次重复取均值。正配对差表示基线减FlowState，置信区间为10000次round分层bootstrap。',
           '', '| 策略 | epoch(ms) | 平均boundary至完成(ms) | 平均请求执行(ms) | 总控制(ms) |', '|---|---:|---:|---:|---:|']
    for p in POLICIES:
        lines.append('| '+p+' | '+' | '.join(f'{primary[p][k]:.3f}' for k in METRICS[:4])+' |')
    lines+=['','## 配对结果','']
    for b in POLICIES[:2]:
        x=comparisons[b][METRICS[0]]
        lines.append(f"相对 {b}：平均改善 {x['mean_difference_ms']:.3f} ms；95% CI {x['bootstrap_95_ci_ms']}；胜/平/负 {x['win']}/{x['tie']}/{x['loss']}。")
        lines.append('负例及平局：'+('、'.join(c['snapshot_id'] for c in x['non_winning_cases']) or '无')+'。')
    lines+=['','## 分段均值','', '| 策略 | 公共观测 | 专属观测 | 输入构造 | 选择 | 协调 | 请求间验证及派发 |', '|---|---:|---:|---:|---:|---:|---:|']
    for p in POLICIES:
        lines.append('| '+p+' | '+' | '.join(f'{primary[p][k]:.3f}' for k in METRICS[4:])+' |')
    lines+=['','## 分层与重复稳定性','']
    for rd,values in rounds.items():
        lines.append('round '+rd+'：'+ '；'.join(p+' '+f"{values['means'][p][METRICS[0]]:.3f} ms" for p in POLICIES)+'。')
    for p,c in cv_summary.items():
        lines.append(f"{p} 的epoch重复CV：中位数 {c['median']:.4f}，P95 {c['p95']:.4f}，最大 {c['max']:.4f}。")
    lines+=['','## 盈亏平衡外推','']
    for b,value in breakeven.items():
        lines.append(b+'：'+json.dumps(value['overall'],ensure_ascii=False))
    lines+=['','完整逐快照配对差、各指标负例、分层置信区间及CV见 summary.json；未删除负例、平局或异常值。',
            '盈亏平衡仅使用本轮测量，属于固定控制成本与平均单请求节省可外推的假设，不代表新增请求数量的实测。']
    (output/'analysis/final_report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='恢复epoch统计')
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    analyze(args.output)
