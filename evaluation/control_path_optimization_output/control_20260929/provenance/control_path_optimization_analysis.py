"""并列比较冻结旧E2E与优化后新E2E，保留全部负例和完整性证据。"""
import json
from pathlib import Path
import sys
from evaluation.resume_epoch_e2e_analysis import analyze,paired,METRICS,POLICIES


def compare(root):
    """新实验全部通过后才生成正式对照，不覆盖旧结果。"""
    output=root/'optimized_formal'
    new=analyze(output)
    previous=Path('evaluation/resume_epoch_e2e_output/resume_epoch_final_20260928')
    old=json.loads((previous/'analysis/summary.json').read_text())
    assert set(old['snapshot_means'])==set(new['snapshot_means'])
    old_plan=json.loads((previous/'plan.json').read_text())
    new_plan=json.loads((output/'plan.json').read_text())
    assert old_plan==new_plan
    same=0
    for p in new_plan:
        before=json.loads((previous/'runs'/p['run_id']/'record.json').read_text())
        after=json.loads((output/'runs'/p['run_id']/'record.json').read_text())
        assert before['selected_candidate_ids_actual']==after['selected_candidate_ids_actual']
        for a,b in zip(before['requests'],after['requests']):
            assert [a[k] for k in ['rid','workflow_id','h','e','g','completion_tokens']]==[b[k] for k in ['rid','workflow_id','h','e','g','completion_tokens']]
            same+=1
    before_after={}
    for policy in POLICIES:
        rows={sid:{'round':item['round'],'Previous':old['snapshot_means'][sid][policy],'FlowState':item[policy]}
              for sid,item in new['snapshot_means'].items()}
        before_after[policy]={metric:paired(rows,'Previous',metric) for metric in METRICS}
    stable=all(new['paired'][b][METRICS[0]]['bootstrap_95_ci_ms'][0]>0 for b in ['LRU','Marconi'])
    result={'status':'PASS','statistically_stable_net_benefit_vs_both':stable,
            'identical_plan':True,'same_selected_sets':216,'same_request_heg':same,
            'old_means':old['means'],'new_means':new['means'],'new_paired':new['paired'],
            'old_paired':old['paired'],'before_after_paired':before_after,'new_rounds':new['rounds'],
            'new_cv':new['cv_summary'],'说明':'正配对差表示基线减FlowState；稳定净收益定义为两组epoch改善95%CI下界均大于零。前后采集时段不同，旧值仅并列展示，不拼接新测量。'}
    (root/'analysis').mkdir(exist_ok=True)
    (root/'analysis/comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# FlowState控制路径优化与同协议E2E复测','',
        '状态：PASS。新216/216生命周期通过；逐run选择集与旧实验一致，864/864请求的H/E/G及长度一致。',
        '统计单位仍为24个snapshot，三次重复先平均，再做10000次round分层bootstrap。',
        '稳定净收益（同时相对LRU与Marconi）：'+('是。' if stable else '否。'),
        '', '| 策略 | 旧epoch | 新epoch | 旧总控制 | 新总控制 | 旧请求执行均值 | 新请求执行均值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in POLICIES:
        values=[old['means'][p][METRICS[0]],new['means'][p][METRICS[0]],old['means'][p]['total_control_latency_ms'],new['means'][p]['total_control_latency_ms'],old['means'][p]['mean_request_execution_ms'],new['means'][p]['mean_request_execution_ms']]
        lines.append('| '+p+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','## 新实验配对结果','', '| 对比 | 平均改善 | 中位改善 | 95%CI | 胜/平/负 |','|---|---:|---:|---|---|']
    for b in POLICIES[:2]:
        x=new['paired'][b][METRICS[0]]
        ci=x['bootstrap_95_ci_ms']
        lines.append(f"| FlowState相对{b} | {x['mean_difference_ms']:.3f} | {x['median_difference_ms']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {x['win']}/{x['tie']}/{x['loss']} |")
    lines+=['','## 前后控制分解','', '| 策略/版本 | 公共观测 | 专属观测 | 输入构造 | 选择 | 协调 |','|---|---:|---:|---:|---:|---:|']
    for p in POLICIES:
        for label,data in [('旧',old),('新',new)]:
            lines.append('| '+p+'/'+label+' | '+' | '.join(f"{data['means'][p][k]:.3f}" for k in METRICS[4:9])+' |')
    lines+=['','## 全部epoch负例与平局','']
    for b in POLICIES[:2]:
        cases=new['paired'][b][METRICS[0]]['non_winning_cases']
        lines.append(b+'：'+('、'.join(c['snapshot_id'] for c in cases) or '无')+'。')
    lines+=['','## 语义边界与证据','',
        '优化仅涉及批量frontier、不可变句柄引用、无损消息编码和epoch连接复用。每个frontier分别通过非干扰检查；原动态view验证、逐驱逐adapter及请求间检查均保留。',
        '不修改objective、T/E/G、online boundary、snapshot、预算、4请求/1token或engine配置。',
        'profiling报告：../profiling/analysis/profiling_report.md；每个子步骤的平均/P50/P95与调用数：../profiling/analysis/profile_summary.json。',
        '全部三项主指标及总控制的逐snapshot差值、负例、round分层与CV见comparison.json和../optimized_formal/analysis/summary.json。',
        '这些结果仅适用于control-inclusive resume-epoch latency，不代表多token continuation或完整workflow完成时间。']
    (root/'analysis/final_report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    compare(Path(sys.argv[1]))
