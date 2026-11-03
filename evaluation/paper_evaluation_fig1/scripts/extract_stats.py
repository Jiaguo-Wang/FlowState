"""VI-B 主实验统计提取与完整性门禁（只读冻结产物，CPU-only）。

数据源：evaluation/c_formal_perf_gpu/ 的 60 个正式测量臂（m[1-5][A-D]_{FS,SA,WFC}），
每臂唯一权威原始文件为 <臂>/run/summary.json（全精度计时）。门禁臂 g1/g2、旧 Pilot、
CPU 模拟与其他计时协议一律不进入统计。

统计口径：
  A1 每策略每 Pending 五轮 TTFT 中位数；
  A2 聚合 TTFT＝每轮四 Pending 等权算术平均 → 五轮均值的中位数；
  A3 同轮配对差（FS−SA、FS−WFC），百分比分"配对口径"与"聚合中位数比值口径"分别列示；
  A4 恢复行为核验（T/E/G、Prefill Tokens 与冻结选集、Native Runtime 一致）；
  A5 完整性门禁，任一失败即报 FAIL 并退出非零，绝不静默过滤。

输出：data/{raw_arms,pending_ttft,aggregate_ttft,paired_comparison,recovery_behavior}.csv
     + checks.json + provenance.json。
"""
import csv,hashlib,json,re,sys,time
from pathlib import Path
from statistics import mean,median

ROOT=Path(__file__).resolve().parents[3]          # 仓库根
CF=ROOT/'evaluation/c_formal_perf_gpu'
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'evaluation/paper_evaluation_fig1'
DATA=OUT/'data'
POLICIES=['FlowState-Greedy','StandaloneTopK','WFC']
SHORT={'FlowState-Greedy':'FS','StandaloneTopK':'SA','WFC':'WFC'}
PENDINGS=['C_A','C_B','C_C','C_D']
ROUNDS=['1','2','3','4','5']
# 冻结参考聚合值（paper_pending_aggregate_audit.md / 任务书），仅作一致性校验，不用于绘图
REF_AGGREGATE={'FlowState-Greedy':77.829,'StandaloneTopK':111.960,'WFC':125.262}
checks=[]
def check(name,ok,detail=None):
    checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':detail})
    return ok

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    # ---------- 原始提取：60 个正式臂的 run/summary.json ----------
    arms=sorted(p for p in CF.iterdir() if re.fullmatch(r'm[1-5][A-D]_(FS|SA|WFC)',p.name))
    raw=[]
    for arm in arms:
        s=json.loads((arm/'run'/'summary.json').read_text())
        raw.append({
            'experiment_id':arm.name,'round':arm.name[1],'pending':'C_'+arm.name[2],
            'policy':s['policy'],'policy_short':SHORT[s['policy']],
            'ttft_ms':s['ttft_ms'],'cuda_prefill_ms':s['cuda_prefill_ms'],
            'prefill_tokens':s['prefill_token_count'],
            'T':s['E_executable_prefix']+s['G_replay_gap'],
            'E':s['E_executable_prefix'],'G':s['G_replay_gap'],'H':s['H_physical_fa_hit'],
            'cold_output_match':s['cold_output_match'],'status':s['status'],
            'source_path':str(arm/'run'/'summary.json'),
            'source_sha256':hashlib.sha256((arm/'run'/'summary.json').read_bytes()).hexdigest()})
    # ---------- A5 完整性门禁 ----------
    check('正式样本数恰好60',len(raw)==60,len(raw))
    ids=[r['experiment_id'] for r in raw]
    check('无重复实验臂',len(set(ids))==len(ids))
    cells={(r['policy'],r['pending']) for r in raw}
    check('策略×Pending 共12格',cells=={(p,q) for p in POLICIES for q in PENDINGS})
    check('每策略每Pending恰五轮',all(sorted(r['round'] for r in raw if r['policy']==p and r['pending']==q)==ROUNDS for p in POLICIES for q in PENDINGS))
    check('全部臂状态PASS',all(r['status']=='PASS' for r in raw))
    check('全部臂Cold参照逐Token一致',all(r['cold_output_match'] for r in raw))
    # A4 恢复行为：T/E/G 与冻结选集合同一致；Prefill=计划E准入+32新Token
    plans=json.loads((CF/'m1A_FS'/'policy_plans.json').read_text())
    ok_eg=ok_pf=True
    for r in raw:
        eg=plans[r['policy']]['expected_E_G'][r['pending']]
        ok_eg&= (r['E'],r['G'],r['T'])==(eg['E'],eg['G'],eg['T'])
        ok_pf&= r['prefill_tokens']==r['G']+32
    check('Native E/G/T 与冻结选集合同逐臂一致',ok_eg)
    check('Prefill Tokens == G+32 逐臂成立',ok_pf)
    check('T 取值恰为四Pending规划目标',sorted({(r['pending'],r['T']) for r in raw})==[(q,t) for q,t in zip(PENDINGS,(4096,5120,6144,7168))])
    # ---------- A1 每策略每 Pending 五轮中位 ----------
    pend_rows=[]
    for p in POLICIES:
        for q in PENDINGS:
            v=[r['ttft_ms'] for r in raw if r['policy']==p and r['pending']==q]
            row={'policy':p,'policy_short':SHORT[p],'pending':q,'n_rounds':len(v),'ttft_median_ms':median(v)}
            row.update({f'ttft_r{r_}':x for r_,x in zip(ROUNDS,v)})
            pend_rows.append(row)
    # ---------- A2 聚合 TTFT：每轮四 Pending 等权均值 → 五轮中位 ----------
    agg_rows=[];round_means={}
    for p in POLICIES:
        rm=[mean([r['ttft_ms'] for r in raw if r['policy']==p and r['round']==r_]) for r_ in ROUNDS]
        round_means[p]=rm
        row={'policy':p,'policy_short':SHORT[p],'aggregate_ttft_ms':median(rm),'n_rounds':len(rm),'round_mean_min_ms':min(rm),'round_mean_max_ms':max(rm)}
        row.update({f'round_mean_r{r_}':x for r_,x in zip(ROUNDS,rm)})
        agg_rows.append(row)
    for p in POLICIES:
        agg=next(x['aggregate_ttft_ms'] for x in agg_rows if x['policy']==p)
        check(f'聚合TTFT与冻结参考一致[{p}]',abs(agg-REF_AGGREGATE[p])<=0.001,{'recomputed':round(agg,4),'reference':REF_AGGREGATE[p]})
    # ---------- A3 同轮配对差 ----------
    pair_rows=[]
    for base in ('StandaloneTopK','WFC'):
        for r_ in ROUNDS:
            fs=round_means['FlowState-Greedy'][int(r_)-1]; bs=round_means[base][int(r_)-1]
            pair_rows.append({'pair':f'FlowState-Greedy - {base}','round':r_,'fs_round_mean_ms':fs,'base_round_mean_ms':bs,
                              'delta_ms':fs-bs,'pct_paired_vs_base':(fs-bs)/bs*100.0})
        fsagg=median(round_means['FlowState-Greedy']);bsagg=median(round_means[base])
        pair_rows.append({'pair':f'FlowState-Greedy - {base}','round':'median','fs_round_mean_ms':fsagg,'base_round_mean_ms':bsagg,
                          'delta_ms':median([x['delta_ms'] for x in pair_rows if x['pair']==f'FlowState-Greedy - {base}']),
                          'pct_paired_vs_base':None,'pct_aggregate_median_ratio':(fsagg/bsagg-1)*100.0,
                          'note':'pct_paired_vs_base=同轮配对差/基线轮均值(逐轮)；pct_aggregate_median_ratio=聚合中位数比值口径；两者口径不同不可混用'})
    # 给逐轮行补配对百分比口径说明列
    for x in pair_rows:
        x.setdefault('pct_aggregate_median_ratio',None)
        x.setdefault('note','pct_paired_vs_base=同轮配对差/该轮基线等权均值（配对口径，负值=FlowState更快）')
    # ---------- A4 recovery_behavior ----------
    rec_rows=[{'policy':p,'pending':q,
               'T':next(r['T'] for r in raw if r['policy']==p and r['pending']==q),
               'E':next(r['E'] for r in raw if r['policy']==p and r['pending']==q),
               'G':next(r['G'] for r in raw if r['policy']==p and r['pending']==q),
               'prefill_tokens':next(r['prefill_tokens'] for r in raw if r['policy']==p and r['pending']==q),
               'ttft_median_ms':next(x['ttft_median_ms'] for x in pend_rows if x['policy']==p and x['pending']==q)}
              for p in POLICIES for q in PENDINGS]
    # ---------- 落盘 ----------
    def table(name,rows_):
        cols=list(rows_[0])
        with (DATA/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
            for r in rows_: w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()})
    table('raw_arms.csv',raw); table('pending_ttft.csv',pend_rows)
    table('aggregate_ttft.csv',agg_rows); table('paired_comparison.csv',pair_rows)
    table('recovery_behavior.csv',rec_rows)
    (OUT/'checks.json').write_text(json.dumps({'checks':checks,'all_pass':all(c['status']=='PASS' for c in checks),'generated_unix_ns':time.time_ns()},ensure_ascii=False,indent=2)+'\n')
    prov={'frozen_inputs':{'c_formal_image_id':json.loads((CF/'m1A_FS'/'source_identity.json').read_text())['image_id'],
                           'lifecycle_results_csv':hashlib.sha256((CF/'lifecycle_results.csv').read_bytes()).hexdigest(),
                           'policy_plans_m1A_FS':hashlib.sha256((CF/'m1A_FS'/'policy_plans.json').read_bytes()).hexdigest(),
                           'reference_audit_md':hashlib.sha256((CF/'paper_pending_aggregate_audit.md').read_bytes()).hexdigest()},
          'arm_summary_sha256':{r['experiment_id']:r['source_sha256'] for r in raw},
          'scope':'仅60正式臂 m[1-5][A-D]_{FS,SA,WFC}；排除 g1/g2 门禁臂与一切其他计时协议','generated_unix_ns':time.time_ns()}
    (OUT/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n')
    for c in checks: print(c['status'],c['check'])
    if not all(c['status']=='PASS' for c in checks):
        print('完整性门禁 FAIL：不生成正式图表基础数据（CSV 已写出供诊断，绘图脚本应拒绝运行）'); sys.exit(1)
    print('统计提取完成：%d 臂 / %d Pending 行 / %d 聚合行 / %d 配对行'%(len(raw),len(pend_rows),len(agg_rows),len(pair_rows)))

if __name__=='__main__':main()
