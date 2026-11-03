"""只汇总已通过完整验证的60个正式主臂；Gate和Smoke不进入性能统计。"""
import csv
import json
import statistics as st
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def main():
    evidence=read(OUT/'takeover/evidence_main.json');base=read(OUT/'formal_verification_main.json')
    assert evidence['status']=='PASS' and base['verdict']=='PASS' and base['arms_checked']==60
    protocol=read(OUT/'protocol.json');analysis=read(OUT/'analysis_summary.json');plans=read(OUT.parent/'t2g1_plans.json')['plans']
    rows=[]
    for a in protocol['main_arms']:
        s=read(OUT/a['name']/'run/summary.json')
        rows.append({'arm':a['name'],'rep':a['rep'],'pending':a['pending'],'policy':a['policy'],'H':s['H_physical_fa_hit'],'E':s['E_executable_prefix'],'G':s['G_replay_gap'],'prefill_tokens':s['prefill_token_count'],'ttft_ms':s['ttft_ms'],'prefill_cuda_ms':s['cuda_prefill_ms'],'request_wall_ms':s['request_wall_ms'],'control_rpc_ms':s['control_rpc_wall_ms'],'epoch_wall_ms':s['epoch_wall_ms'],'cow_cuda_ms':s['cow_copy_cuda_ms'],'cow_host_ms':s['cow_copy_host_ms'],'decode_cuda_ms':s['cuda_decode_ms'],'J_plan_ms':s['plan_recovery_cost_ms']})
    with (OUT/'formal_measurements_full_precision.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# T2-G2 三策略正式GPU对照：接管完成报告','','**READY：3个Gate与60个正式主臂完成，CPU原始证据复核通过。** 结论仅对应冻结C/K3受控合成工作负载、Qwen3.5-9B、SGLang v0.5.17、单H100 PCIe。','','## 1. 接管与执行完整性','','原`check()`参数错误在接管前已经修复保存，本次未再次修改该脚本，也未重跑Gate。原检查器33项及补充原始证据173项均通过后，由检查程序生成gate_passed.json。初始主矩阵无产物、无活动实验进程，GPU0空闲14MiB，随后运行原launch_formal.py main。','','60臂严格沿用原协议顺序；每臂独立Engine/容器。未改策略、选集、Runtime控制或测量窗口。完整验证覆盖原始8 Token Cold对照、准确13项Recurrent-only驱逐、Keep组件保护、Native来源/COW、FA索引/受保护内容、形状预热、请求窗口零新增Triton编译及清理。协调时全量FA内容保护由哈希锁定控制器在返回前断言；没有伪造未单独落盘的全量哈希。','','Gate和Smoke仅作为门禁，不进入任何下列统计。正式分母为3策略×4请求×5轮=60臂；每个策略20臂、每个请求/策略5次。不是60个独立工作负载。','','## 2. 模型层与实际恢复结构','','Marconi-Evict在本文指**Marconi-Evict Adaptation**；KVFlow为**KVFlow-inspired**，均不称完整原论文系统。三策略通过同一原生状态控制路径；本实验执行冻结选集，不把控制路径中的计划验证时间称为在线策略求解时间。','','|策略|四Pending共同模型J（ms）|冻结保留集合|','|---|---:|---|']
    for p,plan in plans.items():text.append(f"|{p}|{plan.get('J_ms_current_scope')}|{', '.join(plan['selected_checkpoint_ids'])}|")
    text+=['','|Pending|策略|H|E|G|实际Prefill Token|','|---|---|---:|---:|---:|---:|']
    for pend in ['C_A','C_B','C_C','C_D']:
        for policy in protocol['policies'].values():
            rr=[r for r in rows if r['pending']==pend and r['policy']==policy];r=rr[0]
            assert len(rr)==5 and all(tuple(x[k] for k in ['H','E','G','prefill_tokens'])==tuple(r[k] for k in ['H','E','G','prefill_tokens']) for x in rr)
            text.append(f"|{pend}|{policy}|{r['H']}|{r['E']}|{r['G']}|{r['prefill_tokens']}|")
    text+=['','J只用于模型化集合评价；G只对应规划目标T−E。实际Prefill包含规划前沿之后的输入尾部，不以G替代实际执行Token。逻辑K=3不表示GPU实际容量不足；原物理池28槽。','','## 3. 逐请求五轮结果','','|Pending|策略|TTFT中位[min,max] ms|CUDA Prefill中位[min,max] ms|请求Wall中位ms|','|---|---|---|---|---:|']
    for pend in ['C_A','C_B','C_C','C_D']:
        for policy in protocol['policies'].values():
            rr=[r for r in rows if r['pending']==pend and r['policy']==policy]
            def fmt(k):
                a=[r[k] for r in rr];return f'{st.median(a):.3f} [{min(a):.3f}, {max(a):.3f}]'
            text.append(f"|{pend}|{policy}|{fmt('ttft_ms')}|{fmt('prefill_cuda_ms')}|{st.median(r['request_wall_ms'] for r in rr):.3f}|")
    text+=['','|比较（FS−对照）|Pending|五个配对TTFT差值中位[min,max] ms|负/零/正|','|---|---|---|---|']
    for pend,obj in analysis['per_pending'].items():
        for comp in ['FS-MC','FS-KV']:
            d=obj[comp+'_ttft_diff_ms'];text.append(f"|{comp}|{pend}|{d['median_ms']:.3f} [{d['min_ms']:.3f}, {d['max_ms']:.3f}]|{d['neg']}/{d['zero']}/{d['pos']}|")
    text+=['','## 4. 四Pending等权轮级比较','','每轮先对同策略四个独立请求的TTFT取等权均值，再对同轮策略作差；负数表示FS更低。不是并发工作流完成时间或吞吐。','','|比较|五轮等权TTFT差值中位[min,max] ms|负/零/正|','|---|---|---|']
    for c,d in analysis['round_equal_weight'].items():text.append(f"|{c}|{d['median_ms']:.3f} [{d['min_ms']:.3f}, {d['max_ms']:.3f}]|{d['neg']}/{d['zero']}/{d['pos']}|")
    text+=['','逐轮完整值见round_equal_weight.csv；40个请求配对差见pair_analysis.csv。五轮只作描述性统计，不声称显著性、生产P95或跨工作负载泛化。','','## 5. 控制成本与测量有效性','','|策略|Control RPC中位ms（20臂）|生命周期内epoch_wall中位ms|','|---|---:|---:|']
    for p in protocol['policies'].values():
        rr=[r for r in rows if r['policy']==p];text.append(f"|{p}|{st.median(r['control_rpc_ms'] for r in rr):.3f}|{st.median(r['epoch_wall_ms'] for r in rr):.3f}|")
    text+=['','控制分段均值完整见analysis_summary.json/control_decomposition。Observe、认证/计划、Apply-time、FA审计和Reconcile均保留；它们是Control RPC子区间，不能再加到父区间中。shape warm、重放与清空为隔离测量设施，epoch_wall不是生产Serving延迟。','','原analyze_formal.py的cow_diff_ms将CUDA与host耗时相加，二者可能重叠，本报告不使用该派生字段作为净开销；原字段保留且不改变冻结分析程序。单独CUDA/host值见formal_measurements_full_precision.csv。TTFT与请求Wall包含CUDA执行，不相加。未记录足够边界以把全部控制后路径独立拆成生产净收益，不能删除研究审计后声称净加速。',f"冻结双峰诊断（decode>160ms）记录慢臂{analysis['decode_slow_arms']}个、配对模式不一致{analysis['mode_mismatch_pairs']}；全部保留，不按胜负剔除。该诊断不等于编译污染；请求窗口Triton新增数均经验证为0。",'','## 6. 三层结论与限制','','1. J差异说明固定目标下的分配取舍；本报告没有把J当成实测GPU时间。','2. Native E/G/Prefill与计划一致，表明冻结保留集被落实；不能因此升级跨Workflow物理血缘，继续PROVENANCE_UNVERIFIED。','3. 实测TTFT/CUDA差异以全部配对数据为准，逐Pending反向与平局一并保留；不能据CPU模型排序预先判胜。','4. 这是适配策略保留集合的受控比较，不是Marconi或KVFlow完整系统复现；也不证明全面在线Select或控制成本后稳定净收益。','','## 7. 证据索引','','- protocol.json / formal_precheck.json / implementation_lock_gate.json / implementation_lock_main.json：冻结与身份。','- formal_verification_gate.json / formal_verification_main.json；takeover/evidence_gate.json / evidence_main.json：原检查器与原始证据复核。','- 每臂run/{summary,output,reconciliation,post_request,engine_cleanup}.json及*_cleanup.json：原始正确性/清理记录。','- formal_measurements_full_precision.csv：60臂全精度测量；formal_arms_main.csv：原检查器摘要。','- pair_analysis.csv / round_equal_weight.csv / pending_policy_matrix.csv / analysis_summary.json：冻结分析脚本输出。','- takeover/recovery_inventory.json / main_takeover_host.log：接管与主矩阵执行。','','完成后停止，不增加工作负载、重复次数或新实验。']
    with (OUT/'formal_gpu_comparison_report.md').open('x') as f:f.write('\n'.join(text)+'\n')

if __name__=='__main__':main()
