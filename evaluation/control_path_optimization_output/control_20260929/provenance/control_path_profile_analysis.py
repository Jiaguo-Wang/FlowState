"""汇总控制路径探针，区分冻结事实和新测量，避免嵌套耗时重复计数。"""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys
from evaluation.resume_epoch_e2e_analysis import percentile


def stats(values):
    """报告每epoch累计值的均值、中位数与P95。"""
    return {'mean':mean(values),'p50':percentile(values,.5),'p95':percentile(values,.95),'min':min(values),'max':max(values)}


def analyze(output):
    """每阶段完整闭合到RPC时间、服务端嵌套调用及客户端本地余量。"""
    plan=json.loads((output/'plan.json').read_text())
    assert len(plan)==72
    stages={'common':'common_observation_ms','policy':'policy_observation_ms','reconciliation':'reconciliation_ms'}
    epochs=defaultdict(list)
    op_epochs=defaultdict(list)
    all_records=[]
    max_closure=0
    for p in plan:
        r=json.loads((output/'runs'/p['run_id']/'record.json').read_text())
        assert r['status']=='PASS' and all(r['correctness'].values())
        calls=json.loads((output/'runs'/p['run_id']/'profile.json').read_text())
        all_records.append(r)
        for phase,field in stages.items():
            group=[c for c in calls if c['phase']==phase]
            assert len(group)==(4 if phase=='policy' else 1)
            part={'before_submit_ms':0,'queue_wait_ms':0,'server_ms':0,'event_handoff_ms':0,'after_submit_ms':0,
                  'connection_ms':0,'client_encode_ms':0,'client_decode_ms':0,'rpc_ms':0,'server_unattributed_ms':0,
                  'request_bytes':0,'response_bytes':0,'rpc_count':len(group)}
            ops=defaultdict(lambda:{'count':0,'inclusive_ms':0,'exclusive_ms':0})
            for c in group:
                times=[c[k] for k in ['client_start_ns','submit_start_ns','server_start_ns','server_end_ns','submit_end_ns','client_end_ns']]
                assert all(a<=b for a,b in zip(times,times[1:])),(p['run_id'],times)
                for key,a,b in zip(['before_submit_ms','queue_wait_ms','server_ms','event_handoff_ms','after_submit_ms'],times,times[1:]):
                    part[key]+=(b-a)/1e6
                part['rpc_ms']+=(times[-1]-times[0])/1e6
                part['connection_ms']+=(c['connected_ns']-c['client_start_ns'])/1e6
                part['client_encode_ms']+=c['encode_ns']/1e6
                part['client_decode_ms']+=c['decode_ns']/1e6
                part['request_bytes']+=c['request_bytes'];part['response_bytes']+=c['response_bytes']
                part['server_unattributed_ms']+=(c['server_end_ns']-c['server_start_ns']-sum(x['exclusive_ns'] for x in c['operations'].values()))/1e6
                for key,v in c['operations'].items():
                    ops[key]['count']+=v['count']
                    ops[key]['inclusive_ms']+=v['inclusive_ns']/1e6
                    ops[key]['exclusive_ms']+=v['exclusive_ns']/1e6
            part['stage_ms']=r[field]
            part['client_local_ms']=r[field]-part['rpc_ms']
            assert part['client_local_ms']>=-1e-5
            closure=abs(part['stage_ms']-sum(part[k] for k in ['before_submit_ms','queue_wait_ms','server_ms','event_handoff_ms','after_submit_ms','client_local_ms']))
            max_closure=max(max_closure,closure)
            epochs[phase].append(part)
            op_epochs[phase].append(ops)
    result={'status':'PASS','epochs':72,'clock':'同主机perf_counter_ns','max_closure_error_ms':max_closure,
            'phases':{},'operations':{},'说明':'子步骤统计为每epoch累计；包含时间具有嵌套关系，只能用独占时间闭合。新profiling不替代旧冻结延迟。'}
    for phase,rows in epochs.items():
        result['phases'][phase]={k:stats([r[k] for r in rows]) for k in rows[0]}
        keys=sorted({k for r in op_epochs[phase] for k in r})
        result['operations'][phase]={k:{metric:stats([r.get(k,{}).get(metric,0) for r in op_epochs[phase]])
                    for metric in ['count','inclusive_ms','exclusive_ms']} for k in keys}
    old=Path('evaluation/resume_epoch_e2e_output/resume_epoch_final_20260928')
    frozen=[]
    for p in json.loads((old/'plan.json').read_text()):
        if p['policy']!='FlowState':continue
        r=json.loads((old/'runs'/p['run_id']/'record.json').read_text())
        e=json.loads((old/'runs'/p['run_id']/'evidence.json').read_text())
        frozen.append({'policy_observation_ms':r['policy_observation_ms'],'common_observation_ms':r['common_observation_ms'],
            'common_server_ms':e['common']['worker_ns']/1e6,'common_outer_ms':r['common_observation_ms']-e['common']['worker_ns']/1e6,
            'reconciliation_ms':r['reconciliation_ms'],'reconciliation_server_ms':e['reconciliation']['worker_ns']/1e6,
            'reconciliation_outer_ms':r['reconciliation_ms']-e['reconciliation']['worker_ns']/1e6,
            'eviction_ms':sum(e['reconciliation']['operation_ns'])/1e6,
            'reconciliation_server_other_ms':(e['reconciliation']['worker_ns']-sum(e['reconciliation']['operation_ns']))/1e6})
    result['frozen_measured']={k:stats([r[k] for r in frozen]) for k in frozen[0]}
    (output/'analysis').mkdir(exist_ok=True)
    (output/'analysis/profile_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# FlowState控制路径瓶颈profiling','', '状态：PASS；72/72独立生命周期全部通过原有E2E门禁。',
           '冻结实现未修改，诊断仅包装函数计时并增加响应元数据。所有状态操作及检查仍调用冻结函数。',
           '本表是新profiling的每epoch累计值，不是历史293.093ms与211.337ms的未记录内部计时。',
           '', '| 阶段 | 总计 | 提交入口前 | 队列等待 | 服务端 | 事件交接 | 返回与解码 | 客户端本地 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for phase,row in result['phases'].items():
        lines.append('| '+phase+' | '+' | '.join(f"{row[k]['mean']:.3f}" for k in ['stage_ms','before_submit_ms','queue_wait_ms','server_ms','event_handoff_ms','after_submit_ms','client_local_ms'])+' |')
    lines+=['','每项平均/P50/P95、调用次数与完整嵌套路径见profile_summary.json；下表展示各子步骤独占时间，单位ms。',
            '', '| 阶段与子步骤 | 平均 | P50 | P95 | 平均调用次数/epoch |','|---|---:|---:|---:|---:|']
    for phase,ops in result['operations'].items():
        for name,x in ops.items():
            s=x['exclusive_ms']
            lines.append(f"| {phase}/{name} | {s['mean']:.6f} | {s['p50']:.6f} | {s['p95']:.6f} | {x['count']['mean']:.3f} |")
    lines+=['','## 调用与安全边界','',
        '每epoch公共读取1次RPC、frontier读取4次串行RPC、协调1次RPC。TP/PP/DP均为1，无跨rank/NCCL调用；调用跨worker与scheduler进程，经本机TCP、服务线程、队列及事件交接。',
        '每个frontier调用读取前后各一次全树、分配器和语义状态，共8份快照；四次查询间没有请求执行或状态变更。公共view读取一次全部候选，协调再读取驱逐前后两份view；候选路径逐checkpoint串行读取，共享祖先索引被重复摘要。',
        '同一scheduler安全点内，相邻frontier查询的前一后态可以作为后一前态，但每个查询仍必须独立比较；不能仅检查整批首尾而遗漏中间状态变化。',
        '不可移出的检查：空闲及运行模式、句柄精确节点/前缀、决策view摘要匹配、观察非干扰、每候选驻留、每次驱逐目标/后置条件、组件隔离、独立容量、全局仅目标状态改变以及最终sanity。',
        '可移出的仅是日志落盘、结果汇总、参考oracle重算及审计展示；冻结E2E已经把这些放在区间外。诊断性输出不等于用于安全验证的状态读取，不能因其位于测试目录而删除。',
        '可安全复用候选token序列、节点预期身份和当前epoch句柄对象；不得跨状态变更复用recurrent驻留、allocator、LRU、view digest或frontier。只读view内重复tensor索引摘要可局部复用，离开该只读操作即失效。',
        '', '## 历史数值归因限制','',
        '历史reconciliation可严格分成逐驱逐和、服务端其他工作、服务端外余量；历史frontier没有内部时钟，无法唯一追溯293.093ms的各子步骤。所有可核实历史统计保存在frozen_measured。新测各阶段闭合误差见max_closure_error_ms。',
        '提交入口前包含连接/编码、服务线程读取与解码及调度影响；事件交接包含唤醒和重新获得执行权。连接/编码/解码另有计时，但没有把剩余时间武断归因于GIL或操作系统。tensor.cpu/tolist可能隐式同步，未新增GPU synchronize，不能宣称测得每次设备同步时间。']
    (output/'analysis/profiling_report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    analyze(Path(sys.argv[1]))
