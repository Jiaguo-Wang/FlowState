"""先独立运行 PCB 门禁，未通过时不允许进入其他策略。"""
from dataclasses import asdict
import json, traceback
from common import *
import pcb

def main():
    before=input_hashes();write('source_integrity_before.json',before)
    results=[]
    try:
        cases=load_cases()
        for case in cases:
            epoch=case['epoch'];selected,trace=pcb.select(epoch);result=score(epoch,selected)
            reverse=Epoch(tuple(reversed(epoch.candidates)),tuple(reversed(epoch.pending)),epoch.k)
            assert pcb.select(reverse)[0]==selected
            fs=case['references']['FlowState'];difference=result['cost_ms']-fs['cost_ms']
            entry={k:v for k,v in case.items() if k!='epoch'}
            entry.update(input=asdict(epoch),PCB=result,pcb_trace=trace,difference_ms=difference,same_set=set(selected)==set(fs['selected']),same_frontiers=frontier_vector(result)==frontier_vector(fs),reaches_all_frontiers=frontier_vector(result)==all_frontier(epoch))
            results.append(entry)
            if abs(difference)>EPS:
                write('pcb_failure_case.json',entry)
                raise ValueError(f"PCB 严格不一致：{case['snapshot_id']} K={epoch.k}，差值 {difference}")
            if case['population']=='AgentX':assert entry['reaches_all_frontiers']
            assert fingerprint(epoch)==case['input_digest']
        after=input_hashes();assert before==after
        with (OUT/'pcb_gate_results.jsonl').open('w') as f:
            for r in results:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        gate=dict(status='PASS',cases=len(results),ties=sum(abs(r['difference_ms'])<=EPS for r in results),same_sets=sum(r['same_set'] for r in results),same_frontiers=sum(r['same_frontiers'] for r in results),different_sets_same_frontiers=sum(not r['same_set'] and r['same_frontiers'] for r in results),max_absolute_difference_ms=max(abs(r['difference_ms']) for r in results),source_integrity='PASS',pcb_sha256=sha(OUT/'pcb.py'),common_sha256=sha(OUT/'common.py'),definition_sha256=sha(ROOT/'evaluation/P1_BASELINE_AND_WORKLOAD_DESIGN_REVIEW.md'))
        write('pcb_gate.json',gate);print(json.dumps(gate,ensure_ascii=False))
    except Exception as e:
        write('validation.json',dict(status='VALIDATION_BLOCKED',pcb_gate='FAIL',error=str(e),traceback=traceback.format_exc(),completed_cases=len(results)))
        (OUT/'final_report.md').write_text('# PCB 门禁失败\n\nVALIDATION_BLOCKED\n\n'+str(e)+'\n\n未实现或执行 LDR/WFC；失败详情见 validation.json 与已有 pcb_failure_case.json。\n')
        raise

if __name__=='__main__':main()
