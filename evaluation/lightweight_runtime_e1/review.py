"""四臂独立核验，不按性能结果决定有效性。"""
import json
from pathlib import Path
from evaluation.exact_greedy_paired_gpu_fix_b5d4a.protocol import compare_outputs,require
from evaluation.exact_greedy_paired_gpu_gate_b5d4b.review import signature

def read(p):return json.loads(Path(p).read_text())
def ref_signature(ref):
 i=ref['initial']
 return signature({'raw_epoch':i['strict_epoch'],'before':i['before'],'fa_map_before':ref['fa_map'],'fa_content_before':ref['fa_content']})
def check(out,name):
 p=out/name;s=read(p/'summary.json');r=read(p/'reconciliation.json');ref=read(p/'reference_audit.json');post=read(p/'post_request.json')
 require(s['status']=='PASS' and r['ok'] and r['reconciliation']['status']=='PASS','执行门禁失败')
 compare=compare_outputs(read(p/'output.json'),read(out/'cold/output.json'))
 require(len(r['operations'])==len(r['plan']['logical_eviction_ids']) and all(o['returned'] for o in r['operations']),'驱逐不完整')
 require(r['plan']['selected_ids']==ref['plan']['selected_ids'] and r['plan']['pending']==ref['plan']['pending'],'轻重计划不同')
 require(post['state']['selected_payload']==ref['selected_payload'] and post['state']['fa_protected_payload']==ref['fa_protected_payload'],'历史内容损坏')
 require(read(p/'native_source_validation.json')['native_source']=='MATCHED','Native匹配失败')
 require(s['provenance_status']=='PROVENANCE_UNVERIFIED' and not s['cross_workflow_resume_certified'],'血缘等级错误')
 tr=post['trace'];require(len(tr['compute'])==len(tr['execution'])==8,'执行次数不完整')
 for outer,inner in zip(tr['execution'],tr['compute']):
  require(outer['native_run_returned'] and inner['returned'] and outer['ns_before']<=inner['ns_before']<inner['ns_after']<=outer['ns_after'] and outer['cuda_stream']==inner['cuda_stream'] and outer['cuda_elapsed_ms']>=inner['cuda_elapsed_ms']>0,'计时边界不可信')
 require(tr['copies'] and all(c['returned'] for c in tr['copies']),'COW不完整')
 if name=='smoke':
  require(read(p/'post_control_audit.json')['ok'],'Smoke内容后验失败')
  require(all(c['source_payload_before']==c['source_payload_after']==c['destination_payload_after'] for c in tr['copies']),'Smoke复制内容错误')
 return {'status':'PASS','run':name,'correctness':compare,'mode':s['mode'],'provenance':'PROVENANCE_UNVERIFIED'}
