"""独立检查配对输入、物理语义及Cold输出；不按性能结果筛选。"""
import hashlib,json
from pathlib import Path
from evaluation.exact_greedy_paired_gpu_fix_b5d4a.protocol import require,compare_outputs

def read(p):return json.loads(Path(p).read_text())
def signature(r):
 raw=r['raw_epoch'];ids={c['checkpoint_id']:c['token_prefix_sha256'] for c in raw['candidates']}
 candidates=[]
 for c in raw['candidates']:
  candidates.append({'tokens':c['token_ids'],'L':c['token_position_L'],'payload':[{k:v for k,v in x.items() if k!='slot'} for x in c['payload']],
   'edges':sorted((e['continuation_id'],e['status'],e['conditions']) for e in raw['edges'] if e['checkpoint_id']==c['checkpoint_id'])})
 return {'configuration':raw['configuration_digest'],'candidates':sorted(candidates,key=lambda c:(c['L'],c['tokens'])),
  'pending':[(p['continuation_id'],p['token_ids'],p['planning_target_T'],p['attention_frontier']) for p in sorted(raw['pendings'],key=lambda p:p['continuation_id'])],
  'fa_map':r['fa_map_before'],'fa_content':r['fa_content_before'],'pool':r['before']['pool_capacity'],'free':len(r['before']['free_slots'])}

def check_pair(out,pair):
 paths=[out/f'p{pair:02d}_{s}' for s in ['Greedy','Exact']]
 records=[read(p/'reconciliation.json') for p in paths]
 require(signature(records[0])==signature(records[1]),'两臂初始状态语义或组件内容不可比')
 cold=read(out/'cold'/'output.json');details=[]
 for p,r in zip(paths,records):
  summary=read(p/'summary.json');post=read(p/'post_request.json')
  require(summary['status']=='PASS' and r['ok'] and r['apply_validation']['status']=='PASS','实际门禁失败')
  compare=compare_outputs(read(p/'output.json'),cold)
  require(r['target']['pending']['continuation_id']=='B5C2_C_TURN_003','目标发生变化')
  require(all(x['returned'] for x in r['operations']) and len(r['plan']['selected_ids'])<=2,'协调失败')
  require(r['selected_payload_before']==r['selected_payload_after']==post['state']['selected_payload'],'Selected内容变化')
  require(r['fa_content_before']==r['fa_content_after'],'FA协调变化')
  compute=post['trace']['compute'];require(compute and any(x['is_extend'] for x in compute) and all(x['returned'] and x['cuda_elapsed_ms']>0 for x in compute),'缺少实际模型CUDA计时')
  execution=post['trace']['execution'];require(execution and any(x['is_extend'] for x in execution),'缺少Prefill计时')
  require(all(x['native_run_returned'] and x['cuda_elapsed_ms']>0 for x in execution),'缺少设备事件区间')
  require(all(x.get('returned') for x in post['trace']['copies']),'COW未完成')
  require(summary['provenance_status']=='PROVENANCE_UNVERIFIED' and summary['cross_workflow_resume_certified'] is False,'不得意外升级血缘')
  require(read(p/'native_source_validation.json')['native_source']=='MATCHED','实际匹配认证缺失')
  details.append({'provenance':'PROVENANCE_UNVERIFIED','cross_workflow_resume_certified':False,'run':p.name,'output':compare,'native_match':summary['native_match'],'source_history':r['target']['source_history']['request_id']})
 return {'status':'PASS','pair':pair,'comparable':True,'details':details}
