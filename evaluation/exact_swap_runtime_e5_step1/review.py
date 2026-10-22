"""核验轻量恢复与实际计时；Smoke额外对拍内容，不放宽安全门禁。"""
from evaluation.lightweight_runtime_e1.review import read,ref_signature,require,compare_outputs
from evaluation.lightweight_runtime_e1.control import equivalent

def check(out,name,cold,pid,smoke=False):
 p=out/name;s=read(p/'summary.json');r=read(p/'reconciliation.json');ref=read(p/'reference_audit.json');post=read(p/'post_request.json');tr=post['trace']
 require(s['status']=='PASS' and r['ok'] and r['reconciliation']['status']=='PASS','协调或请求失败')
 require(s['mode']=='light','不是轻量路径')
 require(r['target']['pending']['continuation_id']==pid,'目标错误')
 require(r['apply_validation']['status']=='PASS' and r['apply_validation']['same_scheduler_call'],'连续Apply-time检查失败')
 require(len(r['operations'])==len(r['plan']['logical_eviction_ids'])==4 and all(x['returned'] for x in r['operations']),'驱逐不完整')
 require(r['plan']['selected_ids']==ref['plan']['selected_ids'] and r['plan']['pending']==ref['plan']['pending'],'轻重计划不一致')
 equivalent(r['raw_epoch'],ref['initial']['strict_epoch'])
 residents=[c['checkpoint_id'] for c in r['raw_epoch']['candidates'] if any(n['token_ids']==c['token_ids'] and n['slots'] for n in r['after']['nodes'])]
 require(set(residents)==set(r['plan']['selected_ids']),'最终驻留集合不正确')
 require(r['before']['attention_digest']==r['after']['attention_digest'] and r['before']['non_recurrent_digest']==r['after']['non_recurrent_digest'],'非目标状态变化')
 require(post['state']['selected_payload']==ref['selected_payload'] and post['state']['fa_protected_payload']==ref['fa_protected_payload'],'历史内容损坏')
 require(read(p/'native_source_validation.json')['native_source']=='MATCHED','Native来源不一致')
 require(len(tr['admission'])==1 and len(tr['compute'])==len(tr['execution'])==8,'执行证据不完整')
 for outer,inner in zip(tr['execution'],tr['compute']):
  require(outer['native_run_returned'] and inner['returned'] and outer['ns_before']<=inner['ns_before']<inner['ns_after']<=outer['ns_after'] and outer['cuda_stream']==inner['cuda_stream'] and outer['cuda_elapsed_ms']>=inner['cuda_elapsed_ms']>0,'CUDA计时无效')
 require(tr['copies'] and all(c['returned'] for c in tr['copies']),'COW未完成')
 if smoke:
  require(read(p/'post_control_audit.json')['ok'],'协调后全内容检查失败')
  require(all(c['source_payload_before'] is not None and c['source_payload_before']==c['source_payload_after']==c['destination_payload_after'] for c in tr['copies']),'Smoke复制内容错误')
 require(s['provenance_status']=='PROVENANCE_UNVERIFIED','血缘越级')
 comparison=compare_outputs(read(p/'output.json'),read(out/cold/'output.json'))
 actual=sum(e['extend_token_count'] for e in tr['execution'] if e['is_extend'])
 require(actual==s['native_match']['input_token_count']-s['native_match']['executable_prefix'],'实际Prefill区间不一致')
 return {'status':'PASS','run':name,'pending':pid,'solver':r['solver'],'mode':'light','output':comparison,'actual_prefill_tokens':actual,'native_match':s['native_match'],'selected':residents,'FA_metadata_unchanged':True,'selected_content_unchanged':True,'cold':cold,'provenance':'PROVENANCE_UNVERIFIED'}
