"""逐Pending独立Cold对照和状态后置认证，不使用性能作门禁。"""
from .prepare import read
from .source_contract import validate_frozen
from evaluation.lightweight_runtime_e1.review import ref_signature
from evaluation.lightweight_runtime_e1.protocol import require,compare_outputs

def check(out,name,cold,pid):
 p=out/name;s=read(p/'summary.json');r=read(p/'reconciliation.json');ref=read(p/'reference_audit.json');post=read(p/'post_request.json');trace=post['trace']
 require(s['status']=='PASS' and r['ok'] and r['reconciliation']['status']=='PASS','恢复或协调未通过')
 require(r['target']['pending']['continuation_id']==pid,'目标变化')
 require(r['apply_validation']['status']=='PASS' and r['apply_validation']['same_scheduler_call'],'Apply-time隔离失败')
 require(len(r['operations'])==5 and all(x['returned'] for x in r['operations']),'实际驱逐不完整')
 require(r['selected_payload_before']==r['selected_payload_after']==post['state']['selected_payload'],'保留状态内容变化')
 require(r['fa_content_before']==r['fa_content_after'] and read(p/'post_control_audit.json')['ok'],'协调FA内容失败')
 require(post['state']['fa_protected_payload']==ref['fa_protected_payload'],'请求修改受保护FA内容')
 require(read(p/'native_source_validation.json')['native_source']=='MATCHED','Native来源未认证')
 require(len(trace['admission'])==1 and len(trace['execution'])==8 and all(x['native_run_returned'] for x in trace['execution']),'Admission或执行证据不足')
 require(trace['copies'] and all(x['returned'] and x['source_payload_before']==x['source_payload_after']==x['destination_payload_after'] for x in trace['copies']),'COW未完成或复制错误')
 comparison=compare_outputs(read(p/'output.json'),read(out/cold/'output.json'))
 require(s['provenance_status']=='PROVENANCE_UNVERIFIED','不得升级物理血缘')
 return {'status':'PASS','run':name,'pending':pid,'solver':r['solver'],'reconciliation':'RECONCILIATION_VALIDATED','resume':'RESUME_VALIDATED','provenance':'PROVENANCE_UNVERIFIED','selected_current':r['plan']['selected_ids'],'J_model_ms':r['plan']['recovery_cost_ms'],'native_match':s['native_match'],'output':comparison,'operations':len(r['operations']),'cold':cold,'admission_state_validated':True,'FA_reconciliation_unchanged':True,'selected_components_preserved':True}
