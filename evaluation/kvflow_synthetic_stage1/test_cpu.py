"""验证生成输入和新Runtime适配的纯CPU前提，不启动Engine。"""
import ast,json,hashlib
from pathlib import Path
from evaluation.kvflow_inspired_retention_cpu.policy import select,compute_ste
OUT=Path(__file__).resolve().parent

def main():
 w=json.loads((OUT/'workload.json').read_text());checks={}
 checks['twenty_warmups']=len(w['history'])==20
 checks['fixed_reused']=all(w['history'][i]['token_ids']==w['history'][i+10]['token_ids'] for i in range(10))
 for j,c in enumerate(w['cycles']):
  checks[f'ste_{j}']=list(compute_ste(c['graph'],c['epoch_id']).values())==list(range(10))
  checks[f'lengths_{j}']=all(len(d['fixed_token_ids'])==4096 and len(d['input_token_ids'])==4128 and d['input_token_ids'][:4096]==d['fixed_token_ids'] for d in c['declarations'])
  checks[f'one_ready_{j}']=sum(n['ready'] for n in c['graph'])==1
  for k in [3,5]:checks[f'selection_{j}_{k}']=select(c['candidates'],c['graph'],c['declarations'],c['epoch_id'],c['configuration_digest'],k)[0]==tuple(f'CP_{i:02d}' for i in range(k))
 checks['suffix_changes']=all(w['cycles'][0]['pendings'][i]['token_ids'][4096:]!=w['cycles'][1]['pendings'][i]['token_ids'][4096:] for i in range(10))
 for p in OUT.glob('*.py'):ast.parse(p.read_text())
 checks['syntax']=True
 checks['runtime_no_hardcoded_node']= 'Node27' not in (OUT/'control.py').read_text()
 checks['preserved_reconciliation']='StateController.reconcile_strict(epoch,plan,runtime)' in (OUT/'control.py').read_text()
 assert all(checks.values()),checks
 (OUT/'cpu_tests.json').write_text(json.dumps({'status':'PASS','checks':checks,'说明':'原KVFlow14项单测另复用；这些检查不是GPU正确性认证'},ensure_ascii=False,indent=2)+'\n')
 print(len(checks),'项CPU检查通过')
if __name__=='__main__':main()
