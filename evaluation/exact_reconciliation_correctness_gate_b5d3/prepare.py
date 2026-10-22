"""核验固定输入与既有控制恢复实现，不修改任何历史锁。"""
from pathlib import Path
from evaluation.multishare_runtime_gate_b5c3.prepare import ROOT,read,write,sha
OUT=Path(__file__).resolve().parent

def main():
 old=ROOT/'evaluation/exact_runtime_shadow_gate_b5d2';source=read(old/'source_identity.json')
 for p,h in source['sha256'].items():assert sha(p)==h,p
 for directory in ['exact_runtime_shadow_gate_b5d2','exact_planner_contract_gate_b5d1']:
  base=ROOT/'evaluation'/directory
  for name,h in read(base/'evidence_manifest.json')['files'].items():
   p=base/name;assert sha(p)==h,str(p);source['sha256'][str(p)]=h
 for directory in ['runtime_control_closure_r2c0','runtime_control_closure_r2c1a','runtime_control_closure_r2c1b2']:
  for p in (ROOT/'evaluation'/directory).glob('*.py'):source['sha256'][str(p)]=sha(p)
 for p in (ROOT/'flowstate').rglob('*.py'):source['sha256'][str(p)]=sha(p)
 (OUT/'prepared_inputs.json').write_bytes((old/'prepared_inputs.json').read_bytes())
 write(OUT/'source_identity.json',source)
 for p in OUT.glob('*.py'):compile(p.read_text(),str(p),'exec')
 print('源身份与新代码语法预检查PASS')
if __name__=='__main__':main()
