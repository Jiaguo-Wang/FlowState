"""核验既有冻结身份，原样复制输入到本轮独立输出。"""
import json
from pathlib import Path
from evaluation.multishare_runtime_gate_b5c3.prepare import sha,read,write,ROOT
OUT=Path(__file__).resolve().parent

def main():
 assert not (OUT/'execution_started.json').exists(),'禁止重跑已开始生命周期'
 old=ROOT/'evaluation/multishare_runtime_gate_b5c3';source=read(old/'source_identity.json')
 for p,h in source['sha256'].items():assert sha(p)==h,p
 for directory,manifest in [('multishare_cpu_input_gate_b5c2','artifact_manifest.json'),('multishare_runtime_gate_b5c3','evidence_manifest.json'),('exact_planner_contract_gate_b5d1','evidence_manifest.json')]:
  base=ROOT/'evaluation'/directory
  for name,h in read(base/manifest)['files'].items():
   p=base/name;assert sha(p)==h,str(p);source['sha256'][str(p)]=h
 for p,h in read(ROOT/'evaluation/exact_planner_contract_gate_b5d1/implementation_lock.json')['sources_sha256'].items():assert sha(p)==h,p
 (OUT/'prepared_inputs.json').write_bytes((old/'prepared_inputs.json').read_bytes())
 write(OUT/'source_identity.json',source)
 for p in OUT.glob('*.py'):compile(p.read_text(),str(p),'exec')
 print('冻结输入、模型、运行接口和D1规划器摘要PASS')
if __name__=='__main__':main()
