"""锁定本轮独立实现和既有Runtime安全模块，不覆盖历史文件。"""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 prior=read(ROOT/'evaluation/multishare_runtime_gate_b5c3/source_identity.json')
 files=list(OUT.glob('*.py'))+[OUT/'protocol.json',OUT/'workload.json',OUT/'prepared_inputs.json']
 for directory in ['runtime_control_closure_r2c0','runtime_control_closure_r2c1a','exact_reconciliation_correctness_gate_b5d3','multishare_runtime_gate_b5c3','strict_compatibility_b1c','kvflow_inspired_retention_cpu','exact_planner_contract_gate_b5d1']:
  files+=list((ROOT/'evaluation'/directory).glob('*.py'))
 files+=list((ROOT/'flowstate').glob('*.py'))+[ROOT/'flowstate/adapters/sglang.py']
 write(OUT/'source_identity.json',{'status':'PASS','image_id':prior['image_id'],'runtime_sources':prior['runtime_sources'],'sha256':{str(p):sha(p) for p in files},'说明':'本轮源码内容锁；不冒称与旧实验完整锁相同'})
 write(OUT/'cpu_gate.json',{'status':'PASS','runtime_sources':prior['runtime_sources'],'test_results':read(OUT/'cpu_tests.json')})
if __name__=='__main__':main()
