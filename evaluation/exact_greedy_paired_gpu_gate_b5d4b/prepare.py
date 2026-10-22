"""只核验既有冻结证据并复制启动输入到新目录，不修改原失败包。"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'evaluation/exact_greedy_paired_gpu_gate_b5d4';FIX=ROOT/'evaluation/exact_greedy_paired_gpu_fix_b5d4a'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (OUT/'execution_started.json').exists(),'执行已开始，不允许重新准备'
 source=read(OLD/'source_identity.json')
 source['sha256'].update({p:x['expected'] for p,x in read(FIX/'source_integrity.json')['checked_files'].items()})
 for directory in [OLD,FIX]:
  manifest=directory/'evidence_manifest.json';source['sha256'][str(manifest)]=sha(manifest)
  for name,h in read(manifest)['files'].items():source['sha256'][str(directory/name)]=h
 for p,h in source['sha256'].items():assert sha(p)==h,p
 tests=read(FIX/'cpu_regression_results.json');assert tests['status']=='PASS' and tests['tests_run']==30
 for name in ['prepared_inputs.json','protocol_lock.json','cpu_gate.json']:(OUT/name).write_bytes((OLD/name).read_bytes())
 (OUT/'original_experiment_protocol.md').write_bytes((OLD/'experiment_protocol.md').read_bytes())
 write(OUT/'source_identity.json',source)
 print('冻结输入、D4原协议、D4a修复与30项CPU回归身份PASS',len(source['sha256']))
if __name__=='__main__':main()
