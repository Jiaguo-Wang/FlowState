"""核验原失败包与生产依赖未变，保存独立修复Diff及内容摘要。"""
import difflib,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1];OLD=ROOT/'evaluation/exact_greedy_paired_gpu_gate_b5d4'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 expected={str(OLD/k):v for k,v in read(OLD/'evidence_manifest.json')['files'].items()}
 expected.update(read(OLD/'implementation_lock.json')['files'])
 expected.update(read(OLD/'source_identity.json')['sha256'])
 checked={p:{'expected':h,'actual':sha(Path(p))} for p,h in expected.items()}
 changed=[p for p,x in checked.items() if x['expected']!=x['actual']]
 patches={}
 for name in ['control.py','runner.py','bridge.py','observation.py','protocol.py']:
  before=OLD/name;after=P/name
  diff=''.join(difflib.unified_diff(before.read_text().splitlines(True),after.read_text().splitlines(True),fromfile=str(before),tofile=str(after)))
  (P/(name+'.diff')).write_text(diff)
  patches[name]={'original':str(before),'original_sha256':sha(before),'patched':str(after),'patched_sha256':sha(after),'changed':bool(diff)}
 for f in P.glob('*.py'):compile(f.read_text(),str(f),'exec')
 result={'status':'PASS' if not changed else 'FAIL','checked_files':checked,'changed_prior_files':changed,'patches':patches,
  'new_modules':{p.name:sha(p) for p in P.glob('*.py') if p.name not in patches},
  'scope':'旧B5-D4代码、执行锁、报告和失败证据保持不变；仅新目录独立副本修复；没有启动GPU'}
 (P/'source_integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 assert not changed,changed
 assert read(P/'cpu_regression_results.json')['status']=='PASS'
 print('摘要核验PASS',len(checked),'项；旧产物未改')
if __name__=='__main__':main()
