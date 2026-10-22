"""只读核对历史输入、允许的生产接线版本及镜像源码身份。"""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
B4=ROOT/'evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    old=read(B4/'execution_lock.json')['sha256'];new=read(ROOT/'evaluation/runtime_control_closure_r2b/source_identity.json')['sha256']
    rows=[];bad=[]
    for p,h in old.items():
        current=sha(p);expected=h;basis='B4-B0冻结锁'
        if p in (str(ROOT/'flowstate/optimizer.py'),str(ROOT/'flowstate/controller.py')):
            expected=new.get(p,new.get(str(Path(p).relative_to(ROOT))));basis='R2-A/R2-B已授权生产接线'
        rows.append({'path':p,'historical_sha256':h,'expected_sha256':expected,'current_sha256':current,'basis':basis,'pass':current==expected})
        if current!=expected:bad.append(p)
    for rel,h in read(B4/'manifest.json')['files'].items():
        p=B4/rel
        if sha(p)!=h:bad.append(str(p))
    image=read(OUT/'image_preflight.json');sources=read(OUT/'runtime_sources.json')
    current_sources={p:hashlib.sha256(v.encode()).hexdigest() for p,v in sources.items()};current_sources.update(image['sources'])
    expected_sources=read(ROOT/'evaluation/strict_compatibility_b1c_output/b1c_g000_20261014_r2/retained/evidence.json')['source_identity']
    for p,h in expected_sources.items():
        if current_sources.get(p)!=h:bad.append(p)
    snap=read(B4/'g001/snapshot.json')
    for n in ('sglang','transformers'):
        if image['versions'][n]!=snap['configuration'][n+'_version']:bad.append(n)
    result={'status':'PASS' if not bad else 'BLOCKED','mismatches':bad,'files':rows,'runtime_sources':current_sources,'versions':image['versions'],'snapshot_path':str(B4/'g001/snapshot.json'),'snapshot_sha256':sha(B4/'g001/snapshot.json'),'configuration':snap['configuration'],'说明':'旧锁不覆盖；仅Optimizer与Controller采用已经完成的R2生产版本。当前GPU状态尚未认证。'}
    write(OUT/'source_identity.json',result)
    print(result['status'],bad)
    if bad:raise SystemExit(1)
if __name__=='__main__':main()
