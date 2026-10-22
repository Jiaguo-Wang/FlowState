"""重新核对冻结输入、既有实现与本轮桥接身份，不启动Engine。"""
from pathlib import Path
from evaluation.runtime_control_closure_r2c0.prepare import ROOT,read,sha,write
OUT=Path(__file__).resolve().parent
PRIOR=ROOT/'evaluation/runtime_control_closure_r2c0'
def main():
    for rel,h in read(PRIOR/'evidence_manifest.json')['files'].items():assert sha(PRIOR/rel)==h,rel
    identity=read(PRIOR/'source_identity.json')
    for row in identity['files']:assert sha(row['path'])==row['current_sha256'],row['path']
    for p,h in read(PRIOR/'implementation_lock.json')['sha256'].items():assert sha(p)==h,p
    source=read(ROOT/'evaluation/runtime_control_closure_r2b/source_identity.json')['sha256']
    for p,h in source.items():assert sha(p)==h,p
    identity['scope']='R2-C1a新生命周期；重新核验C0全部证据及R2-B实现，不复用其物理绑定'
    identity['c0_manifest_sha256']=sha(PRIOR/'evidence_manifest.json')
    identity['r2b_implementation_sha256']=source
    write(OUT/'source_identity.json',identity)
    print('PASS：来源与旧产物身份已重新核验')
if __name__=='__main__':main()
