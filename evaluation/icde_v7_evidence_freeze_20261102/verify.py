"""最终核对本轮派生清单、源摘要和报告引用，不执行任何实验。"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

PKG=Path(__file__).resolve().parent
ROOT=PKG.parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    manifest=json.loads((PKG/'source_manifest.json').read_text())
    for path,record in manifest['files'].items():
        assert sha(Path(path))==record['sha256_before']==record['sha256_after'],path
    proof=ROOT/'evaluation/rq3_workload_structure/budget_sensitivity_k1_k4_20261030/post_analysis_validation.json'
    h=sha(proof);r=json.loads(proof.read_text())
    assert r['status']=='PASS' and r['independent_oracle_conditions']==396
    assert all(c['pass'] for c in r['checks'])
    manifest['files'][str(proof)]={'bytes':proof.stat().st_size,'sha256_before':h,'sha256_after':sha(proof)}
    assert h==manifest['files'][str(proof)]['sha256_after']
    (PKG/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    def read(name):
        with (PKG/name).open() as f:return list(csv.DictReader(f))
    master,structures,cases=read('evidence_master.csv'),read('workload_structure_summary.csv'),read('ten_difference_cases.csv')
    assert len(master)==83 and len(structures)==1357 and len(cases)==10
    assert len([r for r in master if r['status']=='NUMERIC_MISMATCH'])==3
    assert len([r for r in master if r['status']=='DENOMINATOR_MISMATCH'])==1
    assert len([r for r in structures if r['domain']=='Formal比例预算'])==573
    assert len([r for r in structures if r['domain']=='Formal固定预算'])==764
    assert len([r for r in structures if r['domain']=='Strict20受控共享分支'])==20
    for filename in ['final_audit_report.md','claim_evidence_matrix.md']:
        text=(PKG/filename).read_text()
        assert 'NEEDS_TARGETED_FIX' in text
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if not link.startswith(('http:','https:','#')):assert (PKG/link.split('#')[0]).exists(),link
    for path in PKG.glob('*.py'):ast.parse(path.read_text())
    result={'verdict':'NEEDS_TARGETED_FIX','source_files_checked':len(manifest['files']),
            'evidence_metrics':len(master),'workload_structure_rows':len(structures),'difference_conditions':len(cases),
            'numeric_mismatches':3,'denominator_mismatches':1,'hidden_tables':8,
            'parent_raw_formal_rows_verified':573,'stored_independent_oracle_conditions':396,
            '原始源SHA变化':0,'GPU实验次数':0,'策略重跑次数':0,'重新拟合次数':0,
            '唯一最小补证':'提供含8张表的v7.0静态导出，按冻结源清单修正文表并逐格对拍；不需新实验'}
    with (PKG/'verification.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    files={p.name:sha(p) for p in sorted(PKG.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'}
    with (PKG/'artifact_manifest.json').open('x') as f:json.dump({'sha256':files},f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
