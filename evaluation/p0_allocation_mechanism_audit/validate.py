"""最终核对分析产物、案例、图形数据与输入摘要；不写入任何冻结目录。"""
from pathlib import Path
import json, hashlib, xml.etree.ElementTree as ET
OUT=Path(__file__).resolve().parent

def load(name):return json.loads((OUT/name).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(s) for s in (OUT/'all_cases.jsonl').read_text().splitlines()]
lookup={(r['snapshot_id'],r['k']):r for r in rows}
assert len(rows)==573 and len(lookup)==573
checks=0
for r in rows:
    for policy,v in r['policies'].items():
        assert abs(sum(p['cost_ms'] for p in v['per_pending'])-v['cost_ms'])<=1e-9
        if policy!='FlowState':assert abs(sum(v['delta_per_pending_ms'])-r['improvement_ms'][policy])<=1e-9
        checks+=1
for label in 'ABCDEFG':
    r=load('case_'+label+'.json');source=lookup[(r['snapshot_id'],r['k'])]
    for k,v in source.items():assert r[k]==v
    original=json.loads(Path(r['source_file']).read_text().splitlines()[r['source_line']-1]);assert original['snapshot_id']==r['snapshot_id'] and original['k']==r['k']
for fig in ('marginal','cost'):
    ET.parse(OUT/('figure_'+fig+'.svg'))
    for x in load('figure_'+fig+'_data.json'):
        r=load('case_'+x['case']+'.json')
        if fig=='marginal':assert x['first_divergence']==r['first_divergence']
        else:
            for name,values in x['costs'].items():assert values==[p['cost_ms'] for p in r['policies'][name]['per_pending']]
integ=load('integrity.json')
for path,value in integ['before'].items():assert sha(Path(path))==value
result=dict(status='PASS',说明='573 个案例、七个代表性案例及两张矢量图的数据链核对通过；冻结输入末次复核无变化。',cases=len(rows),policy_cost_checks=checks,case_studies=7,vector_figures=2,frozen_files_unchanged=len(integ['before']),gpu_started=False,population_regenerated=False)
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
manifest=dict(status='PARTIALLY_SUPPORTED',说明='独立只读分析产物的 SHA-256；本清单自身不进入摘要。',files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'})
(OUT/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
