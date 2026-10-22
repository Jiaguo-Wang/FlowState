"""核对因果截断前的结构化工具字段，不解析自然语言计划为ASG。"""
import json
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
out=Path(__file__).resolve().parent
x=json.loads((out/'openhands_source_inspection.json').read_text())
refs={r['session']:r for r in x['records']}
rows=pq.read_table(x['dataset_path'],filters=[('session_id','in',list(refs))],columns=['session_id','messages_json']).to_pylist()
fields=Counter();names=Counter();args=Counter();bad=0;calls=0
for row in rows:
    turn=0
    for m in json.loads(row['messages_json']):
        if m.get('role')=='assistant':
            turn+=1
            if turn==refs[row['session_id']]['turn']:break
        raw=m.get('tool_calls_json')
        if not raw:continue
        for c in json.loads(raw):
            calls+=1;fields.update(c.keys());fn=c.get('function',{});names[fn.get('name','UNKNOWN')]+=1
            try:
                a=json.loads(fn.get('arguments') or '{}')
                if isinstance(a,dict):args.update(a.keys())
            except (ValueError,TypeError):bad+=1
result={'calls':calls,'call_fields':dict(fields),'tool_names':dict(names),'argument_keys':dict(args),'unparsed_arguments':bad,'说明':'仅检查当前截断之前的工具调用；工具调用ID不是Agent Step ID，不将自然语言计划转为图'}
(out/'tool_call_inspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
