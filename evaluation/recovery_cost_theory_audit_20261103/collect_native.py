"""CPU容器内只读保存源码摘录及身份；不导入模型、不访问设备。"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT=Path('/sgl-workspace/sglang/python/sglang/srt')
RANGES={
 'mem_cache/unified_cache/unified_tree_core.py':[(597,755)],
 'mem_cache/unified_cache/components/mamba_component.py':[(54,211)],
 'managers/schedule_batch.py':[(1222,1228),(1249,1375),(2281,2359),(2639,2670)],
 'managers/scheduler.py':[(3528,3730)],
 'model_executor/model_runner.py':[(1486,1535)],
 'models/qwen3_5.py':[(620,688),(688,742),(830,890),(1166,1235)],
 'layers/attention/linear/gdn_backend.py':[(470,729)],
 'layers/attention/flashattention_backend.py':[(1122,1475)],
 'layers/attention/hybrid_linear_attn_backend.py':[(885,946)],
}


def main():
    files=[]
    for rel,ranges in RANGES.items():
        path=ROOT/rel;data=path.read_bytes();lines=data.decode().splitlines()
        files.append({'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),
                      'ranges':[{'start':start,'end':end,'lines':lines[start-1:end]} for start,end in ranges]})
    print(json.dumps({'sglang_version':importlib.metadata.version('sglang'),
        'image':'sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926',
        '说明':'原样第三方源码为只读证据，不是新增执行实现；未挂载GPU、禁网、未启动Engine。',
        'files':files},ensure_ascii=False))


if __name__=='__main__':main()
