"""本阶段一次性冻结完整数据、模型和关键源码摘要，不启动Runtime。"""
from .contracts import *

def main():
    require(not (HERE/'execution_lock.json').exists(),'锁已存在，禁止无声重锁')
    preflight=read(ROOT/'evaluation/strict_compatibility_b4a/cpu_preflight.json')
    verify_hashes(preflight['source_hashes'])
    paths={DATA,PROTOCOL,MANIFEST,B1,ROOT/'evaluation/strict_compatibility_b4a/final_report.md'}
    for directory in ('flowstate','evaluation','tests/runtime','motivation/artifacts/wp3b_gate_20260820'):
        paths.update((ROOT/directory).rglob('*.py'))
    paths.update(p for p in Path('/home/wjg/models/qwen3.5-9b').iterdir() if p.is_file())
    for name in ('retained/evidence.json','image_identity.json'):
        paths.add(ROOT/'evaluation/strict_compatibility_b1c_output/b1c_g000_20261014_r2'/name)
    hashes={str(p):sha(p) for p in sorted(paths)}
    write_new(HERE/'execution_lock.json',{'schema':'b4b0.execution-lock.v1','groups':list(GROUPS),'说明':'完整源Parquet、模型权重与配置、协议及Python实现逐文件SHA-256；未来执行前核验，不以mtime替代','sha256':hashes})
    print('已锁定文件数',len(hashes))
if __name__=='__main__':main()
