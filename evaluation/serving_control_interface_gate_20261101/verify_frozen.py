"""只读复核十五个冻结响应；派生记录只能写入本实验目录。"""
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from . import interface

PKG = Path(__file__).resolve().parent
ROOT = PKG.parents[1]
SOURCE = PKG.parent / 'control_path_separation_20261101'


def sha(path):
    """流式读取证据摘要，不加载模型或设备。"""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    manifest = json.loads((PKG / 'source_manifest.json').read_text())
    for path, digest in manifest['sources_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('原始源码摘要变化：' + path)
    rows = []
    sources = dict(manifest['sources_sha256'])
    for repeat in range(1, 6):
        for policy in ('fs', 'mc', 'kv'):
            base = SOURCE / f'arms/t{repeat}_{policy}/run'
            path = base / 'reconciliation.json'
            r = json.loads(path.read_text())
            full = r['admission_baseline']
            compact = interface.project(full, r)
            post = json.loads((base / 'post_request.json').read_text())
            if compact['selected_components'] != post['state']['selected_components']:
                raise ValueError('冻结请求后身份不一致')
            rows.append({'repeat': repeat, 'policy': policy, 'source': str(path.relative_to(ROOT)),
                         'epoch_id': r['epoch']['epoch_id'], 'epoch_digest': r['epoch']['content_digest'],
                         'selected_ids': r['plan']['selected_ids'], 'J_ms': r['plan']['recovery_cost_ms'],
                         'E': [p['E'] for p in r['plan']['pending']],
                         'G': [p['G'] for p in r['plan']['pending']],
                         'full_baseline_json_bytes': len(json.dumps(full, ensure_ascii=False, separators=(',', ':')).encode()),
                         'minimal_baseline_json_bytes': len(json.dumps(compact, ensure_ascii=False, separators=(',', ':')).encode()),
                         'projection_and_post_identity': 'PASS', 'gpu_executed_in_this_task': False})
            for name in ('reconciliation.json', 'post_request.json', 'summary.json'):
                p = base / name
                sources[str(p.relative_to(ROOT))] = sha(p)
    for path in ['flowstate/strict_control.py', 'flowstate/strict_epoch.py', 'flowstate/optimizer.py',
                 'flowstate/controller.py', 'flowstate/adapters/sglang.py',
                 'evaluation/control_path_remaining_attribution_20261101/final_report.md',
                 'evaluation/control_path_separation_20261101/final_report.md',
                 'evaluation/runtime_control_closure_r2c0/bridge.py',
                 'evaluation/runtime_control_closure_r2c1a/bridge.py',
                 'evaluation/exact_reconciliation_correctness_gate_b5d3/control.py',
                 'evaluation/lightweight_runtime_e1/protocol.py',
                 'evaluation/exact_greedy_paired_gpu_fix_b5d4a/persistence.py']:
        sources[path] = sha(ROOT / path)
    patch = ''.join(''.join(difflib.unified_diff((SOURCE / f).read_text().splitlines(True),
                                               (PKG / f).read_text().splitlines(True),
                                               fromfile=str((SOURCE / f).relative_to(ROOT)),
                                               tofile=str((PKG / f).relative_to(ROOT))))
                    for f in ('bridge.py', 'runner.py', 'control.py', 'observation.py', 'protocol.py', 'codec.py', 'device_entry.py'))
    outputs = {'frozen_response_verification.json': {'status': 'PASS', 'rows': rows},
               'evidence_manifest.json': {'sha256': sources, '说明': '真实历史证据只读，摘要不等于当前GPU认证。'},
               'cpu_environment.json': {'python': sys.version, 'platform': platform.platform(),
                                        'GPU运行': False, 'GPU查询': False, '依赖安装': False}}
    for name, data in outputs.items():
        target = PKG / name
        if target.exists():
            if json.loads(target.read_text()) != data:
                raise ValueError('已有派生产物与复核结果不同：' + name)
        else:
            with target.open('x') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    (PKG / 'implementation.diff').write_text(patch)
    print('冻结响应投影及请求后组件对照：15/15通过；未运行GPU。')


if __name__ == '__main__':
    main()
