"""独立验证集合收益恒等式和报告使用的历史内容锁；禁止覆盖已有记录。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from analyze import BASE, ROOT, EPS


def main():
    parser = argparse.ArgumentParser(description='复核统计恒等式，不运行分配算法或GPU')
    parser.add_argument('--input-dir', type=Path, default=BASE)
    args = parser.parse_args()
    out = args.input_dir.resolve()
    with (out / 'natural_workload_per_condition.csv').open() as f:
        rows = {r['condition_id']: r for r in csv.DictReader(f)}
    errors = []
    with (out / 'calculation_evidence.jsonl').open() as f:
        for line in f:
            item = json.loads(line)
            for policy in item['policy_results'].values():
                benefit = sum(max([item['independent_benefit_matrix_ms'][c][p]
                                   for c in policy['selected']] + [0.0])
                              for p in range(len(policy['E'])))
                errors.append(abs(benefit - (float(rows[item['condition_id']]['J_empty_ms']) - policy['J'])))
    if len(errors) != 2292 or max(errors) > EPS:
        raise ValueError('集合收益恒等式或结果域不一致')
    anchors = {}
    for name in ['source_integrity_before.json', 'artifact_manifest.json']:
        path = ROOT / 'evaluation/p1_strong_baseline_validation' / name
        anchors[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    value = {'旧锁交叉核对来源': anchors, '集合收益恒等式检查数量': len(errors),
             '最大误差_ms': max(errors), '范围': '只读原始Frozen Formal数据，不运行策略'}
    dest = out / 'report_anchor_manifest.json'
    if dest.exists():
        if json.loads(dest.read_text()) != value:
            raise ValueError('已有记录与当前核验不一致；拒绝覆盖')
    else:
        with dest.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    print(json.dumps({'状态': 'PASS', '恒等式数量': len(errors), '最大误差_ms': max(errors)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
