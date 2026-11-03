"""离线归因分析（只读 c_pair_timing_gpu 产物）：把 queue/pregap 窗口内的耗时逐函数分解为
A 类研究审计 vs 原生（B/C 类）操作，为正式实验的《审计分类表》提供实测数字。

- 逐生命周期：queue/pregap/execwin（summary 分解）+ match 窗口（前 cache_state 全量哈希 + 原生 match +
  save 落盘）+ copy 窗口（payload 哈希×3 + CUDA 事件 + 原生 copy）+ TRACE['audit'] 各类合计；
- A 类窗口内估计 = admission cache_state(≈match 内同函数一次) + match 窗口审计部分 + copy 窗口审计部分；
  post-request b5d4_state 的全量哈希在窗口外，单列。
"""
import csv
import json
from pathlib import Path

SRC = Path('/home/wjg/code/FlowState/evaluation/c_pair_timing_gpu')
OUT = Path(__file__).resolve().parent


def main():
    rows = []
    for case in sorted(p for p in SRC.iterdir() if p.is_dir() and (p / 'run' / 'summary.json').exists()):
        s = json.loads((case / 'run' / 'summary.json').read_text())
        if s.get('status') != 'PASS':
            continue
        t = json.loads((case / 'run' / 'native_trace.json').read_text())
        d = s['timing_decomposition']
        m = t['matches'][0]
        # match 窗口 = 前 cache_state 全量哈希 + 原生 match_prefix + save() 落盘（A 类为主）
        match_win = (m['ns_after'] - m['ns_before']) / 1e6
        match_native = m['native_call_ns'] / 1e6
        match_audit = match_win - match_native
        # copy 窗口（E>0 才有）= payload 哈希×3 + 同步/事件 + 原生 copy_from
        copy_win = copy_native = copy_hash = 0.0
        for c in t['copies']:
            copy_win += (c['ns_after'] - c['ns_before']) / 1e6
            copy_native += c['native_host_ns'] / 1e6
        copy_hash = sum(a['ns'] for a in t.get('audit', []) if a['kind'] == 'cow_content_hash') / 1e6
        # cache_state 全量哈希逐次时长；窗口内次数 = 总数−1（末次为 post-request b5d4_state，
        # 其审计项不落盘入 native_trace.json 的最后一批 save——如实按“文件内可见”口径统计并注明）
        full_hashes = [a['ns'] / 1e6 for a in t.get('audit', []) if a['kind'] == 'cache_state_full_hash']
        in_window_hashes = full_hashes[:-1] if len(full_hashes) >= 2 else full_hashes
        rows.append({
            'name': case.name, 'policy': s['policy'], 'E': s['E_executable_prefix'],
            'queue_ms': round(d['queue_ms'], 1), 'pregap_ms': round(d['pregap_ms'], 1),
            'execwin_ms': round(d['execwin_ms'], 1), 'ttft_ms': round(s['ttft_ms'], 1),
            'match_window_ms': round(match_win, 1), 'match_native_ms': round(match_native, 2),
            'match_audit_ms': round(match_audit, 1),
            'copy_window_ms': round(copy_win, 1), 'copy_native_ms': round(copy_native, 2),
            'copy_payload_hash_ms': round(copy_hash, 1),
            'cache_state_hashes_in_window_n': len(in_window_hashes),
            'cache_state_hash_ms_each': [round(x, 1) for x in full_hashes],
            'audit_write_total_ms': round(t.get('audit_write_ns', 0) / 1e6, 1),
            'queue_pregap_sum_ms': round(d['queue_ms'] + d['pregap_ms'], 1),
            'A类窗口内估计_ms': round(sum(in_window_hashes) + match_audit - (in_window_hashes[-1] if in_window_hashes else 0)
                                      + copy_hash + (copy_win - copy_native - copy_hash), 1),
        })
    with (OUT / 'audit_attribution.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['name']:12s} E={r['E']:5d} queue={r['queue_ms']:7.1f} pregap={r['pregap_ms']:6.1f} "
              f"match窗={r['match_window_ms']:7.1f}(原生{r['match_native_ms']:.2f}) copy窗={r['copy_window_ms']:6.1f}"
              f"(哈希{r['copy_payload_hash_ms']:6.1f}) 全量哈希次={[round(x) for x in r['cache_state_hash_ms_each']]}")


if __name__ == '__main__':
    main()
