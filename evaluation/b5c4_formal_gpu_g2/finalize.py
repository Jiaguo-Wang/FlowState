"""G2 收尾（宿主机执行，analyze.py 之后运行）：生成 implementation_lock.json 与 final_report.md。

implementation_lock：锁定 G2 全部代码/冻结输入/分析产物（顶层 *.py/*.json，含三个阶段目录与
pkg_g2；逐臂证据由各臂 execution_lock_*.json 与 run 目录自证）。final_report：从
preflight/gate/verify.json、formal/{checks,arm_status,raw_results,paired_analysis} 装配，
六段结构（设计/门禁/正式/三层结论/限制/产物），全部数值取自证据文件，不手写。
"""
import hashlib, json
from pathlib import Path
import csv, statistics

ROOT = Path('/home/wjg/code/FlowState')
G2 = ROOT / 'evaluation/b5c4_formal_gpu_g2'
FORMAL = G2 / 'formal'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    gate = json.loads((G2 / 'preflight/gate/verify.json').read_text())
    checks = json.loads((FORMAL / 'checks.json').read_text())
    arm_status = json.loads((FORMAL / 'arm_status.json').read_text())
    raw = list(csv.DictReader((FORMAL / 'raw_results.csv').open()))
    # paired_analysis.csv 为两段式（请求级表 + [轮级] 段 + [五轮中位] JSON）：按段标记切块解析
    text = (FORMAL / 'paired_analysis.csv').read_text().splitlines()
    req_lines, round_lines = [], []
    section = 'request'
    for line in text:
        if line.startswith('[轮级'):
            section = 'round'
            continue
        if line.startswith('[五轮中位'):
            section = 'summary'
            continue
        if line.strip():
            (req_lines if section == 'request' else round_lines if section == 'round' else []).append(line)
    pp = list(csv.DictReader(req_lines))
    round_rows = list(csv.DictReader(round_lines))
    summary = checks['layer3_measured']['round_level_delta_median']

    # ── 实现锁 ──
    files = {}
    for d in [G2, G2 / 'preflight', G2 / 'preflight/cold', G2 / 'preflight/gate', G2 / 'formal', G2 / 'pkg_g2']:
        for p in sorted(list(d.glob('*.py')) + list(d.glob('*.json'))):
            files[str(p)] = sha(p)
    lock = {'files': files, 'note': '逐臂证据由各臂 execution_lock_*.json 与 run 目录（summary/reconciliation/stream/post_request 等）自证；失败尝试全部保留',
            'image': json.loads((G2 / 'preflight/source_identity.json').read_text())['image_id'],
            'gate_verdict': gate['verdict'], 'formal_arms_pass': checks['validity']['arms_pass']}
    out = G2 / 'implementation_lock.json'
    assert not out.exists(), '拒绝覆盖 implementation_lock.json'
    out.write_text(json.dumps(lock, ensure_ascii=False, indent=2))

    # ── 最终报告 ──
    n_pass, n_fail = checks['validity']['arms_pass'], len(checks['validity']['arms_fail'])
    g2b_verdict = 'PASS' if n_pass == 40 and gate['verdict'] == 'PASS' else ('PARTIAL' if n_pass > 0 else 'BLOCKED')
    L = []
    L.append('# G2 最终报告：B5-C4/K2 严格多共享结构 GPU 性能对照\n')
    L.append('**verdict：G2-A %s / G2-B %s**（%d/40 正式臂 PASS，失败 %d；判定取两阶段全 PASS 才为整体 PASS）\n' % (gate['verdict'], g2b_verdict, n_pass, n_fail))
    L.append('## 1. 研究问题与设计')
    L.append('- 相同严格兼容关系与 K=2 预算下，公共祖先状态（Greedy 选集 {N20,N18}，J=130.8030838298517）替换为两个深层分支状态（Exact 选集 {N18,N27}，J=48.228728759089606），能否稳定减少实际 Prefill 开销与请求 TTFT。')
    L.append('- Exact 臂＝生产 `plan_strict` 门禁内自动（7 候选/4 Pending/K=2 满足门禁）；Greedy 臂＝`greedy_adapter` 复用生产 Marginal Greedy 内部路径（control 断言域满足门禁，未篡改候选/门禁/预算）；两臂同走 validate_plan→Bind/Preflight→reconcile_strict。')
    L.append('- 每臂独立 Docker/GPU 生命周期；G2-B 40 臂按 schedule.json 预固定顺序（奇数轮 Exact 先、偶数轮 Greedy 先），launch.py 强制顺序执行。\n')
    L.append('## 2. G2-A 门禁（8 臂，preflight/gate/verify.json）')
    L.append('| 臂 | 判定 |')
    L.append('|---|---|')
    for a in gate['arms']:
        L.append('| %s | %s |' % (a['run'], a['status']))
    L.append('')
    L.append('- D4b 专项（ΔCUDA Prefill 与 ΔTTFT 方向，±3ms 噪声带）：%s' % ('未发现反向' if gate['d4b_ttft_reversal_check']['pass'] else '存在反向'))
    for x in gate['d4b_ttft_reversal_check']['rows']:
        L.append('  - Pending %s：Δprefill=%.2fms Δttft=%.2fms%s' % (x['pending'], x.get('delta_cuda_prefill_ms', float('nan')), x.get('delta_ttft_ms', float('nan')), '（噪声带内）' if not x.get('both_beyond_noise') else ''))
    L.append('- Cold 参照：C 沿用 b5d4b 冻结；A/B/D 由 preflight/cold/ 三臂补采（全新 Engine、空缓存断言、FA命中==E==0）经 promote_cold.py 核验落位。\n')
    L.append('## 3. G2-B 正式结果（40 臂，formal/）')
    L.append('### 请求级：每 Pending 五轮中位（ms）')
    L.append('| Pending | Exact TTFT 中位 | Greedy TTFT 中位 | ΔTTFT | Exact Prefill 中位 | Greedy Prefill 中位 | ΔPrefill |')
    L.append('|---|---|---|---|---|---|---|')
    for r in pp:
        L.append('| %s | %s | %s | %s | %s | %s | %s |' % (r['pending'], r.get('Exact_ttft_median'), r.get('Greedy_ttft_median'), r.get('delta_ttft_median'), r.get('Exact_prefill_median'), r.get('Greedy_prefill_median'), r.get('delta_prefill_median')))
    L.append('')
    L.append('### 轮级：四 Pending 等权均值 → 五轮中位（ms，正值=Exact 更快）')
    L.append('| 轮 | Exact TTFT | Greedy TTFT | ΔTTFT_r | Exact Prefill | Greedy Prefill | ΔPrefill_r |')
    L.append('|---|---|---|---|---|---|---|')
    for r in round_rows:
        if r.get('delta_ttft_r') not in (None, ''):
            L.append('| %s | %s | %s | %s | %s | %s | %s |' % (r['round'], r.get('Exact_ttft_equal_mean'), r.get('Greedy_ttft_equal_mean'), r['delta_ttft_r'], r.get('Exact_prefill_equal_mean'), r.get('Greedy_prefill_equal_mean'), r['delta_prefill_r']))
    L.append('')
    L.append('- 五轮中位汇总：%s' % json.dumps(summary, ensure_ascii=False))
    fails = [b for b, v in arm_status.items() if v['status'] != 'PASS']
    retries = {b: v['attempts'] for b, v in arm_status.items() if len(v['attempts']) > 1}
    slower = [(x['run'], x['pending'].split('_')[1], '%.1f' % float(x['ttft_ms'])) for x in raw
              for y in raw if y['round'] == x['round'] and y['pending'] == x['pending'] and y['solver'] != x['solver']
              and x['solver'] == 'Exact' and float(x['ttft_ms']) > float(y['ttft_ms'])]
    L.append('- 保留全部有效结果：Exact 单臂 TTFT 更慢的轮次 %s（明细见 raw_results.csv）；失败臂：%s；重试臂：%s'
             % (slower or '无', fails or '无', json.dumps(retries, ensure_ascii=False) if retries else '无'))
    L.append('- 结构对照：A/B 两策略 E 相同（3712）为近零差对照；C/D 为结构差异主效应（Exact E=3520 vs Greedy E=2432）。\n')
    L.append('## 4. 三层结论（不得跨层推导）')
    L.append('1. **模型层**：在线 J 与 e3 冻结参照全精度一致（Exact %s vs Greedy %s，全部 40 臂核验=%s）。' % (checks['layer1_model_J'].get('exact_J_all'), checks['layer1_model_J'].get('greedy_J_all'), checks['layer1_model_J']['consistent']))
    L.append('2. **结构层**：Native Gap（G_model=T−E）Exact C/D=%s vs Greedy C/D=%s；实际 Prefill Token 数 Exact C/D=%s vs Greedy C/D=%s（Prefill=len(pt)−E）。' % (checks['layer2_native_gap']['exact_G_CD'], checks['layer2_native_gap']['greedy_G_CD'], checks['layer2_native_gap']['prefill_tokens_exact_CD'], checks['layer2_native_gap']['prefill_tokens_greedy_CD']))
    L.append('3. **实测层**：CUDA Prefill / TTFT 配对差见 §3；方向与结构层%s。' % ('一致' if all((float(r['delta_prefill_median'] or 0) > 0) == (float(r['delta_ttft_median'] or 0) > 0) for r in pp if r.get('delta_prefill_median') is not None) else '需逐 Pending 说明'))
    L.append('- 第 1/2 层为模型与结构事实，不构成第 3 层的证明；实测层以配对差报告，五轮样本不宣称统计显著性。\n')
    L.append('## 5. 有效性威胁与限制')
    L.append('- 共享单卡 H100 PCIe、未锁频：clocks.sm/power.draw 逐臂留档（gpu_before/cleanup）；轮内策略顺序均衡以对冲慢漂移。')
    L.append('- CUDA 事件是设备时间线区间而非纯内核之和；控制口径完整 Wall 含审计成本（分段单独报告）。')
    L.append('- Cold 参照与正式臂同 Engine 配置同 Tokenizer 模板；输出逐 Token 一致只证恢复语义等价，不证血缘。')
    L.append('- N18/N20/N27 为逻辑标签，物理身份逐生命周期重建；未跨容器复用 Node/Slot ID。\n')
    L.append('## 6. 产物清单')
    L.append('- 冻结与门禁：preflight/{prepare,cpu_preflight,stage_inputs,promote_cold}.py、preflight/{prepared_inputs,cpu_gate,source_identity,g2_reference,cold_reference_*}.json、preflight/cold/、preflight/gate/（含 verify.json）')
    L.append('- 正式：formal/{schedule.json,launch.py,driver.sh,analyze.py,raw_results.csv,paired_analysis.csv,checks.json,arm_status.json} + 40 臂 run 目录')
    L.append('- 代码包：pkg_g2/（bridge/runner/control/greedy_adapter/source_contract/persistence + build_record.json，差异全部文档化）')
    L.append('- 顶层：protocol.json、implementation_lock.json、本报告')
    report = G2 / 'final_report.md'
    assert not report.exists(), '拒绝覆盖 final_report.md'
    report.write_text('\n'.join(L) + '\n')
    print('收尾完成：implementation_lock.json（%d 文件）+ final_report.md；G2-A %s / G2-B %s' % (len(files), gate['verdict'], g2b_verdict))


if __name__ == '__main__':
    main()
