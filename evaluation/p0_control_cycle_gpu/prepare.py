"""P0-4 第一轮 GPU 验证·臂目录装配（宿主机执行）：只读复制冻结输入并锁定身份。

每策略一臂（r1_ck3_fs / r1_ck3_mc / r1_ck3_kv），冻结链（全部只读，拒绝覆盖）：
- prepared_inputs / cpu_candidate_prefix_mapping / cpu_gate / cold_reference_C_{A..D}.json
  ← c_formal_perf_gpu/g1_CB_FS（C/K3 正式臂冻结输入，与 G1 prepare 同源同款）；
- t2g1_plans.json ← evaluation/t2g1_three_policy_gpu/t2g1_plans.json（仅作 T4 之后窗口外的
  oracle 对照输入，控制路径绝不读取）；
- workloads_case_C.json ← kvflow_multiworkflow_cpu_main/workloads.json 的 C 结构 case
  （两 Baseline 内核的声明输入域；抽取后与源文件规范化一致性断言）；
- source_identity.json：镜像ID + 镜像内 Runtime 源码哈希（沿用 g1_CB_FS 冻结值）
  + hybrid_entry_sources（被测生产入口与本包接线当前哈希，仅审计记录）。
CPU 侧预核验：g1_CB_FS 冻结运行记录（census 16=16）、三策略 E 向量与公开冻结值一致、
四 Pending 两两 LCP × 三策略 E 向量的串行准入安全性判定（任意前序请求物化只在自身分支
深端 > LCP，不抬高后续 Pending 准入——理论依据 + 引擎侧单终端物化语义）。
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
SRC = ROOT / 'evaluation/c_formal_perf_gpu/g1_CB_FS'
T2G1 = ROOT / 'evaluation/t2g1_three_policy_gpu'
MAIN = ROOT / 'evaluation/kvflow_multiworkflow_cpu_main'
PKG = ROOT / 'evaluation/p0_control_cycle_gpu'
OUTBASE = PKG / 'arms'
NAMES = ('prepared_inputs.json', 'cpu_candidate_prefix_mapping.json', 'cpu_gate.json',
         'cold_reference_C_A.json', 'cold_reference_C_B.json',
         'cold_reference_C_C.json', 'cold_reference_C_D.json')
# 公开冻结 E 向量（t2g1_plans；启动前防文件损坏的语义 sanity）
FROZEN_E = {'FlowState-Greedy': {'C_A': 0, 'C_B': 5120, 'C_C': 6144, 'C_D': 7168},
            'Marconi-Evict': {'C_A': 1024, 'C_B': 1024, 'C_C': 1024, 'C_D': 1024},
            'KVFlow-inspired': {'C_A': 4032, 'C_B': 2048, 'C_C': 0, 'C_D': 0}}
ARM = {'FlowState-Greedy': 'r1_ck3_fs', 'Marconi-Evict': 'r1_ck3_mc', 'KVFlow-inspired': 'r1_ck3_kv'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_once(p, data):
    assert not p.exists(), '拒绝覆盖既有文件：' + str(p)
    p.write_bytes(data if isinstance(data, bytes)
                  else json.dumps(data, ensure_ascii=False, indent=2).encode() + b'\n')


def lcp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def serial_safety_check():
    """四 Pending 串行准入安全性（声明顺序 [C_A,C_B,C_C,C_D]，三策略同一顺序）。

    判定标准（与引擎物化语义一致）：请求单 chunk（chunked_prefill_size=45056 > 最长请求 7208），
    仅 finished 时把请求终端状态 donate 进 radix（mamba_component.prepare_for_caching_req），
    无沿前缀中间物化。因此先执行请求 i 的物化点位于其自身分支深端（历史证据：物化深度≥T_i，
    如 v1.0 失败臂 warm 执行 5154 Token 物化于 5120=T），安全条件为
    T_i > LCP(i,j)：物化状态不在后续 Pending j 的前缀上，j 的 admission==E 合同保持。

    附带登记最坏情形（若引擎存在沿前缀中间物化——本轮无证据）：LCP(i,j) ≤ E_j(policy)
    逐对记录；Marconi-Evict 的 C_A→C_B（LCP 2048 > E 1024）与 KVFlow 的 C_C/C_D→其他
    （E=0）在该假想情形下不安全，由 GPU 第一轮逐请求 admission==E 门槛实测把守。
    """
    prep = json.loads((SRC / 'prepared_inputs.json').read_text())['pendings']
    pens = {p['continuation_id']: p['token_ids'] for p in prep}
    ts = {p['continuation_id']: p['attention_frontier'] for p in prep}
    order = sorted(pens)
    lcp_matrix = {(a, b): lcp(pens[a], pens[b]) for a in order for b in order if a != b}
    terminal_rows = []
    for j_i, j in enumerate(order):
        for i in order[:j_i]:
            ok = ts[i] > lcp_matrix[(i, j)]
            terminal_rows.append({'earlier': i, 'later': j, 'T_earlier': ts[i],
                                  'lcp': lcp_matrix[(i, j)], 'safe': ok})
            assert ok, ('终端物化安全判定失败：%s→%s T=%d ≤ LCP=%d' %
                        (i, j, ts[i], lcp_matrix[(i, j)]))
    worst_case = []
    for policy, evec in FROZEN_E.items():
        for j_i, j in enumerate(order):
            for i in order[:j_i]:
                worst_case.append({'policy': policy, 'earlier': i, 'later': j,
                                   'lcp': lcp_matrix[(i, j)], 'E_later': evec[j],
                                   'hypothetical_safe': lcp_matrix[(i, j)] <= evec[j]})
    return {'order': order,
            'lcp_matrix': {'%s|%s' % k: v for k, v in lcp_matrix.items()},
            'terminal_checks': terminal_rows, 'all_safe': True,
            'worst_case_registry': worst_case,
            'worst_case_all_safe': all(r['hypothetical_safe'] for r in worst_case),
            'basis': '引擎物化语义=请求单 chunk 仅 finished 终端 donate（无中间物化）；'
                     '物化点≥T_i（v1.0 失败臂实测 5154→5120=T 证据）> 两两 LCP ⇒ '
                     '不落在其他 Pending 前缀上；最坏情形（中间物化假想）仅登记，'
                     '由 GPU 第一轮逐请求 admission==E 门槛实测把守'}


def collect_entry_sources():
    """被测生产入口与本包接线文件的当前哈希（审计记录；运行时校验的是镜像内 runtime_sources）。"""
    entry_sources = {}
    for p in sorted((ROOT / 'flowstate').rglob('*.py')):
        entry_sources[str(p)] = sha(p)
    for name in ('select_entry.py',):
        entry_sources[str(PKG.parent / 'p0_control_inclusive_harness' / name)] = sha(PKG.parent / 'p0_control_inclusive_harness' / name)
    for name in ('control.py', 'bridge.py', 'observation.py', 'protocol.py', 'persistence.py', 'runner.py'):
        entry_sources[str(PKG / name)] = sha(PKG / name)
    return entry_sources


def build_arm(policy, safety, g1_source, entry_sources, attempt=1, fix_note=None):
    """按同源冻结输入装配单个臂目录（attempt≥2 时在 source_identity 记录重试与修复说明）。"""
    arm = ARM[policy]
    out = OUTBASE / arm
    assert not out.exists(), '臂目录已存在，拒绝覆盖：' + str(out)
    out.mkdir()
    for name in NAMES:
        write_once(out / name, (SRC / name).read_bytes())
    write_once(out / 't2g1_plans.json', (T2G1 / 't2g1_plans.json').read_bytes())
    case = next(c for c in json.loads((MAIN / 'workloads.json').read_text())['cases'] if c['structure'] == 'C')
    write_once(out / 'workloads_case_C.json', case)   # 抽取副本；下方与源文件规范化一致性断言
    src_case_again = next(c for c in json.loads((MAIN / 'workloads.json').read_text())['cases']
                          if c['structure'] == 'C')
    assert json.loads((out / 'workloads_case_C.json').read_text()) == src_case_again, 'case 抽取副本与源不一致'
    identity = {
        'image_id': g1_source['image_id'],
        'runtime_sources': g1_source['runtime_sources'],
        'hybrid_entry_sources': entry_sources,
        'frozen_input_hashes': {'g1_CB_FS/' + n: sha(SRC / n) for n in NAMES} |
                               {'t2g1_plans.json': sha(T2G1 / 't2g1_plans.json')},
        'serial_safety': safety,
        'note': 't2g1_plans 仅供 runner 在 T4 之后窗口外 oracle 对照；控制路径（control.py）不读取'}
    if attempt > 1:
        identity = identity | {'attempt': attempt, 'fix_note': fix_note}
    write_once(out / 'source_identity.json', identity)
    print('臂目录就绪', policy, '->', out, '(attempt %d)' % attempt)
    return {'arm': arm, 'dir': str(out)}


def cpu_prechecks():
    """CPU 侧语义预核验（census 16=16、三策略 E 向量与公开冻结值一致、串行准入安全）。"""
    rec = json.loads((SRC / 'run' / 'reconciliation.json').read_text())
    assert len(rec['census']['mapped']) == 16 and not rec['census'].get('unmapped'), 'census非16=16'
    t2g1 = json.loads((T2G1 / 't2g1_plans.json').read_text())['plans']
    for policy, evec in FROZEN_E.items():
        for key in ('selected_checkpoint_ids', 'J_ms_current_scope', 'expected_E_G'):
            assert key in t2g1[policy], 't2g1_plans oracle 字段缺失：%s.%s（对照代码字段名须与冻结 schema 一致）' % (policy, key)
        assert {k: v['E'] for k, v in t2g1[policy]['expected_E_G'].items()} == evec, \
            't2g1_plans E向量与公开冻结值不符：' + policy
    return serial_safety_check()


def main():
    OUTBASE.mkdir(exist_ok=True)
    safety = cpu_prechecks()
    g1_source = json.loads((SRC / 'source_identity.json').read_text())
    entry_sources = collect_entry_sources()
    prepared = {}
    for policy in ARM:
        prepared[policy] = build_arm(policy, safety, g1_source, entry_sources)
    write_once(OUTBASE / 'prepare_record.json', {'prepared': prepared, 'serial_safety': safety,
                                                 'source_identity_note': 'runtime_sources 沿用 g1_CB_FS 冻结哈希'})


def retry_arm(policy):
    """失败臂的单臂重装配（fail-loud 纪律：重试须显式发起并归档失败证据）。

    r1_ck3_fs 累计两轮失败（均已在归档目录留证）：
    - attempt1（2026-10-29 07:17）：observation.wrap_eager.execute 处的 batch 是 ForwardBatch
      （无 reqs 属性），当时版本写 batch.reqs[0].rid，首个目标请求 forward 即 AttributeError →
      Scheduler SIGQUIT 全进程树退出；history/shape warm 因 ACTIVE=False 早退未触发。
      修复：run()（ScheduleBatch 层）暂存 ACTIVE_RID，execute() 同一同步调用栈内读取
      （离线回归 offline_test_observation 3/3）。
    - attempt2（2026-10-29 15:5x）：四请求与全部内容门禁已通过，唯 T0–T4 整数恒等式失败——
      bounds 平坦列表使 sum(gaps) 覆盖全部相邻对（含段内 start→end），与 warm/req_total 双计段长，
      恒等式右侧恒多出 warm+req（实测差 7570.6ms 恰等于 warm_total+req_total）。
      修复：gaps 改为仅段间间隙（rpc_end→预热1、段间、预热末→请求1、请求间）。
    其余两臂（未运行）每次追加 identity 补充记录，登记被修文件新旧哈希。
    """
    arm = ARM[policy]
    out = OUTBASE / arm
    stamp = time.strftime('%Y%m%d_%H%M%S')
    # 运行编号 = 已运行的归档数（failed_attempt* / passed_print_bug*）+ 待归档目录若已运行(+1) + 1；
    # 从未运行的装配目录用 discarded_assembly_ 前缀，不占编号（2026-10-29 两次编号公式教训留档）。
    # 臂目录已被手动归档时允许不存在，直接装配下一 attempt。
    archived_runs = sorted(OUTBASE.glob('failed_attempt*_%s_*' % arm)) + \
                    sorted(OUTBASE.glob('passed_print_bug_%s_*' % arm))
    pending_ran = out.exists() and ((out / 'run' / 'failure.json').exists() or
                                    (out / 'run' / 'summary.json').exists())
    attempt = len(archived_runs) + (1 if pending_ran else 0) + 1
    archive = None   # 臂目录已被手动归档时无待归档目录；retry_record 中如实注明（2026-10-29 record 写入教训）
    if out.exists():
        ran = (out / 'run' / 'failure.json').exists() or (out / 'run' / 'summary.json').exists()
        if ran and (out / 'run' / 'summary.json').exists():
            archive = OUTBASE / ('passed_print_bug_%s_%s' % (arm, stamp))   # 门禁已过、仅收尾异常
        elif ran:
            archive = OUTBASE / ('failed_attempt%d_%s_%s' % (attempt - 1, arm, stamp))
        else:
            archive = OUTBASE / ('discarded_assembly_%s_%s' % (arm, stamp))
        out.rename(archive)
        for suffix in ('.runtime.log', '_command.json', '_process.json', '_cleanup.json'):
            p = OUTBASE / (arm + suffix)
            if p.exists():
                p.rename(archive / (arm + suffix))
        prior_records = sorted(OUTBASE.glob('retry_record_%s*.json' % arm))
        for p in prior_records:
            p.rename(archive / p.name)
        status_p = OUTBASE / 'arms_status.json'
        if status_p.exists():
            status_p.rename(archive / (('arms_status_attempt%d.json' % (attempt - 1)) if ran
                                       else 'arms_status_discarded.json'))
    else:
        print('臂目录不存在（已手动归档），直接装配 attempt %d' % attempt)
    safety = cpu_prechecks()
    g1_source = json.loads((SRC / 'source_identity.json').read_text())
    entry_sources = collect_entry_sources()
    fix_note = ('累计修复：①attempt1 observation.py wrap_eager.execute 于 ForwardBatch（无 reqs）改读 '
                'run() 暂存 ACTIVE_RID（offline_test_observation 3/3）；②attempt2 T0–T4 恒等式段长'
                '双计——sum(gaps) 曾含段内相邻对再另加 warm/req_total，改为仅段间间隙；③attempt3 oracle '
                '对照字段名凭记忆写成 frozen[recovery_cost_ms]，冻结 schema 实为 J_ms_current_scope；'
                '④attempt4 summary 落盘后最终打印行仍用旧名 cycle_ms——改用 cycle_ms_full_incl_warm，'
                '并加结论保护（summary 已存在时异常记 failure_after_summary.json，不写 failure.json）；'
                '证据见归档目录')
    prepared = build_arm(policy, safety, g1_source, entry_sources, attempt=attempt, fix_note=fix_note)
    # 未运行的两臂：追加（不覆盖）identity 补充记录，登记被修文件新旧哈希
    for other in ARM:
        if other == policy:
            continue
        op = OUTBASE / ARM[other] / ('identity_update_after_attempt%d_fix.json' % (attempt - 1))
        if op.exists():
            continue
        old = json.loads((OUTBASE / ARM[other] / 'source_identity.json').read_text())
        updates = {}
        for fname in ('observation.py', 'runner.py'):
            key = str(PKG / fname)
            updates[fname] = {'sha_before': old['hybrid_entry_sources'][key],
                              'sha_after': entry_sources[key]}
        write_once(op, {'updated_files': updates,
                        'reason': 'r1_ck3_fs attempt1（observation.py ForwardBatch 无 reqs）与 attempt2'
                                  '（T0–T4 恒等式段长双计）根因修复；本臂尚未运行',
                        'offline_regression': 'offline_test_observation 3/3 + attempt2 落盘数字复算（差值恰='
                                              'warm_total+req_total，证实双计）'})
    failures = {
        1: {'error': 'SystemExit(0)（Scheduler SIGQUIT 级联）',
            'root_cause': 'observation.py wrap_eager.execute 于 ForwardBatch 访问 batch.reqs（属性不存在）；'
                          '首个目标请求 C_A 的首个 forward 触发，history/shape warm 因 ACTIVE=False 早退未触发',
            'fix': 'run() 暂存 ACTIVE_RID（同一同步调用栈），execute() 读取；offline_test_observation 3/3'},
        2: {'error': "ValueError('T0–T4 恒等式失败：存在重复累加或遗漏')",
            'root_cause': 'runner.py 恒等式 bounds 平坦列表使 sum(gaps) 覆盖全部相邻对（含段内 start→end），'
                          '与 warm_total/req_total 双计段长；实测差 7570.6ms == warm_total(6807.9)+req_total(762.6)，'
                          '且四请求全部内容门禁（cold/E/prefill/COW/FA/admission/audit/triton）在该次已通过',
            'fix': 'gaps 改为仅段间间隙；attempt2 落盘数字复算恒等式精确闭合'},
        3: {'error': "KeyError('recovery_cost_ms')",
            'root_cause': 'runner.py oracle 对照凭记忆引用 frozen[recovery_cost_ms]，t2g1 冻结 schema 实际字段为 '
                          'J_ms_current_scope；该次选集对照已逐位一致、J 数值逐位相等（154.34895561599998），'
                          '且 T0–T4 恒等式已通过',
            'fix': "改为 frozen['J_ms_current_scope']；prepare cpu_prechecks 增加 oracle 必备字段断言"},
        4: {'error': "KeyError('cycle_ms')——发生在 summary.json 成功落盘之后的最终控制台打印行",
            'root_cause': '双口径改名（control_cycle_ms / cycle_ms_full_incl_warm）时漏改 print 行的 '
                          "summary['cycle']['cycle_ms']；该次全部门禁与 oracle 已通过、summary 已落盘"
                          '（launcher 判 PASS，rc=1 仅为打印 re-raise），证据见 passed_print_bug 归档',
            'fix': 'print 行改用 cycle_ms_full_incl_warm；except 块加结论保护——summary 已存在时异常记 '
                   'failure_after_summary.json，不再写 failure.json'}}
    write_once(OUTBASE / ('retry_record_%s_attempt%d.json' % (arm, attempt)), {
        'retry_of': arm, 'attempt': attempt, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'archived_dir': (str(archive) if archive else '无：干净起点装配（臂目录此前已手动归档）'),
        'previous_failures': [failures[i] for i in range(1, attempt) if i in failures],
        'prepared': prepared})


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--retry':
        retry_arm(sys.argv[2])
    else:
        main()

