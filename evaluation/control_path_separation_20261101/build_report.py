"""只读取本轮证据，核对完整性并生成独立结果包；不启动任何实验。"""
import hashlib
import json
from pathlib import Path
import statistics

from . import analyze
from .codec import unpack

PKG = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_all(lock):
    records = []
    initial = set()
    start_ns = __import__('time').monotonic_ns()
    for spec in lock['schedule']:
        arm = PKG/'arms'/spec['arm']
        if not arm.exists():
            continue
        run = arm/'run'
        s = json.loads((run/'summary.json').read_text())
        r = json.loads((run/'reconciliation.json').read_text())
        assert s['status'] == 'PASS' and s['oracle_match'] and s['cycle']['identity_exact']
        assert not (run/'failure.json').exists() and not (run/'failure_after_summary.json').exists()
        assert json.loads((arm/'host_exit.json').read_text())['returncode'] == 0
        assert json.loads((run/'engine_cleanup.json').read_text())['shutdown_returned']
        assert (arm/'gpu_cleanup.json').exists()
        assert json.loads((arm/'device_binding.json').read_text())['host_gpu_uuid'] == lock['gpu_uuid']
        assert len(r['operations']) == 13 and all(o['returned'] for o in r['operations'])
        assert r['reconciliation']['status'] == 'PASS'
        assert len(r['before']['free_slots']) == 12 and len(r['after']['free_slots']) == 25
        assert sum(bool(n['slots']) for n in r['after']['nodes']) == 3
        assert r['selected_payload_before'] == r['selected_payload_after']
        assert all(x['cold_output_match'] for x in s['pending_rows'])
        assert all(x['prefill_token_count'] == x['G']+32 for x in s['pending_rows'])
        assert all(w['new_entries'] == 0 for w in s['triton_new_kernels'] if w['window'] == 'post_warm_block→post_request')
        for name, expected in lock['input_sha256'].items():
            assert hashlib.sha256((arm/name).read_bytes()).hexdigest() == expected
        mapping = {c['physical_checkpoint_id']: c['cpu_checkpoint_id'] for c in r['census']['mapped']}
        candidates = {mapping[c['checkpoint_id']]: {'token_ids': c['token_ids'], 'L': c['token_position_L'],
                       'payload': [(d['component'], d['sha256']) for d in c['payload']]}
                      for c in r['raw_epoch']['candidates']}
        edges = {mapping[e['checkpoint_id']]+'/'+e['continuation_id']: (e['status'], e['conditions'])
                 for e in r['raw_epoch']['edges']}
        assert len(candidates) == 16 and len(edges) == 64
        assert all(v[0] != 'UNKNOWN' for v in edges.values())
        signature = digest({'candidates': candidates, 'edges': edges, 'FA': r['fa_map_before']})
        initial.add(signature)
        rpc = next(p for p in run.glob('rpc_*') if (p/'request.json').exists()
                   and json.loads((p/'request.json').read_text()).get('action') == 'b5d4_apply')
        if spec['mode'] == 'serving':
            decoded, _ = unpack(json.loads((rpc/'response.raw').read_text()))
            assert decoded == r
            assert s['separation']['report_timing']['start_ns'] >= s['cycle']['t4_ns']
        else:
            assert json.loads((rpc/'response.raw').read_text()) == r
        assert len(list((run/'control_evidence').glob('*.json'))) == 6
        evidence = [run/'summary.json', run/'reconciliation.json', run/'post_request.json',
                    run/'engine_cleanup.json', arm/'gpu_cleanup.json', arm/'device_binding.json', rpc/'response.raw']
        records.append({'spec': spec, 'initial_state_signature': signature, 'operations': 13,
                        'edges': 64, 'unknown': 0, 'free_slots': [12, 25],
                        'selected': s['selection']['selected_cpu'],
                        'E': [p['E'] for p in s['pending_rows']], 'G': [p['G'] for p in s['pending_rows']],
                        'prefill': [p['prefill_token_count'] for p in s['pending_rows']],
                        'source_hashes': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in evidence}})
    assert len(initial) == 1, '不同实验臂初始候选/矩阵/内容/FA映射不等价'
    result = {'start_ns': start_ns, 'end_ns': __import__('time').monotonic_ns(),
              'performed_after_gpu_measurements': True, 'records': records, 'initial_signatures': list(initial)}
    (PKG/'verified_evidence_inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return records


def main():
    analyze.main()
    lock = json.loads((PKG/'execution_lock_gpu1.json').read_text())
    records = validate_all(lock)
    stats = json.loads((PKG/'statistics.json').read_text())
    assert len(records) == 26 and stats['n_completed'] == 26
    comparison = stats['paired']['serving-original']
    assert comparison['control_cycle_ms']['n'] == 5
    net = []
    for baseline in ['MC', 'KV']:
        p = stats['paired']['serving_FS-'+baseline]
        assert p['full_cycle_ms']['n'] == 5
        if p['full_cycle_ms']['median'] < 0 and p['full_cycle_ms']['wins'] >= 4 and p['audit_inclusive_ms']['median'] < 0:
            net.append(baseline)
    control_gain = comparison['control_cycle_ms']['median'] < 0
    full_gain = comparison['full_cycle_ms']['median'] < 0
    if net:
        verdict = 'SUCCESS_NET_GAIN'
    elif full_gain and control_gain:
        verdict = 'SUCCESS_FULL_CYCLE'
    elif control_gain:
        verdict = 'SUCCESS_CONTROL'
    else:
        verdict = 'NO_MEASURABLE_GAIN'
    old = json.loads((PKG/'original_source_manifest.json').read_text())['files']
    for path, expected in old.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    for path, expected in lock['code'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected

    lines = ['# 控制路径分离：性能分析', '', '**判定：'+verdict+'。**', '',
             '工作负载为冻结受控C/K3：16候选、4 Pending、K=3，Qwen3.5-9B、SGLang v0.5.17、单H100 PCIe TP=1。所有本轮测量在GPU1的新Engine生命周期中完成，不混入历史GPU0/失败批量路径结果。Smoke仅用于正确性，不进入性能分母。', '',
             '## 新旧FlowState五轮配对（ms）', '',
             '|轮次|Control差|Full差|Audit-inclusive差|', '|---|---:|---:|---:|']
    for i in range(5):
        lines.append('|'+str(i+1)+'|'+'|'.join(f'{comparison[m]["values"][i]:+.3f}' for m in ['control_cycle_ms', 'full_cycle_ms', 'audit_inclusive_ms'])+'|')
    lines += ['', '差值为serving−original，负值表示新路径更短。第五轮Control退化保留。', '',
              '## 五轮原始分布（ms）', '',
              '|阶段/模式/策略|指标|均值|中位数|最小|最大|', '|---|---|---:|---:|---:|---:|']
    for group, values in stats['groups'].items():
        for metric in ['control_cycle_ms', 'full_cycle_ms', 'audit_inclusive_ms', 'sum_ttft_ms', 'sum_cuda_prefill_ms']:
            x = values[metric]
            lines.append(f'|{group}|{metric}|{x["mean"]:.3f}|{x["median"]:.3f}|{x["min"]:.3f}|{x["max"]:.3f}|')
    lines += ['', '## 同轮配对汇总（ms）', '', '|比较|指标|差值中位数|胜/平/负|范围|', '|---|---|---:|---:|---:|']
    for group, values in stats['paired'].items():
        for metric in analyze.METRICS:
            x = values[metric]
            lines.append(f'|{group}|{metric}|{x["median"]:+.3f}|{x["wins"]}/{x["ties"]}/{x["losses"]}|[{x["min"]:+.3f},{x["max"]:+.3f}]|')
    lines += ['', '## 测量边界与分离成本', '',
              'Control为T3−T0，Full为T4−T0，均保留原始定义。T5是在全部原后验/Cold/Oracle检查、报告格式化及附加审计记录完成后、summary写盘和Engine清理前读取；Audit-inclusive为T5−T0。它不是包含模型启动和两轮历史重放的完整生命周期。', '',
              'Control RPC包括控制和RPC尾段，不与Control重复相加；八个timing阶段与clock六段间隙不能混合求和。Selection Call嵌套在Selection阶段。ΣTTFT与ΣCUDA是四个串行受控请求等权总和，不能视为自然并发吞吐。所有Full/T5都来自同一臂直接时间戳，未拼接平均值或扣除估计审计成本。', '',
              'Journal仍在原时刻捕获完整快照，仍在请求前fsync，只改变紧凑JSON格式；完整响应先持久化再解码。唯一延后的是已有完整持久化响应的重复缩进展示文件。其Wall/进程CPU、编码/压缩/解码开销、字节量及起止时间在separation_audit.json和performance_results.csv中；这些都是实际成本，不是免费优化。', '',
              '新格式化在T4后同步执行，不与本臂四个目标请求竞争。编码、压缩和解码均在请求前，仍计入Full。未新增GPU调用或异步取证线程；CPU计时为process_time差，可能包含同期线程，不能声称独占函数CPU时间。原路径未单独记录的编码CPU成本保持UNKNOWN。', '',
              '全量GPU状态SHA、29次协调观察、13次逐项驱逐及前后验全部保留。GPU同步、内容复制和RPC数量没有减少；收益是取证CPU/字节/重复展示路径的组合效果。本轮没有单独消融各项修改，不能按阶段中位数相减给每项分配因果收益。', '',
              '三策略虽然共享取证代码与要求，保留集不同会改变FA保护区等审计工作量；不能把这类研究审计成本直接称作生产必须成本。形状预热仍计入Full，各自编译与预热成本实际保留。', '',
              '## 正确性、适用域和负面结果', '',
              '65项CPU测试、1个GPU Smoke及25个性能臂原合同均通过。26臂初始候选Token/内容、严格矩阵和FA映射签名一致，64条配对无UNKNOWN；每臂13次真实驱逐，3个保留状态及槽位账本正确。逐请求实际E/G/Prefill、Cold、COW与FA保护见pending_results.csv和verified_evidence_inventory.json。跨Workflow物理血缘仍为PROVENANCE_UNVERIFIED。', '',
              'req_generation不是Mamba槽位Generation。未引入状态缓存或减少观测，安全限定原连续Scheduler/独占受控Engine合同；完整复原的ABA不能靠端点SHA证明不存在，不能据此升级为通用并发事务或长期Pin保证。codec SHA仅认证传输完整性，不是GPU状态认证。', '',
              'GPU0初始已有其他SGLang任务，未启动模型或干预；预检失败及首次启动前GPU1绑定修订均保存。GPU1每臂检查进程/显存，另一张卡和CPU未声明独占。五轮只是该共享服务器上的描述性证据，不声称生产P95、显著性或普遍优势。', '',
              '人类可读Baseline名称为Marconi-Evict Adaptation和KVFlow-inspired；均为受控策略级适配，不是原论文完整Serving System。正式FlowState入口实际为n=16下的Marginal Greedy分支，无新求解器。', '',
              '## 证据索引', '',
              '逐臂：performance_results.csv；逐Pending：pending_results.csv；逐轮差：paired_comparison.csv；均值/中位/范围与全部原值：statistics.json；完整物理与来源证据：verified_evidence_inventory.json、arms/*/run；代码/协议锁：execution_lock_gpu1.json、implementation.diff；原版本与历史未改：original_source_manifest.json。']
    (PKG/'performance_analysis.md').write_text('\n'.join(lines)+'\n')

    final = ['# 最终报告', '', '**判定：'+verdict+'。**', '',
             '## 实际实施与安全边界', '',
             '1. 必要在线安全操作全部保留：Scope/静止边界、真实FA前沿、严格Epoch与兼容关系、Node/Slot/组件/引用/别名/容量、Bind/Preflight、13项驱逐和逐项Postcheck。',
             '2. 内容SHA虽昂贵，但当前仍有安全与保护门禁消费者；本轮未证明可以删除或延迟，所有全量FA/Recurrent SHA保持原时刻真实执行。',
             '3. 安全优化仅为Journal紧凑编码、完整apply响应的可校验无损封装，以及将已有持久化副本的重复缩进展示移至T4后。原始完整响应仍在请求前fsync，坏响应或请求前必需的持久化错误继续拒绝请求；T4后重复展示写入失败只能拒绝整臂认证，不能撤销已执行请求。',
             '4. 没有减少GPU同步、RPC次数或全量状态复制；减少的是CPU格式化、线上字节和重复展示的关键路径成本，不涉及上一轮失败的批量读取。',
             '5. 保持Strict Epoch、物理绑定与实际状态检查；不使用Slot Generation替代检查。现有跨窗口Generation缺口没有被修复，不能声称排除任意并发ABA，只在原已验证的连续控制范围内等价。', '',
             '## 定量结果（五轮同轮配对中位差，ms）', '',
             '|比较|Control|Full T0–T4|Audit-inclusive T0–T5|ΣTTFT|ΣCUDA Prefill|', '|---|---:|---:|---:|---:|---:|']
    for label, group in stats['paired'].items():
        final.append('|'+label+'|'+'|'.join(f'{group[m]["median"]:+.3f}' for m in ['control_cycle_ms', 'full_cycle_ms', 'audit_inclusive_ms', 'sum_ttft_ms', 'sum_cuda_prefill_ms'])+'|')
    final += ['', '负值表示左侧更短。全部逐轮正负差、均值、中位数和范围见performance_analysis.md及CSV，未筛掉慢轮次。', '',
              '6. 新旧FlowState控制时间：配对中位差'+f'{comparison["control_cycle_ms"]["median"]:+.3f}'+'ms，胜/平/负='+f'{comparison["control_cycle_ms"]["wins"]}/{comparison["control_cycle_ms"]["ties"]}/{comparison["control_cycle_ms"]["losses"]}。第五轮退化保留。',
              '7. Native恢复语义及8 Token Cold逐Token结果保持；相同选集下CUDA/TTFT仍有实测波动，不能因E/G一致宣称时延完全不变。每个Pending实测值见pending_results.csv。',
              '8. Full的变化与Audit-inclusive的变化分别报告；T4后CPU/Wall未隐去，T5不含summary写盘/Engine退出，不等于完整服务生命周期。',
              '9. 相对Baseline净收益：按预先保守条件（Full中位差<0、至少4/5获胜且Audit-inclusive中位差<0），满足的对象为'+('、'.join(net) if net else '无')+'。不把仅Prefill/TTFT改善升级为完整周期净收益。',
              '10. 仍需同步的大量内容审计是本轮未分离的主要剩余成本；必要Runtime成本下限仍UNKNOWN。尚无跨模型、自然并发或生产低审计路径的外推证据。', '',
              '## 完成范围与证据', '',
              '65项CPU测试通过；26个GPU1独立生命周期（1 Smoke、10新旧配对臂、15三策略臂），338次真实Recurrent-only驱逐，104个目标请求的原Native/输出合同通过。E=0请求为合法完整重算，只有E>0的请求认证COW；不将所有请求都称作Recurrent命中。Cold使用冻结独立参考，本轮未另外新跑Cold生命周期。', '',
              '26臂初始状态物理签名相同；每臂Resident 16→3、Free Slots 12→25，全部响应完整、FA/组件保护及Pool账本通过。三策略同协议、同GPU1、同输入/候选/K/模型，未拼接历史基线。全部负向结果保留；GPU0最初资源预检失败且未启动模型，换卡发生于首次模型启动前。', '',
              '精确实现身份以execution_lock_gpu1.json和最终manifest为准，基线Git提交8cba094d2c5466759b70e4a5ef8f4824b61aade5；当前代码仅新增隔离实验包。旧十五臂、生产源码及上一轮失败结果未修改。完整命令、环境、日志、物理记录和清理见execution文件、arms/*与最终清理记录。', '',
              'Marconi-Evict Adaptation / KVFlow-inspired是适配策略，不是官方完整系统；PROVENANCE_UNVERIFIED保持不变。', '',
              '本轮完成后停止，不修改论文，不继续扩大优化或实验。']
    (PKG/'final_report.md').write_text('\n'.join(final)+'\n')
    (PKG/'decision.json').write_text(json.dumps({'decision': verdict, 'n_lifecycles': 26,
        'online_state_operations_changed': False, 'new_old_pair_control_median_ms': comparison['control_cycle_ms']['median'],
        'new_old_pair_full_median_ms': comparison['full_cycle_ms']['median'],
        'new_old_pair_audit_median_ms': comparison['audit_inclusive_ms']['median'],
        'net_gain_against': net}, ensure_ascii=False, indent=2))
    print('最终判定', verdict)


if __name__ == '__main__':
    main()
