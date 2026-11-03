"""汇总固定五轮原始结果及完整性证据，不改写历史产物。"""
import hashlib
import json
from pathlib import Path
import statistics
from . import analyze

PKG = Path(__file__).resolve().parent


def main():
    analyze.main()
    lock = json.loads((PKG/'execution_lock.json').read_text())
    specs = [s for s in lock['schedule'] if s['phase'] == 'paired']
    assert len(specs) == 10
    data = {}
    inventory = []
    for spec in [s for s in lock['schedule'] if s['phase'] in ['paired', 'smoke']]:
        base = PKG/'arms'/spec['arm']
        s = json.loads((base/'run/summary.json').read_text())
        r = json.loads((base/'run/reconciliation.json').read_text())
        assert s['status'] == 'PASS' and s['oracle_match'] and s['cycle']['identity_exact']
        assert len(r['operations']) == 13 and all(o['returned'] for o in r['operations'])
        assert r['selected_payload_before'] == r['selected_payload_after']
        assert sum(bool(n['slots']) for n in r['after']['nodes']) == 3
        assert [p['E'] for p in s['pending_rows']] == [0, 5120, 6144, 7168]
        assert [p['prefill_token_count'] for p in s['pending_rows']] == [4128, 32, 32, 32]
        assert all(p['cold_output_match'] for p in s['pending_rows'])
        assert all(w['new_entries'] == 0 for w in s['triton_new_kernels']
                   if w['window'] in ['post_warmup2→post_control', 'post_warm_block→post_request'])
        cleanup = base/'gpu_cleanup.json'
        if not cleanup.exists():
            cleanup = base/'gpu_cleanup_recovery.json'
        assert cleanup.exists()
        assert json.loads((base/'host_exit.json').read_text())['returncode'] == 0
        files = ['run/summary.json', 'run/reconciliation.json', 'run/engine_cleanup.json', 'host_exit.json', cleanup.name]
        inventory.append({'spec': spec, 'evidence_sha256': {f: hashlib.sha256((base/f).read_bytes()).hexdigest() for f in files},
                          'cleanup_recovered_after_launcher_exception': cleanup.name.endswith('_recovery.json')})
        if spec['phase'] == 'paired':
            data[spec['round'], spec['mode']] = s
    old = json.loads((PKG/'phase1_sources.json').read_text())
    for path, digest in old.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for path, digest in json.loads((PKG/'original_code_manifest.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for spec in specs:
        for name, digest in lock['inputs'].items():
            assert hashlib.sha256((PKG/'arms'/spec['arm']/name).read_bytes()).hexdigest() == digest
    (PKG/'verified_arm_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2))
    stats = json.loads((PKG/'performance_statistics.json').read_text())['optimized-original']
    assert stats['pairs'] == 5
    assert stats['control_cycle_ms']['median'] >= 0
    assert stats['cycle_ms_full_incl_warm']['median'] >= 0
    lines = ['# 性能结果：NO_MEASURABLE_GAIN', '',
             '一次正确性Smoke及十个性能臂全部完成；五轮原版/批量版独立生命周期交错配对。Qwen3.5-9B、冻结SGLang v0.5.17镜像、单H100 PCIe TP=1、C/K3，16候选、4 Pending。未达到LEVEL 1/2，故按预先协议不启动额外三策略五轮。', '',
             '所有性能值为同一臂直接时钟或原CUDA事件，未扣除审计、未拼接跨臂耗时。Smoke双算不进统计。首对收尾利用率门禁误判及修订见 resource_gate_diagnosis.md；首对负结果保留，未重跑。', '',
             '## 五轮配对原始结果（ms）', '',
             '|轮次|原版Control|批量Control|差值|原版Full|批量Full|差值|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in range(1, 6):
        a, b = data[r, 'original']['cycle'], data[r, 'batched']['cycle']
        lines.append(f'|{r}|{a["control_cycle_ms"]:.3f}|{b["control_cycle_ms"]:.3f}|{b["control_cycle_ms"]-a["control_cycle_ms"]:+.3f}|{a["cycle_ms_full_incl_warm"]:.3f}|{b["cycle_ms_full_incl_warm"]:.3f}|{b["cycle_ms_full_incl_warm"]-a["cycle_ms_full_incl_warm"]:+.3f}|')
    lines += ['', '## 中位数与分布', '', '|指标|原版中位数［最小，最大］|批量版中位数［最小，最大］|配对差中位数|批量胜/平/负|', '|---|---:|---:|---:|---:|']
    def metric(s, name):
        if name == 'sum_ttft_ms':
            return sum(p['ttft_ms'] for p in s['pending_rows'])
        if name == 'sum_cuda_prefill_ms':
            return sum(p['cuda_prefill_ms'] for p in s['pending_rows'])
        return s['cycle'][name]
    for name in ['control_cycle_ms', 'cycle_ms_full_incl_warm', 'control_rpc_wall_ms', 'rpc_tail_ms',
                 'request_exec_total_ms', 'sum_ttft_ms', 'sum_cuda_prefill_ms']:
        groups = [[metric(data[r, m], name) for r in range(1, 6)] for m in ['original', 'batched']]
        v = stats[name]
        cells = [f'{statistics.median(g):.3f}［{min(g):.3f}，{max(g):.3f}］' for g in groups]
        lines.append(f'|{name}|{cells[0]}|{cells[1]}|{v["median"]:+.3f}|{v["wins"]}/{v["ties"]}/{v["losses"]}|')
    lines += ['', '## 阶段分布（五轮中位数，ms）', '', '|阶段|原版|批量版|', '|---|---:|---:|']
    records = {k: json.loads((PKG/'arms'/f'p{k[0]}_{k[1]}_fs/run/reconciliation.json').read_text()) for k in data}
    for stage in records[1, 'original']['timing']:
        name = stage['stage']
        values = [statistics.median(next(t['ms'] for t in records[r, m]['timing'] if t['stage'] == name) for r in range(1, 6)) for m in ['original', 'batched']]
        lines.append(f'|{name}|{values[0]:.3f}|{values[1]:.3f}|')
    lines += ['', '八阶段为顺序子区间；clock六阶段之外的间隙含FA基线及协调，不与八阶段重复相加。逐Pending原始E/G、CUDA、TTFT、请求Wall、COW事件及时间戳见 pending_results.csv；逐臂全阶段见 performance_results.csv；五轮配对差见 paired_comparison.csv。', '',
              '## 内容读取为何没有带来控制收益', '',
              '完整16槽的Recurrent读取每次824,180,736B：原逐槽两组件约32次复制变为2次，但新实现把整组件收集到主机后又逐槽作非连续切片的contiguous复制，再计算SHA；增加主机重排与暂存。全量FA为637,534,208B，64MiB分组限制下仍为16次复制，并未减少该段复制次数；保护FA区间503,316,480B可由16次降为8次，所选3槽154,533,888B由6次降为2次。原始计数见各批量臂reconciliation.json中的optimization_metrics。', '',
              '上述结构开销可由代码确认，阶段耗时显示观察/新鲜度变慢；但未独立测量PCIe、CPU重排与SHA各自耗时或峰值显存，不能把全部退化定量归因于单一原因。未缓存内容、未减少被审计字节量；所有GPU内容保护保留。', '',
              '## 计时与可信范围', '',
              'Control=T3−T0；Full=T4−T0包括RPC尾段、日志刷盘、形状预热和四请求间隙。客户端Control RPC包含控制与尾段，不与Control相加。CUDA为请求原事件；四请求CUDA/TTFT之和不是自然并发吞吐。现有Full是受控实验完整周期，不是生产Serving延迟。', '',
              '全量观察、内容哈希、Journal序列化/写盘、请求后全量审计与独立Oracle均保留。必要Scope、身份、引用/别名/槽位检查、严格矩阵与Epoch认证、Bind/Preflight、逐项驱逐和Postcheck也保留。没有凭历史131.904ms结果生成速度比，没有减去估计审计时间。', '',
              '功能语义一致不能证明请求时延完全不变；逐轮TTFT/CUDA正负均保留，五轮只作描述性比较，不作生产P95或显著性声明。', '',
              '首对第二臂原收尾瞬间的进程列表未记录（断言先触发），之后独立复查清理通过。该异常不隐藏：主表包含五对；另观察第2–5轮的方向，避免借此排除首对负观测。', '',
              '## 结论边界', '',
              '本批量化尝试正确但未降低控制或完整周期，不能部署为性能优化；未满足追加三策略实验条件。本轮没有新Marconi/KVFlow完整周期比较，不能推断本实现相对基线净收益。它也不能否定其他有证据支持的优化机会；尚未证明可以安全移除任何内容检查或采用跨时刻快照缓存。']
    (PKG/'performance_analysis.md').write_text('\n'.join(lines)+'\n')
    control_delta = stats['control_cycle_ms']['median']
    full_delta = stats['cycle_ms_full_incl_warm']['median']
    final = f'''# 最终报告

**判定：NO_MEASURABLE_GAIN。** 保持正确性的本次批量读取修改反而变慢；不是GPU恢复错误，也未达到LEVEL 1、2或3。

1. **原路径为什么慢：** 原十五臂复核确认，完整观察、新鲜度与内容后验为主要成本；FS在线选择约32ms，远小于约10秒控制周期。阶段与嵌套详见 phase1_analysis.md，旧131.904ms为不同Legacy协议，不可比较。
2. **实际修改：** 仅隔离实验包批量读取Recurrent组件与FA内容。原控制、求解、Epoch、K、Phi、逐项驱逐、COW和时间边界不变；完整差异见 implementation.diff 与 batched_payload.py。
3. **正确性：** 59项CPU合同/摘要测试及4项资源门禁测试通过；1次GPU新旧摘要逐调用对照通过；10个性能臂均通过原Native/Cold/FA/组件保护和Oracle检查，共11个独立生命周期，143次真实Recurrent-only驱逐。逐臂证据 verified_arm_inventory.json。跨Workflow物理血缘未另行认证，保持 PROVENANCE_UNVERIFIED。
4. **控制时间：** 五轮批量−原版配对中位数 **{control_delta:+.3f}ms**，批量版{stats['control_cycle_ms']['wins']}胜、{stats['control_cycle_ms']['losses']}负，未下降。
5. **请求恢复：** 选集及J全等；E=[0,5120,6144,7168]、G=[4096,0,0,0]、实际Prefill=[4128,32,32,32]。命中的B/C/D各有Native COW，A无Recurrent命中；四请求8 Token输出均与冻结独立Cold相同。TTFT/CUDA实测波动见完整表，不因语义一致宣称时延不变。
6. **完整周期：** 五轮配对中位数 **{full_delta:+.3f}ms**，批量版{stats['cycle_ms_full_incl_warm']['wins']}胜、{stats['cycle_ms_full_incl_warm']['losses']}负，未改善。
7. **相对基线：** 预设的“控制改善后才追加三策略”条件未满足，因此没有启动额外15臂；本轮不能声称优于Marconi-Evict Adaptation或KVFlow-inspired。
8. **仍有取证成本：** 全树FA内容SHA、逐组件SHA、前后对照、Journal序列化/fsync、大响应RPC、请求后审计与Oracle均实际保留，未从结果扣除。
9. **必要Runtime成本：** 静止Scope、真实FA前沿、组件/槽位身份及引用/别名/容量账本、严格矩阵、在线选择、Apply-time重验和逐项安全协调仍不可省略。未证明当前内容SHA都可被轻检查替代。
10. **优化机会与限制：** 本次减少复制次数却增加主机重排/暂存，全量FA分组又未减少复制次数；没有单独测得重排/PCIe/SHA的份额。后续若获授权，可基于这两个具体瓶颈评估更合适的布局/读取实现；本任务不继续搜索参数、改安全合同或扩展实验。

## 异常与证据完整性

首对批量臂完成后，新增启动器把历史采样利用率4%误判为当前非空闲。原日志保留，未重跑、未删负结果；只修正收尾判据，启动前门禁不变。4项CPU测试确认仍拒绝真实进程/显存占用。原execution_lock与新execution_amendment分别保存，详见 resource_gate_diagnosis.md。首对原收尾进程字段缺失，后续独立清理证据补充该范围，不能冒称原瞬间字段完整。

全部旧十五臂源文件及原控制源码哈希重新核对不变；新十臂输入逐字节等于冻结来源。原始GPU日志、计时、Cold输出、物理状态与清理均在 arms/；完整五轮差值、分布、阶段归因见 performance_analysis.md，源数据为 performance_results.csv、pending_results.csv、paired_comparison.csv。

本轮完成后停止，不推进其他实验或论文修改。
'''
    (PKG/'final_report.md').write_text(final)
    (PKG/'decision.json').write_text(json.dumps({'decision': 'NO_MEASURABLE_GAIN', 'paired_rounds': 5,
        'gpu_lifecycles': 11, 'additional_three_policy_executed': False, 'paired_control_ms': control_delta,
        'paired_full_ms': full_delta}, indent=2))


if __name__ == '__main__':
    main()
