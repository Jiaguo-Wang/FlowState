"""只读取本轮结果，核对隔离条件并生成中文报告和证据清单。"""
import hashlib
import json
from pathlib import Path
from evaluation.strict_compatibility_b1c.protocol import semantic_nodes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evaluation/strict_compatibility_b1c_output/b1c_g000_20261014_r2'
B1 = ROOT / 'evaluation/strict_compatibility_b1_output/b1_openhands_g000_20261014_02/pilot_evidence.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def main():
    arms = {a: json.loads((OUT/a/'evidence.json').read_text()) for a in ('reference','retained','removed')}
    ref, keep, drop = (arms[a] for a in ('reference','retained','removed'))
    assert all(a['status'] == 'PASS' and a['shutdown_returned'] for a in arms.values())
    assert all(a['b1_sha256'] == sha(B1) for a in arms.values())
    assert all(a['configuration'] == keep['configuration'] for a in arms.values())
    assert all(a['output']['output_ids'] == ref['output']['output_ids'] for a in arms.values())
    assert semantic_nodes(keep['barrier']) == semantic_nodes(drop['barrier'])
    assert semantic_nodes(keep['isolated_retained']) == semantic_nodes(drop['isolated_retained'])
    target = next(c for c in keep['candidates'] if c['checkpoint_id'] == 'RQ3_G000_B_TURN_001')
    digest = target['path']['prefix_sha256']
    before = {n['prefix_digest']: n for n in keep['before_request']['nodes']}
    after = {n['prefix_digest']: n for n in drop['before_request']['nodes']}
    assert before.keys() == after.keys()
    changed = [k for k in before if before[k] != after[k]]
    assert changed == [digest]
    a, b = dict(before[digest]), dict(after[digest])
    assert a.pop('mamba_slots') == [6] and b.pop('mamba_slots') == [] and a == b
    for arm, d in arms.items():
        for key in ('engine_initial','barrier','isolated_retained','before_request','after_request'):
            if key in d:
                assert d[key]['observation_noninterference'] and d[key]['scope']['scheduler_fully_idle']
        assert '14 MiB' in (OUT/arm/'gpu_after.csv').read_text()
    summary = {
        'verdict':'PASS', 'target_checkpoint':target['checkpoint_id'],
        'pending':keep['pending']['continuation_id'], 'changed_prefix_only':changed,
        'conditions': {a:{'H':d['request_match']['physical_fa_hit'], 'E':d['request_match']['executable_prefix'],
            'G_H_minus_E':d['request_match']['replay_gap'], 'output_ids':d['output']['output_ids'],
            'request_id':d['request_match']['request_id']} for a,d in arms.items()},
        'direct_match_and_native_copy':keep['request_evidence'],
        'configuration_equal':True, 'initial_barrier_equal':True, 'only_target_recurrent_changed':True,
        'output_equal_8_tokens':True, 'gpu_cleanup':'PASS', 'slot_generation':'UNKNOWN',
        'scope':'仅认证本轮受控 B1 到 A Pending 恢复边，不认证全部跨 Workflow 关系',
    }
    write('comparison.json', summary)
    rows = '\n'.join(f"| {a} | {d['request_match']['physical_fa_hit']} | {d['request_match']['executable_prefix']} | {d['request_match']['replay_gap']} | 8/8 一致 |" for a,d in arms.items())
    events = '\n'.join(f"| {arm} | {e['event']} | {e['unix_ns']} |" for arm in ('retained','removed') for e in arms[arm]['events'])
    report = f'''# B1-C 跨 Workflow Checkpoint 隔离恢复验证

## A. 判定与范围

**PASS**。Workflow B 执行时物化的 `RQ3_G000_B_TURN_001` 在本轮静止边界隔离条件下，被 Workflow A 的 `RQ3_G000_A_PENDING_TURN_003` 实际匹配并通过原生 COW 复制到工作状态。仅移除该 recurrent state，FA 命中保持不变，执行前沿从 2432 降至 0。两条件均与独立全量重算参考的 8 个确定性输出 Token 完全一致。

这是一个已知输入的受控回放案例，不是自然 Agent Fork 认证、容量竞争证明或性能比较。不将结论推广到另两条 B1 跨 Workflow 边或旧正式矩阵。

## B. 输入、状态和隔离证据

来源为 `{B1}`。本轮没有覆盖该文件；采集时与汇总时 SHA-256 均为 `{sha(B1)}`。模型为 Qwen3.5-9B，SGLang 0.5.17，单 H100 PCIe、TP=1；复用原 Docker 镜像及全部引擎配置。模型配置、Tokenizer 文件、Chat Template 和引擎参数身份保存在每个 evidence.json 的 configuration；容器镜像身份在 image_identity.json。GPU 执行使用历史 Docker 环境，CPU 核对使用 uv312。

复用 group 0 / round 2 的 8 次历史请求、8 个候选与4个已知 Pending；逐一核对全部 Pending Token IDs 与 B1 相同，并核对历史请求实际 Runtime 输入摘要。仅提交 A Pending 做本轮验证。Pending 输入长度 5020，Planning Target 与 FA Frontier 均为4352；前缀摘要为 `{keep['pending']['token_prefix_sha256']}`。这是冻结回放边界已准备输入，不使用 Pending 输出或后续轮次消息；不声称证明自然 Agent 的原生 Ready 事件。

目标 Checkpoint 长度2432，Token Prefix 摘要 `{digest}`。保留与移除生命周期均重新定位为 node20 / slot6；未沿用上一轮 node29。`history_census[event=after_B1]` 记录 B 请求 `rq3e-g000-b-turn-001` 执行后新增 node20；`materialization_evidence` 保存其完整来源。它与 A 的 Workflow ID 不同，但2432个实际 Token 逐项相同。物理节点可能服务多个逻辑 Workflow，B 来源指物化事件归属，不宣称永久独占的 Workflow 所有权。

本轮每个生命周期的 barrier / isolated_retained / before_request 各自具有 monotonic_ns、scope 与完整 nodes。继承的 pending.barrier_id 是旧 B1 来源标识；本轮实际边界以“生命周期名 + 状态字段 + monotonic_ns”区分。全部状态采集时 running_requests=waiting_requests=0、无 chunked request、scheduler_fully_idle=true；TP/DP/PP均为1，无HiCache与session cache。

遍历所有物理驻留 recurrent 节点，要求均由8个候选覆盖、无共享 slot 别名，核对完整 Token 路径。先经原 Adapter 安全驱逐 A2（4352）和 A1（3904），不改 FA，确认 B1 是唯一最深兼容状态；移除条件在相同隔离状态上再驱逐 B1。两个生命周期的初始状态与隔离状态一致；最终提交前逐节点差异仅为 B1 的 mamba_slots 从[6]变为空。检查包括节点身份、Prefix Digest、引用、FA不变、仅目标 recurrent 改变、路径/树不变与原生 sanity check。

物理组件：conv 为 bfloat16 `[24,29,8192,3]`，temporal 为 float32 `[24,29,32,128,128]`。28个可分配槽位另含哨兵索引；初始8个驻留、20个可用、0个保护槽位。未证明真实紧张预算。组件依据同一 pool 槽位与原生复制路径，未单独做张量字节一致性测试；Slot Generation 仍为 UNKNOWN。

## C. 恢复与输出

H、E、G单位均为tokens；G=H−E，与5020−E的请求完整待计算长度不同。冷参考 H=0，不应把其 G=0理解为不需要计算输入。

| 条件 | FA命中 H | 执行前沿 E | G=H−E | 与全量重算输出比较 |
|---|---:|---:|---:|---|
{rows}

输出 Token IDs 均为 `{ref['output']['output_ids']}`。采样 temperature=0、max_new_tokens=8、ignore_eos=true。这是8 Token受控正确性检查，不宣称任意长度输出等价。保留与移除的 FA tree 摘要均保持 `{keep['barrier']['tree']['full_tree_sha256']}`。未检测到历史或验证请求期间的原有 recurrent 节点非预期驱逐/替换、FA级联、OOM、回撤或截断。安全保护检查均通过。

## D. 直接物理归因

`retained/evidence.json → request_match`：真实请求 `b1c_retained_A_pending_turn003` 原始匹配返回 best_match_node=20，E=2432，cow_source_slots=[6]，working_slots=[19]。`request_evidence.copies` 记录原生 copy_from([6],[19]) 已返回，关联同一请求；随后请求执行完成且8 Token输出与冷参考一致。移除条件的 best_match_node=9（根），E=0、无源槽位，也无目标 COW调用。

因此具备本生命周期内直接 Request→Node→源槽位→工作槽位的调用链证据，而非按位置推断或仅受控差分。观察器对原始 match_prefix 和 copy_from 各调用一次，返回原对象，不新增前缀 lookup，不修改匹配/恢复决策。索引读取可能引入同步，故本轮不报告延迟性能。COW调用返回不等于独立的 GPU 张量校验；执行完成和输出对照提供端到端佐证，不宣称跨生命周期的槽位代际认证。

本地代码入口：`evaluation/strict_compatibility_b1c/runner.py:28` install_observation；`:75` state_snapshot；`:218` evict；`:231` 开始历史回放与隔离；`:285` 后续请求结果门禁。`protocol.py` 的 isolation_plan 检查全树候选覆盖、Token身份与槽位别名。最小观察器只在本 Pilot 子进程内安装，不修改容器原生源码文件。

容器原生证据根路径 `/sgl-workspace/sglang/python/sglang/srt/`，每个 evidence.json 的 source_identity 保存以下源文件 SHA-256：

- `mem_cache/unified_cache/unified_tree_core.py:597–622` 原始 match_prefix；`:662–676` 所有组件验证后的最佳节点；`:745–765` 返回匹配与Node ID。
- `mem_cache/unified_cache/components/mamba_component.py:130–141` recurrent驻留验证；`:173–202` 从best_match_node取源槽位、分配目标并设置mamba_cow_src_index。
- `managers/schedule_batch.py:1310–1343` 请求匹配与前缀索引；`:2640–2660` 传递延迟COW源与目标。
- `model_executor/model_runner.py:1506–1523` 消费延迟COW并调用物理池复制。
- `mem_cache/memory_pool.py:939–975` copy_from 覆盖conv与temporal组件。

## E. 对 Design 与下一阶段的影响

这条 B1→A 的 EPOCH_ADMISSIBLE 边获得真实恢复与安全驱逐反事实支持：Workflow ID 不同本身不排除共享，必须依赖实际输入前缀、执行配置、同边界完整状态与安全引用条件。现有候选集合内的这一边可用于严格模型，不据此认证其他边，也不新增或修改旧正式矩阵。

**本 Snapshot 可进入独立 CPU-only 的 EPOCH_ADMISSIBLE 矩阵比较准备阶段**：输入与物理候选边界已采集；仍需将准入认证和逐边实际恢复认证分别标注。B1原始矩阵11条准入边中，本轮只强化验证其中一条跨Workflow边。这里没有策略胜负、物理容量竞争或全体兼容边恢复正确性的结论；本轮不执行分配比较。

## F. 执行记录、异常和清理

共3个成功独立生命周期：冷参考、保留、移除。另保留首次失败启动 `../b1c_g000_20261014/reference/`：新观察器把根节点空list当作Tensor读取长度，尚未提交请求即停止。修正观察器为与既有探针一致的len读取，未放宽任何语义门禁，在本新目录重新执行。不得将首次启动描述为成功实验或隐藏。

5项CPU测试通过：正确隔离集、未管理状态拒绝、槽位别名拒绝、Token冲突拒绝、原生match/copy恰调用一次。结果存于cpu_tests.log。三个成功容器均正常shutdown并退出；各gpu_after.csv记录两卡均14MiB，结束后再次查询相同，无残留本轮容器。旧生产代码、算法与冻结B1未修改；仓库挂载只读，输出独立挂载。完整文件哈希见 evidence_manifest.json（清单不包含自身哈希或JIT缓存）。

| 生命周期 | 事件 | Unix时间ns |
|---|---|---:|
{events}

详细原始记录：reference/evidence.json、retained/evidence.json、removed/evidence.json，以及各自 runtime.log、gpu_before.csv、gpu_after.csv。comparison.json 提供独立CPU汇总。复现入口 `evaluation/strict_compatibility_b1c/run.sh` 会拒绝覆盖已有证据；未来复跑须另设输出目录并另获任务授权，本轮不再执行。
'''
    (OUT/'final_report.md').write_text(report)
    paths = list((ROOT/'evaluation/strict_compatibility_b1c').glob('*.py')) + [ROOT/'evaluation/strict_compatibility_b1c/run.sh', B1]
    paths += [p for p in OUT.rglob('*') if p.is_file() and 'triton_cache' not in p.parts and p.name != 'evidence_manifest.json']
    failed = OUT.parent/'b1c_g000_20261014/reference'
    paths += [p for p in failed.glob('*') if p.is_file()]
    write('evidence_manifest.json', {'verdict':'PASS','说明':'记录输入、脚本与本轮输出；原生容器源码哈希在各 evidence.json 的 source_identity；不收录JIT缓存与本清单自身',
        'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(paths))]})
    print('PASS：隔离反事实、输出、配置、清理与原始B1身份核对通过')

if __name__ == '__main__':
    main()
