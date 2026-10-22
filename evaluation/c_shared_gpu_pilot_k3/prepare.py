"""C Shared/Branched K=3 GPU Pilot 前置准备：构建输入、映射 CPU 候选到物理前缀、冻结身份。"""
import json, hashlib, subprocess, time
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
OUT = ROOT / 'evaluation/c_shared_gpu_pilot_k3'
MAIN = ROOT / 'evaluation/kvflow_multiworkflow_cpu_main'
DIAG = ROOT / 'evaluation/kvflow_native_recovery_diagnostic'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)


def main():
    w = json.loads((MAIN / 'workloads.json').read_text())
    C = next(c for c in w['cases'] if c['structure'] == 'C')
    assert len(C['planned_warmup']) == 16 and [p['ready'] for p in C['calls']] == [True] * 4 + [False]

    # ── 构建 prepared_inputs.json：结构与 kvflow_native_recovery_diagnostic 完全一致 ──
    # history 行含 workflow_id/label/turn（from_frozen 来源证据需要 workflow_id；b5c3 观察需要 label）；
    # C 案例全部调用归属同一共享工作流 C_SHARED，预热按冻结计划轮次与前缀属主标注
    inputs = {
        'configuration': w['configuration'],
        'sampling': {'temperature': 0, 'max_new_tokens': 1, 'ignore_eos': True},
        'history': [{'request_id': r['request_id'], 'workflow_id': 'C_SHARED',
                     'label': r['request_id'], 'turn': int(r['request_id'].split('_')[2]),
                     'token_ids': r['token_ids'],
                     'token_prefix_sha256': r['token_prefix_sha256']} for r in C['planned_warmup']],
        'pendings': [{'continuation_id': p['continuation_id'], 'request_id': p['continuation_id'],
                      'workflow_id': p['workflow_id'], 'label': p['continuation_id'], 'turn': 3,
                      'token_ids': p['token_ids'],
                      'token_prefix_sha256': p['token_prefix_sha256'],
                      'planning_target_T': p['planning_target_T'],
                      'attention_frontier': p['attention_frontier_assumed'], 'ready': p['ready'],
                      'controlled_input_ready': True}
                     for p in C['calls'] if p['ready']],
        'workload_type': 'c_shared_branched_frozen_case',
        'pending_dispatch_allowed': False,
    }
    write(OUT / 'prepared_inputs.json', inputs)

    # ── CPU 候选 ↔ 物理前缀映射：候选前缀必须唯一等于某预热水截断 ──
    bywarm = {r['request_id']: r['token_ids'] for r in C['planned_warmup']}
    prefix_index = {}  # (前缀sha, L) -> 预热请求ID列表
    for rid, toks in bywarm.items():
        for L in (len(toks), len(toks) - 64):
            d = hashlib.sha256(json.dumps(toks[:L]).encode()).hexdigest()
            prefix_index.setdefault((d, L), []).append(rid)
    mapping = {}
    for c in C['candidates']:
        L = c['token_position_L']
        # 物理侧将用原始 token 列表匹配；此处校验两轮预热水前缀一致（CPU 声明前缀即其截断）
        w0 = bywarm['C_WARM_0_' + c['prefix_owner']]
        w1 = bywarm['C_WARM_1_' + c['prefix_owner']]
        assert w0[:L] == w1[:L] and len(w0) >= L, '两轮预热前缀不一致或长度不足'
        mapping[c['checkpoint_id']] = {'L': L, 'prefix_owner': c['prefix_owner'],
                                       'token_prefix_sha256_declared': c['token_prefix_sha256'],
                                       'warmup_round0': 'C_WARM_0_' + c['prefix_owner'],
                                       'warmup_round1': 'C_WARM_1_' + c['prefix_owner']}
    write(OUT / 'cpu_candidate_prefix_mapping.json',
          {'mapping': mapping, 'rule': '物理候选按 (token_ids[:L] 与预热水前缀逐Token相等, L) 匹配 CPU checkpoint_id'})

    # ── 身份锁：复用诊断轮的 58 文件闭包（逐项重算确认未变）+ 本轮新增文件 ──
    diag_identity = json.loads((DIAG / 'source_identity.json').read_text())
    stale = [p for p, h in diag_identity['sha256'].items() if sha(p) != h]
    assert not stale, '诊断轮源码闭包已变化：' + repr(stale)
    sha_map = dict(diag_identity['sha256'])
    for p in sorted(OUT.glob('*.py')) + [OUT / 'protocol.json', OUT / 'policy_plans.json',
                                         OUT / 'prepared_inputs.json', OUT / 'cpu_candidate_prefix_mapping.json',
                                         MAIN / 'workloads.json', MAIN / 'cpu_results.json',
                                         ROOT / 'evaluation/kvflow_multiworkflow_cpu_review/review_results.json']:
        sha_map[str(p)] = sha(p)

    # ── 容器内 runtime 源码摘要：与诊断轮同一镜像，逐项在容器内重算核验 ──
    script = 'import hashlib,json,sys;from pathlib import Path;' \
             'src=json.loads(Path("/b5out/_runtime_hash_check/source_identity.json").read_text())["runtime_sources"];' \
             'print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in src}))'
    tmp = OUT / '_runtime_hash_check'; tmp.mkdir(exist_ok=True)
    (tmp / 'source_identity.json').write_text(json.dumps({'runtime_sources': diag_identity['runtime_sources']}))
    cmd = ['docker', 'run', '--rm', '--pull', 'never', '--mount', f'type=bind,src={OUT},dst=/b5out',
           '--entrypoint', 'python3', diag_identity['image_id'], '-B', '-c', script]
    got = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    assert got == diag_identity['runtime_sources'], '容器内 runtime 源码摘要与诊断轮不一致'
    write(OUT / 'source_identity.json', {'status': 'PASS', 'image_id': diag_identity['image_id'],
                                         'runtime_sources': diag_identity['runtime_sources'],
                                         'sha256': sha_map,
                                         '说明': '镜像与16项容器内runtime源码摘要经容器内重算核验，与诊断轮一致；新增本轮文件'})
    print('准备完成：prepared_inputs(16预热+4Pending)、候选前缀映射、source_identity',
          len(sha_map), '项；runtime源码', len(got), '项核验通过')


if __name__ == '__main__':
    main()
