"""G2 装配记录（宿主机执行）：逐文件哈希 + 与已验证基线的差异清单，供执行锁与复核使用。

pkg_g2 为手写组合包（非逐字节复制），基线与差异如下（全部只加不改核心语义）：
- bridge.py ← c_formal_perf_gpu/bridge.py（60 正式臂已验证测量桥接）；差异＝cold 臂最小扩展×3：
  允许 B5D4_ARM=cold、cold 模式 admission 钩子（仅记录、不要求协调）、新动作 g2_set_cold_target
  （只设 TARGET 观察目标，不设任何缓存/调度状态）；strict 路径逐行不变。
- runner.py ← c_formal_perf_gpu/runner.py 计时骨架 + b5d4a/fix_a 的 B5-C4 合同；差异＝G2 双臂：
  strict 臂 8 历史单轮重放（B5-C4 单轮即物化 7 候选）、b5d4_apply 后置 solver/选集断言、
  事后 g2_reference Oracle、形状预热 B 按 B5-C4 语义 L=len(pt)−E（去掉 C/K3 的 T+32 合同）、
  Cold 参照逐 Token 对照、B5-C4 Prefill=len(pt)−E/G_model=T−E 分列；新增 cold 臂补采流程。
- control.py ← paper_hybrid_gpu_fix_a/control.py（gate_a 已验收 B5-C4 Exact 分支）；差异＝
  G2_SOLVER 双臂分派（Exact=生产 plan_strict 自动；Greedy=greedy_adapter 生产 Marginal Greedy
  内部路径，control 断言域满足 Exact 门禁以证明未篡改规模）＋7 候选/4 Pending 域断言。
- greedy_adapter.py：新增。逐行复用 plan_strict 超域分支（同一 frontiers 闭包/_select_candidates/
  StrictAllocationPlan 构造），仅跳过 Exact 枚举；不改 flowstate 核心。
- source_contract.py ← fix_a/source_contract.py；差异＝目标 Pending 由 G2_PENDING 环境变量指定
  （B5C2_A/B/C/D_TURN_003），其余（唯一最深已准入选中态、support≥2、PROVENANCE_UNVERIFIED）不变。
- persistence.py ← fix_a/persistence.py 逐字节复制（Journal/call_saved）。
"""
import hashlib, json
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
PKG = ROOT / 'evaluation/b5c4_formal_gpu_g2/pkg_g2'
BASES = {'bridge.py': 'evaluation/c_formal_perf_gpu/bridge.py',
         'runner.py': 'evaluation/c_formal_perf_gpu/runner.py',
         'control.py': 'evaluation/paper_hybrid_gpu_fix_a/control.py',
         'source_contract.py': 'evaluation/paper_hybrid_gpu_fix_a/source_contract.py',
         'persistence.py': 'evaluation/paper_hybrid_gpu_fix_a/persistence.py'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    names = sorted(p.name for p in PKG.glob('*.py'))
    expected = sorted(list(BASES) + ['greedy_adapter.py', '__init__.py', 'prepare_pkg.py'])
    assert set(names) == set(expected), 'pkg_g2 文件集与预期不符：' + repr((set(names), set(expected)))
    record = {'package': str(PKG), 'files': {}, 'bases': {k: {'path': str(ROOT / v), 'sha256': sha(ROOT / v)} for k, v in BASES.items()},
              'identical': [], 'handwritten': ['runner.py', 'control.py', 'bridge.py', 'source_contract.py', 'greedy_adapter.py']}
    for name in names:
        text = (PKG / name).read_text()
        compile(text, str(PKG / name), 'exec')  # 语法自检，不执行
        record['files'][name] = {'sha256': sha(PKG / name), 'bytes': len(text.encode())}
        if name in BASES and sha(PKG / name) == sha(ROOT / BASES[name]):
            record['identical'].append(name)
    record['identical_note'] = 'persistence.py 应为逐字节复制；其余为文档化差异（见本文件 docstring）'
    assert record['identical'] == ['persistence.py'], '预期仅 persistence.py 与基线逐字节一致：' + repr(record['identical'])
    out = PKG / 'build_record.json'
    assert not out.exists(), '拒绝覆盖 build_record.json'
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print('pkg_g2 装配记录完成：%d 文件，persistence.py 逐字节同 fix_a，其余差异已文档化' % len(names))


if __name__ == '__main__':
    main()
