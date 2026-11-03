"""CPU spawn 重入回归（不启动 GPU、不触碰 /b5out）：防护 2026-10-27 GPU1 首臂失败机制。

失败机制回顾：SGLang Engine 以 multiprocessing spawn 起 scheduler 子进程时，
spawn._fixup_main_from_path 把入口文件以 __mp_main__ 语义重新执行；无 __main__
保护的入口会在子进程重复执行顶层副作用（独占写 /b5out/device_binding.json）。

三项检查（全部 CPU-only，宿主无 torch 故用 stub）：
1 结构（AST）：本批次 device_entry.py 顶层除 import/def/常量赋值/docstring 外
  不得有可执行语句，且必须存在 `if __name__ == '__main__'` 保护；
2 行为-新入口 GREEN：torch stub + runpy.run_path(run_name='__mp_main__') 模拟
  子进程重入，断言退出码 0（保护生效 → 重入为无操作、无任何写入）；
3 行为-旧入口 RED（对照）：对 formal_gpu1/device_entry.py（无保护原版）做同样
  重入模拟，断言必然失败（宿主无 /b5out → FileNotFoundError；容器内为
  FileExistsError——与线上失败同族），证明本测试确能捕获该缺陷类别。
"""
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
T2G1 = ROOT / 'evaluation/t2g1_three_policy_gpu'
NEW_ENTRY = T2G1 / 'formal_gpu1b' / 'device_entry.py'
OLD_ENTRY = T2G1 / 'formal_gpu1' / 'device_entry.py'
RESULT = T2G1 / 'formal_gpu1b' / 'spawn_regression.json'

# 顶层允许的安全形态：import / 函数与类定义 / 常量赋值 / pass / docstring
SAFE_KINDS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
              ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Pass)


def is_main_guard(node):
    """判断顶层 If 是否为 `if __name__ == '__main__':` 保护。"""
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__'
            and any(isinstance(c, ast.Constant) and c.value == '__main__'
                    for c in node.test.comparators))


def structure_check(path):
    """返回（未保护的可执行顶层语句列表, 是否存在 __main__ 保护）。"""
    tree = ast.parse(Path(path).read_text())
    bad = [type(n).__name__ for n in tree.body
           if not isinstance(n, SAFE_KINDS) and not is_main_guard(n)
           and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    guarded = any(is_main_guard(n) for n in tree.body)
    return bad, guarded


def child_reentry(entry, stub_dir):
    """以 __mp_main__ 语义在子进程重入入口（等价 spawn 子进程对主模块的重新执行）。"""
    code = ('import sys; sys.path.insert(0, %r); import runpy; '
            "runpy.run_path(%r, run_name='__mp_main__')" % (str(stub_dir), str(entry)))
    r = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=60)
    tail = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''
    return r.returncode, tail


def main():
    # 安全前提：宿主无 /b5out（旧入口对照在此环境下抛 FileNotFoundError，绝无真实写入副作用）
    assert not Path('/b5out').exists(), '宿主存在 /b5out，对照测试前提不成立'

    stub = Path(tempfile.mkdtemp(prefix='t2g2_spawn_stub_'))
    (stub / 'torch.py').write_text(
        "class _P:\n"
        "    uuid = 'GPU-051c9abc-864a-9f31-c4f7-a737aed997ff'\n"
        "    name = 'NVIDIA H100 PCIe'\n"
        "class cuda:\n"
        "    @staticmethod\n    def device_count():\n        return 1\n"
        "    @staticmethod\n    def get_device_properties(i):\n        return _P()\n")

    bad_new, guarded_new = structure_check(NEW_ENTRY)
    bad_old, guarded_old = structure_check(OLD_ENTRY)
    rc_new, err_new = child_reentry(NEW_ENTRY, stub)
    rc_old, err_old = child_reentry(OLD_ENTRY, stub)

    checks = [
        {'check': '新入口顶层无可执行语句（全部在 main() 内）', 'pass': not bad_new,
         'detail': '残留：%s' % bad_new},
        {'check': '新入口存在 __main__ 保护', 'pass': guarded_new, 'detail': ''},
        {'check': '新入口 __mp_main__ 重入退出码 0（无操作）', 'pass': rc_new == 0,
         'detail': 'rc=%d %s' % (rc_new, err_new)},
        {'check': '旧入口顶层存在未保护可执行语句（对照前提）', 'pass': bool(bad_old) and not guarded_old,
         'detail': '未保护语句：%s' % bad_old},
        {'check': '旧入口 __mp_main__ 重入必然失败（RED 对照，测试有效性）',
         'pass': rc_old != 0 and ('FileNotFoundError' in err_old or 'FileExistsError' in err_old),
         'detail': 'rc=%d %s' % (rc_old, err_old)},
    ]
    status = 'PASS' if all(c['pass'] for c in checks) else 'FAIL'
    RESULT.write_text(json.dumps({'status': status, 'checks': checks,
                                  'stub_dir': str(stub),
                                  'old_entry': str(OLD_ENTRY), 'new_entry': str(NEW_ENTRY)},
                                 ensure_ascii=False, indent=2))
    print('spawn 重入回归：%s（%d 检查）' % (status, len(checks)))
    for c in checks:
        print(' %s %s %s' % ('PASS' if c['pass'] else 'FAIL', c['check'], c['detail']))
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
