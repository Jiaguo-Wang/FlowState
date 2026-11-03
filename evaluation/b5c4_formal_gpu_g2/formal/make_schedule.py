"""G2-B 正式阶段安排预固定（宿主机执行，任何正式臂启动前运行一次）：生成 schedule.json。

配对安排：2 策略 × 4 Pending × 5 轮 = 40 臂。轮内策略顺序均衡——奇数轮 Exact 先、
偶数轮 Greedy 先；策略块内 Pending 顺序固定 A,B,C,D。配对单位=同轮同 Pending 的
(Exact, Greedy) 两臂（ΔTTFT_r / ΔCUDA Prefill_r 按同轮同编号配对）。安排一经生成
不得改动；实际执行顺序由 launch.py 按本安排逐臂强制（前序臂未完成不得启动后续）。
"""
import json
from pathlib import Path

OUT = Path('/home/wjg/code/FlowState/evaluation/b5c4_formal_gpu_g2/formal')
LETTERS = 'ABCD'


def main():
    schedule = []
    for r in range(1, 6):
        solvers = ['Exact', 'Greedy'] if r % 2 == 1 else ['Greedy', 'Exact']
        for solver in solvers:
            for letter in LETTERS:
                schedule.append({'round': r, 'solver': solver, 'pending': 'B5C2_%s_TURN_003' % letter,
                                 'letter': letter, 'run': 'r%d_%s_%s' % (r, solver.lower(), letter.lower()),
                                 'solver_first_in_round': solver == solvers[0]})
    out = OUT / 'schedule.json'
    assert not out.exists(), '拒绝覆盖 schedule.json（安排一经固定不得改动）'
    out.write_text(json.dumps({'rounds': 5, 'arms': 40, 'ordering': '奇数轮Exact先、偶数轮Greedy先；策略块内A,B,C,D',
                               'pairing': '同轮同Pending的(Exact,Greedy)两臂', 'schedule': schedule},
                              ensure_ascii=False, indent=2))
    first = ['%s:%s' % (r, s) for r, s in [(a['round'], a['solver']) for a in schedule] if s == 'Exact']
    print('schedule.json 固定：40 臂；每轮首策略 =', {r: ('Exact' if r % 2 else 'Greedy') for r in range(1, 6)})


if __name__ == '__main__':
    main()
