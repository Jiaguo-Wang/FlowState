"""observation.py 请求期 rid 归属的离线回归测试（宿主机、无 GPU、无 sglang 依赖）。

背景（r1_ck3_fs attempt1 失败根因，2026-10-29）：wrap_eager.execute 处的 batch 是
ForwardBatch（eager runner 的参数），没有 .reqs 属性；attempt1 版本在该处写
batch.reqs[0].rid，导致首个目标请求 forward 即 AttributeError → Scheduler 崩溃。
修复：run()（ScheduleBatch 层）在进入独占目标批次时暂存 ACTIVE_RID（同一同步调用栈），
execute() 读取暂存值。本测试用 stub torch 与假批次对象复现该路径：
1) 目标批次经 run → original → execute（forward_batch 无 reqs）不再崩溃且 COMPUTE 行带正确 rid；
2) ACTIVE=False 时 execute 早退、不触碰批次对象（无 reqs/无任何属性也能通过）；
3) 结束后 ACTIVE/ACTIVE_RID 复位。
"""
import sys
import types


class _FakeEvent:
    def record(self, stream=None):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.0


class _FakeStream:
    cuda_stream = 0


_fake_torch = types.ModuleType('torch')
_fake_torch.cuda = types.SimpleNamespace(
    synchronize=lambda: None, Event=lambda enable_timing=False: _FakeEvent(),
    current_stream=lambda: _FakeStream())
sys.modules['torch'] = _fake_torch   # observation.py 仅在函数内延迟 import torch，stub 即可满足

import evaluation.p0_control_cycle_gpu.observation as obs   # noqa: E402


class _FakeReq:
    def __init__(self, rid):
        self.rid = rid
        self.origin_input_ids = [1] * 8
        self.full_untruncated_fill_ids = [1] * 8
        self.prefix_indices = [1] * 4
        self.output_ids = []


class _FakeMode:
    def __init__(self, extend):
        self._extend = extend

    def is_extend(self):
        return self._extend

    def __str__(self):
        return 'EXTEND' if self._extend else 'DECODE'


class _FakeScheduleBatch:
    """ScheduleBatch 替身：具备 observe_batch 所需字段（Decode 模式走非 Extend 分支）。"""

    def __init__(self, rid):
        self.reqs = [_FakeReq(rid)]
        self.forward_mode = _FakeMode(extend=False)
        self.prefix_lens = [4]
        self.extend_lens = [0]


class _FakeForwardBatch:
    """ForwardBatch 替身：刻意不提供 reqs 属性（复现 attempt1 崩溃条件）。"""

    def __init__(self):
        self.forward_mode = _FakeMode(extend=False)


def main():
    # ── 情形1：目标批次 run → original（内部触发 execute，批次无 reqs）──
    obs.TARGETS = frozenset({'C_A'})
    obs.ACTIVE = False
    obs.ACTIVE_RID = None
    obs.COMPUTE.clear()
    sched = types.SimpleNamespace(enable_overlap=False)
    batch = _FakeScheduleBatch('C_A')
    forward_batch = _FakeForwardBatch()
    execute = obs.wrap_eager(lambda runner, b, *a, **k: ('eager_ok', b is forward_batch),
                             persist=lambda: None)

    def original(scheduler, b, *args, **kwargs):
        return execute(object(), forward_batch)   # 引擎在 run_batch 内部调用 eager execute

    run = obs.wrap_run_batch(original, [], lambda: None)
    ret = run(sched, batch)
    assert ret == ('eager_ok', True), 'run 返回值被改变：%r' % (ret,)
    assert len(obs.COMPUTE) == 1 and obs.COMPUTE[0]['rid'] == 'C_A', \
        'COMPUTE 行 rid 归属错误：%r' % (obs.COMPUTE,)
    assert obs.ACTIVE is False and obs.ACTIVE_RID is None, 'ACTIVE/ACTIVE_RID 未复位'

    # ── 情形2：非目标批次（rid 不在 TARGETS）→ run 直接透传，不进入计时/execute 分支 ──
    obs.COMPUTE.clear()
    run2 = obs.wrap_run_batch(lambda scheduler, b, *a, **k: 'passthrough', [], lambda: None)
    assert run2(sched, _FakeScheduleBatch('OTHER')) == 'passthrough'
    assert not obs.COMPUTE and obs.ACTIVE is False

    # ── 情形3：ACTIVE=False 时 execute 早退，批次对象无任何属性也不崩溃 ──
    assert execute(object(), object()) == ('eager_ok', False)
    print('offline_test_observation: 3/3 通过（ForwardBatch 无 reqs 路径已修复）')


if __name__ == '__main__':
    main()
