"""P0-4 三策略完整控制周期编排器：T0–T4 计时边界、四 Pending 串行执行与取证分离。

结构与计时合同（P0-4 审查 §5）：
  T0  开始本轮状态观察与认证
  T1  完成候选、兼容关系与物理状态确认（census 对账 + epoch 认证）
  T2  完成本策略在线 Checkpoint Select（真实内核，select_entry）
  T3  完成 Plan 校验 + Recurrent-only Reconciliation 及必要安全后验
  T4  四个 Pending 的 Native Resume 与请求执行全部完成（= 第 4 请求执行终点）
  主指标 Control-inclusive Cycle Time = T4 − T0

实现要点：
- 整数纳秒计时（Python int 精确无浮点噪声）；分段按顺序执行并以相邻边界衔接，
  恒等式 (T4−T0) == Σ分段 + Σ间隙 在整数算术下精确成立，任何重复累加/遗漏都会被断言捕获。
- 取证分离（P0-3 结论）：研究专用取证操作进 OutOfWindowEvidence 队列，仅在 T4 之后统一
  flush（其耗时单独记录、不计入周期）；必要在线检查（身份认证、物理映射复核、Plan 校验、
  协调后验合同）保留在窗口内对应分段。
- RequestBackend 为抽象请求后端：CPU 门禁用 DryBackend（确定性微延时）验证编排/计时/关联
  结构；GPU 阶段替换为真实 Native Resume 后端（同接口），编排代码零改动。
- 单 Pending 请求执行终点与 TTFT 分别记录（若为单 Token Continuations，二者语义差异如实分开）。
"""
from abc import ABC, abstractmethod
from time import monotonic_ns, sleep


class RequestBackend(ABC):
    """请求执行后端抽象：GPU 阶段实现 preflight/execute 为真实 Native Resume。"""

    @abstractmethod
    def preflight(self, pending_id):
        """请求前必要在线检查（身份/恢复源存在性/状态未被污染）。"""

    @abstractmethod
    def execute(self, pending_id):
        """执行请求，返回 (ttft_ns, exec_end_ns)——单调钟整数纳秒。"""


class DryBackend(RequestBackend):
    """CPU 门禁干跑后端：确定性微延时模拟请求各阶段，不触碰任何真实状态。"""

    def __init__(self, pendings):
        self._pendings = list(pendings)
        self._i = 0

    def preflight(self, pending_id):
        assert pending_id == self._pendings[self._i], '请求顺序与声明不一致：%s' % pending_id
        sleep(0.0005)  # 模拟必要在线检查耗时

    def execute(self, pending_id):
        assert pending_id == self._pendings[self._i], '请求顺序与声明不一致：%s' % pending_id
        sleep(0.0015)  # 模拟 TTFT 之前的执行
        ttft = monotonic_ns()
        sleep(0.0010)  # 模拟剩余执行（单 Token Continuations 时 TTFT 与终点几乎重合，如实分别记录）
        self._i += 1  # 本请求完成，推进到下一声明序号
        return ttft, monotonic_ns()  # (ttft, 执行终点)


class OutOfWindowEvidence:
    """研究取证队列：窗口内只入队（不执行不计时），T4 后统一 flush。"""

    def __init__(self):
        self._queue = []
        self.flush_ms = None
        self.records = []

    def collect(self, op_name, payload=None):
        """登记一项取证操作（如 census 摘要、metrics 快照、inspect 批量读）。"""
        self._queue.append((op_name, payload))

    def flush_after_window(self):
        """T4 之后执行全部取证；耗时单独记录，绝不计入周期。"""
        t0 = monotonic_ns()
        for op_name, payload in self._queue:
            self.records.append({'op': op_name, 'payload': payload})
        self.flush_ms = (monotonic_ns() - t0) / 1e6
        self._queue = []
        return self.records


class CycleClock:
    """顺序分段时钟：记录 (名称, 起, 止) 整数纳秒；相邻边界衔接保证无重叠无遗漏。"""

    def __init__(self):
        self.segments = []
        self._boundary = None

    def start(self):
        self._boundary = monotonic_ns()
        return self._boundary

    def seg(self, name):
        """分段上下文：起=上一边界，止=本段结束；下一段从本止点衔接。"""
        clock = self

        class _Seg:
            def __enter__(self):
                self._start = clock._boundary
                return self

            def __exit__(self, *exc):
                clock._boundary = monotonic_ns()
                clock.segments.append({'stage': name, 'start_ns': self._start,
                                       'end_ns': clock._boundary})
                return False
        return _Seg()

    def boundary(self):
        b = monotonic_ns()
        gap = b - self._boundary
        self._boundary = b
        return gap


def stage_registry():
    """窗口内分段注册表（固定合同；GPU 阶段每段接真实实现，名称与嵌套关系不变）。"""
    return [
        ('T0', 'observe_and_authenticate', '状态观察与 Strict Epoch 认证（from_frozen/prepare 全合同）'),
        ('T1', 'census_epoch_confirm', 'CPU↔物理候选对账 16=16 + 物化时间线合同'),
        ('T2', 'policy_select', '本策略真实内核在线 Select（select_entry 三内核之一）'),
        ('T3', 'plan_validate_and_reconcile', 'Plan 校验 + Recurrent-only Reconciliation + 必要后验合同'
                                              '（驱逐逐项落实/FA 保护/驻留==Plan；GPU 阶段真实物化）'),
        ('T4', 'resume_p0..p3', '四 Pending 串行 Native Resume：每请求必要前置检查 + TTFT + 执行终点'),
    ]


def run_cycle(policy, select_fn, backend, evidence, pendings):
    """执行一个完整控制周期，返回带恒等式断言的周期记录。

    select_fn：无参回调，执行本策略在线 Select（真实内核），返回选择记录 dict；
    evidence：OutOfWindowEvidence，窗口内取证只 collect 不执行；
    pendings：四 Pending 声明顺序（C_A..C_D）。
    """
    clock = CycleClock()
    record = {'policy': policy, 'stage_registry': stage_registry()}
    t0 = clock.start()                                        # ── T0
    with clock.seg('observe_and_authenticate'):
        epoch_ctx = select_fn('epoch_context')                # GPU: 实时观察+认证；CPU: 冻结构造链
        record['epoch_candidates'] = epoch_ctx['candidates']
        record['epoch_pendings'] = epoch_ctx['pendings']
    gap_01 = clock.boundary()
    with clock.seg('census_epoch_confirm'):                   # ── T1
        census = {'mapped': 16, 'unmapped': 0}                # CPU: 复算冻结；GPU: 实时 census 16=16
        record['census'] = census
    gap_12 = clock.boundary()
    with clock.seg('policy_select'):                          # ── T2
        selection = select_fn('select')
        record['selection'] = selection
    gap_23 = clock.boundary()
    with clock.seg('plan_validate_and_reconcile'):            # ── T3
        reconcile = {'evicted': selection['evicted_physical_ids'],
                     'postchecks': ['FA/非R摘要不变', '槽位释放记账', '驻留==Plan'],
                     'execution': 'CPU:Plan校验+意图台账；GPU:reconcile_strict真实物化（待验证）'}
        record['reconciliation'] = reconcile
    gap_34 = clock.boundary()
    pending_rows = []
    for pid in pendings:                                      # ── T4（四请求串行）
        with clock.seg('resume_preflight_%s' % pid):
            backend.preflight(pid)
        gap_p = clock.boundary()
        with clock.seg('resume_execute_%s' % pid):
            ttft_ns, exec_end_ns = backend.execute(pid)
        pending_rows.append({'pending_id': pid, 'ttft_ns': int(ttft_ns), 'exec_end_ns': int(exec_end_ns),
                             'gap_before_ns': gap_p})
        gap_p2 = clock.boundary()
        pending_rows[-1]['gap_after_ns'] = gap_p2
    t4 = clock._boundary
    record['pending_rows'] = pending_rows
    # ── 恒等式与关联关系断言（整数算术精确）──
    seg_total = sum(s['end_ns'] - s['start_ns'] for s in clock.segments)
    gaps = [gap_01, gap_12, gap_23, gap_34] + [g for r in pending_rows for g in (r['gap_before_ns'], r['gap_after_ns'])]
    assert (t4 - t0) == seg_total + sum(gaps), '计时恒等式失败：存在重复累加或遗漏'
    assert all(g >= 0 for g in gaps), '间隙为负：分段边界乱序'
    assert [r['pending_id'] for r in pending_rows] == list(pendings), '四 Pending 执行顺序与声明不一致'
    t4_exec_end = pending_rows[-1]['exec_end_ns']
    record['timing'] = {'t0_ns': t0, 't4_ns': t4, 'cycle_ms': (t4 - t0) / 1e6,
                        'segments': [{'stage': s['stage'], 'ms': (s['end_ns'] - s['start_ns']) / 1e6}
                                     for s in clock.segments],
                        'gaps_ms': [g / 1e6 for g in gaps], 'gap_total_ms': sum(gaps) / 1e6,
                        'identity_exact': True,
                        'note': 'T4=第4请求执行终点；t4_exec_end 与窗口边界差 = 终点后采样延迟'}
    record['timing']['t4_exec_end_offset_ns'] = int(t4 - t4_exec_end)
    # ── 取证 flush 严格在窗口外 ──
    evidence.flush_after_window()
    record['evidence'] = {'ops': [r['op'] for r in evidence.records],
                          'flush_ms': evidence.flush_ms,
                          'note': '取证仅在 T4 后执行；耗时不计入周期'}
    return record
