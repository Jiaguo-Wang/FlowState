"""只在本轮隔离Scheduler进程复用已验证Native观测；不改安装源码。"""
def scheduler_process(*args,**kwargs):
    from evaluation.exact_reconciliation_correctness_gate_b5d3 import bridge,control
    from .control import apply
    control.apply=apply
    return bridge.scheduler_process(*args,**kwargs)
