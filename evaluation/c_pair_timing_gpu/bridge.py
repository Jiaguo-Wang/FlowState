"""实验层接线：在 e1 原生观测（含 CUDA 事件计时/Admission门禁/COW审计）之上替换为本轮 Pilot 控制路径。"""
import os
from evaluation.lightweight_runtime_e1 import bridge as native
from evaluation.lightweight_runtime_e1 import control as original_control
from . import control


def scheduler_process(*args, **kwargs):
    original_control.apply = control.apply
    if native.ARM == 'cold':
        # Cold 臂目标由环境变量指定，而非 e1 默认的 B5C2 轮次标识
        native.TARGET = os.environ['PILOT_PENDING']
    return native.scheduler_process(*args, **kwargs)
