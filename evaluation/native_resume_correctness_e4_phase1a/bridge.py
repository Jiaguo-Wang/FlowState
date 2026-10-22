"""仅替换实验固定目标与冻结语义核验；沿用既有Scheduler、协调和观测安全路径。"""
import os
from evaluation.lightweight_runtime_e1 import bridge as native
from evaluation.lightweight_runtime_e1 import control
from evaluation.exact_greedy_paired_gpu_fix_b5d4a import control as heavy
from .source_contract import choose_source

def scheduler_process(*args,**kwargs):
    control.choose_source=choose_source
    heavy.choose_source=choose_source
    if native.ARM=='cold':native.TARGET=os.environ['E4_PENDING']
    return native.scheduler_process(*args,**kwargs)
