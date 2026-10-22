"""实验层接线，保留原E1 Scheduler隔离、Native观测与安全合同。"""
import os
from evaluation.lightweight_runtime_e1 import bridge as native
from evaluation.lightweight_runtime_e1 import control as original_control
from . import control

def scheduler_process(*args,**kwargs):
 def reference(scheduler,request):
  result=control.reference(scheduler,request)
  original_control.REFERENCE=control.REFERENCE
  return result
 original_control.reference=reference
 original_control.apply=control.apply
 if native.ARM=='cold':native.TARGET=os.environ['E4_PENDING']
 return native.scheduler_process(*args,**kwargs)
