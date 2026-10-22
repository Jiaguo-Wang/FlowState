"""直接复用冻结B5-C3只读采集入口；每个容器独立挂载当前工作负载目录。"""
from evaluation.multishare_runtime_gate_b5c3.runner import main
if __name__=='__main__':main()
