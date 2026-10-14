#!/usr/bin/env bash
# 只运行独立 CPU 情景分析，不改写协议，不调用 GPU 或回放。
set -euo pipefail
cd /home/wjg/code/FlowState
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
/home/wjg/uv312/bin/python evaluation/submission_p1_allocation_structure/analyze.py
/home/wjg/uv312/bin/python evaluation/submission_p1_allocation_structure/verify_and_report.py
