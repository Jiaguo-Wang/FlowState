#!/usr/bin/env bash
# 只读取既有数据并在本目录重建审计，不启动推理引擎。
set -euo pipefail
cd /home/wjg/code/FlowState
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
/home/wjg/uv312/bin/python evaluation/openhands_cross_workflow_compatibility_audit/extract_inputs.py
/home/wjg/uvgood/bin/python evaluation/openhands_cross_workflow_compatibility_audit/reconstruct_tokens.py
/home/wjg/uv312/bin/python evaluation/openhands_cross_workflow_compatibility_audit/finalize.py
