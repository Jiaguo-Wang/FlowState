#!/usr/bin/env bash
# 独立可复现入口（任务 A5：LMCache lmcache_10 真实消息恢复验证）：
#   ./reproduce.sh          # GPU 冒烟（输入为冻结文件，无需构造步骤）
#
# 约束：共享 GPU 机器——执行前后都用 nvidia-smi 确认；只占用一张空闲卡。
# 输入只读：agentic_workflow_dataset_research/phase2_submission_inputs/
# （不需要 agentx 数据挂载）。产物写入本目录，不触碰上一轮 Task A 证据。
set -euo pipefail

IMAGE="lmsysorg/sglang:v0.5.17-cu129-runtime"
REPO="/home/wjg/code/FlowState"
MODEL="/home/wjg/models/qwen3.5-9b"
PORT="${FLOWSTATE_STEP5D_PORT:-49939}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_CACHE="$SCRIPT_DIR/artifacts/.triton_cache"
mkdir -p "$TRITON_CACHE"

echo "== 执行前 GPU 状态 =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

pick_gpu() {
  # 选择第一张显存占用 < 1000 MiB 的卡
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 1000 {print $1; exit}'
}

GPU="$(pick_gpu)"
if [ -z "$GPU" ]; then
  echo "没有空闲 GPU，拒绝启动（共享机器保护）" >&2
  exit 1
fi
echo "== 使用 GPU $GPU =="

docker run --rm --network host --shm-size 32g \
  --gpus "\"device=$GPU\"" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$REPO:$REPO" \
  -v "$MODEL:/model:ro" \
  -v "$MODEL:$MODEL:ro" \
  -e "FLOWSTATE_STEP5D_PORT=$PORT" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e "TRITON_CACHE_DIR=$TRITON_CACHE" \
  -w "$REPO" \
  "$IMAGE" \
  python evaluation/submission_p0_lmcache_recovery/lmcache_recovery_smoke.py

echo "== 执行后 GPU 状态 =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
