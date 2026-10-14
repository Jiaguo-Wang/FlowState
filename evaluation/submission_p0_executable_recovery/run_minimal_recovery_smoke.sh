#!/usr/bin/env bash
# 独立可复现冒烟入口（任务 A）：
#   ./run_minimal_recovery_smoke.sh            # 构造样本 + GPU 冒烟（完整流程）
#   ./run_minimal_recovery_smoke.sh build      # 仅构造样本（CPU）
#   ./run_minimal_recovery_smoke.sh smoke      # 仅 GPU 冒烟（样本已存在）
#
# 约束：共享 GPU 机器——执行前后都用 nvidia-smi 确认；只占用一张空闲卡。
set -euo pipefail

IMAGE="lmsysorg/sglang:v0.5.17-cu129-runtime"
REPO="/home/wjg/code/FlowState"
MODEL="/home/wjg/models/qwen3.5-9b"
DATA="/home/wjg/data/agentx"
PORT="${FLOWSTATE_STEP5D_PORT:-49939}"
MODE="${1:-all}"

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

COMMON_ARGS=(
  --rm --network host --shm-size 32g
  -v "$REPO:$REPO"
  -v "$MODEL:/model:ro"
  -v "$MODEL:$MODEL:ro"
  -v "$DATA:$DATA:ro"
  -e "FLOWSTATE_STEP5D_PORT=$PORT"
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e "TRITON_CACHE_DIR=$TRITON_CACHE"
  -w "$REPO"
)

run_build() {
  docker run "${COMMON_ARGS[@]}" "$IMAGE" \
    python evaluation/submission_p0_executable_recovery/build_minimal_cases.py
}

run_smoke() {
  docker run "${COMMON_ARGS[@]}" --gpus "\"device=$GPU\"" \
    -e CUDA_VISIBLE_DEVICES=0 \
    "$IMAGE" \
    python evaluation/submission_p0_executable_recovery/minimal_recovery_smoke.py
}

case "$MODE" in
  build) run_build ;;
  smoke) run_smoke ;;
  all)   run_build && run_smoke ;;
  *) echo "未知模式: $MODE（可用: all/build/smoke）" >&2; exit 1 ;;
esac

echo "== 执行后 GPU 状态 =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
