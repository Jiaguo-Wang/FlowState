#!/usr/bin/env bash
set -euo pipefail

# 只接受正式采集或单次诊断，避免误启动其他实验。
RQ6_MODE="${1:?必须指定 preflight 或 formal}"
RQ6_OUTPUT_ROOT="${2:?必须指定绝对输出目录}"
RQ6_GPU_DEVICE="${3:-1}"
RQ6_REPOSITORY="/home/wjg/code/FlowState"
RQ6_FORMAL_ROOT="${RQ6_REPOSITORY}/evaluation/runtime_artifacts/rq3_openhands_main_formal_20260904_001017"

if [[ "${RQ6_MODE}" != "preflight" && "${RQ6_MODE}" != "formal" ]]; then
  echo "模式必须是 preflight 或 formal" >&2
  exit 2
fi

RQ6_EXTRA_ARGS=()
if [[ "${RQ6_MODE}" == "preflight" ]]; then
  RQ6_EXTRA_ARGS+=(--preflight)
fi

docker run --rm \
  --gpus "device=${RQ6_GPU_DEVICE}" \
  --network host \
  --shm-size 32g \
  -v "${RQ6_REPOSITORY}:${RQ6_REPOSITORY}" \
  -v /home/wjg/data:/home/wjg/data:ro \
  -v /home/wjg/models:/home/wjg/models:ro \
  -v /home/wjg/models/qwen3.5-9b:/model:ro \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e "PYTHONPATH=${RQ6_REPOSITORY}:${RQ6_REPOSITORY}/tests/runtime" \
  -e FLOWSTATE_STEP12H9A_PORT=49948 \
  -e "TRITON_CACHE_DIR=${RQ6_OUTPUT_ROOT}/cache/triton" \
  -w "${RQ6_REPOSITORY}" \
  --entrypoint python3 \
  lmsysorg/sglang:v0.5.17-cu129-runtime \
  -m evaluation.rq6_runtime_overhead \
  --output-root "${RQ6_OUTPUT_ROOT}" \
  --formal-root "${RQ6_FORMAL_ROOT}" \
  "${RQ6_EXTRA_ARGS[@]}"
