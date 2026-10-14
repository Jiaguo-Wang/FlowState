#!/usr/bin/env bash
# 三个容器分别执行全量重算参考、保留条件和移除条件，任一失败立即停止。
set -euo pipefail
B1C_REPO=/home/wjg/code/FlowState
B1C_MODEL=/home/wjg/models/qwen3.5-9b
B1C_DATA=/home/wjg/data/agentic_coding_trajectories
B1C_OUT="$B1C_REPO/evaluation/strict_compatibility_b1c_output/b1c_g000_20261014_r2"
B1C_IMAGE=lmsysorg/sglang:v0.5.17-cu129-runtime
B1C_INPUT=evaluation/strict_compatibility_b1_output/b1_openhands_g000_20261014_02/pilot_evidence.json
mkdir -p "$B1C_OUT"
docker image inspect "$B1C_IMAGE" > "$B1C_OUT/image_identity.json"
sha256sum "$B1C_REPO/evaluation/strict_compatibility_b1c/runner.py" "$B1C_REPO/evaluation/strict_compatibility_b1c/protocol.py" "$B1C_REPO/$B1C_INPUT" > "$B1C_OUT/protocol_lock.sha256"
for B1C_ARM in reference retained removed; do
    mkdir -p "$B1C_OUT/$B1C_ARM"
    test ! -e "$B1C_OUT/$B1C_ARM/evidence.json"
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$B1C_OUT/$B1C_ARM/gpu_before.csv"
    B1C_MEM=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$B1C_MEM" -ge 1000 ]; then
        echo "GPU 0 非空闲，停止本 Pilot"
        exit 1
    fi
    echo "开始 $B1C_ARM"
    docker run --rm --pull never --name "flowstate-b1c-$B1C_ARM" --gpus '"device=0"' --network host --shm-size 32g \
      --mount "type=bind,src=$B1C_REPO,dst=$B1C_REPO,readonly" \
      --mount "type=bind,src=$B1C_MODEL,dst=$B1C_MODEL,readonly" \
      --mount "type=bind,src=$B1C_DATA,dst=$B1C_DATA,readonly" \
      --mount "type=bind,src=$B1C_OUT,dst=/b1c" \
      --env CUDA_VISIBLE_DEVICES=0 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
      --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 --env FLOWSTATE_STEP5D_PORT=49985 \
      --env "PYTHONPATH=$B1C_REPO:$B1C_REPO/tests/runtime:$B1C_REPO/motivation/artifacts/wp3b_gate_20260820" \
      --env TRITON_CACHE_DIR=/b1c/triton_cache --workdir "$B1C_REPO" --entrypoint python3 "$B1C_IMAGE" \
      -B -m evaluation.strict_compatibility_b1c.runner --arm "$B1C_ARM" --out "/b1c/$B1C_ARM/evidence.json" \
      --b1 "$B1C_INPUT" --reference /b1c/reference/evidence.json --retained /b1c/retained/evidence.json \
      > "$B1C_OUT/$B1C_ARM/runtime.log" 2>&1
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$B1C_OUT/$B1C_ARM/gpu_after.csv"
    echo "完成 $B1C_ARM"
done
