#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wjg/code/newfind"
ART="$ROOT/flowstate/motivation/artifacts/wp3b_gate_20260820"

GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT="$ART/formal_k4_${RUN_TAG}"

mkdir -p "$OUT"

echo "[FORMAL-K4] output=$OUT"
echo "[FORMAL-K4] gpu=$GPU_ID"

cat > "$OUT/order_plan.tsv" <<EOF
repquota orderpolicy
11prompt_lru
12workflow_k
2to3-2.7 1workflow_k
2to3-2.7 2prompt_lru
31prompt_lru
32workflow_k
411toppm 1workflow_k
411toppm 2prompt_lru
51prompt_lru
52workflow_k
EOF


run_arm() {
    local rep="$1"
    local order="$2"
    local policy="$3"

    local log="$OUT/r$(printf '%02d' "$rep")_o${order}_${policy}.log"
    local cname="fs-k4-r${rep}-o${order}-${policy//_/-}"

    echo
    echo "============================================================"
    echo "[FORMAL-K4] START rep=$rep order=$order policy=$policy"
    echo "[FORMAL-K4] log=$log"
    echo "============================================================"

    docker run --rm \
      --name "$cname" \
      --gpus "\"device=${GPU_ID}\"" \
      --network host \
      --shm-size 32g \
      -v "$ROOT:/repo" \
      -v /home/wjg/models/qwen3.5-9b:/model:ro \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e WP3D_CTRL_PORT=49935 \
      -e TRITON_CACHE_DIR=/repo/flowstate/motivation/artifacts/wp3b_gate_20260820/.triton_cache \
      -e HF_HUB_OFFLINE=1 \
      -e TRANSFORMERS_OFFLINE=1 \
      lmsysorg/sglang:v0.5.17-cu129-runtime \
      bash -lc "
        set -e

        cd /sgl-workspace/sglang/python/sglang/srt
        patch -p1 < /repo/flowstate/motivation/artifacts/runtime_validation_gap_replay_20260819/instrumentation.patch

        cd /sgl-workspace/sglang
        patch -p1 < /repo/flowstate/motivation/artifacts/replay_cost_20260819/recovery_timing.patch

        cd /repo

        exec python3 -u \
          flowstate/motivation/artifacts/wp3b_gate_20260820/gate_of_gate.py \
          --policy ${policy}
      " 2>&1 | tee "$log"

    # ------------------------------------------------------------
    # Hard post-run guards.
    # ------------------------------------------------------------
    if ! grep -q '\[WP3B-GOG\] STATUS=complete' "$log"; then
        echo "[FORMAL-K4] ERROR: missing STATUS=complete"
        exit 1
    fi

    if ! grep -q '\[WP3B-GOG\] PRE_MEASURE' "$log"; then
        echo "[FORMAL-K4] ERROR: missing PRE_MEASURE"
        exit 1
    fi

    if ! grep -q "\[FSVAL\] match_end req=wp3b_gog_${policy}_p1_child_b" "$log"; then
        echo "[FORMAL-K4] ERROR: missing measured Child-B FSVAL path"
        exit 1
    fi

    if ! grep -q "\[FSWP2\] request_timing req=wp3b_gog_${policy}_p1_child_b" "$log"; then
        echo "[FORMAL-K4] ERROR: missing measured Child-B timing"
        exit 1
    fi

    echo "[FORMAL-K4] DONE rep=$rep order=$order policy=$policy"

    grep -E \
      "\[WP3B-GOG\].*(PRE_MEASURE|STATUS)|\[FSVAL\] match_end req=wp3b_gog_${policy}_p1_child_b|\[FSWP2\] request_timing req=wp3b_gog_${policy}_p1_child_b" \
      "$log" \
      | tee -a "$OUT/compact_results.log"
}


# Alternating order controls systematic temporal/order effects.
run_arm 1 1 prompt_lru
run_arm 1 2 workflow_k

run_arm 2 1 workflow_k
run_arm 2 2 prompt_lru

run_arm 3 1 prompt_lru
run_arm 3 2 workflow_k

run_arm 4 1 workflow_k
run_arm 4 2 prompt_lru

run_arm 5 1 prompt_lru
run_arm 5 2 workflow_k


echo
echo "============================================================"
echo "[FORMAL-K4] ALL 10 ARMS COMPLETE"
echo "============================================================"

python3 \
  "$ART/parse_formal_k4.py" \
  "$OUT"

echo
echo "[FORMAL-K4] SUMMARY"
cat "$OUT/formal_k4_summary.md"

echo
echo "[FORMAL-K4] RESULT_DIR=$OUT"
