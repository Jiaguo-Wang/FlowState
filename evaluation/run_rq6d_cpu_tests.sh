#!/usr/bin/env bash
set -euo pipefail

# 只在完整副本中运行会生成历史路径报告的既有测试。
RQ6D_REPOSITORY=/home/wjg/code/FlowState
RQ6D_OUTPUT="${1:?必须提供新的诊断目录绝对路径}"
RQ6D_MIRROR=$(mktemp -d /tmp/flowstate-rq6d-cpu.XXXXXX)
mkdir -p "${RQ6D_MIRROR}/repo" "${RQ6D_MIRROR}/audits"
cp -a "${RQ6D_REPOSITORY}/." "${RQ6D_MIRROR}/repo/"
cp -a /home/wjg/data/agentx/audits/. "${RQ6D_MIRROR}/audits/"
echo "完整 CPU 测试副本：${RQ6D_MIRROR}"
docker run --rm --network none \
  -v "${RQ6D_MIRROR}/repo:${RQ6D_REPOSITORY}" \
  -v /home/wjg/data:/home/wjg/data:ro \
  -v "${RQ6D_MIRROR}/audits:/home/wjg/data/agentx/audits" \
  -v /home/wjg/models:/home/wjg/models:ro \
  -v "${RQ6D_OUTPUT}:/output" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e "PYTHONPATH=${RQ6D_REPOSITORY}" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -w "${RQ6D_REPOSITORY}" \
  --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime \
  -c 'import subprocess; from pathlib import Path; log=Path("/output/full_cpu_suite.log").open("w"); result=subprocess.run(["python3","-m","pytest","-q","-p","no:cacheprovider","tests"],stdout=log,stderr=subprocess.STDOUT); print("完整 CPU 测试退出码：",result.returncode); raise SystemExit(result.returncode)'
