#!/usr/bin/env bash
# 仅CPU重建消息输入，禁用网络和GPU，独立目录保存核验结果。
set -eu
cd /home/wjg/code/FlowState
/home/wjg/uv312/bin/python -B -m evaluation.agentic_executable_workload_cpu_gate.prepare
docker run --rm --pull never --name flowstate-agentic-ab-cpu --runtime=runc --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --tmpfs /tmp:rw,size=512m --user 1045:1045 -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e TOKENIZERS_PARALLELISM=false -e HF_HOME=/tmp/hf --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/agentic_executable_workload_cpu_gate,dst=/home/wjg/code/FlowState/evaluation/agentic_executable_workload_cpu_gate --mount type=bind,src=/home/wjg/models/qwen3.5-9b,dst=/home/wjg/models/qwen3.5-9b,readonly --entrypoint python3 sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926 -B -m evaluation.agentic_executable_workload_cpu_gate.rebuild_tokens
python3 -B -m unittest -v evaluation.agentic_executable_workload_cpu_gate.test_contracts
python3 -B -m evaluation.agentic_executable_workload_cpu_gate.verify
