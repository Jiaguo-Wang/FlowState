#!/usr/bin/env bash
# 仅使用冻结镜像做CPU输入核验；不挂载GPU，不联网，不启动引擎。
set -eu
cd /home/wjg/code/FlowState
docker run --rm --name flowstate-imw-cpu --runtime=runc --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --tmpfs /tmp:rw,size=512m --user 1045:1045 -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e TOKENIZERS_PARALLELISM=false -e HF_HOME=/tmp/hf -v /home/wjg/code/FlowState:/home/wjg/code/FlowState:ro -v /home/wjg/code/FlowState/evaluation/independent_multishare_workloads:/home/wjg/code/FlowState/evaluation/independent_multishare_workloads:rw -v /home/wjg/models/qwen3.5-9b:/home/wjg/models/qwen3.5-9b:ro --entrypoint python3 sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926 -B /home/wjg/code/FlowState/evaluation/independent_multishare_workloads/tokenize_cpu.py
