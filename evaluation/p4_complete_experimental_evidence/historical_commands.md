# 历史命令记录

仅转录文件中已有记录，未执行。容器路径保持原文；不能假定在宿主机直接可用。

### 02_long：server_command.txt

来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/recovery_profiler_128k_20260826_133712_596813/server_command.txt`。以下是历史记录，不代表本轮执行。

```text
进程入口：python -m evaluation.recovery_profiler_128k
SGLang Engine 参数：chunked_prefill_size=45056 context_length=131200 disable_cuda_graph=True disable_overlap_schedule=True enable_request_time_stats_logging=True log_level='info' mamba_max_states_per_path=-1 mamba_radix_cache_strategy='extra_buffer' mamba_track_interval=256 max_mamba_cache_size=24 mem_fraction_static=0.4 model_path='/model' stream_interval=1 tp_size=1
说明：context_length 仅从 Step 9D 的 45056 提高到 131200；模型语义与 cache policy 不变。
```

### 02_failed：server_command.txt

来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/recovery_profiler_128k_20260826_133234_448476/server_command.txt`。以下是历史记录，不代表本轮执行。

```text
进程入口：python -m evaluation.recovery_profiler_128k
SGLang Engine 参数：chunked_prefill_size=45056 context_length=131200 disable_cuda_graph=True disable_overlap_schedule=True enable_request_time_stats_logging=True log_level='info' mamba_max_states_per_path=-1 mamba_radix_cache_strategy='extra_buffer' mamba_track_interval=256 max_mamba_cache_size=24 mem_fraction_static=0.4 model_path='/model' stream_interval=1 tp_size=1
说明：context_length 仅从 Step 9D 的 45056 提高到 131200；模型语义与 cache policy 不变。
```

### 02_position：server_command.txt

来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/recovery_position_audit_20260826_144654_852303/server_command.txt`。以下是历史记录，不代表本轮执行。

```text
进程入口：python -m evaluation.recovery_position_audit
Legacy 引擎参数：chunked_prefill_size=45056 context_length=45056 disable_cuda_graph=True disable_overlap_schedule=True enable_request_time_stats_logging=True log_level='info' mamba_max_states_per_path=-1 mamba_radix_cache_strategy='extra_buffer' mamba_track_interval=256 max_mamba_cache_size=24 mem_fraction_static=0.4 model_path='/model' stream_interval=1 tp_size=1
Position matrix 引擎参数：chunked_prefill_size=45056 context_length=131200 disable_cuda_graph=True disable_overlap_schedule=True enable_request_time_stats_logging=True log_level='info' mamba_max_states_per_path=-1 mamba_radix_cache_strategy='extra_buffer' mamba_track_interval=256 max_mamba_cache_size=24 mem_fraction_static=0.4 model_path='/model' stream_interval=1 tp_size=1
两个阶段均使用正式 Mamba-only eviction primitive。
```

### 02_kimi：launch_command.txt

来源：`/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555/launch_command.txt`。以下是历史记录，不代表本轮执行。

```text
docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair01-A --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 1 --condition A

docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair01-B --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 1 --condition B

docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair02-B --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 2 --condition B

docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair02-A --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 2 --condition A

docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair03-A --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 3 --condition A

docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair03-B --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 3 --condition B

docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair04-B --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 4 --condition B

docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair04-A --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 4 --condition A

docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair05-A --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 5 --condition A

docker run --rm --pull never --name flowstate-rq5b3-20260922-220555-pair05-B --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase3_output/rq5b3_20260922_220555,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49981 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase3 --output-root /output --pair 5 --condition B
```

### 03_model：server_command.txt

来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/recovery_model_freeze_20260826_154235_266020/server_command.txt`。以下是历史记录，不代表本轮执行。

```text
进程入口：python -m evaluation.recovery_model_freeze
Held-out 引擎参数：chunked_prefill_size=45056 context_length=131200 disable_cuda_graph=True disable_overlap_schedule=True enable_request_time_stats_logging=True log_level='info' mamba_max_states_per_path=-1 mamba_radix_cache_strategy='extra_buffer' mamba_track_interval=256 max_mamba_cache_size=24 mem_fraction_static=0.4 model_path='/model' stream_interval=1 tp_size=1
正式 mutation primitive：flowstate.adapters.sglang.SGLangAdapter.evict_mamba_only
```

### 09_kimi1：launch_command.txt

来源：`/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase1_output/rq5b_20260922_064043/launch_command.txt`。以下是历史记录，不代表本轮执行。

```text
实际运行命令（初始 smoke；三次复现仅将末尾运行标识依次改为 rep01、rep02、rep03）：
docker run --rm --pull never --gpus '"device=0,1"' --network host --shm-size 32g \
  --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly \
  --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly \
  --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase1_output/rq5b_20260922_064043,dst=/output \
  --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 \
  --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor \
  --env FLOWSTATE_RQ5B_PORT=49961 --workdir /home/wjg/code/FlowState \
  --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime \
  -m evaluation.rq5b_kimi_phase1 --output-root /output --run-id smoke

内部引擎启动参数见 runs/smoke/engine_args.json。采用 Engine API；所有对应的 CLI 参数均已通过 launch_server_help.txt 核实。本轮未启动独立 HTTP 服务。
```

### 09_kimi2：launch_command.txt

来源：`/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527/launch_command.txt`。以下是历史记录，不代表本轮执行。

```text
docker run --rm --pull never --name flowstate-rq5b2-20260922-152527-smoke --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49971 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase2 --output-root /output --run-id smoke

docker run --rm --pull never --name flowstate-rq5b2-20260922-152527-rep01 --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49971 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase2 --output-root /output --run-id rep01

docker run --rm --pull never --name flowstate-rq5b2-20260922-152527-rep02 --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49971 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase2 --output-root /output --run-id rep02

docker run --rm --pull never --name flowstate-rq5b2-20260922-152527-rep03 --gpus '"device=0,1"' --network host --shm-size 32g --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly --mount type=bind,src=/data1/models/Kimi-Linear-48B-A3B-Instruct,dst=/model,readonly --mount type=bind,src=/home/wjg/code/FlowState/evaluation/rq5b_kimi_phase2_output/rq5b2_20260922_152527,dst=/output --env HF_HOME=/output/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/home/wjg/code/FlowState:/home/wjg/code/FlowState/tests/runtime:/home/wjg/code/FlowState/motivation/artifacts/wp3b_gate_20260820 --env TRITON_CACHE_DIR=/output/cache/triton --env TORCHINDUCTOR_CACHE_DIR=/output/cache/torchinductor --env FLOWSTATE_RQ5B_PORT=49971 --workdir /home/wjg/code/FlowState --entrypoint python3 lmsysorg/sglang:v0.5.17-cu129-runtime -m evaluation.rq5b_kimi_phase2 --output-root /output --run-id rep03
```

### 09_rerun：server_command.txt

来源：`/home/wjg/code/FlowState/evaluation/runtime_artifacts/formal_model_latency_rerun_20260827_015154_313413/server_command.txt`。以下是历史记录，不代表本轮执行。

```text
容器内入口：
python3 -m evaluation.formal_model_latency_rerun

SGLang 使用 FormalEndToEndGateEngine 内嵌启动。
引擎参数：
{
  "chunked_prefill_size": 45056,
  "context_length": 45056,
  "disable_cuda_graph": true,
  "disable_overlap_schedule": true,
  "enable_request_time_stats_logging": true,
  "log_level": "info",
  "mamba_max_states_per_path": -1,
  "mamba_radix_cache_strategy": "extra_buffer",
  "mamba_track_interval": 256,
  "max_mamba_cache_size": 24,
  "mem_fraction_static": 0.4,
  "model_path": "/model",
  "stream_interval": 1,
  "tp_size": 1
}
```
