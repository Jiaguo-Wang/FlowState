#!/usr/bin/env python3
"""执行一次正式 SGLangAdapter 单节点驱逐图形处理器验证。"""

from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIRECTORY = (
    _REPOSITORY_ROOT
    / "motivation"
    / "artifacts"
    / "wp3b_gate_20260820"
)
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_ARTIFACT_DIRECTORY))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flowstate.adapters.sglang import RuntimeCheckpointHandle
from single_node_eviction_transport import FormalEvictionGateEngine
from targeted_probe import ControlClient


VOCAB_SIZE = 248_320
PARENT_LENGTH = 32_768
CONTROL_PORT = int(os.environ.get("FLOWSTATE_STEP5C_PORT", "49936"))
SAMPLING_PARAMETERS = {
    "max_new_tokens": 1,
    "temperature": 0,
    "ignore_eos": True,
}
ENGINE_CONFIGURATION = {
    "model_path": "/model",
    "context_length": 45_056,
    "mem_fraction_static": 0.40,
    "tp_size": 1,
    "chunked_prefill_size": 45_056,
    "max_mamba_cache_size": 16,
    "mamba_radix_cache_strategy": "extra_buffer",
    "disable_cuda_graph": True,
    "disable_overlap_schedule": True,
    "stream_interval": 1,
    "log_level": "info",
}


def make_tokens(seed: int, count: int) -> list[int]:
    """构造确定且不依赖分词器的令牌序列。"""
    return [(seed + index * 7_919) % VOCAB_SIZE for index in range(count)]


def token_digest(token_ids: tuple[int, ...]) -> str:
    """计算与正式适配器相同的前缀摘要。"""
    return hashlib.sha256(array("q", token_ids).tobytes()).hexdigest()


def wait_for_probe(client: ControlClient, timeout_seconds: float = 300.0) -> None:
    """等待调度器进程内的测试传输层就绪。"""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = client.ping()
            if response.get("ok"):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("等待单节点驱逐 transport 超时")


def inspect(client: ControlClient, token_ids: tuple[int, ...]) -> dict:
    """通过测试传输层读取目标节点状态。"""
    return client.checkpoint_control(
        nonce="flowstate_step5c:inspect:before",
        label="flowstate_step5c:before",
        action="inspect",
        token_ids=token_ids,
    )


def evict_with_formal_adapter(
    client: ControlClient,
    handle: RuntimeCheckpointHandle,
) -> dict:
    """请求调度器线程调用正式适配器的公开驱逐接口。"""
    return client._call(
        {
            "op": "checkpoint_control",
            "nonce": "flowstate_step5c:evict:formal",
            "label": "flowstate_step5c:formal",
            "action": "flowstate_evict_mamba_only",
            "checkpoint_id": handle.checkpoint_id,
            "token_ids": list(handle.token_ids),
            "extra_key": handle.extra_key,
            "expected_node_id": handle.expected_node_id,
            "expected_prefix_sha256": handle.expected_prefix_digest,
        }
    )


def main() -> int:
    """运行一次 32K 检查点构造与单节点驱逐验证。"""
    engine = None
    try:
        engine = FormalEvictionGateEngine(**ENGINE_CONFIGURATION)
        client = ControlClient(CONTROL_PORT)
        wait_for_probe(client)
        engine.flush_cache()

        parent = tuple(make_tokens(51_001, PARENT_LENGTH))
        request_id = "flowstate_step5c_parent"
        result = engine.generate(
            input_ids=list(parent),
            sampling_params=SAMPLING_PARAMETERS,
            rid=request_id,
        )
        output_ids = result.get("output_ids") or []
        if len(output_ids) != 1:
            raise RuntimeError(f"32K 请求输出数量异常：{output_ids}")

        before_response = inspect(client, parent)
        before_path = before_response["after"]["path"]
        handle = RuntimeCheckpointHandle(
            checkpoint_id="STEP5C_PARENT",
            token_ids=parent,
            expected_node_id=int(before_path["node_id"]),
            expected_prefix_digest=token_digest(parent),
        )
        eviction = evict_with_formal_adapter(client, handle)
        after_path = eviction["after"]["path"]
        proof = eviction["proof"]

        checks = {
            "before_fa": bool(before_path["target_full_present"]),
            "before_mamba": bool(before_path["target_mamba_present"]),
            "after_fa": bool(after_path["target_full_present"]),
            "after_mamba_absent": not bool(
                after_path["target_mamba_present"]
            ),
            **proof,
        }
        if not all(checks.values()):
            raise RuntimeError(f"单节点驱逐安全条件失败：{checks}")

        summary = {
            "status": "PASS",
            "before": {
                "node_id": before_path["node_id"],
                "fa_resident": before_path["target_full_present"],
                "mamba_resident": before_path["target_mamba_present"],
            },
            "after": {
                "node_id": after_path["node_id"],
                "fa_resident": after_path["target_full_present"],
                "mamba_resident": after_path["target_mamba_present"],
            },
            "safety": proof,
            "formal_primitive": eviction["formal_primitive"],
        }
        print(
            "[STEP5C-GATE] RESULT=" + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        return 0
    except Exception:
        traceback.print_exc()
        print("[STEP5C-GATE] STATUS=FAIL", flush=True)
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
