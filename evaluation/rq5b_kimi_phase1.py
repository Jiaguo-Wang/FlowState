#!/usr/bin/env python3
"""在独立引擎生命周期中验证 Kimi 双卡运行时与 KDA 单组件隔离。"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/runtime"))
sys.path.insert(0, str(ROOT / "motivation/artifacts/wp3b_gate_20260820"))


ENGINE_ARGS = {
    "model_path": "/model",
    "trust_remote_code": True,
    "tp_size": 2,
    "context_length": 32768,
    "attention_backend": "triton",
    "linear_attn_backend": "triton",
    "mamba_radix_cache_strategy": "extra_buffer",
    "mamba_track_interval": 256,
    "mamba_max_states_per_path": -1,
    "max_mamba_cache_size": 16,
    "disable_overlap_schedule": True,
    "cuda_graph_backend_decode": "disabled",
    "cuda_graph_backend_prefill": "disabled",
    "chunked_prefill_size": 2048,
    "mem_fraction_static": 0.80,
    "log_level": "info",
}
SAMPLING = {"max_new_tokens": 1, "temperature": 0, "ignore_eos": True}
VOCAB_SIZE = 163840
C1 = 8192
C2 = 16384


def save(path: Path, value: object) -> None:
    """把阶段一的原始证据写成可审计的中文 JSON。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tokens(seed: int, length: int) -> tuple[int, ...]:
    """生成固定、有效且避开特殊令牌的测试前缀。"""
    return tuple(1000 + ((seed + index * 7919) % (VOCAB_SIZE - 2000)) for index in range(length))


def digest(values: tuple[int, ...]) -> str:
    """生成与运行时句柄一致的令牌摘要。"""
    return hashlib.sha256(array("q", values).tobytes()).hexdigest()


def call(client: object, action: str, **fields: object) -> dict:
    """向对应 rank 的调度器安全时点提交单个控制请求。"""
    response = client._call({
        "op": "checkpoint_control",
        "nonce": f"rq5b:{action}:{time.monotonic_ns()}",
        "action": action,
        **fields,
    })
    if not response.get("ok"):
        raise RuntimeError(f"运行时控制失败：{response}")
    return response


def wait(client: object, seconds: float = 300.0) -> None:
    """等待对应 rank 的本地控制服务完成启动。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if client.ping().get("ok"):
                return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("Kimi 调度器控制端口等待超时")


def generate(engine: object, request_id: str, values: tuple[int, ...]) -> dict:
    """用一次确定性单令牌生成在缓存中建立前缀分叉。"""
    output = engine.generate(input_ids=list(values), sampling_params=SAMPLING, rid=request_id)
    if not isinstance(output, dict) or len(output.get("output_ids") or ()) != 1:
        raise RuntimeError(f"请求 {request_id} 未完成单令牌生成")
    metadata = output.get("meta_info") or {}
    if int(metadata.get("num_retractions", 0) or 0) != 0:
        raise RuntimeError(f"请求 {request_id} 发生调度回撤")
    return {"request_id": request_id, "input_tokens": len(values), "output_ids": output["output_ids"], "metadata": metadata}


def heg(view: dict) -> dict:
    """只用两个逻辑检查点的实际驻留事实计算 H、E、G。"""
    paths = view["paths"]
    h = C2 if paths["c2"]["path_full_all_present"] else C1 if paths["c1"]["path_full_all_present"] else 0
    e = C2 if paths["c2"]["target_mamba_present"] else C1 if paths["c1"]["target_mamba_present"] else 0
    return {"H": h, "E": e, "G": h - e}


def rank_facts(view: dict) -> dict:
    """抽取可跨 rank 比对的逻辑事实，保留每个物理节点标识。"""
    return {
        "scope": view["scope"],
        "H_E_G": heg(view),
        "tree_structure_digest": view["tree"]["structure_sha256"],
        "attention_tree_digest": view["tree"]["full_tree_sha256"],
        "recurrent_tree_digest": view["tree"]["mamba_tree_sha256"],
        "checkpoints": {
            name: {
                "node_id": int(path["node_id"]),
                "prefix_sha256": path["prefix_sha256"],
                "attention_resident": bool(path["target_full_present"] and path["path_full_all_present"]),
                "recurrent_resident": bool(path["target_mamba_present"]),
                "recurrent_slots": path["target_mamba_slots"],
                "path_node_ids": path["path_node_ids"],
                "path_recurrent_positions": path["path_mamba_positions"],
            }
            for name, path in view["paths"].items()
        },
    }


def logical_consistency(views: dict[int, dict]) -> bool:
    """核对两个 rank 的逻辑驻留与前缀身份，不假设物理槽号相同。"""
    a, b = (rank_facts(views[rank]) for rank in (0, 1))
    return a["H_E_G"] == b["H_E_G"] and all(
        a["checkpoints"][name][field] == b["checkpoints"][name][field]
        for name in ("c1", "c2")
        for field in ("prefix_sha256", "attention_resident", "recurrent_resident")
    )


def main() -> int:
    """先执行运行时门禁，再完成一次双 rank 的单组件隔离。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = Path(args.output_root) / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    save(out / "engine_args.json", ENGINE_ARGS)
    record = {"run_id": args.run_id, "status": "INCOMPLETE", "phase": "启动"}
    engine = None
    try:
        from targeted_probe import ControlClient
        from rq5b_kimi_phase1_transport import KimiPhase1Engine

        engine = KimiPhase1Engine(**ENGINE_ARGS)
        clients = {rank: ControlClient(int(os.environ.get("FLOWSTATE_RQ5B_PORT", "49961")) + rank, timeout_s=300.0) for rank in (0, 1)}
        for client in clients.values():
            wait(client)
        gates = {rank: call(client, "runtime_gate") for rank, client in clients.items()}
        save(out / "runtime_gate.json", gates)
        assert all(gates[rank]["scope"]["tp_rank"] == rank for rank in (0, 1))
        assert all(gates[rank]["scope"]["track_interval"] == 256 for rank in (0, 1))
        assert all(gates[rank]["tree"]["mamba_node_count"] == 0 for rank in (0, 1))
        record["phase"] = "构造检查点"
        prefix8 = tokens(1101, C1)
        prefix16 = prefix8 + tokens(2202, C2 - C1)
        requests = [
            generate(engine, f"{args.run_id}:main", prefix16 + tokens(3303, 256)),
            generate(engine, f"{args.run_id}:branch8", prefix8 + tokens(4404, 256)),
            generate(engine, f"{args.run_id}:branch16", prefix16 + tokens(5505, 256)),
        ]
        save(out / "requests.json", requests)
        handles = [
            {"checkpoint_id": "c1", "token_ids": list(prefix8), "expected_prefix_digest": digest(prefix8)},
            {"checkpoint_id": "c2", "token_ids": list(prefix16), "expected_prefix_digest": digest(prefix16)},
        ]
        before = {rank: call(client, "inspect", handles=handles) for rank, client in clients.items()}
        for rank in (0, 1):
            for row in handles:
                path = before[rank]["paths"][row["checkpoint_id"]]
                assert path["prefix_sha256"] == row["expected_prefix_digest"]
                assert path["target_mamba_present"] and path["path_full_all_present"]
            assert heg(before[rank]) == {"H": C2, "E": C2, "G": 0}
        assert logical_consistency(before)
        save(out / "before.json", {rank: rank_facts(view) for rank, view in before.items()})
        record["phase"] = "单组件驱逐"
        for rank in (0, 1):
            for row in handles:
                row[f"expected_node_id_rank{rank}"] = before[rank]["paths"][row["checkpoint_id"]]["node_id"]
        results = {}
        for rank, client in clients.items():
            local_handles = [{**row, "expected_node_id": row[f"expected_node_id_rank{rank}"]} for row in handles]
            results[rank] = call(client, "evict_target", handles=local_handles, target_id="c2")
            save(out / f"eviction_rank{rank}.json", results[rank])
        after = {rank: call(client, "inspect", handles=handles) for rank, client in clients.items()}
        save(out / "after.json", {rank: rank_facts(view) for rank, view in after.items()})
        checks = {
            "runtime_gate": True,
            "rank_consistency_before": logical_consistency(before),
            "rank_consistency_after": logical_consistency(after),
            "heg_before": all(heg(before[rank]) == {"H": C2, "E": C2, "G": 0} for rank in (0, 1)),
            "heg_after": all(heg(after[rank]) == {"H": C2, "E": C1, "G": C1} for rank in (0, 1)),
            "recurrent_only_eviction": all(all(results[rank]["checks"].values()) for rank in (0, 1)),
            "attention_preserved": all(before[rank]["tree"]["full_tree_sha256"] == after[rank]["tree"]["full_tree_sha256"] and before[rank]["accounting"]["full_allocator"] == after[rank]["accounting"]["full_allocator"] for rank in (0, 1)),
            "no_native_recurrent_eviction": all(results[rank]["checks"]["recurrent_change_exact"] for rank in (0, 1)),
            "no_unexpected_rematerialization": all(not after[rank]["paths"]["c2"]["target_mamba_present"] for rank in (0, 1)),
        }
        record.update({"status": "PASS" if all(checks.values()) else "FAIL", "phase": "完成", "checks": checks, "heg_before": {rank: heg(before[rank]) for rank in (0, 1)}, "heg_after": {rank: heg(after[rank]) for rank in (0, 1)}})
        if record["status"] != "PASS":
            raise RuntimeError(f"组件隔离门禁失败：{checks}")
        return 0
    except Exception as error:
        record.update({"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()})
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                record["engine_shutdown"] = "PASS"
            except Exception as error:
                record["engine_shutdown"] = f"FAIL: {error!r}"
                record["status"] = "FAIL"
        save(out / "record.json", record)


if __name__ == "__main__":
    raise SystemExit(main())
