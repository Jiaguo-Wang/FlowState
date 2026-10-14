"""只验证一个冻结跨 Workflow 边，记录原生匹配和复制调用的证据。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

from evaluation.strict_compatibility_b1c.protocol import (
    isolation_plan, require, semantic_nodes, token_digest,
)

TARGET = "RQ3_G000_B_TURN_001"
PENDING = "RQ3_G000_A_PENDING_TURN_003"
MATCHES = {}
COPIES = []
COPY_OWNERS = {}


def tensor_ids(value):
    return [] if value is None else [int(x) for x in value.detach().cpu().reshape(-1).tolist()]


def install_observation(cache):
    """原始匹配和复制各调用一次，只记录现有返回值及请求字段。"""
    original_match = cache.match_prefix
    pool = cache.req_to_token_pool.mamba_pool
    original_copy = pool.copy_from

    def copy_from(src, dst):
        source, destination = tensor_ids(src), tensor_ids(dst)
        result = original_copy(src, dst)
        COPIES.append({
            "source_slots": source, "destination_slots": destination,
            "request_ids": [COPY_OWNERS.get((s, d)) for s, d in zip(source, destination)],
            "native_call_returned": True, "monotonic_ns": time.monotonic_ns(),
        })
        return result

    def match_prefix(params):
        result = original_match(params)
        req = params.req
        if req is not None:
            source = tensor_ids(req.mamba_cow_src_index)
            destination = tensor_ids(req.mamba_pool_idx)
            if params.cow_mamba and source:
                for s, d in zip(source, destination):
                    COPY_OWNERS[(s, d)] = str(req.rid)
            row = {
                "request_id": str(req.rid), "monotonic_ns": time.monotonic_ns(),
                "physical_fa_hit": int(result.full_kv_hit_length),
                "executable_prefix": int(result.device_indices.numel()),
                "replay_gap": int(result.full_kv_hit_length) - int(result.device_indices.numel()),
                "best_match_node": int(result.best_match_node),
                "last_device_node": int(result.last_device_node),
                "last_host_node": int(result.last_host_node),
                "mamba_host_hit_length": int(result.mamba_host_hit_length),
                "mamba_branching_seqlen": result.mamba_branching_seqlen,
                "cow_requested": bool(params.cow_mamba),
                "cow_source_slots": source, "working_slots": destination,
                "input_token_count": len(req.origin_input_ids),
                "input_token_digest": token_digest(req.origin_input_ids),
            }
            MATCHES.setdefault(str(req.rid), []).append(row)
        return result

    cache.match_prefix = match_prefix
    pool.copy_from = copy_from


def state_snapshot(scheduler):
    """在调度器静止时直接读取树和物理池，不执行前缀匹配。"""
    import targeted_probe as probe
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
    from evaluation.barrier_fa_frontier_control import semantic_cache_snapshot
    cache = scheduler.tree_cache
    scope = probe._validate_runtime_scope(scheduler, cache)
    before = semantic_cache_snapshot(cache, component_type=CT,
        tree_snapshot=probe._global_maps(cache), accounting_snapshot=probe._accounting_snapshot(cache))
    nodes = []
    for node_id, node in probe._tree_items(cache.tree_core):
        segments, current = [], node
        while current.parent is not None:
            require(not current.key.is_bigram, "本 Pilot 不支持 bigram key")
            segments.append(list(current.key))
            current = current.parent
        tokens = [int(x) for seg in reversed(segments) for x in seg]
        full, mamba = node.component_data[CT.FULL], node.component_data[CT.MAMBA]
        nodes.append({"node_id": node_id, "token_ids": tokens, "position": len(tokens),
            "prefix_digest": token_digest(tokens), "full_present": full.value is not None,
            "full_count": 0 if full.value is None else len(full.value),
            "mamba_slots": tensor_ids(mamba.value), "mamba_host_present": mamba.host_value is not None,
            "references": {"full_lock": int(full.lock_ref), "mamba_lock": int(mamba.lock_ref),
                "mamba_host_lock": int(mamba.host_lock_ref), "mamba_session": int(mamba.session_ref)}})
    state = cache.req_to_token_pool.mamba_pool.mamba_cache
    components = {"conv": [{"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)} for t in state.conv],
        "temporal": {"shape": list(state.temporal.shape), "dtype": str(state.temporal.dtype), "device": str(state.temporal.device)}}
    after = semantic_cache_snapshot(cache, component_type=CT,
        tree_snapshot=probe._global_maps(cache), accounting_snapshot=probe._accounting_snapshot(cache))
    require(before == after, "只读树观测改变了运行时语义状态")
    return {"scope": scope, "nodes": nodes, "components": components,
        "accounting": probe._accounting_snapshot(cache), "tree": probe._global_maps(cache),
        "observation_noninterference": True, "monotonic_ns": time.monotonic_ns()}


def scheduler_process(*args, **kwargs):
    """仅在本次隔离进程安装观察器，原始源码文件保持不变。"""
    import wp3b_end_to_end_transport as transport
    import targeted_probe as probe
    original_control = transport._run_checkpoint_control

    def control(scheduler, request):
        action = request.get("action")
        if action == "b1c_state":
            return {"ok": True, "state": state_snapshot(scheduler)}
        if action in ("b1c_records", "flowstate_runtime_metrics"):
            probe._validate_runtime_scope(scheduler, scheduler.tree_cache)
            rid = str(request["request_id"])
            records = MATCHES.get(rid, [])
            require(bool(records), "缺少请求原生匹配记录")
            return {"ok": True, "metrics": records[-1], "records": records,
                "copies": [r for r in COPIES if rid in r["request_ids"]]}
        return original_control(scheduler, request)

    transport._install_match_instrumentation = install_observation
    transport._run_checkpoint_control = control
    return transport._wrapped_run_scheduler_process(*args, **kwargs)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="冻结 B1 跨 Workflow 隔离恢复验证")
    parser.add_argument("--arm", choices=("reference", "retained", "removed"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--retained", type=Path)
    args = parser.parse_args()
    require(not args.out.exists(), "不得覆盖已有证据")
    result = {"arm": args.arm, "status": "RUNNING", "events": [], "evictions": []}
    engine = None

    def event(name):
        result["events"].append({"event": name, "monotonic_ns": time.monotonic_ns(), "unix_ns": time.time_ns()})
        write_json(args.out, result)

    try:
        from sglang.srt.entrypoints.engine import Engine
        from transformers import AutoTokenizer
        from targeted_probe import ControlClient
        from evaluation.controlled_multiworkflow_v1.runtime_gate import wait_for_transport
        from evaluation.rq3_openhands_neutral_collector import (
            SGLangGroupRuntime, materialize_group_requests, replay_group_to_barrier,
        )
        from evaluation.strict_compatibility_b1.run_pilot import _load_frozen_group
        from evaluation.openhands_single_workflow_smoke import (
            DATASET_PATH, TOKENIZER_PATH, normalize_message, _template_input_ids,
        )
        from evaluation.rq3_openhands_population import load_session_messages

        b1 = json.loads(args.b1.read_text())
        frozen = b1["positive_arm"]
        pending = next(p for p in frozen["pendings"] if p["continuation_id"] == PENDING)
        target = next(c for c in frozen["checkpoints"] if c["checkpoint_id"] == TARGET)
        require(target["workflow_id"] != pending["workflow_id"], "目标不是跨 Workflow 边")
        identity = b1["configuration_identity"]
        config = identity["engine_configuration"]
        for version in ("sglang", "transformers"):
            require(importlib.metadata.version(version) == identity[version + "_version"], "软件版本与 B1 不同")
        for row in identity["model_files"] + identity["tokenizer_files"]:
            if "sha256" in row:
                require(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"], "模型或 Tokenizer 配置身份变化")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        require(hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest() == identity["chat_template_sha256"], "模板身份变化")
        manifest = Path(b1["source_population_manifest"])
        group = _load_frozen_group(manifest, b1["group_ordinal"])
        requests, boundary = materialize_group_requests(tokenizer,
            {label: load_session_messages(sid, DATASET_PATH) for label, sid in group.session_by_label.items()},
            group=group, normalize_message=normalize_message, template_input_ids=_template_input_ids)
        for row in boundary:
            for key in ("pending_turn_output_read", "r_plus_2_message_consumed", "r_plus_2_request_materialized", "future_timing_read", "future_checkpoint_read"):
                require(row[key] is False, "输入超过当前因果边界")
        for p in frozen["pendings"]:
            require(requests[(p["workflow_label"], 3)]["input_ids"] == p["token_ids"], "Pending Token 与 B1 不一致")
        result.update({"b1_sha256": hashlib.sha256(args.b1.read_bytes()).hexdigest(), "configuration": identity,
            "boundary_audit": boundary, "pending": pending, "target_checkpoint_id": TARGET,
            "sampling_params": {"max_new_tokens": 8, "temperature": 0, "ignore_eos": True},
            "slot_generation": "UNKNOWN：原生接口未暴露代际", "source_identity": {}})
        import sglang
        source_root = Path(sglang.__file__).parent / "srt"
        for rel in ("mem_cache/unified_cache/unified_tree_core.py", "mem_cache/unified_cache/components/mamba_component.py", "mem_cache/memory_pool.py", "managers/schedule_batch.py", "model_executor/model_runner.py"):
            p = source_root / rel
            result["source_identity"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        event("输入与配置核对完成")

        class ObservedEngine(Engine):
            run_scheduler_process_func = staticmethod(scheduler_process)

        engine = ObservedEngine(**config)
        client = ControlClient(int(os.environ["FLOWSTATE_STEP5D_PORT"]))
        wait_for_transport(client)
        runtime = SGLangGroupRuntime(engine, client)

        def state():
            return client._call({"op": "checkpoint_control", "action": "b1c_state", "nonce": str(time.time_ns())})["state"]

        def records(rid):
            return client._call({"op": "checkpoint_control", "action": "b1c_records", "request_id": rid, "nonce": str(time.time_ns())})

        def evict(row):
            p = runtime.inspect_checkpoint("b1c_before_evict", row["token_ids"])["after"]["path"]
            require(p["node_id"] == row["path"]["node_id"] and p["prefix_sha256"] == token_digest(row["token_ids"]), "驱逐前身份变化")
            require(p["target_mamba_present"] and all(p[k] == 0 for k in ("target_full_lock_ref", "target_mamba_lock_ref", "target_mamba_host_lock_ref", "target_mamba_session_ref")), "保护引用冲突")
            response = client._call({"op": "checkpoint_control", "action": "flowstate_evict_mamba_only",
                "nonce": str(time.time_ns()), "label": "b1c", "checkpoint_id": row["checkpoint_id"],
                "token_ids": row["token_ids"], "extra_key": None, "expected_node_id": p["node_id"], "expected_prefix_sha256": p["prefix_sha256"]})
            require(response["ok"] and all(response["proof"][k] for k in ("same_node", "fa_unchanged", "only_target_mamba_changed", "path_unchanged", "tree_unchanged", "sanity_check")), "驱逐安全证明失败")
            result["evictions"].append({"checkpoint_id": row["checkpoint_id"], "response": response})
            event("安全驱逐 " + row["checkpoint_id"])

        result["engine_initial"] = state()
        if args.arm != "reference":
            trace = replay_group_to_barrier(runtime, group, requests)
            require(trace.fa_query_side_effect_free and not (trace.native_mamba_eviction or trace.fa_cascade or trace.oom or trace.truncation), "历史 replay 正确性门禁失败")
            result["history_requests"] = [dict(row) for row in trace.request_rows]
            result["history_census"] = [dict(row) for row in trace.census_rows]
            result["history_inputs"] = []
            for (label, turn), req in requests.items():
                if turn <= 2:
                    evidence = records(req["rid"])
                    require(all(row["input_token_digest"] == token_digest(req["input_ids"]) for row in evidence["records"]), "Runtime 历史输入身份不符")
                    result["history_inputs"].append({**req, "runtime_match": evidence})
            result["barrier"] = state()
            candidates = []
            for old in frozen["checkpoints"]:
                actual = next(c for c in trace.checkpoints if c.checkpoint_id == old["checkpoint_id"])
                require(actual.token_pos == old["token_position_L"], "候选物化位置改变")
                require(requests[(actual.workflow_label, actual.turn)]["input_ids"][:actual.token_pos] == old["token_ids"], "候选原始输入改变")
                path = runtime.inspect_checkpoint(old["checkpoint_id"], old["token_ids"])["after"]["path"]
                require(path["target_mamba_present"] and path["path_full_all_present"] and path["prefix_sha256"] == old["token_prefix_sha256"], "候选未完整驻留或身份改变")
                candidates.append({"checkpoint_id": old["checkpoint_id"], "token_ids": old["token_ids"], "path": path})
            result["candidates"] = candidates
            plan = isolation_plan(candidates, result["barrier"]["nodes"], pending["token_ids"], TARGET)
            target_row = next(c for c in candidates if c["checkpoint_id"] == TARGET)
            target_node = target_row["path"]["node_id"]
            b1_creation = next(r for r in trace.census_rows if r["event"] == "after_B1")
            require(target_node in b1_creation["added_mamba_node_ids"], "不能证明目标节点的 recurrent state 在 B1 执行时物化")
            result["materialization_evidence"] = b1_creation
            result["isolation_checkpoint_ids"] = [c["checkpoint_id"] for c in plan]
            event("原始 barrier 与物化来源确认")
            for row in plan:
                evict(row)
            result["isolated_retained"] = state()
            eligible = [n for n in result["isolated_retained"]["nodes"] if n["mamba_slots"] and n["token_ids"] == pending["token_ids"][:n["position"]]]
            require(max(eligible, key=lambda n: n["position"])["node_id"] == target_node, "目标未成为最深兼容状态")
            require(sum(n["position"] >= len(target["token_ids"]) for n in eligible) == 1, "存在同等或更深干扰状态")
            require(result["isolated_retained"]["tree"]["full_tree_sha256"] == result["barrier"]["tree"]["full_tree_sha256"], "隔离改变 FA 状态")
            if args.arm == "removed":
                retained = json.loads(args.retained.read_text())
                require(semantic_nodes(result["barrier"]) == semantic_nodes(retained["barrier"]), "两个 lifecycle 初始 barrier 不可比")
                require(semantic_nodes(result["isolated_retained"]) == semantic_nodes(retained["isolated_retained"]), "两个条件隔离后的状态不一致")
                evict(target_row)
            result["before_request"] = state()
            event("隔离与条件设置完成")
        else:
            result["before_request"] = result["engine_initial"]

        rid = "b1c_" + args.arm + "_A_pending_turn003"
        event("提交 A Pending")
        output = engine.generate(input_ids=pending["token_ids"], sampling_params=result["sampling_params"], rid=rid)
        result["output"] = output
        result["request_evidence"] = records(rid)
        result["after_request"] = state()
        require(len(output.get("output_ids", [])) == 8 and output["meta_info"]["completion_tokens"] == 8, "输出被截断或长度不符")
        require(int(output["meta_info"].get("num_retractions", 0) or 0) == 0, "请求出现回撤")
        require(all(r["input_token_digest"] == pending["token_prefix_sha256"] for r in result["request_evidence"]["records"]), "实际 pending 输入身份不符")
        matches = [r for r in result["request_evidence"]["records"] if r["cow_requested"]]
        require(bool(matches), "缺少实际请求的 COW 匹配记录")
        metric = matches[-1]
        result["request_match"] = metric
        before_nodes = {r["prefix_digest"]: r["mamba_slots"] for r in result["before_request"]["nodes"] if r["mamba_slots"]}
        after_nodes = {r["prefix_digest"]: r["mamba_slots"] for r in result["after_request"]["nodes"] if r["mamba_slots"]}
        require(all(after_nodes.get(k) == v for k, v in before_nodes.items()), "请求期间出现未预期的已有 recurrent 替换或驱逐")
        if args.arm == "reference":
            require(metric["physical_fa_hit"] == 0 and metric["executable_prefix"] == 0, "全量重算参考出现缓存命中")
            require(output["output_ids"][0] == b1["reference_outputs"][PENDING]["output_token_id"], "参考首 Token 与 B1 不一致")
        else:
            reference = json.loads(args.reference.read_text())
            require(output["output_ids"] == reference["output"]["output_ids"], "恢复输出与全量重算不一致，停止")
            require(metric["physical_fa_hit"] == pending["attention_frontier"], "FA 命中与冻结边界不一致")
            if args.arm == "retained":
                require(metric["executable_prefix"] == len(target["token_ids"]), "隔离后未从目标前沿恢复")
                require(metric["best_match_node"] == target_node and metric["cow_source_slots"] == target_row["path"]["target_mamba_slots"], "请求匹配与目标物理身份不符")
                require(any(r["source_slots"] == metric["cow_source_slots"] and r["destination_slots"] == metric["working_slots"] for r in result["request_evidence"]["copies"]), "缺少原生 COW 复制调用证据")
            else:
                require(metric["executable_prefix"] < retained["request_match"]["executable_prefix"], "移除 B1 后恢复前沿未下降")
                require(metric["physical_fa_hit"] == retained["request_match"]["physical_fa_hit"], "两个条件 FA 命中不同")
        result["status"] = "PASS"
        event("输出与状态门禁全部通过")
    except BaseException as exc:
        result["status"] = "BLOCKED"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        write_json(args.out, result)
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        result["shutdown_returned"] = engine is not None
        write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
