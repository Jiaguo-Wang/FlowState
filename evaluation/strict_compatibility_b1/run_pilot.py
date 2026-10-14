#!/usr/bin/env python3
"""从冻结 OpenHands group 采集严格兼容证据并执行最小恢复验证。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.openhands_common_barrier_snapshot_gate import (
    ENGINE_CONFIGURATION_COMMON_BARRIER,
)
from evaluation.openhands_single_workflow_smoke import (
    DATASET_PATH,
    TOKENIZER_PATH,
    _template_input_ids,
    normalize_message,
)
from evaluation.rq3_openhands_neutral_collector import (
    WorkflowGroup,
    materialize_group_requests,
    replay_group_to_barrier,
    sglang_runtime_factory,
)
from evaluation.rq3_openhands_population import load_session_messages


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_tokens(token_ids: Sequence[int]) -> str:
    import array

    return _sha256_bytes(array.array("q", map(int, token_ids)).tobytes())


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "status": "MISSING"}
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def _load_frozen_group(manifest_path: Path, group_ordinal: int) -> WorkflowGroup:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(
        item
        for item in manifest["groups"]
        if int(item["group_ordinal"]) == group_ordinal
    )
    return WorkflowGroup(
        group_ordinal=int(row["group_ordinal"]),
        population_segment=str(row["population_segment"]),
        allocation_round=int(row["allocation_round"]),
        session_ids=tuple(str(item) for item in row["session_ids"]),
    )


def _request_output(engine: object, request_id: str, tokens: Sequence[int]) -> dict:
    from evaluation.controlled_multiworkflow_v1.runtime_gate import (
        SAMPLING_PARAMETERS,
    )

    result = engine.generate(
        input_ids=[int(item) for item in tokens],
        sampling_params=SAMPLING_PARAMETERS,
        rid=request_id,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"请求未返回结构化结果：{request_id}")
    output_ids = result.get("output_ids") or []
    metadata = result.get("meta_info") or {}
    if len(output_ids) != 1 or int(metadata.get("completion_tokens", 1)) != 1:
        raise RuntimeError(f"请求输出长度不符合单令牌协议：{request_id}")
    if int(metadata.get("num_retractions", 0) or 0) != 0:
        raise RuntimeError(f"请求发生回撤：{request_id}")
    return {"output_token_id": int(output_ids[0]), "metadata": metadata}


def _path_evidence(path: Mapping[str, object]) -> dict[str, object]:
    ref_fields = (
        "target_full_lock_ref",
        "target_mamba_lock_ref",
        "target_mamba_host_lock_ref",
        "target_mamba_session_ref",
    )
    refs = {name: path.get(name) for name in ref_fields}
    refs_known = all(value is not None for value in refs.values())
    refs_clear = refs_known and all(int(value) == 0 for value in refs.values())
    return {
        "node_id": path.get("node_id"),
        "prefix_tokens": path.get("prefix_tokens"),
        "prefix_sha256": path.get("prefix_sha256"),
        "target_full_present": path.get("target_full_present"),
        "path_full_all_present": path.get("path_full_all_present"),
        "target_mamba_present": path.get("target_mamba_present"),
        "target_mamba_slots": path.get("target_mamba_slots"),
        "target_full_value_sha256": path.get("target_full_value_sha256"),
        "references": refs,
        "references_clear": refs_clear if refs_known else None,
        "slot_generation": "UNKNOWN：当前探针未暴露代际账本",
    }


def _run_epoch(
    *,
    group: WorkflowGroup,
    requests: Mapping[tuple[str, int], Mapping[str, object]],
    config: Mapping[str, object],
    port: int,
    mode: str,
    output_dir: Path,
    reference_outputs: Mapping[str, int] | None = None,
    chosen_edge: Mapping[str, object] | None = None,
    boundary_audit: Sequence[Mapping[str, object]] = (),
    configuration_identity_sha256: str = "",
) -> dict[str, object]:
    from evaluation.controlled_multiworkflow_v1.runtime_gate import (
        query_runtime_metrics,
    )

    os.environ["FLOWSTATE_STEP5D_PORT"] = str(port)
    runtime = sglang_runtime_factory(config, port)()
    engine = runtime._engine
    client = runtime._client
    run_id = f"b1_{mode}_{time.time_ns()}"
    result: dict[str, object] = {"run_id": run_id, "mode": mode, "events": []}
    try:
        engine.flush_cache()
        started_ns = time.monotonic_ns()
        trace = replay_group_to_barrier(runtime, group, requests)
        barrier_ns = time.monotonic_ns()
        result["events"].append({"event": "history_replay_complete_and_barrier_entered", "monotonic_ns": barrier_ns})
        result["epoch"] = {
            "barrier_id": f"b1-g{group.group_ordinal:03d}-{mode}",
            "allocation_round": group.allocation_round,
            "history_request_count": len(trace.request_rows),
            "pending_count": len(trace.pendings),
            "started_monotonic_ns": started_ns,
            "barrier_monotonic_ns": barrier_ns,
            "history_request_ids": [str(row["rid"]) for row in trace.request_rows],
            "future_boundary_audit": list(boundary_audit),
            "state_census_rows": [dict(row) for row in trace.census_rows],
            "fa_frontier_queries_side_effect_free": trace.fa_query_side_effect_free,
        }

        checkpoints = []
        for candidate in trace.checkpoints:
            source_request = requests[(candidate.workflow_label, candidate.turn)]
            prefix_tokens = tuple(
                int(item) for item in source_request["input_ids"][: candidate.token_pos]
            )
            if len(prefix_tokens) != candidate.token_pos:
                raise RuntimeError("候选 Token 前缀长度与物化位置不符")
            inspected = runtime.inspect_checkpoint(
                f"{run_id}:{candidate.checkpoint_id}:barrier",
                prefix_tokens,
            )
            path = inspected["after"]["path"]
            result["events"].append({
                "event": "checkpoint_identity_and_residency_inspected",
                "checkpoint_id": candidate.checkpoint_id,
                "monotonic_ns": time.monotonic_ns(),
            })
            evidence = _path_evidence(path)
            if (
                int(evidence["prefix_tokens"]) != candidate.token_pos
                or evidence["prefix_sha256"] != _digest_tokens(prefix_tokens)
                or int(evidence["node_id"]) != candidate.node_id
            ):
                raise RuntimeError(f"候选物理身份在 barrier 不一致：{candidate.checkpoint_id}")
            checkpoints.append(
                {
                    "checkpoint_id": candidate.checkpoint_id,
                    "workflow_id": candidate.workflow_id,
                    "workflow_label": candidate.workflow_label,
                    "request_id": str(source_request["rid"]),
                    "token_position_L": candidate.token_pos,
                    "token_ids": list(prefix_tokens),
                    "token_count": len(prefix_tokens),
                    "token_prefix_sha256": _digest_tokens(prefix_tokens),
                    "state_identity": evidence,
                }
            )

        pendings = []
        pending_by_label = {item.workflow_label: item for item in trace.pendings}
        for label, workflow_id in group.session_by_label.items():
            request = requests[(label, group.allocation_round + 1)]
            tokens = tuple(int(item) for item in request["input_ids"])
            pending = pending_by_label[label]
            result["events"].append({
                "event": "pending_runtime_token_identity_captured",
                "continuation_id": pending.continuation_id,
                "monotonic_ns": time.monotonic_ns(),
            })
            pendings.append(
                {
                    "request_id": str(request["rid"]),
                    "continuation_id": pending.continuation_id,
                    "workflow_id": workflow_id,
                    "workflow_label": label,
                    "barrier_id": result["epoch"]["barrier_id"],
                    "token_ids": list(tokens),
                    "token_count": len(tokens),
                    "token_prefix_sha256": _digest_tokens(tokens),
                    "planning_target_T": min(
                        int(pending.anchor_pos),
                        int(pending.resident_fa_frontier),
                    ),
                    "attention_frontier": pending.resident_fa_frontier,
                    "known_before_dispatch": True,
                    "source": "冻结回放中截至当前轮完成消息构造；未读取 pending 输出或后续轮消息",
                }
            )

        edges = []
        candidate_by_id = {item["checkpoint_id"]: item for item in checkpoints}
        pending_by_id = {item["continuation_id"]: item for item in pendings}
        for candidate in checkpoints:
            for pending in pendings:
                length = int(candidate["token_position_L"])
                target = int(pending["planning_target_T"])
                prefix_equal = (
                    tuple(candidate["token_ids"])
                    == tuple(pending["token_ids"][:length])
                )
                state = candidate["state_identity"]
                physical_ok = bool(
                    state["target_full_present"]
                    and state["path_full_all_present"]
                    and state["target_mamba_present"]
                    and state["target_mamba_slots"]
                )
                config_ok = bool(configuration_identity_sha256)
                epoch_ok = (
                    candidate["workflow_id"] != ""
                    and pending["barrier_id"] == result["epoch"]["barrier_id"]
                    and result["epoch"]["barrier_monotonic_ns"]
                    >= result["epoch"]["started_monotonic_ns"]
                )
                conditions = {
                    "L_le_T": length <= target,
                    "exact_token_prefix": prefix_equal,
                    "same_loaded_model_tokenizer_config": config_ok,
                    "complete_recurrent_and_attention_residency": physical_ok,
                    "same_observation_boundary": epoch_ok,
                    "physical_identity_matches": (
                        int(state["prefix_tokens"]) == length
                        and state["prefix_sha256"] == candidate["token_prefix_sha256"]
                    ),
                    "reference_protection_clear": state["references_clear"],
                }
                if any(value is False for value in conditions.values()):
                    status = "REJECTED"
                elif all(value is True for value in conditions.values()):
                    status = "EPOCH_ADMISSIBLE"
                else:
                    status = "UNKNOWN"
                edges.append(
                    {
                        "checkpoint_id": candidate["checkpoint_id"],
                        "continuation_id": pending["continuation_id"],
                        "request_id": pending["request_id"],
                        "status": status,
                        "conditions": conditions,
                        "actual_resume_verified": False,
                    }
                )

        if mode == "positive" and chosen_edge is None:
            admissible = [edge for edge in edges if edge["status"] == "EPOCH_ADMISSIBLE"]
            if admissible:
                selected = max(
                    admissible,
                    key=lambda edge: int(candidate_by_id[str(edge["checkpoint_id"])]["token_position_L"]),
                )
                selected_checkpoint = candidate_by_id[str(selected["checkpoint_id"])]
                selected_pending = pending_by_id[str(selected["continuation_id"])]
                chosen_edge = {
                    **selected,
                    "checkpoint_digest": selected_checkpoint["token_prefix_sha256"],
                    "checkpoint_L": selected_checkpoint["token_position_L"],
                    "pending_digest": selected_pending["token_prefix_sha256"],
                }
                result["chosen_edge"] = chosen_edge

        if mode == "evicted" and chosen_edge is not None:
            checkpoint = candidate_by_id[str(chosen_edge["checkpoint_id"])]
            pending = pending_by_id[str(chosen_edge["continuation_id"])]
            if (
                checkpoint["token_prefix_sha256"] != chosen_edge.get("checkpoint_digest")
                or int(checkpoint["token_position_L"]) != int(chosen_edge.get("checkpoint_L", -1))
                or pending["token_prefix_sha256"] != chosen_edge.get("pending_digest")
            ):
                raise RuntimeError("驱逐复现实例与正例 arm 的冻结输入身份不一致")
            state = checkpoint["state_identity"]
            if state["references_clear"] is not True:
                raise RuntimeError("候选保护引用非零或未知，拒绝执行驱逐反事实")
            response = client._call(
                {
                    "op": "checkpoint_control",
                    "nonce": f"{run_id}:evict:{checkpoint['checkpoint_id']}",
                    "label": run_id,
                    "action": "flowstate_evict_mamba_only",
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "token_ids": checkpoint["token_ids"],
                    "extra_key": None,
                    "expected_node_id": int(state["node_id"]),
                    "expected_prefix_sha256": state["prefix_sha256"],
                }
            )
            checkpoint["eviction_result"] = response
            verified = runtime.inspect_checkpoint(
                f"{run_id}:after_evict:{checkpoint['checkpoint_id']}",
                tuple(checkpoint["token_ids"]),
            )["after"]["path"]
            checkpoint["after_eviction"] = _path_evidence(verified)
            if verified.get("target_mamba_present"):
                raise RuntimeError("驱逐后目标 Recurrent 状态仍驻留")

        requests_to_resume = []
        if mode == "positive" and chosen_edge is not None:
            requests_to_resume.append(
                ("positive", pending_by_id[str(chosen_edge["continuation_id"])])
            )
        if mode == "evicted" and chosen_edge is not None:
            requests_to_resume.append(
                ("evicted", pending_by_id[str(chosen_edge["continuation_id"])])
            )
        if mode == "positive" and reference_outputs:
            incompatible = next(
                (
                    edge
                    for edge in edges
                    if chosen_edge is not None
                    and edge["checkpoint_id"] == chosen_edge["checkpoint_id"]
                    if edge["status"] == "REJECTED"
                    and edge["conditions"].get("L_le_T") is True
                    and edge["conditions"].get("exact_token_prefix") is False
                ),
                None,
            )
            if incompatible is not None:
                requests_to_resume.append(
                    ("incompatible", pending_by_id[str(incompatible["continuation_id"])])
                )
        resumes = []
        for case, pending in requests_to_resume:
            request_id = f"{run_id}:{case}:{pending['request_id']}"
            started = time.monotonic_ns()
            output = _request_output(engine, request_id, pending["token_ids"])
            completed = time.monotonic_ns()
            result["events"].append({
                "event": "pending_request_completed",
                "case": case,
                "request_id": request_id,
                "monotonic_ns": completed,
            })
            metrics = query_runtime_metrics(client, request_id)
            reference = None if reference_outputs is None else reference_outputs.get(str(pending["continuation_id"]))
            expected = None if reference is None else int(reference["output_token_id"])
            checkpoint = None
            if chosen_edge is not None:
                checkpoint = candidate_by_id.get(str(chosen_edge["checkpoint_id"]))
            checkpoint_length = None if checkpoint is None else int(checkpoint["token_position_L"])
            if case == "positive" and checkpoint_length is not None:
                if int(metrics.get("executable_prefix", -1)) < checkpoint_length:
                    raise RuntimeError(
                        f"正例未证明 Runtime 使用目标 Recurrent 前沿：{metrics}，L={checkpoint_length}"
                    )
                if int(metrics.get("physical_fa_hit", -1)) < checkpoint_length:
                    raise RuntimeError(
                        f"正例物理前缀命中短于 checkpoint：{metrics}，L={checkpoint_length}"
                    )
            if case == "evicted" and checkpoint_length is not None:
                if int(metrics.get("executable_prefix", -1)) >= checkpoint_length:
                    raise RuntimeError(
                        f"驱逐反事实后仍达到被驱逐检查点前沿：{metrics}，L={checkpoint_length}"
                    )
            resumes.append(
                {
                    "case": case,
                    "request_id": request_id,
                    "continuation_id": pending["continuation_id"],
                    "submit_monotonic_ns": started,
                    "completion_monotonic_ns": completed,
                    "output_token_id": output["output_token_id"],
                    "reference_output_token_id": expected,
                    "output_matches_full_recompute": (
                        None if expected is None else output["output_token_id"] == expected
                    ),
                    "target_checkpoint_prefix_is_incompatible": (
                        case == "incompatible"
                    ),
                    "target_checkpoint_attribution": (
                        "UNKNOWN：请求匹配记录仅提供前沿长度，未提供所用 Node ID"
                        if case == "incompatible"
                        else "前沿长度达到候选位置；节点级请求映射仍为 UNKNOWN"
                    ),
                    "runtime_metrics": metrics,
                }
            )
        result.update(
            {
                "snapshot_id": f"b1-openhands-g{group.group_ordinal:03d}-round{group.allocation_round}",
                "checkpoint_count": len(checkpoints),
                "pending_count": len(pendings),
                "edge_count": len(edges),
                "allocation_boundary": {
                    "barrier_id": result["epoch"]["barrier_id"],
                    "capture_started_monotonic_ns": result["epoch"]["started_monotonic_ns"],
                    "capture_completed_monotonic_ns": result["epoch"]["barrier_monotonic_ns"],
                    "concurrent_request_execution_during_capture": False,
                    "atomic_runtime_transaction_claimed": False,
                },
                "edge_status_counts": {
                    key: sum(edge["status"] == key for edge in edges)
                    for key in ("EPOCH_ADMISSIBLE", "REJECTED", "UNKNOWN")
                },
                "checkpoints": checkpoints,
                "pendings": pendings,
                "edges": edges,
                "resumes": resumes,
                "runtime_correctness": "PASS" if all(
                    row["output_matches_full_recompute"] is True
                    for row in resumes
                    if row["case"] in ("positive", "incompatible", "evicted")
                ) else "FAIL",
                "c7_status": "PARTIAL：正例恢复和不兼容例输出检查；驱逐反事实单独记录",
                "candidate_node_to_request_mapping": "由 materialization request 与同一 Token 前缀的 Runtime inspect 关联",
                "slot_generation": "UNKNOWN：Runtime 未暴露槽位代际账本",
            }
        )
        return result
    finally:
        runtime.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenHands 严格兼容性最小 Pilot")
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--group-ordinal", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/model"))
    parser.add_argument("--port", type=int, default=49983)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from evaluation.rq3_openhands_neutral_collector import census_projection

    if importlib.metadata.version("sglang") != "0.5.17":
        raise RuntimeError("Pilot 要求 SGLang 0.5.17")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Pilot 未发现可用 CUDA 设备")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("Pilot 输出目录必须为空")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    group = _load_frozen_group(args.population_manifest, args.group_ordinal)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    messages = {
        label: load_session_messages(session_id, DATASET_PATH)
        for label, session_id in group.session_by_label.items()
    }
    requests, boundary_audit = materialize_group_requests(
        tokenizer,
        messages,
        group=group,
        normalize_message=normalize_message,
        template_input_ids=_template_input_ids,
    )
    if any(
        row["pending_turn_output_read"]
        or row["r_plus_2_message_consumed"]
        or row["r_plus_2_request_materialized"]
        or row["future_timing_read"]
        or row["future_checkpoint_read"]
        for row in boundary_audit
    ):
        raise RuntimeError("Pending 构造越过当前信息边界")

    config = dict(ENGINE_CONFIGURATION_COMMON_BARRIER)
    config["model_path"] = str(args.model_path)
    os.environ["FLOWSTATE_STEP5D_PORT"] = str(args.port)
    baseline_runtime = sglang_runtime_factory(config, args.port)()
    references = {}
    try:
        engine = baseline_runtime._engine
        for label in group.session_by_label:
            request = requests[(label, group.allocation_round + 1)]
            request_id = f"b1_ref_g{group.group_ordinal:03d}_{label}"
            output = _request_output(engine, request_id, request["input_ids"])
            from evaluation.controlled_multiworkflow_v1.runtime_gate import (
                query_runtime_metrics,
            )

            metrics = query_runtime_metrics(baseline_runtime._client, request_id)
            if int(metrics.get("physical_fa_hit", -1)) != 0:
                raise RuntimeError(f"全量重算参考出现缓存命中：{request_id} {metrics}")
            references[str(request_id_for_pending(group, label))] = {
                "output_token_id": output["output_token_id"],
                "runtime_metrics": metrics,
            }
            engine.flush_cache()
    finally:
        baseline_runtime.shutdown()

    tokenizer_files = [
        Path(TOKENIZER_PATH) / name
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json")
    ]
    model_files = [args.model_path / name for name in ("config.json", "generation_config.json")]
    config_identity = {
        "model_path": str(args.model_path),
        "model_files": [_file_identity(path) for path in model_files],
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_files": [_file_identity(path) for path in tokenizer_files],
        "chat_template_sha256": _sha256_bytes(str(tokenizer.chat_template).encode("utf-8")),
        "transformers_version": importlib.metadata.version("transformers"),
        "sglang_version": importlib.metadata.version("sglang"),
        "engine_configuration": config,
        "engine_configuration_sha256": _sha256_bytes(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }

    positive = _run_epoch(
        group=group,
        requests=requests,
        config=config,
        port=args.port,
        mode="positive",
        output_dir=args.output_dir,
        reference_outputs=references,
        boundary_audit=boundary_audit,
        configuration_identity_sha256=config_identity["engine_configuration_sha256"],
    )
    chosen = positive.get("chosen_edge")
    if chosen is None:
        _write_json(
            args.output_dir / "pilot_evidence.json",
            {
                "status": "BLOCKED_NO_EPOCH_ADMISSIBLE_EDGE",
                "configuration_identity": config_identity,
                "future_boundary_audit": boundary_audit,
                "positive_arm": positive,
            },
        )
        return 2
    evicted = _run_epoch(
        group=group,
        requests=requests,
        config=config,
        port=args.port + 1,
        mode="evicted",
        output_dir=args.output_dir,
        reference_outputs=references,
        chosen_edge=chosen,
        boundary_audit=boundary_audit,
        configuration_identity_sha256=config_identity["engine_configuration_sha256"],
    )
    _write_json(
        args.output_dir / "pilot_evidence.json",
        {
            "status": "PASS" if positive["runtime_correctness"] == "PASS" and evicted["runtime_correctness"] == "PASS" else "FAIL",
            "schema": "flowstate.strict_compatibility_b1.v1",
            "source_population_manifest": str(args.population_manifest),
            "group_ordinal": group.group_ordinal,
            "session_ids": list(group.session_ids),
            "configuration_identity": config_identity,
            "future_boundary_audit": boundary_audit,
                "reference_outputs": references,
            "positive_arm": positive,
            "evicted_arm": evicted,
        },
    )
    return 0 if (
        positive["runtime_correctness"] == "PASS"
        and evicted["runtime_correctness"] == "PASS"
    ) else 1


def request_id_for_pending(group: WorkflowGroup, label: str) -> str:
    from evaluation.rq3_openhands_neutral_collector import continuation_id_for

    return continuation_id_for(group.group_ordinal, label, group.allocation_round + 1)


if __name__ == "__main__":
    raise SystemExit(main())
