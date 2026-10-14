#!/usr/bin/env python3
"""LMCache 真实消息的 Executable Recovery 验证（Task A5，案例 lmcache_10）。

在冻结 SGLang v0.5.17 runtime（docker 镜像内、FormalEndToEndGateEngine
+ WP3B 测试传输层）上，使用 LMCache Agentic Traces 冻结的**完整多轮消息
Token 输入**（不经 decode → re-template，直接提交保存的 input_ids）验证：

  Phase INIT（输入核验）：逐位核验冻结输入的全部先验声明——
    P1=16338 / P2=21520 / LCP=10206 / [0,9984) 逐位一致 / LCP 后分歧 /
    文件 sha256 与 manifest 一致 / 引擎模型文件 sha 与 Phase 2 身份合同一致；
    派生 CONTROLLED 负例（位置 4992 翻转一个 token，显式标记）；
    写出 input_identity.json；
  Phase REF（参考）：全新引擎上全量重算 P1 / P2 / P_neg（互相 flush 隔离，
    断言零前缀命中）——隔离的正确性基线；
  Phase MAT1（物化）：materialize(C0=prefix[:9984])——9984=39×256 为
    track 边界，但物化与否以 runtime inspect 实测为准；检查 exact 节点、
    digest、FA-KV 与 Recurrent 双驻留、lock 引用（合同 C4–C6 准入）；
  Phase RES1 / RES2（独立恢复）：每次 flush 后**重新只物化 C0**，再分别
    恢复 P1 / P2——两条 continuation 各自从"只包含 C0 的干净状态"出发，
    互不沾染对方缓存；断言 physical/executable ≈ 9984、gap ≤ 1、
    输出与全量重算一致、C0 物理身份在同树内稳定；
  Phase DUAL（同引擎双分支，补充观测）：同一棵树先后恢复 P1、P2——
    如实记录 P2 可能因 P1 留下的合法 FA-KV 段获得比 9984 更深的物理命中
    （真实来源=P1 分支段，不归功于 C0）；检查 C0 源状态未被修改；
    分支终止点 16338/21520 均非 256 边界，终止状态预期不落盘（Task A
    发现②），只记录实测、不设断言；
  Phase NEG（负例）：P_neg 在 9984 之前分歧，物理命中必须停在翻转位置；
  Phase EVICT（驱逐反事实）：单独环境重新物化 C0 → 正式 primitive 仅驱逐
    C0 的 Recurrent 状态（FA-KV 保留）→ 恢复 P1：executable 必须后退、
    gap 必须增大，记录实际数值（不预设具体 gap）；
  Phase 谓词判定：复用 Task A 合同谓词给出 P1/P2/P_neg 的 C1–C7 判定，
    写出 compatibility_results.json。

产物：runtime_evidence.json / input_identity.json / compatibility_results.json
（无论成败均落盘，失败保留现场）。

本脚本必须在具备 GPU 的 lmsysorg/sglang:v0.5.17-cu129-runtime 容器内、
由 reproduce.sh 启动；不修改任何冻结输入、正式矩阵与数据。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from array import array
from datetime import datetime, timezone
from pathlib import Path

_DIRECTORY = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _DIRECTORY.parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT / "tests" / "runtime"))
sys.path.insert(
    0,
    str(_REPOSITORY_ROOT / "motivation" / "artifacts" / "wp3b_gate_20260820"),
)

os.environ.setdefault("FLOWSTATE_STEP5D_PORT", "49939")

# 冻结输入位置（只读）
PHASE2_DIR = (
    _REPOSITORY_ROOT
    / "agentic_workflow_dataset_research"
    / "phase2_submission_inputs"
)
P1_TOKENS_PATH = PHASE2_DIR / "inputs" / "lmcache_10_p1_tokens.json"
P2_TOKENS_PATH = PHASE2_DIR / "inputs" / "lmcache_10_p2_tokens.json"
MANIFEST_PATH = PHASE2_DIR / "runtime_candidate_manifest.jsonl"
TOKENIZER_IDENTITY_PATH = PHASE2_DIR / "tokenizer_identity.json"

# 预选 checkpoint：共同前缀内部、256-token tracking 边界（39×256）。
# 边界对齐只是候选理由，物化与否以 runtime inspect 实测为准。
C0_LENGTH = 9984
# 先验声明（来自 runtime_candidate_manifest.jsonl，逐项核验）
EXPECTED_P1_LENGTH = 16338
EXPECTED_P2_LENGTH = 21520
EXPECTED_LCP = 10206
# CONTROLLED 负例：在 C0 之前翻转一个 token（位置取 C0 长度的一半）
NEGATIVE_POSITION = C0_LENGTH // 2

EVIDENCE_PATH = _DIRECTORY / "runtime_evidence.json"
INPUT_IDENTITY_PATH = _DIRECTORY / "input_identity.json"
COMPATIBILITY_RESULTS_PATH = _DIRECTORY / "compatibility_results.json"

# ±1 容差：mamba 状态按 track_interval 取整到边界，与上一轮一致的宽容度。
TOLERANCE = 1

# 容器内模型挂载路径（与上一轮 run_minimal_recovery_smoke.sh 相同）
MODEL_MOUNT = "/model"


class SmokeFailure(RuntimeError):
    """冒烟过程中的断言失败（决定 PARTIAL 与具体失败阶段）。"""


class RuntimeMechanismBlocked(RuntimeError):
    """runtime 机制无法执行定义的合法恢复（决定 BLOCKED）。"""


def _log(stage: str, message: str) -> None:
    print(f"[A5-SMOKE][{stage}] {message}", flush=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def token_digest(token_ids) -> str:
    """与正式运行时句柄相同的 token 摘要算法。"""
    return hashlib.sha256(array("q", token_ids).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest_entry() -> dict:
    """从 manifest 中取出 lmcache_10 条目（冻结声明的权威来源）。"""
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("case_id") == "lmcache_10":
            return entry
    raise RuntimeMechanismBlocked("manifest 中找不到 lmcache_10 条目")


def _load_tokens(path: Path) -> tuple[tuple[int, ...], dict]:
    """读取冻结的 input_ids（直接提交，不做任何重模板化）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = {
        key: value
        for key, value in data.items()
        if key not in ("input_ids", "messages")
    }
    tokens = tuple(int(x) for x in data["input_ids"])
    return tokens, meta


def _measure_lcp(p1: tuple[int, ...], p2: tuple[int, ...]) -> int:
    """逐位计算最长公共前缀长度。"""
    lcp = 0
    for a, b in zip(p1, p2):
        if a != b:
            break
        lcp += 1
    return lcp


def _mamba_rows_by_node(census: dict) -> dict[int, list]:
    """把 census 树的 mamba 行转为按节点索引的映射。"""
    tree = census.get("tree", {}) if isinstance(census, dict) else {}
    rows = tree.get("mamba_rows", [])
    if isinstance(rows, dict):
        return {int(node): slots for node, slots in rows.items()}
    return {int(row[0]): row[1] for row in rows}


def _tree_digests(census: dict) -> dict:
    """提取 census 树的结构摘要（若存在）。"""
    tree = census.get("tree", {}) if isinstance(census, dict) else {}
    return {
        key: tree.get(key)
        for key in ("structure_sha256", "full_tree_sha256")
        if key in tree
    }


def _serialize_verdict(verdict) -> dict:
    """把谓词判定结果转为可 JSON 化的结构。"""
    return {
        "checkpoint_id": verdict.checkpoint_id,
        "continuation_id": verdict.continuation_id,
        "is_admissible": verdict.is_admissible,
        "conditions": [
            {
                "condition_id": item.condition_id,
                "status": item.status.value,
                "detail": item.detail,
            }
            for item in verdict.conditions
        ],
        "non_pass_reasons": verdict.reasons(),
    }


def _materialize_c0(engine, client, tag: str) -> dict:
    """物化 C0（anchor 请求）并返回 inspect 路径快照与确定性输出。

    anchor 的生成 token 不进入树路径（Task A 发现①），exact 节点位于
    输入边界 9984——与正式采集的 token_pos 约定一致。
    """
    h_tokens = _materialize_c0.h_tokens  # type: ignore[attr-defined]
    request_id = f"a5_anchor_{tag}"
    anchor_output, _ = engine_generate(engine, request_id, h_tokens)
    metrics = dict(query_runtime_metrics(client, request_id))
    inspect = inspect_checkpoint(client, "lmcache10_C0", h_tokens)
    path = inspect["after"]["path"]
    return {
        "tag": tag,
        "request_id": request_id,
        "anchor_output": anchor_output,
        "anchor_metrics": metrics,
        "path": path,
    }


def engine_generate(engine, request_id: str, token_ids: tuple[int, ...]):
    """runtime_gate.generate 的薄封装（保持本文件自包含可读）。"""
    from evaluation.controlled_multiworkflow_v1.runtime_gate import generate

    return generate(engine, request_id, token_ids)


def inspect_checkpoint(client, checkpoint_id: str, token_ids: tuple[int, ...]):
    """runtime_gate.inspect_checkpoint 的薄封装。"""
    from evaluation.controlled_multiworkflow_v1.runtime_gate import (
        inspect_checkpoint as _inspect,
    )

    return _inspect(client, checkpoint_id, token_ids)


def query_runtime_metrics(client, request_id: str):
    """runtime_gate.query_runtime_metrics 的薄封装。"""
    from evaluation.controlled_multiworkflow_v1.runtime_gate import (
        query_runtime_metrics as _metrics,
    )

    return _metrics(client, request_id)


def _within(value: int, target: int) -> bool:
    """数值是否落在目标的 ±1 容差内。"""
    return abs(int(value) - target) <= TOLERANCE


def main() -> int:
    stage = "初始化"
    report: dict[str, object] = {
        "task": "FlowState 投稿冲刺任务 A5：LMCache lmcache_10 真实消息恢复验证",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED",
        "failure_stage": stage,
        "engine_config": None,
        "environment": {
            "flowstate_step5d_port": os.environ.get("FLOWSTATE_STEP5D_PORT"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sglang_version": None,
        },
    }
    engine = None
    try:
        import sglang

        report["environment"]["sglang_version"] = sglang.__version__
        if not sglang.__version__.startswith("0.5.17"):
            raise RuntimeMechanismBlocked(
                f"需要冻结 SGLang 0.5.17，实际 {sglang.__version__}"
            )

        # ================= Phase INIT：输入核验与身份冻结 =================
        stage = "输入核验"
        manifest = _load_manifest_entry()
        contract_identity = json.loads(
            TOKENIZER_IDENTITY_PATH.read_text(encoding="utf-8")
        )
        p1_tokens, p1_meta = _load_tokens(P1_TOKENS_PATH)
        p2_tokens, p2_meta = _load_tokens(P2_TOKENS_PATH)

        file_sha = {
            "p1": _sha256_file(P1_TOKENS_PATH),
            "p2": _sha256_file(P2_TOKENS_PATH),
        }
        manifest_sha = manifest["input_sha256"]
        _require(
            file_sha["p1"] == manifest_sha[0] and file_sha["p2"] == manifest_sha[1],
            f"冻结文件 sha256 与 manifest 不一致: {file_sha} vs {manifest_sha}",
        )
        _require(
            len(p1_tokens) == EXPECTED_P1_LENGTH
            and len(p2_tokens) == EXPECTED_P2_LENGTH,
            f"输入长度异常: P1={len(p1_tokens)}, P2={len(p2_tokens)}",
        )
        measured_lcp = _measure_lcp(p1_tokens, p2_tokens)
        _require(
            measured_lcp == EXPECTED_LCP,
            f"实测 LCP={measured_lcp} 与声明 {EXPECTED_LCP} 不一致",
        )
        _require(
            p1_tokens[:C0_LENGTH] == p2_tokens[:C0_LENGTH],
            "[0,9984) 未逐位一致",
        )
        _require(
            len(p1_tokens) > measured_lcp and len(p2_tokens) > measured_lcp,
            "LCP 之后应仍有不同 token",
        )
        _require(
            p1_tokens[measured_lcp] != p2_tokens[measured_lcp],
            "LCP 位置之后首 token 应不同",
        )
        _require(C0_LENGTH % 256 == 0, "9984 应为 256 的整数倍")
        _require(
            C0_LENGTH < measured_lcp,
            "C0 应位于共同前缀内部（不在分叉点）",
        )

        h_tokens = p1_tokens[:C0_LENGTH]
        c0_digest = token_digest(h_tokens)

        # ---- 引擎模型文件身份与 Phase 2 合同比对（Step 1 检查 5）----
        engine_files_sha = {
            name: _sha256_file(Path(MODEL_MOUNT) / name)
            for name in (
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "chat_template.jinja",
            )
        }
        contract_files_sha = contract_identity.get("files_sha256", {})
        identity_compatible = all(
            engine_files_sha.get(name) == sha
            for name, sha in contract_files_sha.items()
        )
        _require(
            identity_compatible,
            "引擎模型文件与 Phase 2 输入合同身份不一致: "
            f"{engine_files_sha} vs {contract_files_sha}",
        )
        model_identity = f"qwen3.5-9b@{engine_files_sha['config.json'][:16]}"
        tokenizer_identity = (
            f"qwen3.5-9b-tokenizer@{engine_files_sha['tokenizer_config.json'][:16]}"
        )
        _log(
            stage,
            f"P1={len(p1_tokens)} P2={len(p2_tokens)} LCP={measured_lcp} "
            f"C0={C0_LENGTH}（{C0_LENGTH // 256}×256），digest={c0_digest[:16]}…，"
            f"身份合同一致={identity_compatible}",
        )

        # ---- CONTROLLED 负例：C0 之前翻转一个 token（不冒充原始数据）----
        neg_original = int(p1_tokens[NEGATIVE_POSITION])
        neg_mutated = neg_original + 1
        if not (0 <= neg_mutated < 248077):
            neg_mutated = neg_original - 1
        _require(neg_mutated != neg_original, "负例翻转未生效")
        p_neg_tokens = (
            p1_tokens[:NEGATIVE_POSITION]
            + (neg_mutated,)
            + p1_tokens[NEGATIVE_POSITION + 1 :]
        )
        _require(
            _measure_lcp(p_neg_tokens, h_tokens) == NEGATIVE_POSITION,
            "负例与 C0 的分歧位置异常",
        )

        input_identity = {
            "schema_version": "1",
            "case_id": manifest["case_id"],
            "dataset_name": manifest["dataset_name"],
            "source_task_id": manifest["source_task_id"],
            "source_session_ids": manifest["source_session_ids"],
            "source_type": manifest["source_type"],
            "source_type_evidence": manifest.get("source_type_evidence"),
            "session_metadata": {"p1": p1_meta, "p2": p2_meta},
            "input_files": {
                "p1": {"path": str(P1_TOKENS_PATH), "sha256": file_sha["p1"]},
                "p2": {"path": str(P2_TOKENS_PATH), "sha256": file_sha["p2"]},
            },
            "tokenization_method": manifest["tokenization_method"],
            "submission_mode": (
                "直接提交冻结 input_ids（runtime_gate.generate 以 input_ids "
                "发送，不经 decode/re-template；见 runtime_gate.py:152-173）"
            ),
            "lengths": {
                "p1": len(p1_tokens),
                "p2": len(p2_tokens),
                "measured_lcp": measured_lcp,
                "c0": C0_LENGTH,
            },
            "verified_claims": {
                "file_sha256_match_manifest": True,
                "lengths_match": True,
                "lcp_match": True,
                "prefix9984_identical": True,
                "diverge_after_lcp": True,
                "c0_inside_prefix": True,
                "c0_is_track_boundary": True,
                "engine_identity_matches_contract": identity_compatible,
            },
            "first_divergence": {
                "position": measured_lcp,
                "p1_token": int(p1_tokens[measured_lcp]),
                "p2_token": int(p2_tokens[measured_lcp]),
            },
            "token_digests": {
                "c0_prefix": c0_digest,
                "p1_full": token_digest(p1_tokens),
                "p2_full": token_digest(p2_tokens),
                "negative_full": token_digest(p_neg_tokens),
            },
            "checkpoint": {
                "checkpoint_id": "lmcache10_C0",
                "length": C0_LENGTH,
                "digest": c0_digest,
                "note": (
                    "9984=39×256 为 track 边界对齐候选；物化以 runtime "
                    "inspect 实测为准（Task A 发现①：exact 节点位于请求输入边界）"
                ),
            },
            "negative_case": {
                "kind": "illegal_prefix_mutation",
                "marked": "CONTROLLED",
                "derived_from": "P1（冻结输入派生，非原始 LMCache 数据）",
                "flip_position": NEGATIVE_POSITION,
                "original_token": neg_original,
                "mutated_token": neg_mutated,
            },
            "input_contract_identity": contract_identity,
            "engine_model_files_sha256": engine_files_sha,
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        INPUT_IDENTITY_PATH.write_text(
            json.dumps(input_identity, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log(stage, f"input_identity.json 写入 {INPUT_IDENTITY_PATH}")

        _materialize_c0.h_tokens = h_tokens  # type: ignore[attr-defined]

        # ================= 引擎启动 =================
        stage = "引擎启动"
        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )
        from evaluation.controlled_multiworkflow_v1.runtime_gate import (
            ENGINE_CONFIGURATION,
            wait_for_transport,
        )
        from evaluation.submission_p0_executable_recovery.executable_compatibility import (  # noqa: E501
            CheckpointRecord,
            ContinuationRecord,
            evaluate_executable_compatibility,
        )

        config = dict(ENGINE_CONFIGURATION)
        config["model_path"] = MODEL_MOUNT
        report["engine_config"] = {
            key: value for key, value in config.items() if key != "model_path"
        }
        engine = FormalEndToEndGateEngine(**config)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        engine.flush_cache()
        _log(stage, "engine 与控制传输层就绪，cache 已清空")

        # ================= Phase REF：全量重算参考（互相隔离） =================
        stage = "REF 全量重算参考"
        references: dict[str, dict] = {}
        for name, tokens in (
            ("P1", p1_tokens),
            ("P2", p2_tokens),
            ("P_neg", p_neg_tokens),
        ):
            request_id = f"a5_ref_{name.lower()}"
            output_token, metadata = engine_generate(
                engine, request_id, tokens
            )
            references[name] = {
                "request_id": request_id,
                "input_length": len(tokens),
                "output_token": output_token,
                "metrics": dict(query_runtime_metrics(client, request_id)),
                "e2e_latency_s": metadata.get("e2e_latency"),
            }
            physical = int(references[name]["metrics"]["physical_fa_hit"])
            _require(
                physical <= TOLERANCE,
                f"参考路径 {name} 不应有前缀命中：physical_fa_hit={physical}",
            )
            _log(
                stage,
                f"{name}: output={output_token}，physical_fa_hit={physical}",
            )
            engine.flush_cache()

        # ================= Phase MAT1：物化并核验 C0 =================
        stage = "MAT1 物化 C0"
        mat1 = _materialize_c0(engine, client, "mat1")
        path_mat1 = mat1["path"]
        _require(
            path_mat1["prefix_sha256"] == c0_digest,
            "C0 物化节点摘要与共同前缀 [0,9984) 不一致",
        )
        _require(
            int(path_mat1["prefix_tokens"]) == C0_LENGTH,
            f"C0 节点前缀长度异常: {path_mat1['prefix_tokens']}",
        )
        if not (
            path_mat1["target_full_present"]
            and path_mat1["target_mamba_present"]
        ):
            raise RuntimeMechanismBlocked(
                f"C0 未同时驻留 FA 与 Mamba（9984 为 track 边界但仍未物化）: "
                f"{path_mat1}"
            )
        lock_refs_mat1 = {
            "target_full_lock_ref": int(path_mat1["target_full_lock_ref"]),
            "target_mamba_lock_ref": int(path_mat1["target_mamba_lock_ref"]),
            "target_mamba_host_lock_ref": int(
                path_mat1["target_mamba_host_lock_ref"]
            ),
            "target_mamba_session_ref": int(
                path_mat1["target_mamba_session_ref"]
            ),
        }
        _require(
            all(value == 0 for value in lock_refs_mat1.values()),
            f"C0 物化后应无残留锁引用（合同 C6）: {lock_refs_mat1}",
        )
        census_mat1 = client.census(f"a5:census:mat1:{os.getpid()}")
        _log(
            stage,
            f"anchor_output={mat1['anchor_output']}，C0 node="
            f"{path_mat1['node_id']}，FA/Mamba 双驻留，mamba 位置="
            f"{path_mat1.get('path_mamba_positions')}，lock 全零",
        )

        materializations: dict[str, dict] = {"mat1": mat1}
        census_snapshots: dict[str, dict] = {}

        def _rematerialize(tag: str) -> dict:
            """flush 后重新只物化 C0，并断言与 MAT1 的确定性一致。"""
            engine.flush_cache()
            mat = _materialize_c0(engine, client, tag)
            _require(
                mat["anchor_output"] == mat1["anchor_output"],
                f"{tag} 的 anchor 输出与 MAT1 不一致（确定性破坏）",
            )
            _require(
                mat["path"]["prefix_sha256"] == c0_digest,
                f"{tag} 的 C0 摘要不一致",
            )
            _require(
                mat["path"]["target_full_present"]
                and mat["path"]["target_mamba_present"],
                f"{tag} 重新物化后未双驻留",
            )
            _require(
                mat["path"]["target_full_value_sha256"]
                == path_mat1["target_full_value_sha256"],
                f"{tag} 的 FA-KV 内容摘要跨物化不稳定",
            )
            materializations[tag] = mat
            return mat

        # ================= Phase RES1：独立恢复 P1（只含 C0 的干净状态） =================
        stage = "RES1 独立恢复 P1"
        mat2 = _rematerialize("mat2")
        res1_request = "a5_res1_p1"
        p1_output, p1_meta_res = engine_generate(engine, res1_request, p1_tokens)
        res1_metrics = dict(query_runtime_metrics(client, res1_request))
        inspect_res1 = inspect_checkpoint(client, "lmcache10_C0", h_tokens)
        path_res1 = inspect_res1["after"]["path"]
        census_res1 = client.census(f"a5:census:res1:{os.getpid()}")
        c0_stable_res1 = {
            "same_node": path_res1["node_id"] == mat2["path"]["node_id"],
            "same_slots": (
                path_res1["target_mamba_slots"]
                == mat2["path"]["target_mamba_slots"]
            ),
            "digest_stable": path_res1["prefix_sha256"] == c0_digest,
            "mamba_present": path_res1["target_mamba_present"],
        }
        _require(
            all(c0_stable_res1.values()),
            f"RES1 后 C0 物理身份不稳定: {c0_stable_res1}",
        )
        independent_resumes: dict[str, dict] = {
            "P1": {
                "materialization": "mat2",
                "request_id": res1_request,
                "output_token": p1_output,
                "metrics": res1_metrics,
                "c0_stability": c0_stable_res1,
                "output_matches_full_recompute": (
                    p1_output == references["P1"]["output_token"]
                ),
            }
        }
        census_snapshots["res1"] = census_res1
        _log(
            stage,
            f"P1: output={p1_output}（REF={references['P1']['output_token']}），"
            f"metrics={res1_metrics}",
        )

        # ================= Phase RES2：独立恢复 P2（只含 C0 的干净状态） =================
        stage = "RES2 独立恢复 P2"
        mat3 = _rematerialize("mat3")
        res2_request = "a5_res2_p2"
        p2_output, p2_meta_res = engine_generate(engine, res2_request, p2_tokens)
        res2_metrics = dict(query_runtime_metrics(client, res2_request))
        inspect_res2 = inspect_checkpoint(client, "lmcache10_C0", h_tokens)
        path_res2 = inspect_res2["after"]["path"]
        c0_stable_res2 = {
            "same_node": path_res2["node_id"] == mat3["path"]["node_id"],
            "same_slots": (
                path_res2["target_mamba_slots"]
                == mat3["path"]["target_mamba_slots"]
            ),
            "digest_stable": path_res2["prefix_sha256"] == c0_digest,
            "mamba_present": path_res2["target_mamba_present"],
        }
        _require(
            all(c0_stable_res2.values()),
            f"RES2 后 C0 物理身份不稳定: {c0_stable_res2}",
        )
        independent_resumes["P2"] = {
            "materialization": "mat3",
            "request_id": res2_request,
            "output_token": p2_output,
            "metrics": res2_metrics,
            "c0_stability": c0_stable_res2,
            "output_matches_full_recompute": (
                p2_output == references["P2"]["output_token"]
            ),
        }
        census_snapshots["res2"] = client.census(
            f"a5:census:res2:{os.getpid()}"
        )
        _log(
            stage,
            f"P2: output={p2_output}（REF={references['P2']['output_token']}），"
            f"metrics={res2_metrics}",
        )

        # 独立恢复的核心断言（C7 主证据，来源=各自只含 C0 的环境）
        for name, metrics, output, ref_output in (
            (
                "P1",
                res1_metrics,
                p1_output,
                references["P1"]["output_token"],
            ),
            (
                "P2",
                res2_metrics,
                p2_output,
                references["P2"]["output_token"],
            ),
        ):
            physical = int(metrics["physical_fa_hit"])
            executable = int(metrics["executable_prefix"])
            gap = int(metrics["replay_gap"])
            _require(
                _within(physical, C0_LENGTH),
                f"独立恢复 {name} 的物理命中应≈{C0_LENGTH}，实际 {physical}",
            )
            _require(
                _within(executable, C0_LENGTH),
                f"独立恢复 {name} 的可执行前缀应≈{C0_LENGTH}，实际 {executable}",
            )
            _require(
                gap <= TOLERANCE,
                f"独立恢复 {name} 的 replay_gap 应≤{TOLERANCE}，实际 {gap}",
            )
            _require(
                output == ref_output,
                f"独立恢复 {name} 的输出与全量重算不一致",
            )
        _log(stage, "两条独立恢复均达到 C0 全长且输出与全量重算一致")

        # ================= Phase DUAL：同引擎双分支（补充观测） =================
        stage = "DUAL 同引擎双分支"
        mat4 = _rematerialize("mat4")
        census_dual_before = client.census(
            f"a5:census:dual0:{os.getpid()}"
        )
        dual_p1_request = "a5_dual_p1"
        dual_p1_output, _ = engine_generate(engine, dual_p1_request, p1_tokens)
        dual_p1_metrics = dict(query_runtime_metrics(client, dual_p1_request))
        inspect_dual_p1 = inspect_checkpoint(client, "lmcache10_C0", h_tokens)
        path_dual_p1 = inspect_dual_p1["after"]["path"]
        dual_p2_request = "a5_dual_p2"
        dual_p2_output, _ = engine_generate(engine, dual_p2_request, p2_tokens)
        dual_p2_metrics = dict(query_runtime_metrics(client, dual_p2_request))
        inspect_dual_p2 = inspect_checkpoint(client, "lmcache10_C0", h_tokens)
        path_dual_p2 = inspect_dual_p2["after"]["path"]
        census_dual_after = client.census(
            f"a5:census:dual1:{os.getpid()}"
        )

        dual_c0_stability = {
            "after_p1": {
                "same_node": path_dual_p1["node_id"] == mat4["path"]["node_id"],
                "same_slots": (
                    path_dual_p1["target_mamba_slots"]
                    == mat4["path"]["target_mamba_slots"]
                ),
                "digest_stable": path_dual_p1["prefix_sha256"] == c0_digest,
                "mamba_present": path_dual_p1["target_mamba_present"],
            },
            "after_p2": {
                "same_node": path_dual_p2["node_id"] == mat4["path"]["node_id"],
                "same_slots": (
                    path_dual_p2["target_mamba_slots"]
                    == mat4["path"]["target_mamba_slots"]
                ),
                "digest_stable": path_dual_p2["prefix_sha256"] == c0_digest,
                "mamba_present": path_dual_p2["target_mamba_present"],
            },
        }
        _require(
            all(all(v.values()) for v in dual_c0_stability.values()),
            f"DUAL 双分支后 C0 物理身份不稳定: {dual_c0_stability}",
        )
        dual_p2_physical = int(dual_p2_metrics["physical_fa_hit"])
        deeper_hit = dual_p2_physical > C0_LENGTH + TOLERANCE
        rows_dual_before = _mamba_rows_by_node(census_dual_before)
        rows_dual_after = _mamba_rows_by_node(census_dual_after)
        new_nodes_dual = sorted(set(rows_dual_after) - set(rows_dual_before))
        dual_branch = {
            "p1": {
                "request_id": dual_p1_request,
                "output_token": dual_p1_output,
                "output_matches_full_recompute": (
                    dual_p1_output == references["P1"]["output_token"]
                ),
                "metrics": dual_p1_metrics,
            },
            "p2": {
                "request_id": dual_p2_request,
                "output_token": dual_p2_output,
                "output_matches_full_recompute": (
                    dual_p2_output == references["P2"]["output_token"]
                ),
                "metrics": dual_p2_metrics,
                "physical_deeper_than_c0": deeper_hit,
                "deeper_hit_attribution": (
                    "P2 与 P1 的真实 LCP=10206；若物理命中深于 9984，"
                    "来源是 P1 分支留下的合法 FA-KV 段 [9984,10206)，"
                    "与 C0 无关（C0 只覆盖 [0,9984)）"
                    if deeper_hit
                    else None
                ),
            },
            "c0_stability": dual_c0_stability,
            "census": {
                "before_count": len(rows_dual_before),
                "after_count": len(rows_dual_after),
                "new_mamba_nodes": new_nodes_dual,
            },
            "terminal_boundary_note": (
                "P1 终点 16338=63×256+150、P2 终点 21520=84×256+16 均非 "
                "track 边界；按 Task A 发现②，分支终止状态预期不物化为"
                "独立节点，故此处只记录实测、不设新节点断言"
            ),
        }
        _require(
            dual_p2_physical >= C0_LENGTH - TOLERANCE,
            f"DUAL P2 的物理命中至少应覆盖 C0: {dual_p2_physical}",
        )
        _require(
            _within(int(dual_p2_metrics["executable_prefix"]), C0_LENGTH),
            f"DUAL P2 的可执行前缀应≈{C0_LENGTH}: {dual_p2_metrics}",
        )
        _require(
            dual_p1_output == references["P1"]["output_token"]
            and dual_p2_output == references["P2"]["output_token"],
            "DUAL 双分支输出与全量重算不一致",
        )
        _log(
            stage,
            f"P1: physical={dual_p1_metrics['physical_fa_hit']}, "
            f"executable={dual_p1_metrics['executable_prefix']}；"
            f"P2: physical={dual_p2_physical}"
            f"（{'深于 C0，来源=P1 分支 FA 段' if deeper_hit else '未深于 C0'}），"
            f"executable={dual_p2_metrics['executable_prefix']}；"
            f"C0 全程稳定；新增 mamba 节点 {new_nodes_dual}",
        )

        # ================= Phase NEG：负例（C0 之前分歧） =================
        stage = "NEG 负例"
        neg_request = "a5_neg_pneg"
        neg_output, _ = engine_generate(engine, neg_request, p_neg_tokens)
        neg_metrics = dict(query_runtime_metrics(client, neg_request))
        neg_physical = int(neg_metrics["physical_fa_hit"])
        _require(
            _within(neg_physical, NEGATIVE_POSITION),
            f"负例物理命中应停在翻转位置 {NEGATIVE_POSITION}，实际 {neg_physical}",
        )
        _require(
            neg_output == references["P_neg"]["output_token"],
            "负例输出与自身全量重算不一致（确定性破坏）",
        )
        negative_case = {
            "request_id": neg_request,
            "marked": "CONTROLLED（冻结输入派生，非原始 LMCache 数据）",
            "flip_position": NEGATIVE_POSITION,
            "output_token": neg_output,
            "metrics": neg_metrics,
            "physical_hit_stops_at_flip": _within(
                neg_physical, NEGATIVE_POSITION
            ),
            "output_matches_full_recompute": (
                neg_output == references["P_neg"]["output_token"]
            ),
        }
        _log(
            stage,
            f"physical_fa_hit 停在 {neg_physical}（翻转位置 {NEGATIVE_POSITION}），"
            f"executable={neg_metrics['executable_prefix']}，"
            f"gap={neg_metrics['replay_gap']}",
        )

        # ================= Phase EVICT：Recurrent-only 驱逐反事实 =================
        stage = "EVICT 驱逐反事实"
        mat5 = _rematerialize("mat5")
        path_mat5 = mat5["path"]
        eviction_response = client._call(
            {
                "op": "checkpoint_control",
                "nonce": f"a5:evict:lmcache10_C0:{os.getpid()}",
                "label": "a5:lmcache10_C0",
                "action": "flowstate_evict_mamba_only",
                "checkpoint_id": "lmcache10_C0",
                "token_ids": list(h_tokens),
                "extra_key": path_mat5.get("extra_key"),
                "expected_node_id": int(path_mat5["node_id"]),
                "expected_prefix_sha256": str(path_mat5["prefix_sha256"]),
            }
        )
        inspect_evicted = inspect_checkpoint(client, "lmcache10_C0", h_tokens)
        path_evicted = inspect_evicted["after"]["path"]
        _require(
            not path_evicted["target_mamba_present"],
            f"驱逐后 C0 仍持有 Recurrent 状态: {path_evicted}",
        )
        _require(
            path_evicted["target_full_present"],
            f"驱逐不应影响 FA-KV 驻留: {path_evicted}",
        )
        evict_request = "a5_evict_p1"
        evict_output, _ = engine_generate(engine, evict_request, p1_tokens)
        evict_metrics = dict(query_runtime_metrics(client, evict_request))
        evict_physical = int(evict_metrics["physical_fa_hit"])
        evict_executable = int(evict_metrics["executable_prefix"])
        evict_gap = int(evict_metrics["replay_gap"])
        mamba_dependency_counterfactual = {
            "eviction_proof": eviction_response.get("proof", {}),
            "inspect_after_eviction": {
                "mamba_present": path_evicted["target_mamba_present"],
                "full_present": path_evicted["target_full_present"],
            },
            "paired_comparison": {
                "before_independent_resume_P1": {
                    "physical_fa_hit": int(res1_metrics["physical_fa_hit"]),
                    "executable_prefix": int(res1_metrics["executable_prefix"]),
                    "replay_gap": int(res1_metrics["replay_gap"]),
                },
                "after_eviction_P1": {
                    "physical_fa_hit": evict_physical,
                    "executable_prefix": evict_executable,
                    "replay_gap": evict_gap,
                },
            },
            "p1_physical_fa_hit": evict_physical,
            "p1_executable_prefix": evict_executable,
            "p1_replay_gap": evict_gap,
            "executable_retreated": evict_executable
            < C0_LENGTH - TOLERANCE,
            "gap_grew": evict_gap > int(res1_metrics["replay_gap"]) + TOLERANCE,
            "p1_output_matches_full_recompute": (
                evict_output == references["P1"]["output_token"]
            ),
        }
        _require(
            mamba_dependency_counterfactual["executable_retreated"]
            and mamba_dependency_counterfactual["gap_grew"],
            "驱逐 C0 的 Recurrent 状态后 P1 的可执行前缀未后退——"
            f"无法证明恢复依赖 Recurrent State: "
            f"{mamba_dependency_counterfactual['paired_comparison']}",
        )
        _require(
            evict_output == references["P1"]["output_token"],
            "驱逐后恢复输出与全量重算不一致",
        )
        _log(
            stage,
            f"驱逐后 P1: physical={evict_physical}, "
            f"executable={evict_executable}, gap={evict_gap}"
            f"（对比独立恢复: executable={res1_metrics['executable_prefix']}, "
            f"gap={res1_metrics['replay_gap']}）——恢复确实依赖 Recurrent State",
        )

        # ================= C7 认证汇总（主证据=独立恢复） =================
        stage = "C7 认证汇总"
        certification: dict[str, dict] = {}
        for name, metrics, output, ref_output in (
            ("P1", res1_metrics, p1_output, references["P1"]["output_token"]),
            ("P2", res2_metrics, p2_output, references["P2"]["output_token"]),
        ):
            physical = int(metrics["physical_fa_hit"])
            executable = int(metrics["executable_prefix"])
            gap = int(metrics["replay_gap"])
            entry = {
                "physical_fa_hit": physical,
                "executable_prefix": executable,
                "replay_gap": gap,
                "mamba_branching_seqlen_observed": metrics.get(
                    "mamba_branching_seqlen"
                ),
                "physical_fa_hit_at_L": _within(physical, C0_LENGTH),
                "executable_prefix_at_L": _within(executable, C0_LENGTH),
                "replay_gap_small": gap <= TOLERANCE,
                "output_matches_full_recompute": output == ref_output,
                "evidence_source": "独立恢复（只含 C0 的干净状态）",
                "resume_verified": False,
            }
            entry["resume_verified"] = all(
                entry[key]
                for key in (
                    "physical_fa_hit_at_L",
                    "executable_prefix_at_L",
                    "replay_gap_small",
                    "output_matches_full_recompute",
                )
            )
            certification[name] = entry
            _require(
                entry["resume_verified"],
                f"{name} 的 C7 动态认证未通过: {entry}",
            )

        runtime_gate_pass = (
            all(item["resume_verified"] for item in certification.values())
            and mamba_dependency_counterfactual["executable_retreated"]
            and mamba_dependency_counterfactual["gap_grew"]
            and negative_case["physical_hit_stops_at_flip"]
        )

        # ================= Phase 谓词判定 =================
        stage = "谓词判定"
        checkpoint_record = CheckpointRecord(
            checkpoint_id="lmcache10_C0",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            token_prefix=h_tokens,
            state_position=C0_LENGTH,
            node_id=int(path_mat1["node_id"]),
            prefix_digest=str(path_mat1["prefix_sha256"]),
            resolution_source="exact",
            recurrent_resident=bool(path_mat1["target_mamba_present"]),
            fa_resident=bool(path_mat1["target_full_present"]),
            fa_frontier=int(path_mat1["prefix_tokens"]),
            generation=None,  # runtime 未暴露 slot generation 账本，如实留空
            lock_refs=lock_refs_mat1,
        )
        verdicts = {}
        for name, tokens in (
            ("P1", p1_tokens),
            ("P2", p2_tokens),
            ("P_neg", p_neg_tokens),
        ):
            runtime_evidence = None
            if name in certification:
                entry = certification[name]
                runtime_evidence = {
                    "resume_verified": entry["resume_verified"],
                    "detail": (
                        f"physical={entry['physical_fa_hit']}, "
                        f"executable={entry['executable_prefix']}, "
                        f"gap={entry['replay_gap']}, "
                        f"输出与全量重算一致="
                        f"{entry['output_matches_full_recompute']}, "
                        f"驱逐反事实证明依赖 Recurrent State="
                        f"{mamba_dependency_counterfactual['executable_retreated']}"
                    ),
                }
            continuation = ContinuationRecord(
                continuation_id=name,
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                token_ids=tuple(tokens),
            )
            verdicts[name] = _serialize_verdict(
                evaluate_executable_compatibility(
                    checkpoint_record,
                    continuation,
                    runtime_evidence=runtime_evidence,
                )
            )
        p1_admissible = verdicts["P1"]["is_admissible"]
        p2_admissible = verdicts["P2"]["is_admissible"]
        pneg_rejected = (
            not verdicts["P_neg"]["is_admissible"]
            and any(
                item["condition_id"] == "C2_token_history"
                and item["status"] == "FAIL"
                for item in verdicts["P_neg"]["conditions"]
            )
        )
        _require(
            p1_admissible and p2_admissible,
            f"合法 continuation 未通过准入: P1={p1_admissible}, "
            f"P2={p2_admissible}",
        )
        _require(pneg_rejected, "负例未被谓词在准入阶段拒绝（C2 应 FAIL）")
        unknown_conditions = sorted(
            {
                item["condition_id"]
                for verdict in verdicts.values()
                for item in verdict["conditions"]
                if item["status"] == "UNKNOWN"
            }
        )

        # ================= 结论 =================
        stage = "结论"
        report["status"] = "PASS" if runtime_gate_pass else "PARTIAL"
        report["failure_stage"] = None
        report["gate"] = {
            "runtime_resume_certified": all(
                item["resume_verified"] for item in certification.values()
            ),
            "illegal_prefix_rejected": pneg_rejected,
            "mamba_dependency_proven": (
                mamba_dependency_counterfactual["executable_retreated"]
                and mamba_dependency_counterfactual["gap_grew"]
            ),
            "contract_note": (
                "谓词层：P1/P2 为 ADMISSIBLE 且 C7 全 PASS；"
                "C5 因 runtime 未暴露 slot generation 账本而保持 UNKNOWN"
                "（诚实标注，沿用 Task A 合同）"
                if unknown_conditions
                else "谓词层全部条件 PASS"
            ),
            "unknown_conditions": unknown_conditions,
        }
        _log(stage, f"RESULT={report['status']} gate={report['gate']}")
        return_code = 0
    except RuntimeMechanismBlocked as error:
        report["status"] = "BLOCKED"
        report["failure_stage"] = stage
        report["error"] = repr(error)
        traceback.print_exc()
        return_code = 2
    except SmokeFailure as error:
        # 输入/前缀/兼容性已验证但 resume 未认证 → PARTIAL；保留失败现场
        report["status"] = "PARTIAL"
        report["failure_stage"] = stage
        report["error"] = repr(error)
        traceback.print_exc()
        return_code = 1
    except Exception as error:  # noqa: BLE001
        report["status"] = "BLOCKED"
        report["failure_stage"] = stage
        report["error"] = repr(error)
        traceback.print_exc()
        return_code = 2
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001
                pass

    # ================= 证据落盘（无论成败；未到达的阶段为 null） =================
    evidence_locals = locals()
    evidence: dict[str, object] = {
        **report,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "input_verification": (
            input_identity if "input_identity" in evidence_locals else None
        ),
        "materializations": (
            {
                tag: {
                    "anchor_output": mat["anchor_output"],
                    "anchor_metrics": mat["anchor_metrics"],
                    "path": mat["path"],
                }
                for tag, mat in materializations.items()
            }
            if "materializations" in evidence_locals
            else None
        ),
        "references_full_recompute": (
            references if "references" in evidence_locals else None
        ),
        "independent_resumes": (
            independent_resumes
            if "independent_resumes" in evidence_locals
            else None
        ),
        "dual_branch": (
            dual_branch if "dual_branch" in evidence_locals else None
        ),
        "negative_case": (
            negative_case if "negative_case" in evidence_locals else None
        ),
        "mamba_dependency_counterfactual": (
            mamba_dependency_counterfactual
            if "mamba_dependency_counterfactual" in evidence_locals
            else None
        ),
        "certification": (
            certification if "certification" in evidence_locals else None
        ),
        "census": {
            "mat1": {
                "tree_digests": (
                    _tree_digests(census_mat1)
                    if "census_mat1" in evidence_locals
                    else None
                ),
                "mamba_node_count": (
                    len(_mamba_rows_by_node(census_mat1))
                    if "census_mat1" in evidence_locals
                    else None
                ),
            },
            "res1": {
                "tree_digests": (
                    _tree_digests(census_snapshots["res1"])
                    if "census_snapshots" in evidence_locals
                    and "res1" in census_snapshots
                    else None
                ),
                "mamba_node_count": (
                    len(_mamba_rows_by_node(census_snapshots["res1"]))
                    if "census_snapshots" in evidence_locals
                    and "res1" in census_snapshots
                    else None
                ),
            },
            "res2": {
                "tree_digests": (
                    _tree_digests(census_snapshots["res2"])
                    if "census_snapshots" in evidence_locals
                    and "res2" in census_snapshots
                    else None
                ),
                "mamba_node_count": (
                    len(_mamba_rows_by_node(census_snapshots["res2"]))
                    if "census_snapshots" in evidence_locals
                    and "res2" in census_snapshots
                    else None
                ),
            },
        },
        "predicate_verdicts": (
            verdicts if "verdicts" in evidence_locals else None
        ),
    }
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if "verdicts" in evidence_locals:
        compatibility_results = {
            "case_id": "lmcache_10",
            "checkpoint_id": "lmcache10_C0",
            "state_position_L": C0_LENGTH,
            "contract_reference": (
                "evaluation/submission_p0_executable_recovery/"
                "compatibility_contract.md（Task A 合同，未修改）"
            ),
            "verdicts": verdicts,
            "unknown_conditions": report.get("gate", {}).get(
                "unknown_conditions"
            )
            if isinstance(report.get("gate"), dict)
            else None,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        COMPATIBILITY_RESULTS_PATH.write_text(
            json.dumps(
                compatibility_results, ensure_ascii=False, indent=2, default=str
            ),
            encoding="utf-8",
        )
    print(f"[A5-SMOKE] 证据写入 {EVIDENCE_PATH}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
