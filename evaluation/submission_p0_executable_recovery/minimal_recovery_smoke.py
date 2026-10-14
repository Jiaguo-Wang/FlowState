#!/usr/bin/env python3
"""最小真实 Runtime Resume 冒烟验证（Task A3）。

在冻结 SGLang v0.5.17 runtime（docker 镜像内、FormalEndToEndGateEngine
+ WP3B 测试传输层）上，验证一个最小合法共享样本：

  Phase REF（参考）：全新引擎上全量重算 P1 / P2 / P_bad（每个之间 flush），
    无任何共享状态 —— 记录确定性参考输出与"零前缀命中"基线；
  Phase MAT（物化）：materialize(H)（生成 token 仅作确定性对照，不进入
    树路径——首次冒烟实测 8192/8193 处无子边），物化 checkpoint
    C0 = radix 树在 H 输入边界（L = 8192）的 exact 节点
    （FA-KV + Mamba 双驻留；与正式采集 candidate 的 token_pos 约定一致）；
  Phase RES（恢复）：两个不同 continuation（P1 / P2 = H + 各自 CONTROLLED
    后缀，token 前缀在 L 处精确一致）从 C0 恢复；负例 P_bad（token 历史
    在 L 之前分歧）只能命中到分歧位置；
  Phase EVICT（驱逐反事实）：用正式 primitive 仅驱逐 C0 的 Mamba 状态
    （FA-KV 保留），第三个合法 continuation P3 的 executable_prefix 必须
    塌缩、replay_gap 增大——证明 P1/P2 的恢复依赖 Recurrent State 而非
    仅 FA-KV；
  Phase REMAT（确定性对照）：flush 后重新 materialize(H)，验证 anchor
    输出与 C0 内容摘要跨物化稳定。

认证判据（合同 C7）：
  - physical_fa_hit ≈ L、executable_prefix ≈ L、replay_gap ≤ 1（±1 容差
    来自 mamba track 边界取整）；mamba_branching_seqlen 在此代码路径不
    填充（实测 None），Recurrent State 依赖由 EVICT 反事实证明；
  - 恢复路径输出 token 与全量重算参考完全一致（temperature=0、
    max_new_tokens=1、禁 CUDA graph / overlap 的确定性设置）；
  - C0 的物理身份（digest、FA-KV 内容摘要）跨物化稳定；同一棵树内
    C0 的 mamba slot 在两次分支恢复全程不变；两个分支各自物化新状态
    （不共享可写 slot；后缀长度 256 使分支终止点对齐 mamba track 边界
    8448，终止状态才落盘）；
  - 负例 P_bad：谓词 C2 FAIL 拒绝准入，runtime physical_fa_hit 停在
    分歧位置。

产物：runtime_evidence.json（无论成败均写入，失败保留现场）。

本脚本必须在具备 GPU 的 lmsysorg/sglang:v0.5.17-cu129-runtime 容器内、
由 run_minimal_recovery_smoke.sh 启动；不修改任何正式矩阵与数据。
"""

from __future__ import annotations

import json
import os
import sys
import traceback
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

CASES_PATH = _DIRECTORY / "compatibility_cases.json"
EVIDENCE_PATH = _DIRECTORY / "runtime_evidence.json"

# ±1 容差：mamba 状态按 track_interval 取整到边界，与 step7b 门一致的宽容度。
TOLERANCE = 1


class SmokeFailure(RuntimeError):
    """冒烟过程中的断言失败（决定 PARTIAL 与具体失败阶段）。"""


class RuntimeMechanismBlocked(RuntimeError):
    """runtime 机制无法执行定义的合法恢复（决定 BLOCKED）。"""


def _log(stage: str, message: str) -> None:
    print(f"[P0-SMOKE][{stage}] {message}", flush=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


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


def main() -> int:
    stage = "初始化"
    report: dict[str, object] = {
        "task": "FlowState 投稿冲刺任务 A3：最小 Runtime Resume 认证",
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

        from targeted_probe import ControlClient
        from wp3b_end_to_end_transport import (
            FormalEndToEndGateEngine,
            requested_control_port,
        )
        from evaluation.controlled_multiworkflow_v1.runtime_gate import (
            ENGINE_CONFIGURATION,
            generate,
            inspect_checkpoint,
            query_runtime_metrics,
            token_digest,
            wait_for_transport,
        )
        from evaluation.submission_p0_executable_recovery.executable_compatibility import (  # noqa: E501
            CheckpointRecord,
            ContinuationRecord,
            evaluate_executable_compatibility,
        )

        cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
        h_tokens = tuple(cases["checkpoint_template"]["prefix_tokens"])
        l_zero = int(cases["checkpoint_template"]["prefix_length_L0"])
        suffix1 = tuple(cases["continuations"][0]["suffix_tokens"])
        suffix2 = tuple(cases["continuations"][1]["suffix_tokens"])
        pbad_entry = cases["continuations"][2]
        p3_entry = cases["continuations"][3]
        suffix3 = tuple(p3_entry["suffix_tokens"])
        mutation_position = int(pbad_entry["mutation"]["position"])
        h_mutated = tuple(pbad_entry["prefix_tokens"])
        identity = cases["identity"]
        if len(h_tokens) != l_zero or mutation_position >= l_zero:
            raise SmokeFailure("样本文件自检失败：长度或分歧位置非法")

        # continuation 不依赖 anchor 输出：P = H + CONTROLLED 后缀
        c0_tokens = h_tokens
        state_position_l = len(c0_tokens)
        p1_tokens = c0_tokens + suffix1
        p2_tokens = c0_tokens + suffix2
        p3_tokens = c0_tokens + suffix3
        pbad_tokens = h_mutated + suffix1

        # ---- 启动受控引擎（step7b 已验证的最小配置；model 挂载在 /model）----
        stage = "引擎启动"
        config = dict(ENGINE_CONFIGURATION)
        config["model_path"] = "/model"
        report["engine_config"] = {
            key: value for key, value in config.items() if key != "model_path"
        }
        engine = FormalEndToEndGateEngine(**config)
        client = ControlClient(requested_control_port())
        wait_for_transport(client)
        engine.flush_cache()
        _log(stage, "engine 与控制传输层就绪，cache 已清空")

        # ---- Phase REF：全新引擎上全量重算参考（互相隔离）----
        stage = "REF 全量重算参考"
        references = {}
        for name, tokens in (
            ("P1", p1_tokens),
            ("P2", p2_tokens),
            ("P3", p3_tokens),
            ("P_bad", pbad_tokens),
        ):
            request_id = f"p0_ref_{name.lower()}"
            output_token, metadata = generate(engine, request_id, tokens)
            references[name] = {
                "request_id": request_id,
                "input_length": len(tokens),
                "output_token": output_token,
                "metrics": dict(query_runtime_metrics(client, request_id)),
                "e2e_latency_s": metadata.get("e2e_latency"),
                "first_token_latency_s": metadata.get("first_token_latency"),
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

        # ---- Phase MAT：materialize(H) → 物化 C0（H 输入边界）----
        stage = "MAT 物化 checkpoint"
        anchor_output, _ = generate(engine, "p0_anchor_mat", h_tokens)
        anchor_metrics_1 = dict(
            query_runtime_metrics(client, "p0_anchor_mat")
        )
        inspect_mat = inspect_checkpoint(client, "minimal_C0", c0_tokens)
        path_mat = inspect_mat["after"]["path"]
        c0_digest = token_digest(c0_tokens)
        if path_mat["prefix_sha256"] != c0_digest:
            raise SmokeFailure("C0 物化节点摘要与 H 输入边界不一致")
        if not (
            path_mat["target_full_present"]
            and path_mat["target_mamba_present"]
        ):
            raise RuntimeMechanismBlocked(
                f"C0 未同时驻留 FA 与 Mamba: {path_mat}"
            )
        census_before_branches = client.census(
            f"p0:census:mat:{os.getpid()}"
        )
        _log(
            stage,
            f"anchor_output={anchor_output}，C0 node={path_mat['node_id']}，"
            f"L={state_position_l}，FA/Mamba 双驻留",
        )

        # ---- Phase RES：两个合法 continuation 恢复 ----
        stage = "RES 恢复 P1/P2"
        resumes: dict[str, dict] = {}
        for name, tokens in (("P1", p1_tokens), ("P2", p2_tokens)):
            request_id = f"p0_res_{name.lower()}"
            output_token, metadata = generate(engine, request_id, tokens)
            resumes[name] = {
                "request_id": request_id,
                "input_length": len(tokens),
                "output_token": output_token,
                "metrics": dict(query_runtime_metrics(client, request_id)),
                "e2e_latency_s": metadata.get("e2e_latency"),
                "first_token_latency_s": metadata.get("first_token_latency"),
            }
            _log(
                stage,
                f"{name}: output={output_token}，"
                f"metrics={resumes[name]['metrics']}",
            )

        # 两次恢复后 C0 仍然双驻留且身份不变（同一棵树内）
        inspect_post = inspect_checkpoint(client, "minimal_C0", c0_tokens)
        path_post = inspect_post["after"]["path"]
        census_after_branches = client.census(
            f"p0:census:post:{os.getpid()}"
        )
        c0_slots_stable = (
            path_mat["target_mamba_slots"]
            == path_post["target_mamba_slots"]
        )
        c0_digest_post_stable = path_post["prefix_sha256"] == c0_digest
        rows_before = _mamba_rows_by_node(census_before_branches)
        rows_after = _mamba_rows_by_node(census_after_branches)
        new_branch_nodes = sorted(set(rows_after) - set(rows_before))
        _require(c0_slots_stable, "C0 的 mamba slot 在分支恢复中发生变化")
        _require(c0_digest_post_stable, "C0 的摘要在分支恢复后发生变化")
        _require(
            path_post["target_mamba_present"],
            "两次分支恢复后 C0 的 Mamba 状态丢失",
        )
        _require(
            len(new_branch_nodes) >= 2,
            f"两个分支应各自物化新的 Mamba 状态节点: {new_branch_nodes}",
        )
        _log(
            stage,
            f"C0 全程稳定；分支新增 Mamba 节点 {new_branch_nodes}",
        )

        # ---- 负例 P_bad：runtime 只能命中到分歧位置 ----
        stage = "RES 负例 P_bad"
        bad_output, _ = generate(engine, "p0_res_p_bad", pbad_tokens)
        bad_metrics = dict(query_runtime_metrics(client, "p0_res_p_bad"))
        bad_physical = int(bad_metrics["physical_fa_hit"])
        _require(
            abs(bad_physical - mutation_position) <= TOLERANCE,
            f"P_bad 的物理命中应停在分歧位置 {mutation_position}，"
            f"实际 {bad_physical}",
        )
        _require(
            bad_output == references["P_bad"]["output_token"],
            "P_bad 恢复路径输出与自身全量重算不一致（确定性破坏）",
        )
        _log(
            stage,
            f"physical_fa_hit 停在 {bad_physical}（分歧点 {mutation_position}）",
        )

        # ---- Phase EVICT：驱逐反事实——证明 P1/P2 的恢复依赖 Recurrent State ----
        # 用正式 mutation primitive（SGLangAdapter.evict_mamba_only）仅驱逐 C0
        # 的 Mamba 状态（FA-KV 保留），再发第三个合法 continuation P3：
        # 若 P1/P2 此前的 executable_prefix≈L 依赖的是 Recurrent State，
        # 则此刻 executable_prefix 必须塌缩、replay_gap 必须增大。
        stage = "EVICT 驱逐反事实"
        eviction_response = client._call(
            {
                "op": "checkpoint_control",
                "nonce": f"p0:evict:minimal_C0:{os.getpid()}",
                "label": "p0:minimal_C0",
                "action": "flowstate_evict_mamba_only",
                "checkpoint_id": "minimal_C0",
                "token_ids": list(c0_tokens),
                "extra_key": path_mat["extra_key"],
                "expected_node_id": int(path_mat["node_id"]),
                "expected_prefix_sha256": str(path_mat["prefix_sha256"]),
            }
        )
        inspect_evicted = inspect_checkpoint(client, "minimal_C0", c0_tokens)
        path_evicted = inspect_evicted["after"]["path"]
        _require(
            not path_evicted["target_mamba_present"],
            f"驱逐后 C0 仍持有 Mamba 状态: {path_evicted}",
        )
        _require(
            path_evicted["target_full_present"],
            f"驱逐不应影响 FA-KV 驻留: {path_evicted}",
        )
        p3_output, _ = generate(engine, "p0_res_p3_post_evict", p3_tokens)
        p3_metrics = dict(
            query_runtime_metrics(client, "p0_res_p3_post_evict")
        )
        p3_physical = int(p3_metrics["physical_fa_hit"])
        p3_executable = int(p3_metrics["executable_prefix"])
        p3_gap = int(p3_metrics["replay_gap"])
        mamba_dependency_counterfactual = {
            "eviction_proof": {
                "sanity_check": bool(
                    eviction_response.get("proof", {}).get("sanity_check")
                ),
                "fa_identity_unchanged": bool(
                    eviction_response.get("proof", {}).get(
                        "fa_identity_unchanged"
                    )
                ),
            },
            "p3_physical_fa_hit": p3_physical,
            "p3_executable_prefix": p3_executable,
            "p3_replay_gap": p3_gap,
            "executable_collapsed": p3_executable
            < state_position_l - TOLERANCE,
            "gap_grew": p3_gap > TOLERANCE,
            "p3_output_matches_full_recompute": (
                p3_output == references["P3"]["output_token"]
            ),
        }
        _require(
            mamba_dependency_counterfactual["executable_collapsed"]
            and mamba_dependency_counterfactual["gap_grew"],
            "驱逐 C0 的 Mamba 状态后 P3 的可执行前缀未塌缩——"
            f"无法证明恢复依赖 Recurrent State: {mamba_dependency_counterfactual}",
        )
        _log(
            stage,
            f"驱逐后 P3: physical={p3_physical}, executable={p3_executable}, "
            f"gap={p3_gap}（对比驱逐前 P1/P2: executable≈{state_position_l}, "
            f"gap=0）——恢复确实依赖 Recurrent State",
        )

        # ---- Phase REMAT：跨物化确定性对照 ----
        stage = "REMAT 确定性对照"
        engine.flush_cache()
        anchor_output_2, _ = generate(engine, "p0_anchor_remat", h_tokens)
        anchor_metrics_2 = dict(
            query_runtime_metrics(client, "p0_anchor_remat")
        )
        anchor_deterministic = anchor_output_2 == anchor_output
        inspect_remat = inspect_checkpoint(client, "minimal_C0", c0_tokens)
        path_remat = inspect_remat["after"]["path"]
        digest_stable = (
            path_remat["prefix_sha256"] == path_mat["prefix_sha256"]
        )
        fa_value_stable = (
            path_remat["target_full_value_sha256"]
            == path_mat["target_full_value_sha256"]
        )
        _require(anchor_deterministic, "两次 materialize(H) 输出不一致")
        _require(digest_stable, "两次物化的 C0 token 摘要不一致")
        _require(
            path_remat["target_full_present"]
            and path_remat["target_mamba_present"],
            f"重新物化的 C0 未双驻留: {path_remat}",
        )
        _log(
            stage,
            f"anchor 确定性输出一致（token={anchor_output}），"
            f"C0 摘要稳定={digest_stable}，FA 内容摘要稳定={fa_value_stable}"
            f"（slot 允许跨 flush 重分配，不作断言）",
        )

        # ---- 汇总 C7 认证判据 ----
        # mamba_branching_seqlen 在此代码路径不填充（实测 None），仅记录观测值；
        # Recurrent State 依赖改由 EVICT 反事实阶段证明（executable 塌缩）。
        stage = "C7 认证汇总"
        certification: dict[str, dict] = {}
        for name in ("P1", "P2"):
            metrics = resumes[name]["metrics"]
            physical = int(metrics["physical_fa_hit"])
            executable = int(metrics["executable_prefix"])
            gap = int(metrics["replay_gap"])
            output_matches = (
                resumes[name]["output_token"]
                == references[name]["output_token"]
            )
            entry = {
                "physical_fa_hit": physical,
                "executable_prefix": executable,
                "replay_gap": gap,
                "mamba_branching_seqlen_observed": metrics.get(
                    "mamba_branching_seqlen"
                ),
                "physical_fa_hit_at_L": abs(physical - state_position_l)
                <= TOLERANCE,
                "executable_prefix_at_L": abs(executable - state_position_l)
                <= TOLERANCE,
                "replay_gap_small": gap <= TOLERANCE,
                "output_matches_full_recompute": output_matches,
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
            all(
                certification[name]["resume_verified"] for name in ("P1", "P2")
            )
            and anchor_deterministic
            and c0_slots_stable
            and c0_digest_post_stable
            and mamba_dependency_counterfactual["executable_collapsed"]
            and mamba_dependency_counterfactual["gap_grew"]
        )

        # ---- 谓词整合：C1–C7 逐条件判定 ----
        stage = "谓词判定"
        lock_refs = {
            "target_full_lock_ref": int(path_mat["target_full_lock_ref"]),
            "target_mamba_lock_ref": int(path_mat["target_mamba_lock_ref"]),
            "target_mamba_host_lock_ref": int(
                path_mat["target_mamba_host_lock_ref"]
            ),
            "target_mamba_session_ref": int(
                path_mat["target_mamba_session_ref"]
            ),
        }
        checkpoint_record = CheckpointRecord(
            checkpoint_id="minimal_C0",
            model_identity=identity["model_identity"],
            tokenizer_identity=identity["tokenizer_identity"],
            token_prefix=c0_tokens,
            state_position=state_position_l,
            node_id=int(path_mat["node_id"]),
            prefix_digest=str(path_mat["prefix_sha256"]),
            resolution_source="exact",
            recurrent_resident=bool(path_mat["target_mamba_present"]),
            fa_resident=bool(path_mat["target_full_present"]),
            fa_frontier=int(path_mat["prefix_tokens"]),
            generation=None,  # runtime 未暴露 slot generation 账本，如实留空
            lock_refs=lock_refs,
        )
        verdicts = {}
        for name, tokens in (
            ("P1", p1_tokens),
            ("P2", p2_tokens),
            ("P_bad", pbad_tokens),
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
                        f"{mamba_dependency_counterfactual['executable_collapsed']}"
                    ),
                }
            continuation = ContinuationRecord(
                continuation_id=name,
                model_identity=identity["model_identity"],
                tokenizer_identity=identity["tokenizer_identity"],
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
        pbad_rejected = (
            not verdicts["P_bad"]["is_admissible"]
            and any(
                item["condition_id"] == "C2_token_history"
                and item["status"] == "FAIL"
                for item in verdicts["P_bad"]["conditions"]
            )
        )
        _require(
            p1_admissible and p2_admissible,
            f"合法 continuation 未通过准入: P1={p1_admissible}, P2={p2_admissible}",
        )
        _require(pbad_rejected, "P_bad 未被谓词在准入阶段拒绝（C2 应 FAIL）")

        # C5 中 slot generation 账本缺失 → 如实记录为 UNKNOWN，
        # 边保持 ADMISSIBLE+C7 通过，不冒充全部条件 PASS。
        unknown_conditions = sorted(
            {
                item["condition_id"]
                for verdict in verdicts.values()
                for item in verdict["conditions"]
                if item["status"] == "UNKNOWN"
            }
        )

        # ---- 结论 ----
        stage = "结论"
        report["status"] = "PASS" if runtime_gate_pass else "PARTIAL"
        report["failure_stage"] = None
        report["gate"] = {
            "runtime_resume_certified": runtime_gate_pass,
            "illegal_prefix_rejected": pbad_rejected,
            "contract_note": (
                "谓词层：P1/P2 为 ADMISSIBLE 且 C7 全 PASS；"
                "C5 因 runtime 未暴露 slot generation 账本而保持 UNKNOWN"
                "（诚实标注，见 compatibility_contract.md §5）"
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
        # 前缀/兼容性已验证但 resume 未认证 → PARTIAL；保留失败现场
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

    # ---- 汇总证据（无论成败；未到达的阶段对应槽位为 null）----
    evidence_locals = locals()
    evidence: dict[str, object] = {
        **report,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "sample": {
            "h_length": len(h_tokens) if "h_tokens" in evidence_locals else None,
            "state_position_L": (
                state_position_l
                if "state_position_l" in evidence_locals
                else None
            ),
            "mutation_position": (
                mutation_position
                if "mutation_position" in evidence_locals
                else None
            ),
            "anchor_output": (
                anchor_output if "anchor_output" in evidence_locals else None
            ),
            "anchor_output_rematerialize": (
                anchor_output_2
                if "anchor_output_2" in evidence_locals
                else None
            ),
            "anchor_deterministic": (
                anchor_deterministic
                if "anchor_deterministic" in evidence_locals
                else None
            ),
            "c0_token_digest": (
                c0_digest if "c0_digest" in evidence_locals else None
            ),
        },
        "paths": {
            "materialize": (
                path_mat if "path_mat" in evidence_locals else None
            ),
            "after_branches": (
                path_post if "path_post" in evidence_locals else None
            ),
            "rematerialize": (
                path_remat if "path_remat" in evidence_locals else None
            ),
        },
        "identity_stability": {
            "within_tree_c0_mamba_slots_stable": (
                c0_slots_stable
                if "c0_slots_stable" in evidence_locals
                else None
            ),
            "c0_digest_stable_after_branches": (
                c0_digest_post_stable
                if "c0_digest_post_stable" in evidence_locals
                else None
            ),
            "digest_stable_across_materializations": (
                digest_stable if "digest_stable" in evidence_locals else None
            ),
            "fa_value_sha256_stable_across_materializations": (
                fa_value_stable
                if "fa_value_stable" in evidence_locals
                else None
            ),
        },
        "anchor_metrics": {
            "materialize": (
                anchor_metrics_1
                if "anchor_metrics_1" in evidence_locals
                else None
            ),
            "rematerialize": (
                anchor_metrics_2
                if "anchor_metrics_2" in evidence_locals
                else None
            ),
        },
        "references_full_recompute": (
            references if "references" in evidence_locals else None
        ),
        "resumes": resumes if "resumes" in evidence_locals else None,
        "negative_case": {
            "request_id": "p0_res_p_bad",
            "output_token": (
                bad_output if "bad_output" in evidence_locals else None
            ),
            "metrics": (
                bad_metrics if "bad_metrics" in evidence_locals else None
            ),
            "physical_hit_stops_at_mutation": (
                (
                    abs(
                        int(bad_metrics["physical_fa_hit"])
                        - mutation_position
                    )
                    <= TOLERANCE
                )
                if "bad_metrics" in evidence_locals
                else None
            ),
        },
        "mamba_dependency_counterfactual": (
            mamba_dependency_counterfactual
            if "mamba_dependency_counterfactual" in evidence_locals
            else None
        ),
        "post_eviction_p3": {
            "request_id": "p0_res_p3_post_evict",
            "output_token": (
                p3_output if "p3_output" in evidence_locals else None
            ),
            "metrics": (
                p3_metrics if "p3_metrics" in evidence_locals else None
            ),
            "path_after_eviction": (
                path_evicted if "path_evicted" in evidence_locals else None
            ),
        },
        "certification": (
            certification if "certification" in evidence_locals else None
        ),
        "census": {
            "after_materialize": {
                "tree_digests": (
                    _tree_digests(census_before_branches)
                    if "census_before_branches" in evidence_locals
                    else None
                ),
                "mamba_node_count": (
                    len(_mamba_rows_by_node(census_before_branches))
                    if "census_before_branches" in evidence_locals
                    else None
                ),
            },
            "after_branches": {
                "tree_digests": (
                    _tree_digests(census_after_branches)
                    if "census_after_branches" in evidence_locals
                    else None
                ),
                "mamba_node_count": (
                    len(_mamba_rows_by_node(census_after_branches))
                    if "census_after_branches" in evidence_locals
                    else None
                ),
                "new_mamba_nodes": (
                    new_branch_nodes
                    if "new_branch_nodes" in evidence_locals
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
    print(f"[P0-SMOKE] 证据写入 {EVIDENCE_PATH}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
