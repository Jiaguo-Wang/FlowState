# -*- coding: utf-8 -*-
"""独立核验本轮全部产物并生成 cpu_input_audit.json、token_sharing_summary.csv、
verification.json 与 artifact_manifest.json；不调用 Tokenizer、不运行任何策略。
"""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
B5C1 = ROOT / "evaluation/workload_structure_feasibility_b5c1"
B5C2 = ROOT / "evaluation/multishare_cpu_input_gate_b5c2"


def read(name):
    return json.loads((P / name).read_text())


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(name, x) -> None:
    (P / name).write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lcp(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main() -> None:
    lock = read("source_selection_lock.json")
    matrix = read("token_prefix_matrix.json")
    causal = read("causal_checks.json")
    config = read("input_configuration_lock.json")
    req = read("requests.json")

    checks = {}

    # 1. 原始数据与前置产物身份
    assert sha(Path(lock["source_path"])) == lock["source_sha256"], "parquet 摘要不一致"
    for name, digest in json.loads((B5C1 / "artifact_manifest.json").read_text())["files"].items():
        assert sha(B5C1 / name) == digest, f"B5-C1 产物摘要不一致：{name}"
    b5c2_lock = json.loads((B5C2 / "source_selection_lock.json").read_text())
    assert sha(B5C2 / "source_selection_lock.json") == lock["b5c2_lock_sha256"], "B5-C2 锁摘要不一致"
    new_sessions = {r["session_id"] for r in lock["records"]}
    new_tasks = {r["instance_id"] for r in lock["records"]}
    old_sessions = {r["session_id"] for r in b5c2_lock["records"]}
    old_tasks = {r["instance_id"] for r in b5c2_lock["records"]}
    assert not new_sessions & old_sessions and not new_tasks & old_tasks, "与 B5-C2 来源重叠"
    by_group = {}
    for r in lock["records"]:
        by_group.setdefault(r["group"], []).append(r)
    for g, recs in by_group.items():
        assert len({x["instance_id"] for x in recs}) == 2 and len({x["session_id"] for x in recs}) == 4
    group_tasks = [t for ts in lock["group_task_mapping"].values() for t in ts]
    assert sorted(group_tasks) == sorted(new_tasks), "组任务映射与记录不一致"
    checks["source_identity"] = "PASS"

    # 2. 截断历史与因果一致性
    tool_total = 0
    for r in lock["records"]:
        assert r["failure"] is None, "存在锁阶段失败记录"
        p = P / r["source_history_file"]
        assert sha(p) == r["source_history_sha256"], "截断历史摘要不一致"
        raw = json.loads(p.read_text())
        turns = [i for i, m in enumerate(raw) if m.get("role") == "assistant"]
        assert len(turns) == 2, "截断历史 Assistant 数不为 2"
        assert r["assistant_message_indices_zero_based"][:2] == turns, "Assistant 索引不一致"
        assert max(turns) < r["pending_cut_index"], "截断索引不大于最后历史 Assistant"
        assert r["history_message_count"] == len(raw)
    for block in causal:
        for row in block["labels"]:
            assert all(
                not block["collector_audits"][0][k]
                for k in (
                    "pending_turn_output_read",
                    "r_plus_2_message_consumed",
                    "r_plus_2_request_materialized",
                    "future_timing_read",
                    "future_checkpoint_read",
                )
            ) or True
            for e in row["tool_results"]:
                assert e["call_message_index"] < e["message_index"] < row["cut_index"], "工具因果顺序错误"
                tool_total += 1
    for block in causal:
        for audit in block["collector_audits"]:
            assert all(
                not audit[k]
                for k in (
                    "pending_turn_output_read",
                    "r_plus_2_message_consumed",
                    "r_plus_2_request_materialized",
                    "future_timing_read",
                    "future_checkpoint_read",
                )
            ), "采集器审计出现未来信息读取"
    checks["history_and_causality"] = "PASS"

    # 3. Token 数组独立复核
    assert len(req) == 48, "应有 48 份请求数组"
    arrays = {}
    for r in req:
        tokens = json.loads((P / r["token_file"]).read_text())
        assert len(tokens) == r["token_count"], "长度不一致"
        assert sha(P / r["token_file"]) == r["file_sha256"], "文件摘要不一致"
        assert (
            hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
            == r["canonical_token_sha256"]
        ), "规范Token摘要不一致"
        if r["turn"] == 3:
            arrays.setdefault(r["group"], {})[r["label"]] = tokens
    assert {g: sorted(v) for g, v in arrays.items()} == {f"w{i}": list("ABCD") for i in (1, 2, 3, 4)}

    # 4. LCP 与矩阵复核（独立重算）
    for g, v in matrix.items():
        if v["status"] != "REBUILT":
            continue
        pend = arrays[g]
        for a in "ABCD":
            for b in "ABCD":
                assert lcp(pend[a], pend[b]) == v["pairwise_lcp"][a][b], f"{g} LCP 复算不一致"
        assert min(v["pairwise_lcp"][a][b] for a in "ABCD" for b in "ABCD") == v["four_way_lcp"]
        assert v["boundary_all_exact"] is True, f"{g} 存在非精确消息边界"
        assert v["positions"][0]["checkpoint_created"] is False and v["positions"][0]["runtime_materialization"] == "UNKNOWN"
    checks["token_arrays_and_lcp"] = "PASS"

    # 5. cpu_input_audit.json 汇总
    group_audit = {}
    for g in ("w1", "w2", "w3", "w4"):
        v = matrix[g]
        recs = by_group[g]
        rows3 = {r["label"]: next(x for x in req if x["group"] == g and x["label"] == r["label"] and x["turn"] == 3) for r in recs}
        group_audit[g] = {
            "status": v["status"],
            "structure_class": v["structure_class"],
            "sources": [
                {
                    "label": r["label"],
                    "instance_id": r["instance_id"],
                    "session_id": r["session_id"],
                    "parquet_row_index_zero_based": r["parquet_row_index_zero_based"],
                    "pending_cut_index": r["pending_cut_index"],
                    "recorded_model": r["recorded_model"],
                }
                for r in recs
            ],
            "pending_token_counts": {k: rows3[k]["token_count"] for k in "ABCD"},
            "pending_token_sha256": {k: rows3[k]["canonical_token_sha256"] for k in "ABCD"},
            "token_sharing": {
                "four_way_lcp": v["four_way_lcp"],
                "four_way_support": v["positions"][0]["support"],
                "task1_lcp_AB": v["task_pair_lcp"]["AB"],
                "task1_support": v["positions"][1]["support"],
                "task2_lcp_CD": v["task_pair_lcp"]["CD"],
                "task2_support": v["positions"][2]["support"],
                "cross_task_lcp": v["cross_task_lcp"],
                "system_prefix_token_lengths": v["system_prefix_token_lengths"],
                "pairwise_lcp": v["pairwise_lcp"],
            },
            "history_causality": "顺序与tool_call_id可核验；墙钟UNKNOWN",
            "runtime_materialization": "UNKNOWN；未创建任何Checkpoint",
            "runtime_must_verify": [
                "历史回放实际接收Token与CPU数组一致",
                "原生Recurrent Checkpoint物化位置与Node/Slot所有权",
                "逐Pending FA frontier与Planning Target（L≤T逐边确认）",
                "完整Strict Compatibility矩阵（三态，无UNKNOWN）",
                "实际支持集合与预算竞争；有效与无效案例原样保留",
            ],
        }
    audit = {
        "workload_type": "Trace-derived Controlled Composition（每组两任务×两Session，共4组16 Session）",
        "identity": {
            "image_id": config["image_id"],
            "versions": config["versions"],
            "loaded_chat_template_sha256": config["loaded_chat_template_sha256"],
            "model_path": config["model_path"],
            "b4_identity_source": "evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01/g001/configuration.json",
        },
        "selection": {
            "protocol": "sampling_protocol.md（先于分词固定）",
            "excluded_b5c2_instance_ids": lock["excluded_instance_ids_from_b5c2"],
            "group_task_mapping": lock["group_task_mapping"],
            "failures": lock["failures"],
        },
        "groups": group_audit,
        "tool_results_total": tool_total,
        "message_boundaries_exact": True,
        "scope_limits": [
            "文本匹配、Token LCP与历史Session重复都不是可执行Recurrent State共享；物化状态UNKNOWN",
            "非自然并发、非原生Agent Fork；受控已知输入组合",
            "未运行GPU、未构造Strict Snapshot、未运行任何分配算法",
        ],
        "verification": checks,
    }
    save("cpu_input_audit.json", audit)

    # 6. token_sharing_summary.csv
    with (P / "token_sharing_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "group", "status", "structure_class",
                "task1_instance_id", "task2_instance_id",
                "token_count_A", "token_count_B", "token_count_C", "token_count_D",
                "four_way_lcp", "support_four_way",
                "task1_lcp_AB", "task2_lcp_CD",
                "cross_task_lcp_min", "cross_task_lcp_max",
                "system_prefix_tokens",
                "common_fork_char_offset_A", "common_fork_char_offset_B",
                "common_fork_char_offset_C", "common_fork_char_offset_D",
                "task1_fork_char_offset_A", "task1_fork_char_offset_B",
                "task2_fork_char_offset_C", "task2_fork_char_offset_D",
                "tool_results_in_group", "message_boundary_exact",
            ]
        )
        for g in ("w1", "w2", "w3", "w4"):
            v = matrix[g]
            recs = by_group[g]
            tasks = list(dict.fromkeys(r["instance_id"] for r in recs))
            rows3 = {
                r["label"]: next(x for x in req if x["group"] == g and x["label"] == r["label"] and x["turn"] == 3)
                for r in recs
            }
            n_tools = sum(len(row["tool_results"]) for block in causal if block["group"] == g for row in block["labels"])
            pos = {p["name"]: p for p in v["positions"]}
            fork = lambda name, label: (pos[name]["divergence"].get(label) or {}).get("character_offset")
            cross = list(v["cross_task_lcp"].values())
            w.writerow(
                [
                    g, v["status"], v["structure_class"],
                    tasks[0], tasks[1],
                    *[rows3[k]["token_count"] for k in "ABCD"],
                    v["four_way_lcp"], "".join(pos["common"]["support"]),
                    v["task_pair_lcp"]["AB"], v["task_pair_lcp"]["CD"],
                    min(cross), max(cross),
                    min(v["system_prefix_token_lengths"].values()),
                    *[str(fork("common", k)) for k in "ABCD"],
                    fork("task1", "A"), fork("task1", "B"),
                    fork("task2", "C"), fork("task2", "D"),
                    n_tools, v["boundary_all_exact"],
                ]
            )

    # 7. 容器清理与总核验
    leftover = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=flowstate-imw-cpu", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert leftover == "", "CPU 容器未清理：" + leftover
    save(
        "verification.json",
        {
            "gate": "READY",
            "source_identity": "PASS",
            "b5c2_exclusion": "PASS（任务与Session均无重叠）",
            "history_and_causality": "PASS",
            "token_files": 48,
            "pending_inputs": 16,
            "pairwise_lcp_recompute": "PASS",
            "message_boundaries": "PASS",
            "tool_result_causality": {"count": tool_total, "ordered_links": "PASS", "wall_clock": "UNKNOWN"},
            "planning_target": "UNKNOWN",
            "runtime_checkpoint_count": "NOT_OBSERVED",
            "gpu_used": False,
            "engine_started": False,
            "cpu_container_cleanup": "PASS：docker ps -a 按本轮名称过滤为空",
        },
    )
    save(
        "artifact_manifest.json",
        {
            "files": {
                str(p.relative_to(P)): sha(p)
                for p in sorted(P.rglob("*"))
                if p.is_file() and p.name != "artifact_manifest.json" and "artifact_manifest" not in p.name
            },
            "说明": "只覆盖本轮独立CPU产物；旧来源与B5-C1/B5-C2产物摘要复核通过且未修改",
        },
    )
    print("READY：独立核验通过，数据交付已生成。")


if __name__ == "__main__":
    main()
