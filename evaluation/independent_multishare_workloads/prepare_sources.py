# -*- coding: utf-8 -*-
"""按预注册协议锁定 4 组×2 任务×2 Session 来源；只保存第三次请求之前的消息。

规则已在 sampling_protocol.md 固定：B5-C1 字典序中剔除 B5-C2 已占用的 2 个
instance_id 后依序取前 8 个任务，连续两任务成组；每任务取字典序前 2 个
Session。合法性失败只记录，不替换、不跳过。
"""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
B5C1 = ROOT / "evaluation/workload_structure_feasibility_b5c1"
B5C2 = ROOT / "evaluation/multishare_cpu_input_gate_b5c2"
DATA = Path("/home/wjg/data/agentic_coding_trajectories/sessions.parquet")

# 预注册排除项（与 B5-C2 来源锁交叉校验）
EXCLUDE_INSTANCE_IDS = ("0b01001001__spectree-64", "12rambau__sepal_ui-411")
N_GROUPS = 4
TASKS_PER_GROUP = 2
SESSIONS_PER_TASK = 2


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    # 数据与清单身份核验
    old = json.loads((B5C1 / "source_manifest.json").read_text())
    assert sha(DATA) == old[str(DATA)]["sha256"], "原始 parquet 摘要与 B5-C1 不一致"
    manifest = json.loads((B5C1 / "artifact_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha(B5C1 / name) == digest, f"B5-C1 产物摘要不一致：{name}"
    b5c2_lock = json.loads((B5C2 / "source_selection_lock.json").read_text())
    b5c2_instances = {r["instance_id"] for r in b5c2_lock["records"]}
    b5c2_sessions = {r["session_id"] for r in b5c2_lock["records"]}
    assert b5c2_instances == set(EXCLUDE_INSTANCE_IDS), "B5-C2 排除项与硬编码值不一致"

    # 规范顺序与固定选择
    summary = json.loads((B5C1 / "audit_summary.json").read_text())
    groups_map = summary["openhands"]["multi_session_task_groups"]
    ordered = sorted(k for k in groups_map if k not in b5c2_instances)
    chosen_tasks = ordered[: N_GROUPS * TASKS_PER_GROUP]
    assert len(chosen_tasks) == N_GROUPS * TASKS_PER_GROUP, "候选任务不足"
    task_sessions = {t: sorted(groups_map[t])[:SESSIONS_PER_TASK] for t in chosen_tasks}
    # 本轮组间与 B5-C2 均不重叠
    assert len(set(chosen_tasks)) == len(chosen_tasks)
    all_sessions = [s for ss in task_sessions.values() for s in ss]
    assert len(set(all_sessions)) == len(all_sessions), "组间 Session 重叠"
    assert not set(all_sessions) & b5c2_sessions, "与 B5-C2 Session 重叠"

    rows = pq.read_table(DATA).to_pylist()
    by_session = {}
    for i, r in enumerate(rows):
        by_session.setdefault(r["session_id"], []).append((i, r))

    records = []
    failures = []
    for gi in range(N_GROUPS):
        group_name = f"w{gi + 1}"
        group_dir = OUT / group_name
        group_dir.mkdir(exist_ok=True)
        labels = ("A", "B", "C", "D")
        picks = []
        for ti in range(TASKS_PER_GROUP):
            task = chosen_tasks[gi * TASKS_PER_GROUP + ti]
            picks.extend((task, sid) for sid in task_sessions[task])
        for label, (task, sid) in zip(labels, picks):
            entry = {
                "group": group_name,
                "label": label,
                "instance_id": task,
                "session_id": sid,
                "failure": None,
            }
            try:
                matches = by_session.get(sid, [])
                assert len(matches) == 1, f"session_id 命中 {len(matches)} 行"
                index, row = matches[0]
                assert json.loads(row["ground_truth_meta_json"])["instance_id"] == task, (
                    "instance_id 与行元数据不一致"
                )
                raw = json.loads(row["messages_json"])
                assert isinstance(raw, list) and raw, "messages_json 非列表或为空"
                turns = [i for i, m in enumerate(raw) if m.get("role") == "assistant"]
                assert len(turns) >= 3, f"只有 {len(turns)} 条 Assistant，无第三次请求"
                cut = turns[2]
                history = raw[:cut]
                name = f"{group_name}_{label}_source_history.json"
                path = group_dir / name
                assert not path.exists(), "截断历史文件已存在，禁止覆盖"
                path.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                entry.update(
                    source_id=row["source_id"],
                    source_dataset=row["source_dataset"],
                    parquet_row_index_zero_based=index,
                    assistant_message_indices_zero_based=turns[:3],
                    pending_cut_index=cut,
                    history_message_count=len(history),
                    source_history_file=str(path.relative_to(OUT)),
                    source_history_sha256=sha(path),
                    recorded_model=row["recorded_model"],
                    third_assistant_content_read=False,
                    source_wall_clock="UNKNOWN",
                )
            except AssertionError as exc:
                entry["failure"] = str(exc)
                failures.append({"group": group_name, "label": label, "reason": str(exc)})
            records.append(entry)

    lock = {
        "workload_type": "Trace-derived Controlled Composition",
        "rule": (
            "sampling_protocol.md 预注册：B5-C1 multi_session_task_groups 的 "
            "instance_id 字典序中剔除 B5-C2 已用 2 任务后依序取前 8 任务，连续两任务"
            "成组 w1..w4；每任务取 session_id 字典序前 2 Session（A,B 属任务一，"
            "C,D 属任务二）；合法性失败记录不替换"
        ),
        "source_path": str(DATA),
        "source_sha256": sha(DATA),
        "excluded_instance_ids_from_b5c2": sorted(b5c2_instances),
        "excluded_sessions_from_b5c2": sorted(b5c2_sessions),
        "chosen_task_order": chosen_tasks,
        "group_task_mapping": {
            f"w{gi + 1}": chosen_tasks[gi * TASKS_PER_GROUP : (gi + 1) * TASKS_PER_GROUP]
            for gi in range(N_GROUPS)
        },
        "records": records,
        "failures": failures,
        "prior_protocol_sha256": sha(OUT / "sampling_protocol.md"),
        "b5c1_source_manifest_entry": old[str(DATA)],
        "b5c2_lock_sha256": sha(B5C2 / "source_selection_lock.json"),
        "说明": (
            "选择顺序先于分词固定；原始行号只作溯源；截断历史已随锁落盘，"
            "Token 重建只消费这些文件"
        ),
    }
    target = OUT / "source_selection_lock.json"
    assert not target.exists(), "来源锁已存在，禁止覆盖"
    target.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for r in records:
        status = "OK" if not r["failure"] else f"FAIL: {r['failure']}"
        print(f"{r['group']}/{r['label']} {r['instance_id']} cut={r.get('pending_cut_index')} {status}")
    print(f"失败数={len(failures)}；来源锁已写出：{target}")


if __name__ == "__main__":
    main()
