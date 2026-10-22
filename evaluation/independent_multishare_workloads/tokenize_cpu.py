# -*- coding: utf-8 -*-
"""在冻结镜像中只做 CPU 分词与结构核验；不导入 SGLang、不创建引擎、不访问 GPU。

与 B5-C2 的 tokenize_cpu.py 同一冻结路径：同一 B4 身份锁、同一 AST 加载的
normalize_message/_parse_tool_calls/_template_input_ids/materialize_group_requests、
同一模板调用参数。差异仅在于：一次处理 w1..w4 四组来源，逐组记录前缀矩阵并
只做结构分类（不把任何结构当作门禁断言）。
"""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

from transformers import AutoTokenizer

ROOT = Path("/home/wjg/code/FlowState")
OUT = ROOT / "evaluation/independent_multishare_workloads"
MODEL = Path("/home/wjg/models/qwen3.5-9b")
B4 = ROOT / "evaluation/strict_compatibility_b4b1_output/capture_g001_g006_run01"
GROUPS = ("w1", "w2", "w3", "w4")
LABELS = "ABCD"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def token_sha(tokens) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def save(path: Path, x) -> None:
    x = json.loads(json.dumps(x, ensure_ascii=False))
    if path.exists():
        assert json.loads(path.read_text()) == x, "已有本轮文件内容冲突：" + str(path)
        return
    with path.open("x") as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
        f.write("\n")


def lcp(*xs) -> int:
    for i, vs in enumerate(zip(*xs)):
        if len(set(vs)) > 1:
            return i
    return min(map(len, xs))


def load_functions(path: Path, names, env) -> dict:
    """AST 只读加载冻结函数，不执行模块其余内容。"""
    tree = ast.parse(path.read_text())
    body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ] + [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert len(body) == len(names) + 1
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), env)
    return {n: env[n] for n in names}


def classify(common: int, matrix: dict, ab: int, cd: int) -> str:
    """只描述结构，不作为门禁：嵌套/扁平/部分嵌套/退化。"""
    cross = [matrix[a][b] for a in "AB" for b in "CD"]
    if common == 0:
        return "degenerate_no_common_prefix"
    if ab > common and cd > common and all(v == common for v in cross):
        return "nested_two_task"
    if ab == common and cd == common:
        return "flat_all_equal"
    return "partial_nested"


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not list(Path("/dev").glob("nvidia*")), "不应挂载GPU设备"

    source = json.loads((OUT / "source_selection_lock.json").read_text())
    identity = json.loads((B4 / "g001/configuration.json").read_text())
    versions = {n: importlib.metadata.version(n) for n in ("transformers", "tokenizers", "sglang", "jinja2")}
    assert versions["transformers"] == identity["transformers_version"]
    assert versions["sglang"] == identity["sglang_version"] == "0.5.17"
    for item in identity["tokenizer_files"] + identity["model_files"]:
        if "sha256" in item:
            assert sha(Path(item["path"])) == item["sha256"], item["path"]
        else:
            assert not Path(item["path"]).exists()
    locked = json.loads((B4 / "execution_lock.json").read_text())["sha256"]
    paths = [
        ROOT / "evaluation/openhands_single_workflow_smoke.py",
        ROOT / "evaluation/rq3_openhands_neutral_collector.py",
    ]
    for p in paths:
        assert sha(p) == locked[str(p)], str(p)

    env = {"json": json, "Mapping": Mapping, "Sequence": Sequence, "Callable": Callable}
    load_functions(paths[0], ["_parse_tool_calls", "normalize_message", "_template_input_ids"], env)
    load_functions(paths[1], ["request_id_for", "materialize_group_requests", "CollectionAbort"], env)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    assert hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest() == identity["chat_template_sha256"]

    config = {
        "workload_type": source["workload_type"],
        "model_path": str(MODEL),
        "versions": versions,
        "image_id": "sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926",
        "b4_identity": identity,
        "files": {
            str(p): sha(p)
            for p in paths + list(MODEL.glob("*token*json")) + [MODEL / "chat_template.jinja", MODEL / "config.json"]
        },
        "loaded_chat_template_sha256": identity["chat_template_sha256"],
        "tokenizer_class": type(tokenizer).__name__,
        "apply_chat_template": {
            "tokenize": True,
            "add_generation_prompt": True,
            "tools": "未传入，与B4一致",
            "enable_thinking": "未传入，使用冻结模板默认值",
            "truncation": False,
        },
        "normalization": "原样AST加载冻结normalize_message、_parse_tool_calls与_template_input_ids；保留role/content/tool_call_id，工具参数JSON转换为对象",
        "tools_definition_source": "模板不额外注入tools列表；保留原System消息内的工具说明和已有tool_calls，与B4完全一致",
        "engine_started": False,
        "gpu_devices": [],
        "sglang_imported": False,
        "model_weights_loaded": False,
        "difference_from_B5C2": "仅来源组合（4组×2任务×2Session）与逐组结构分类不同；分词、规范化、模板与截断语义完全一致",
    }
    save(OUT / "input_configuration_lock.json", config)

    request_rows = []
    causal_rows = []
    group_results = {}
    context_length = identity["engine_configuration"]["context_length"]

    for gi, group in enumerate(GROUPS, start=1):
        recs = [r for r in source["records"] if r["group"] == group]
        gdir = OUT / group
        # 锁阶段已失败的组只记录，不重建
        if any(r["failure"] for r in recs):
            group_results[group] = {
                "status": "LOCK_STAGE_FAILURE",
                "failures": [r["failure"] for r in recs if r["failure"]],
            }
            continue
        histories = {}
        normalized = {}
        causal = []
        for row in recs:
            label = row["label"]
            p = OUT / row["source_history_file"]
            assert sha(p) == row["source_history_sha256"]
            raw = json.loads(p.read_text())
            assert sum(m.get("role") == "assistant" for m in raw) == 2
            seen_calls = {}
            tool_evidence = []
            for i, m in enumerate(raw):
                calls = m.get("tool_calls_json")
                if calls:
                    for c in env["_parse_tool_calls"](calls):
                        if c.get("id"):
                            seen_calls[c["id"]] = i
                if m.get("role") == "tool":
                    call = m.get("tool_call_id")
                    assert call in seen_calls and seen_calls[call] < i, "工具结果无法对应先前调用"
                    tool_evidence.append(
                        {
                            "message_index": i,
                            "tool_call_id": call,
                            "call_message_index": seen_calls[call],
                            "before_pending_cut": i < row["pending_cut_index"],
                            "wall_clock": "UNKNOWN",
                        }
                    )
            histories[label] = raw + [{"role": "assistant"}]
            normalized[label] = [env["normalize_message"](m) for m in raw]
            causal.append(
                {
                    "group": group,
                    "label": label,
                    "cut_index": row["pending_cut_index"],
                    "history_message_count": len(raw),
                    "role_sequence": [m.get("role") for m in raw],
                    "tool_results": tool_evidence,
                    "historical_assistant_outputs": 2,
                    "third_output_available_to_builder": False,
                    "later_messages_available_to_builder": False,
                    "ordering_evidence": "原始消息索引及tool_call_id；无原始墙钟时间",
                    "target_native_outputs": False,
                }
            )

        g = SimpleNamespace(
            group_ordinal=gi,
            allocation_round=2,
            session_by_label={r["label"]: r["session_id"] for r in recs},
        )
        requests, audits = env["materialize_group_requests"](
            tokenizer,
            histories,
            group=g,
            normalize_message=env["normalize_message"],
            template_input_ids=env["_template_input_ids"],
        )
        assert len(requests) == 12

        pending = {}
        for (label, turn), r in requests.items():
            tokens = r["input_ids"]
            assert tokens and all(isinstance(t, int) for t in tokens)
            assert len(tokens) < context_length, "超出引擎上下文长度"
            rid = f"IMW{gi}_{label}_TURN_{turn:03d}"
            name = f"{rid}_tokens.json"
            save(gdir / name, tokens)
            request_rows.append(
                {
                    "group": group,
                    "label": label,
                    "turn": turn,
                    "request_id": rid,
                    "token_file": str((gdir / name).relative_to(OUT)),
                    "token_count": len(tokens),
                    "canonical_token_sha256": token_sha(tokens),
                    "file_sha256": sha(gdir / name),
                    "legacy_builder_request_id": r["rid"],
                }
            )
            if turn == 3:
                pending[label] = tokens

        boundaries = {}
        for label in LABELS:
            msgs = normalized[label]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            (gdir / f"{label}_pending_rendered.txt").write_text(text)
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            assert encoded["input_ids"] == pending[label]
            save(gdir / f"{label}_pending_offsets.json", encoded["offset_mapping"])
            parts = []
            for count in range(1, len(msgs) + 1):
                if count == 1:
                    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
                    prefix = text[: text.index("<|im_start|>user\n")]
                else:
                    prefix = tokenizer.apply_chat_template(msgs[:count], tokenize=False, add_generation_prompt=False)
                is_prefix = text.startswith(prefix)
                pt = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                parts.append(
                    {
                        "after_message_count": count,
                        "last_role": msgs[count - 1]["role"],
                        "rendered_character_count": len(prefix),
                        "rendered_is_prefix": is_prefix,
                        "tokenized_prefix_length": len(pt),
                        "token_lcp_with_full_input": lcp(pt, pending[label]),
                        "exact_token_prefix": pending[label][: len(pt)] == pt,
                    }
                )
            boundaries[label] = parts

        matrix = {a: {b: lcp(pending[a], pending[b]) for b in LABELS} for a in LABELS}
        common = lcp(*pending.values())
        ab = matrix["A"]["B"]
        cd = matrix["C"]["D"]
        offsets = {
            label: json.loads((gdir / f"{label}_pending_offsets.json").read_text()) for label in LABELS
        }
        positions = []
        for name, origin, pos in (("common", "A", common), ("task1", "A", ab), ("task2", "C", cd)):
            prefix = pending[origin][:pos]
            supports = [label for label, ids in pending.items() if len(ids) >= pos and ids[:pos] == prefix]
            divergence = {
                label: {
                    "next_token_id": ids[pos] if pos < len(ids) else None,
                    "next_token_text": tokenizer.decode(ids[pos : pos + 1]),
                    "character_offset": offsets[label][pos] if pos < len(ids) else None,
                }
                for label, ids in pending.items()
                if label in supports
            }
            positions.append(
                {
                    "name": name,
                    "position": pos,
                    "origin": origin,
                    "support": supports,
                    "prefix_token_sha256": token_sha(prefix),
                    "divergence": divergence,
                    "runtime_materialization": "UNKNOWN",
                    "planning_target": "UNKNOWN",
                    "position_within_input": all(pos <= len(pending[x]) for x in supports),
                    "required_future_target": f"T_p >= {pos}；本轮不假设FA或T",
                    "checkpoint_created": False,
                }
            )
        system_tokens = {label: boundaries[label][0]["token_lcp_with_full_input"] for label in LABELS}
        group_results[group] = {
            "status": "REBUILT",
            "counts": {k: len(v) for k, v in pending.items()},
            "pairwise_lcp": matrix,
            "four_way_lcp": common,
            "task_pair_lcp": {"AB": ab, "CD": cd},
            "cross_task_lcp": {f"{a}{b}": matrix[a][b] for a in "AB" for b in "CD"},
            "system_prefix_token_lengths": system_tokens,
            "positions": positions,
            "structure_class": classify(common, matrix, ab, cd),
            "boundary_all_exact": all(
                boundaries[label][c]["exact_token_prefix"]
                for label in LABELS
                for c in range(len(boundaries[label]))
            ),
            "scope": "真实目标Tokenizer构造的CPU内容输入；非原始Runtime已接收Token；无Checkpoint物化或严格边认证",
        }
        causal_rows.append({"group": group, "labels": causal, "collector_audits": audits})

    save(OUT / "requests.json", request_rows)
    save(OUT / "causal_checks.json", causal_rows)
    save(OUT / "token_prefix_matrix.json", group_results)
    print(json.dumps({g: {"status": v["status"], "class": v.get("structure_class")} for g, v in group_results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
