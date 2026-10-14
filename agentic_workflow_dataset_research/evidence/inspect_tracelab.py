# -*- coding: utf-8 -*-
# TraceLab 公开 trace 字段核查脚本：只读取少量样本行，统计顶层字段与 session 结构
import json
import sys
from collections import Counter

gz_path = "/tmp/tracelab_check/syfi_coding_trace.jsonl"

top_keys = Counter()          # 顶层字段名计数
n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
sessions = Counter()          # session_id -> 行数（仅统计抽样范围内）
providers = Counter()
models = Counter()
example_printed = False
timing_kinds = Counter()      # timing_events 中的事件类型计数

with open(gz_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= n_sample:
            break
        try:
            rec = json.loads(line)
        except Exception:
            continue
        for k in rec.keys():
            top_keys[k] += 1
        sid = rec.get("session_id") or rec.get("sessionid") or "?"
        sessions[sid] += 1
        providers[rec.get("provider", "?")] += 1
        models[rec.get("model", "?")] += 1
        # 统计 timing_events 内部结构
        te = rec.get("timing_events")
        if isinstance(te, list):
            for ev in te:
                if isinstance(ev, dict):
                    timing_kinds[ev.get("type") or ev.get("event") or "?"] += 1
        if not example_printed and i == 0:
            # 打印第一条记录的结构（截断长字符串，避免刷屏）
            def shrink(o, depth=0):
                if isinstance(o, dict):
                    return {k: shrink(v, depth+1) for k, v in o.items()}
                if isinstance(o, list):
                    return [shrink(o[0], depth+1), f"...(共{len(o)}项)"] if o else []
                if isinstance(o, str) and len(o) > 120:
                    return o[:120] + f"...(长{len(o)})"
                return o
            print("=== 第一条记录结构（截断） ===")
            print(json.dumps(shrink(rec), ensure_ascii=False, indent=1)[:4000])
            example_printed = True

print("\n=== 顶层字段出现次数（抽样 %d 行） ===" % min(n_sample, i+1))
for k, c in top_keys.most_common():
    print(f"  {k}: {c}")
print("\n=== provider 分布 ===", dict(providers))
print("=== model 分布(top8) ===", models.most_common(8))
print("\n=== 抽样范围内 session 数 ===", len(sessions))
print("=== 每个 session 的行数分布(top10) ===", sessions.most_common(10))
import statistics
if sessions:
    vals = list(sessions.values())
    print("  rounds/session: mean=%.1f median=%d max=%d" % (
        statistics.mean(vals), statistics.median(vals), max(vals)))
print("\n=== timing_events 事件类型分布(top15) ===")
for k, c in timing_kinds.most_common(15):
    print(f"  {k}: {c}")
