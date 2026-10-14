# -*- coding: utf-8 -*-
# lmcache-agentic-traces 结构核查（纯标准库版，不依赖 pandas/numpy）：
# 1) session 与 任务×模型 矩阵；2) 同任务多模型 continuation 是否存在；
# 3) 对若干同任务多模型 session 对，找 input messages 的第一个分歧点（近似量化共享前缀深度）
import duckdb
import json
from collections import defaultdict, Counter

con = duckdb.connect()
files = [f"/tmp/lmcache_check/{i}.parquet" for i in range(5)]

# ---- 统计 1：session 维度的任务/模型矩阵 ----
rows = con.execute(
    "SELECT session_id, model, count(*) FROM read_parquet(?) GROUP BY 1,2", [files]
).fetchall()
sess_models = defaultdict(set)   # session -> {model}
sess_iters = Counter()          # session -> 迭代数
for sid, model, c in rows:
    sess_models[sid].add(model)
    sess_iters[sid] += c
print("总行数(迭代):", sum(sess_iters.values()), " session 数:", len(sess_models))

task_of = {}                     # session -> 任务名（去掉末尾模型段）
for sid in sess_models:
    task_of[sid] = sid.rsplit("__", 1)[0]

task_models = defaultdict(set)   # 任务 -> {模型}
task_sessions = defaultdict(list)
for sid, ms in sess_models.items():
    task_models[task_of[sid]] |= ms
    task_sessions[task_of[sid]].append(sid)

mat = Counter(len(ms) for ms in task_models.values())
print("\n== 任务 × 模型 矩阵 ==")
print("任务总数:", len(task_models))
print("每任务不同模型数的分布:", dict(sorted(mat.items())))
print("多模型(>=2)任务数:", sum(v for k, v in mat.items() if k >= 2))

model_sess = Counter()
for sid, ms in sess_models.items():
    for m in ms:
        model_sess[m] += 1
print("\n== 每模型 session 数 ==")
for m, c in model_sess.most_common():
    print(f"  {m}: {c}")

src_sess = Counter(sid.split("__")[0] for sid in sess_models)
print("\n== 来源分布 (session 数) ==", dict(src_sess))

# ---- 统计 2：取前 8 个多模型任务，比较最后一个 input 的第一条分歧消息 ----
multi_tasks = [t for t, ms in task_models.items() if len(ms) >= 2]
print(f"\n== 抽查 {min(8, len(multi_tasks))} 个多模型任务的分歧点（用各自最长累积 input） ==")

def first_divergence(ma, mb):
    """返回第一个不同消息的下标；None 表示一方为另一方前缀"""
    for i in range(min(len(ma), len(mb))):
        if json.dumps(ma[i], sort_keys=True) != json.dumps(mb[i], sort_keys=True):
            return i
    return None

checked = 0
div_stats = []
for task in sorted(multi_tasks):
    if checked >= 8:
        break
    sessions = sorted(task_sessions[task])
    pairs = []
    for sid in sessions:
        # 每个session取迭代数最多的模型（避免同名歧义），并取最后一行 input
        # 取该 session 累积消息最长的一条 input（近似为最终状态）
        row = con.execute(
            """SELECT input FROM read_parquet(?)
               WHERE session_id = ? ORDER BY len(input) DESC LIMIT 1""",
            [files, sid],
        ).fetchone()
        if row:
            pairs.append((sid, list(sess_models[sid])[0], row[0]))
    if len(pairs) < 2:
        continue
    (sid_a, m_a, ia), (sid_b, m_b, ib) = pairs[0], pairs[-1]
    if m_a == m_b and len(pairs) < 2:
        continue
    div = first_divergence(ia, ib)
    common = div if div is not None else min(len(ia), len(ib))
    chars = sum(len(m.get("content") or "") for m in ia[:common])
    div_stats.append(common)
    print(f"任务[{task.split('__',1)[1][:40]}] {m_a}({len(ia)}轮) vs {m_b}({len(ib)}轮): "
          f"首个分歧轮={div}, 共同消息数={common}, 共同前缀≈{chars/1000:.0f}K字符")
    checked += 1

print("\n结论要点已输出。")
