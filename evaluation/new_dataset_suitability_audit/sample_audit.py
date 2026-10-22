# -*- coding: utf-8 -*-
"""FreeInference 日分片小样本流式审计（任务 D1 第二阶段）。

输入：1–2 个按天划分的 .jsonl.zst 轨迹文件（已按 100 MB 上限下载到 raw/ 下各独立目录）。
处理：zstd 子进程流式解压 + 逐行 json.loads，任何时刻只在内存保留一个 session 与其指标，
     不把整份文件载入内存。初步检查不超过 500 个 sessions（每天文件按文件顺序取前 250 个，
     不足则全取），选择方式在输出中显式记录。

统计口径（全部限定在本轮实际读取的样本内，不得外推为全数据集结论）：
  - 嵌套 subagent：递归计数 tool_calls[].subagent 内嵌 session 的数量、最大嵌套深度；
  - 请求时间重叠：展平全部请求（含嵌套）后按 [start_ms, end_ms] 做扫描线，
    统计同 session 内互相重叠的请求对数与同时在线最大并发；
  - block 前缀结构：同 session 内全部请求的 block_ids 插入前缀 trie，
    记录分叉点（同一前缀后出现 ≥2 个不同后续块）与"前缀包含"对
    （A 的 block 序列是 B 的前缀，即潜在可复用 checkpoint 形状）；
    父请求与其 subagent 首请求的单独特判；
  - 跨 session 前导共享：同一天文件内各 session 首请求前 N 块的相同序列计数
    （block id 按周局部，同周内跨 session 比较才有意义）；
  - 提供商侧缓存：provider_tokens.cached / prompt 比值分布。

不越界声明：block id 是 o200k_base 16-token 块的周局部整数编号，不是原始 token 文本；
本脚本不做、也不能做 exact token compatibility 或 runtime 物化状态认证。
"""

import csv
import json
import subprocess
import sys
from collections import Counter

# 每个日文件最多读取的 session 数（两天合计 500，符合任务上限）
PER_FILE_SESSION_CAP = 250
# 首 request 用于跨 session 前导共享比较的块数
LEADING_BLOCKS = 32
# 展平请求后做重叠对计数的每 session 上限（超出则只做扫描线并发，不做 O(n^2) 对计数）
OVERLAP_PAIR_N_CAP = 2000


def iter_sessions(zst_path):
    """用 zstd 子进程流式解压并逐行产出 session 字典，避免整文件进内存。"""
    proc = subprocess.Popen(
        ["zstd", "-dc", zst_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1024 * 1024,
    )
    try:
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)
    finally:
        proc.stdout.close()
        proc.wait()


def flatten_requests(session, depth=0, out=None):
    """展平 session 树：产出 (请求字典, 嵌套深度, 所属 session 引用, 父请求或 None)。"""
    if out is None:
        out = []
    for req in session.get("requests") or []:
        out.append((req, depth, session, None))
        for tc in req.get("tool_calls") or []:
            sa = tc.get("subagent")
            if sa:
                # 子 session 的每个请求挂到触发它的父请求上
                for sub_req in sa.get("requests") or []:
                    out.append((sub_req, depth + 1, sa, req))
                flatten_requests(sa, depth + 1, out)
    return out


def count_subagents(session):
    """递归统计内嵌 subagent session 总数与最大嵌套深度。"""

    def walk(sess, d):
        n, deepest = 0, d
        for req in sess.get("requests") or []:
            for tc in req.get("tool_calls") or []:
                sa = tc.get("subagent")
                if sa:
                    c1, d1 = walk(sa, d + 1)
                    n += 1 + c1
                    deepest = max(deepest, d1)
        return n, deepest

    return walk(session, 0)


def max_concurrency(intervals):
    """扫描线求同时在线请求的最大并发数。intervals 为 (start, end) 列表。"""
    events = []
    for s, e in intervals:
        if s is None or e is None:
            continue
        events.append((s, 1))
        events.append((e + 1, -1))  # 端点相接视为不重叠
    if not events:
        return 0, 0
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak, len(events) // 2


def overlap_pairs(intervals):
    """同 session 内 [start,end] 互相重叠的请求对数（小规模才做 O(n^2)）。"""
    ivs = sorted((s, e) for s, e in intervals if s is not None and e is not None)
    n = len(ivs)
    if n > OVERLAP_PAIR_N_CAP:
        return None  # 超上限则放弃对计数，只保留扫描线结果
    pairs = 0
    for i in range(n):
        si, ei = ivs[i]
        for j in range(i + 1, n):
            sj, ej = ivs[j]
            if sj > ei:
                break  # 已按起点排序，后面不可能重叠
            if sj <= ei and si <= ej:
                pairs += 1
    return pairs


def prefix_trie_stats(block_lists):
    """对同 session 的全部 block_ids 建前缀 trie。

    返回：(分叉点数, 最大共享前缀块数, “A 是 B 的前缀”的对数)。
    分叉点 = 某前缀之后出现 ≥2 个不同后续块，是多请求共享历史后分叉的直接证据；
    前缀包含对 = 某请求的 block 序列恰为另一请求的前缀（recurrent checkpoint
    可复用形状的必要结构条件，但不是物化证据）。
    """
    root = {}
    for blocks in block_lists:
        node = root
        node.setdefault("__pass__", 0)
        root["__pass__"] = root.get("__pass__", 0) + 1
        for b in blocks:
            child = node.get(b)
            if child is None:
                child = {}
                node[b] = child
            node = child
            node["__pass__"] = node.get("__pass__", 0) + 1
        node["__end__"] = node.get("__end__", 0) + 1

    forks = 0
    max_shared = 0
    contain_pairs = 0

    # trie 深度可达数万块，必须用显式栈迭代遍历
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        passing = node.get("__pass__", 0)
        if passing >= 2:
            max_shared = max(max_shared, depth)
        # 节点键混有整数 block id 与 "__pass__"/"__end__" 字符串标记，先判类型
        branches = [
            k for k in node if not (isinstance(k, str) and k.startswith("__"))
        ]
        real_children = [node[k] for k in branches if node[k].get("__pass__", 0) > 0]
        # 两个不同后续块 → 分叉点（多请求共享该前缀后各自延伸）
        if len(real_children) >= 2:
            forks += 1
        # 某请求在此终止且子树仍有请求延伸 → “A 是 B 的前缀”的包含对
        ends = node.get("__end__", 0)
        if ends:
            below = passing - ends
            contain_pairs += ends * below
        for k in branches:
            stack.append((node[k], depth + 1))

    return forks, max_shared, contain_pairs


def lcp_len(a, b):
    """两个 block 列表的最长公共前缀块数。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def audit_day_file(zst_path, day_label, cap):
    """审计单个日文件，返回 (per_session_metrics 列表, 汇总字典)。"""
    per_session = []
    leading_counter = Counter()  # 首 request 前 LEADING_BLOCKS 块序列 -> session 数
    n_lines = 0
    for sess in iter_sessions(zst_path):
        n_lines += 1
        if len(per_session) >= cap:
            break
        flat = flatten_requests(sess)
        top_reqs = sess.get("requests") or []
        sub_total, sub_depth = count_subagents(sess)

        # 时间重叠：展平全部请求（含嵌套 subagent 的请求）
        intervals = []
        for req, depth, owner, parent in flat:
            s, e = req.get("start_ms"), req.get("end_ms")
            if s is not None and e is not None:
                intervals.append((s, e))
        peak, n_timed = max_concurrency(intervals)
        pairs = overlap_pairs(intervals)

        # block 前缀结构（同 session 内，含嵌套请求）
        block_lists = []
        for req, depth, owner, parent in flat:
            bi = req.get("block_ids")
            if isinstance(bi, list) and bi:
                block_lists.append(bi)
        forks, max_shared, contain_pairs = prefix_trie_stats(block_lists)

        # 父请求 vs subagent 首请求的前缀关系（单独记录，作为委派共享历史证据）
        parent_child_lcp = []
        for req, depth, owner, parent in flat:
            if parent is None:
                continue
            pbi = parent.get("block_ids")
            for tc in parent.get("tool_calls") or []:
                sa = tc.get("subagent")
                if sa and sa is owner:
                    first = (sa.get("requests") or [{}])[0]
                    fbi = first.get("block_ids")
                    if isinstance(pbi, list) and isinstance(fbi, list) and pbi and fbi:
                        parent_child_lcp.append(
                            {
                                "lcp_blocks": lcp_len(pbi, fbi),
                                "parent_blocks": len(pbi),
                                "child_blocks": len(fbi),
                                "child_is_prefix_extension": fbi[: len(pbi)] == pbi,
                            }
                        )
                    break

        # 提供商侧缓存命中率（仅 top-level 请求，避免重复计嵌套）
        cached_ratio = []
        for req in top_reqs:
            pt = req.get("provider_tokens") or {}
            p, c = pt.get("prompt"), pt.get("cached")
            if p and c is not None and p > 0:
                cached_ratio.append(c / p)

        # 首 request 前导块（用于同周跨 session 前导共享）
        first_bi = (top_reqs[0].get("block_ids") if top_reqs else None)
        lead_key = None
        if isinstance(first_bi, list) and len(first_bi) >= LEADING_BLOCKS:
            lead_key = tuple(first_bi[:LEADING_BLOCKS])
            leading_counter[lead_key] += 1

        per_session.append(
            {
                "day": day_label,
                "session_id": sess.get("session_id"),
                "harness": sess.get("harness"),
                "duration_ms": (sess.get("end_ms") or 0) - (sess.get("start_ms") or 0),
                "n_top_requests": len(top_reqs),
                "n_flat_requests": len(flat),
                "subagent_sessions": sub_total,
                "max_nesting_depth": sub_depth,
                "max_concurrency": peak,
                "overlap_pairs": pairs,
                "prefix_forks": forks,
                "max_shared_prefix_blocks": max_shared,
                "prefix_contain_pairs": contain_pairs,
                "parent_child_lcp": parent_child_lcp,
                "cached_ratio_mean": (
                    sum(cached_ratio) / len(cached_ratio) if cached_ratio else None
                ),
                "models": sorted(
                    {r.get("model") for r in top_reqs if r.get("model")}
                ),
                "leading_blocks_key": str(lead_key[:4]) if lead_key else None,
            }
        )

    cross_leading = {
        "distinct_leading_keys": len(leading_counter),
        "keys_shared_by_ge2_sessions": sum(
            1 for v in leading_counter.values() if v >= 2
        ),
        "max_sessions_sharing_one_leading": max(leading_counter.values(), default=0),
    }
    return per_session, cross_leading, n_lines


def summarize(per_session, cross_leading, day_label):
    """把 per-session 指标汇成一行摘要。"""
    n = len(per_session)
    if n == 0:
        return None
    with_sub = sum(1 for m in per_session if m["subagent_sessions"] > 0)
    with_overlap = sum(1 for m in per_session if (m["overlap_pairs"] or 0) > 0)
    with_fork = sum(1 for m in per_session if m["prefix_forks"] > 0)
    with_contain = sum(1 for m in per_session if m["prefix_contain_pairs"] > 0)
    ratios = [m["cached_ratio_mean"] for m in per_session if m["cached_ratio_mean"] is not None]
    return {
        "day": day_label,
        "sessions": n,
        "total_top_requests": sum(m["n_top_requests"] for m in per_session),
        "total_flat_requests": sum(m["n_flat_requests"] for m in per_session),
        "sessions_with_subagent": with_sub,
        "max_nesting_depth": max(m["max_nesting_depth"] for m in per_session),
        "sessions_with_overlap_pairs": with_overlap,
        "total_overlap_pairs": sum(m["overlap_pairs"] or 0 for m in per_session),
        "max_concurrency": max(m["max_concurrency"] for m in per_session),
        "sessions_with_prefix_fork": with_fork,
        "max_shared_prefix_blocks": max(
            m["max_shared_prefix_blocks"] for m in per_session
        ),
        "sessions_with_prefix_contain": with_contain,
        "total_prefix_contain_pairs": sum(
            m["prefix_contain_pairs"] for m in per_session
        ),
        "cross_session_leading": cross_leading,
        "cached_ratio_mean_of_means": sum(ratios) / len(ratios) if ratios else None,
        "cached_ratio_min": min(ratios) if ratios else None,
    }


def main():
    if len(sys.argv) < 2:
        print("用法: sample_audit.py <zst文件1> [zst文件2 ...]", file=sys.stderr)
        sys.exit(2)
    # 输出写到 raw/<day>/ 的上两级（即审计目录本身），与交付文件放在一起
    out_dir = sys.argv[1].rsplit("/", 3)[0]
    all_metrics = []
    summaries = []
    for path in sys.argv[1:]:
        day_label = path.rsplit("/", 1)[-1].replace(".jsonl.zst", "")
        per_session, cross, n_lines = audit_day_file(path, day_label, PER_FILE_SESSION_CAP)
        s = summarize(per_session, cross, day_label)
        s["file_total_session_lines"] = n_lines
        summaries.append(s)
        all_metrics.extend(per_session)
        print(f"[{day_label}] 文件内 session 行数={n_lines} 实际审计={len(per_session)}")

    with open(f"{out_dir}/session_metrics.jsonl", "w", encoding="utf-8") as f:
        for m in all_metrics:
            mm = dict(m)
            mm.pop("parent_child_lcp", None)  # 明细单独落盘，控制行宽
            f.write(json.dumps(mm, ensure_ascii=False) + "\n")

    # 父子 LCP 明细单独保存（候选案例的证据链素材）
    pc = [
        {"day": m["day"], "session_id": m["session_id"], **x}
        for m in all_metrics
        for x in m["parent_child_lcp"]
    ]
    with open(f"{out_dir}/parent_child_lcp.jsonl", "w", encoding="utf-8") as f:
        for x in pc:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(f"{out_dir}/day_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=1)
    print(json.dumps(summaries, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
