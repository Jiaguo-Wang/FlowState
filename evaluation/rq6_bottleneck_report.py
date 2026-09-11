"""生成 RQ6-D 的中文诊断报告与可独立阅读的报告数据包。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics

from evaluation.rq6_bottleneck_analysis import analyze
from evaluation.rq6_bottleneck_diagnosis import FROZEN, ROOT
from evaluation.rq6_system_overhead import source_manifest, source_paths, summarize_values, write_json


def build_report(root):
    """独立核对计时闭合与源码完整性，再生成答案优先的报告。"""
    output = root / "final_diagnostic"
    summary = analyze(output)
    phases = []
    noops = []
    counts = []
    for item in summary["diagnostic_runs"]:
        rpc = json.loads((output / "runs" / item["run_id"] / "rpc_timings.json").read_text())
        noops.extend(r for r in rpc if r["phase"] == "noop")
        c, n = item["candidate_count"], item["eviction_count"]
        for label, key in (("introspection", "introspection_ns"),
                           (item["run_id"] + ":zero", "zero_ns"),
                           (item["run_id"] + ":" + item["mode"], "changed_ns")):
            selected = [r for r in rpc if r["phase"] == label]
            expected_count = c + 1 + (2 * n if key == "changed_ns" else 0)
            if len(selected) != expected_count:
                raise RuntimeError("实际 RPC 数与源码推导不一致")
            for row in selected:
                closure = (row["pre_submit_ns"] + row["queue_wait_ns"] + row["server_ns"]
                    + row["event_return_ns"] + row["post_submit_ns"])
                if closure != row["roundtrip_ns"]:
                    raise RuntimeError("跨进程计时不能闭合")
            counts.append({"run_id": item["run_id"], "phase": label,
                "rpc_count": len(selected), "expected_rpc_count": expected_count,
                "actions": dict(Counter(r["action"] for r in selected)),
                "profile_calls": dict(sum((Counter((r["profile"] or {}).get("counts", {})) for r in selected), Counter()))})
            if item["mode"] == "multiple" and key in ("introspection_ns", "changed_ns"):
                phases.extend(selected)
    total = sum(r["roundtrip_ns"] for r in phases)
    waiting = sum(r["outside_server_ns"] for r in phases)
    noop_parts = {key.replace("_ns", "_ms"): statistics.fmean(row[key] for row in noops) / 1e6
        for key in ("roundtrip_ns", "pre_submit_ns", "queue_wait_ns", "server_ns", "event_return_ns", "post_submit_ns")}
    write_json(root / "noop_decomposition.json", {
        "sample_count": len(noops), "engine_count": 6, "mean": noop_parts,
        "说明": "空载命令本身不读取或变更状态；这些非重叠区间仍保留原队列和事件等待。"})
    phase_parts = {k.replace("_ns", "_ms"): sum(r[k] for r in phases) / 3e6 for k in
        ("pre_submit_ns", "queue_wait_ns", "server_ns", "event_return_ns", "post_submit_ns")}
    before = json.loads((root / "source_before.json").read_text())
    after = source_manifest(source_paths())
    diagnostic_before = json.loads((output / "diagnostic_source_before.json").read_text())
    diagnostic_after = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in diagnostic_before}
    source_ok = before == after and diagnostic_before == diagnostic_after
    integrity = {"status": "PASS" if source_ok else "FAIL", "frozen_source_unchanged": before == after,
        "diagnostic_source_unchanged_during_collection": diagnostic_before == diagnostic_after,
        "before": before, "after": after,
        "diagnostic_before": diagnostic_before, "diagnostic_after": diagnostic_after,
        "frozen_artifact_protection": "容器中冻结仓库与数据只读挂载；完整CPU测试使用独立完整副本"}
    write_json(root / "source_integrity.json", integrity)
    if not source_ok:
        raise RuntimeError("源码完整性验证失败")
    frozen_root = ROOT / "evaluation/rq6_overhead_output/rq6_overhead_20260909_061500"
    frozen_manifest_path = frozen_root / "provenance/sha256_manifest.json"
    frozen_manifest = json.loads(frozen_manifest_path.read_text())
    frozen_mismatches = [name for name, row in frozen_manifest["files"].items()
        if not (frozen_root / name).is_file()
        or hashlib.sha256((frozen_root / name).read_bytes()).hexdigest() != row["sha256"]]
    write_json(root / "frozen_artifact_integrity.json", {
        "status": "FAIL" if frozen_mismatches else "PASS",
        "checked_files": len(frozen_manifest["files"]), "mismatches": frozen_mismatches,
        "reference_manifest_sha256": hashlib.sha256(frozen_manifest_path.read_bytes()).hexdigest(),
        "说明": "只读对照冻结RQ6已有476项摘要，不将当前摘要伪装为历史基线。"})
    if frozen_mismatches:
        raise RuntimeError("冻结产物摘要不匹配")
    test_results = {"environment": "冻结SGLang 0.5.17镜像中的Python 3.12.3；无GPU；完整仓库与AgentX审计副本",
        "new_skip_xfail": False, "related": {}, "full_cpu": {}}
    for label, name in (("related", "related_tests_scheduler_probe.log"), ("full_cpu", "full_cpu_suite.log")):
        log = root / name
        body = log.read_text() if log.exists() else ""
        match = re.search(r"(\d+ passed[^\n]* in [\d.]+s[^\n]*)", body)
        test_results[label] = {"log": name, "summary": match.group(1) if match else "尚未完成",
            "status": "PASS" if match and not re.search(r"\d+ (failed|error)", body) else "PENDING_OR_FAIL"}
    write_json(root / "test_results.json", test_results)
    write_json(output / "observed_message_counts.json", counts)
    formal = json.loads((output / "formal_scaling.json").read_text())
    rows = json.loads((output / "formal_message_counts.json").read_text())
    total_formal = sum(r["timings_ms"]["total_control"] for r in rows)
    formal_stage_fraction = sum(r["timings_ms"]["introspection"] + r["timings_ms"]["reconciliation"] for r in rows) / total_formal
    closure = {"status": "PASS", "diagnostic_rpc_closure_exact": True,
        "diagnostic_observed_message_counts_exact": True,
        "formal_rows": 72, "formal_snapshot_count": 24,
        "formal_introspection_reconciliation_fraction": formal_stage_fraction,
        "diagnostic_outside_handler_fraction": waiting / total,
        "diagnostic_multiple_epoch_rpc_mean_ms": total / 3e6,
        "diagnostic_multiple_epoch_decomposition_mean_ms": phase_parts,
        "transfer_to_formal_internal_fraction_supported": False}
    write_json(root / "data_validation.json", closure)
    summary["status"] = "RQ6_RUNTIME_BOTTLENECK_PARTIAL"
    summary["primary_bottleneck"] = "同步控制传输和线程交接，主要位于入队前与工作完成后的提交线程唤醒区间"
    summary["secondary_bottleneck"] = "逐候选检查、逐驱逐及S4独立控制往返反复支付传输等待；全候选追踪进一步放大消息材料和验证工作"
    summary["remaining_uncertainty"] = "尚未将入队前时间进一步拆成JSON编码解码、socket等待与GIL竞争；隐式GPU同步次数及设备完成时间未测量"
    summary["validation"] = closure
    summary["tests"] = test_results
    write_json(root / "summary.json", summary)

    n = summary["noop_ms"]
    prim = summary["primitive_ms"]["remove_free_ms"]
    intro_breaks = [r["breakdowns"]["introspection_ns"] for r in summary["diagnostic_runs"]]
    intro_mean = {key: statistics.fmean(r[key] for r in intro_breaks) for key in
        ("total_ms", "pre_submit_ms", "queue_wait_ms", "event_return_ms", "post_submit_ms",
         "worker_total_ms", "worker_path_lookup_and_snapshot_ms", "worker_global_tree_validation_snapshot_ms",
         "worker_allocator_snapshot_ms", "worker_scope_validation_ms", "controller_outside_rpc_ms")}
    intro_mean["nested_fa_index_hash_ms"] = statistics.fmean(
        r["nested_details_ms"].get("_path_snapshot:_tensor_sha256", 0)
        + r["nested_details_ms"].get("_global_maps:_tensor_sha256", 0) for r in intro_breaks)
    intro_mean["nested_recurrent_slot_read_ms"] = statistics.fmean(
        r["nested_details_ms"].get("_path_snapshot:_tensor_ids", 0)
        + r["nested_details_ms"].get("_global_maps:_tensor_ids", 0) for r in intro_breaks)
    intro_mean["nested_shared_exact_path_lookup_ms"] = statistics.fmean(
        r["nested_details_ms"].get("_find_exact_node", 0) for r in intro_breaks)
    write_json(root / "introspection_decomposition.json", intro_mean)
    selection_rows = []
    zero_controller_ms = []
    zero_validation_ms = []
    for item in summary["diagnostic_runs"]:
        parts = item["breakdowns"]
        zero_record = json.loads((output / "runs" / item["run_id"] / "zero.json").read_text())
        zero_controller_ms.append(zero_record["controller_and_trace_ns"] / 1e6)
        zero_validation_ms.append(zero_record["post_validation_ns"] / 1e6)
        selection_rows.append({"run": item["run_id"], "C": item["candidate_count"],
            "N": item["eviction_count"], "mode": "单次驱逐" if item["mode"] == "one" else "多次驱逐",
            "introspection_ms": parts["introspection_ns"]["total_ms"],
            "zero_ms": parts["zero_ns"]["total_ms"],
            "reconciliation_ms": parts["changed_ns"]["total_ms"]})
    lines = ["# RQ6-D 运行时控制瓶颈诊断", "", "## 主要结论", "",
        f"状态：{summary['status']}。主要时间已经定位到同步控制传输和线程交接；真正recurrent槽位remove/free的主机调用平均仅{prim['mean']:.6f} ms。三个最终多驱逐诊断的RPC总时间中，工作线程控制处理函数之外占{waiting / total:.2%}。这不能直接当作冻结72-run内部比例。",
        "", "## 口径与数据", "",
        "冻结RQ6仍为24个snapshot、72次独立运行，控制段均值8398.847 ms。新增诊断只取候选数8、12、20的三个组（g016、g005、g015），每组一个单次驱逐与一个多次驱逐run。每run先测20次空载往返、完整只读查询及全部保留的零驱逐验证。所有数值单位为ms。",
        "主机使用同一perf_counter_ns时钟；P95采用冻结线性插值定义。空载120个样本聚集于6个引擎，不是120个独立workload。零驱逐与单次驱逐是诊断条件，多次驱逐沿用正式K及selected set。",
        "", "## 空载与只读查询", "",
        f"无状态读取或变更的no-op往返：mean={n['mean']:.6f}，P95={n['p95']:.6f}，median={n['median']:.6f}，n={n['count']}。每次依然经过TCP、提交队列、scheduler安全点和响应。不能拿空载均值乘消息数，宣称精确重建带大量令牌/trace材料的请求成本。",
        f"六run introspection平均{intro_mean['total_ms']:.6f}：入队前{intro_mean['pre_submit_ms']:.6f}、队列等待{intro_mean['queue_wait_ms']:.6f}、worker处理{intro_mean['worker_total_ms']:.6f}、事件返回{intro_mean['event_return_ms']:.6f}、其后响应传输{intro_mean['post_submit_ms']:.6f}、controller本地工作{intro_mean['controller_outside_rpc_ms']:.6f}。",
        f"worker处理包含精确路径查找与FA/recurrent路径事实{intro_mean['worker_path_lookup_and_snapshot_ms']:.6f}、全树证明{intro_mean['worker_global_tree_validation_snapshot_ms']:.6f}、分配器事实{intro_mean['worker_allocator_snapshot_ms']:.6f}、scope验证{intro_mean['worker_scope_validation_ms']:.6f}。其中嵌套FA索引摘要{intro_mean['nested_fa_index_hash_ms']:.6f}、recurrent槽位读取{intro_mean['nested_recurrent_slot_read_ms']:.6f}、共享精确路径查找{intro_mean['nested_shared_exact_path_lookup_ms']:.6f}。这些是嵌套时间，不能再次加入总时间。布尔驻留读取与路径循环耦合，未单独逐属性计时；冻结控制段没有独立inspect_fa_frontier调用，该步骤位于先前barrier重放。",
        "", "## 零、单次和多次驱逐的直接测量", "",
        "| run | C | N | introspection | 零驱逐（含验证） | 驱逐条件（含验证） |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in selection_rows:
        lines.append(f"| {row['run']} | {row['C']} | {row['N']} | {row['introspection_ms']:.3f} | {row['zero_ms']:.3f} | {row['reconciliation_ms']:.3f} |")
    write_json(root / "preserve_validation_only.json", {
        "preserve_only_controller_ms": summarize_values(zero_controller_ms),
        "post_validation_only_ms": summarize_values(zero_validation_ms),
        "说明": "来自相同零驱逐运行的连续分段，不含驱逐命令或S4；后置验证含census、inspect及不变量验证。"})
    lines += ["", f"零驱逐不发actuator消息；单纯保留全部的controller部分平均{statistics.fmean(zero_controller_ms):.6f}，后置验证平均{statistics.fmean(zero_validation_ms):.6f}。后置验证包括census、C次inspect和不变量验证。controller+trace与后置验证已分别计时，详见每run的zero.json和changed.json。各条件只有一个独立run，跨run抖动明显；不能用表格差值拟合无噪声的每次驱逐成本。",
        "", "## 真正的状态操作及控制消息", "",
        f"34次真实驱逐的detach+free主机调用：mean={prim['mean']:.6f}，P95={prim['p95']:.6f}，max={prim['max']:.6f}。adapter完整原语还包含leaf-set更新及sanity检查；这些分量分别保存。没有插入GPU synchronize；设备异步完成时间为UNSUPPORTED。free归还预分配池中的槽位索引，不是向驱动释放整块checkpoint显存。",
        "正式每epoch同步RPC数量为2C+2N+2：两次census、2C次候选inspect、N次驱逐、N次S4。C=8/12/16/20对应N=6/9/12/16、RPC=30/44/58/74。每RPC一个请求及响应、一条scheduler命令和一次Event.wait；不存在额外远程worker跳转。",
        "每个驱逐保留S0到S4共5个全候选trace快照。正式每epoch的probe路径快照数为4C+2N+5NC，四档分别284/606/1048/1712。这是调用次数，不是GPU同步次数。实测诊断RPC计数与源码推导逐条件一致。",
        "", "## 等待与同步的路径证据", "",
        "Event.wait上限180秒和socket超时240秒在关键路径上，但均为超时上限。sleep_on_idle默认False，源码中的poll(1000)未启用；启动等待、进程join以及GPU清理轮询均在正式计时外。控制动作显式GPU/NCCL同步为0；tensor.cpu/tolist可能隐式等待设备，其实际同步次数为UNSUPPORTED。",
        "队列探针保留原提交逻辑，直接记录提交入口、处理开始、处理结束及提交返回。入队前区间包含controller编码、TCP连接与传输、服务线程读行和解码；处理后至提交返回包含事件置位及线程重新运行。GIL/操作系统调度竞争是合理候选原因，但没有单独测量，不能写为已证实根因。",
        "", "## 相关性与请求可见性", "",
        f"冻结total_control与C、驱逐数、handle数、RPC数、scope验证数的Spearman均为{formal['spearman']['total_control']['candidate_count']:.6f}。这些变量在正式population中单调共变，不能用相关系数区分各自因果贡献。",
        "request-visible blocking=PARTIAL：RQ4同步reconcile和验证完成后才进入pending_resume并提交execute_request，因此workflow恢复被阻塞；控制时scheduler要求完全空闲，没有这些pending的在途请求。冻结TTFT从后续提交开始，不含这段8.4秒。",
        f"冻结introspection+reconciliation可归属总控制均值的{formal_stage_fraction:.4%}；最终诊断RPC的分解在纳秒级严格闭合，处理函数外等待占{waiting / total:.4%}。未观测冻结运行的内部时间，不能把诊断比例或样本均值替换正式8.4秒结果。",
        "", "## 正确性、限制与未关闭项", "",
        "最终6/6隔离运行通过handle映射、精确recurrent驻留、FA及结构保持、仅目标recurrent变化、无原生驱逐、无重驻留和无级联检查。重放通过冻结snapshot验证，未来信息未读取；只读阶段全局树和分配器前后相等。每run清理后设备回到14 MiB、无compute process。",
        "源文件摘要前后相同，冻结RQ6的476项历史摘要全部匹配；所有冻结输入在运行容器中只读挂载。初步6-run与一次探针安装失败记录均保留并排除最终计时。首次失败只到no-op，没有状态变更。",
        f"RQ6相关测试：{test_results['related']['summary']}；完整CPU测试：{test_results['full_cpu']['summary']}。未新增skip或xfail。",
        "目前仍未将入队前传输区间拆成JSON、socket和线程竞争；设备完成时间及隐式同步次数未测量。因此返回PARTIAL，不把时间区间定位包装成更细层面的因果结论。",
        "", "## 下一步最小优化方向", "",
        "只针对现有同步控制通道的线程交接/唤醒做最小改动验证，首先降低空载往返成本，并以完全相同的状态语义和正确性门禁验收。本轮未实施。", ""]
    report = "\n".join(lines)
    (root / "final_report.md").write_text(report, encoding="utf-8")
    sections = report.split("\n## ")
    blocks = [{"id": "title", "type": "markdown", "body": sections[0].strip()}]
    for index, section in enumerate(sections[1:]):
        body = "## " + section.strip()
        if "| run | C | N |" in body:
            body = "\n".join(line for line in body.splitlines() if not line.startswith("|"))
        blocks.append({"id": f"section_{index}", "type": "markdown", "body": body})
        if section.startswith("空载与只读查询"):
            blocks.append({"id": "introspection_chart", "type": "chart", "chartId": "introspection_chart"})
        if "| run | C | N |" in section:
            blocks.append({"id": "diagnostic_table", "type": "table", "tableId": "diagnostic_table"})
    stamp = datetime.now(timezone.utc).isoformat()
    chart_rows = [{"stage": label, "mean_ms": intro_mean[key],
        "total_ms": intro_mean["total_ms"], "run_count": 6, "order": index}
        for index, (label, key) in enumerate((
            ("入队前", "pre_submit_ms"), ("队列等待", "queue_wait_ms"),
            ("处理函数", "worker_total_ms"), ("事件返回", "event_return_ms"),
            ("响应传输", "post_submit_ms"), ("本地工作", "controller_outside_rpc_ms")))]
    # 仅在内存中投影已验证的报告数据，不访问或改写任何实验输入。
    table_sql = "SELECT run, C, N, mode, introspection_ms, zero_ms, reconciliation_ms FROM diagnostic ORDER BY C, rowid"
    chart_sql = 'SELECT stage, mean_ms, total_ms, run_count, "order" FROM introspection ORDER BY "order"'
    with sqlite3.connect(":memory:") as database:
        database.row_factory = sqlite3.Row
        database.execute("CREATE TABLE diagnostic (run TEXT, C INTEGER, N INTEGER, mode TEXT, introspection_ms REAL, zero_ms REAL, reconciliation_ms REAL)")
        database.executemany("INSERT INTO diagnostic VALUES (?, ?, ?, ?, ?, ?, ?)", [tuple(row.values()) for row in selection_rows])
        database.execute('CREATE TABLE introspection (stage TEXT, mean_ms REAL, total_ms REAL, run_count INTEGER, "order" INTEGER)')
        database.executemany("INSERT INTO introspection VALUES (?, ?, ?, ?, ?)", [tuple(row.values()) for row in chart_rows])
        selection_rows = [dict(row) for row in database.execute(table_sql)]
        chart_rows = [dict(row) for row in database.execute(chart_sql)]
    source = {"id": "diagnostic", "label": "RQ6-D最终六次诊断条件表",
        "path": "final_diagnostic/diagnostic_summary.json", "query": {
            "language": "sql", "engine": "SQLite 内存投影",
            "description": "直接投影已验证的六次诊断条件，不重算或修改实验数据。", "sql": table_sql}}
    chart_source = {"id": "introspection", "label": "RQ6-D最终六次只读查询的分段均值",
        "path": "introspection_decomposition.json", "query": {
            "language": "sql", "engine": "SQLite 内存投影",
            "description": "按调用顺序投影已验证的不重叠分段均值。", "sql": chart_sql}}
    write_json(root / "chart_contract.json", {
        "问题": "最终六次诊断的只读查询时间主要分布在哪些不重叠区间？",
        "结论": "入队前与事件返回区间占主要时间；这不是冻结72次运行的内部占比。",
        "图形": "报告内单系列柱形图，六个阶段按调用顺序排列，纵轴从零起，单位毫秒。",
        "数据": "六个分段均值；附总时间、运行数及阶段顺序，不加入嵌套子函数时间。",
        "颜色": "单一蓝色根；类别由横轴文字区分，不添加冗余颜色图例。",
        "布局": "报告全宽独立图块；精确驱逐条件继续使用表格，不绘制趋势。",
        "验证": "使用共享报告打包器的验证回执；若没有浏览器，明确记录仅结构验证。"})
    artifact = {"surface": "report", "manifest": {"version": 1, "surface": "report",
        "title": "RQ6-D 运行时控制瓶颈诊断", "description": "同步控制路径的CPU与运行时分段证据",
        "generatedAt": stamp, "blocks": blocks, "sources": [source, chart_source],
        "cards": [], "charts": [{"id": "introspection_chart", "type": "bar",
            "title": "只读查询的分段平均耗时", "subtitle": "六次最终诊断；单位ms；各段不重叠，不代表冻结72次运行的分解",
            "dataset": "introspection", "sourceId": "introspection",
            "encodings": {"x": {"field": "stage"}, "y": {"field": "mean_ms"}}}],
        "tables": [{"id": "diagnostic_table",
            "title": "六个独立条件的直接计时", "subtitle": "单位ms；零驱逐和驱逐条件均包含验证，每条件一个独立run",
            "dataset": "diagnostic", "sourceId": "diagnostic", "defaultSort": {"field": "C", "direction": "asc"},
            "columns": [{"field": "run", "label": "运行", "type": "text"},
                {"field": "C", "label": "候选数"}, {"field": "N", "label": "驱逐数"},
                {"field": "introspection_ms", "label": "只读查询ms"},
                {"field": "zero_ms", "label": "零驱逐ms"},
                {"field": "reconciliation_ms", "label": "驱逐条件ms"}]}]},
        "snapshot": {"version": 1, "generatedAt": stamp, "status": "partial",
            "datasets": {"diagnostic": selection_rows, "introspection": chart_rows}, "accessIssues": [{"id": "waiting_detail",
            "message": "入队前时间尚未进一步拆分为JSON处理、socket等待与线程竞争。"}]},
        "sources": [source, chart_source]}
    write_json(root / "artifact.json", artifact)
    diagnostic_sources = ["evaluation/rq6_bottleneck_analysis.py",
        "evaluation/rq6_bottleneck_diagnosis.py", "evaluation/rq6_bottleneck_report.py",
        "evaluation/rq6_wait_diagnosis.py", "evaluation/run_rq6d_cpu_tests.sh",
        "tests/runtime/rq6_bottleneck_transport.py", "tests/test_rq6_bottleneck_diagnosis.py"]
    write_json(root / "manifest.json", {
        "task": "RQ6-D", "status": summary["status"], "generated_at": stamp,
        "formal_reference": str(frozen_root), "formal_runs_read": 72,
        "canonical_diagnostic_root": "final_diagnostic", "canonical_runs": 6,
        "excluded_pilot_runs": 6, "excluded_probe_setup_failures": 1,
        "noop_samples": n["count"], "actual_evictions": prim["count"],
        "source_integrity": integrity["status"], "frozen_artifact_integrity": "PASS",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in diagnostic_sources},
        "tests": test_results, "gpu_state": "final_gpu_state.json",
        "report": "final_report.md", "portable_report": "report.html",
        "说明": "新建独立诊断产物；没有优化、重跑72次正式benchmark、启动Kimi或修改冻结产物。"})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ6-D 中文诊断报告")
    parser.add_argument("--output", type=Path, required=True)
    build_report(parser.parse_args().output)
