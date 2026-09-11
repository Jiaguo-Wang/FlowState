"""在独立补充诊断中细分请求到达、队列事件等待和响应返回。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

from evaluation import rq6_bottleneck_diagnosis as diagnosis
from evaluation.rq6_system_overhead import write_json


def install_wait_timing(probe, client_class):
    """包裹原有提交函数，不改变队列、事件或超时配置。"""
    original = probe.ProbeState.submit

    def timed_submit(self, request, *args, **kwargs):
        entered = perf_counter_ns()
        result = original(self, request, *args, **kwargs)
        exited = perf_counter_ns()
        if "diagnostic_timing" in result:
            result["diagnostic_timing"]["submit_enter_ns"] = entered
            result["diagnostic_timing"]["submit_exit_ns"] = exited
        return result

    probe.ProbeState.submit = timed_submit

    class WaitClient(client_class):
        """将同一主机单调时钟上的队列边界附加到既有计时记录。"""

        def _call(self, request):
            response = super()._call(request)
            if self.phase is not None:
                timing = response["diagnostic_timing"]
                start = int(timing["client_sent_ns"])
                self.rows[-1]["pre_submit_ns"] = timing["submit_enter_ns"] - start
                self.rows[-1]["queue_wait_ns"] = timing["server_started_ns"] - timing["submit_enter_ns"]
                self.rows[-1]["event_return_ns"] = timing["submit_exit_ns"] - timing["server_ended_ns"]
                self.rows[-1]["post_submit_ns"] = self.rows[-1]["return_ns"] - self.rows[-1]["event_return_ns"]
            return response

    return WaitClient


def main():
    """只执行预先列明的补充定位条件，每次运行后验证设备清理。"""
    parser = argparse.ArgumentParser(description="RQ6-D 队列等待补充诊断")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        import targeted_probe as probe
        diagnosis.MeasuredClient = install_wait_timing(probe, diagnosis.MeasuredClient)
        return diagnosis.worker(args.output, args.worker)
    for row in json.loads((args.output / "plan.json").read_text()):
        run_id = row["run_id"]
        diagnosis.base.wait_gpu_stable(gpu_index=0)
        with (args.output / f"{run_id}.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "evaluation.rq6_wait_diagnosis",
                "--output", str(args.output), "--worker", run_id], stdout=log,
                stderr=subprocess.STDOUT, timeout=1800)
        cleanup = diagnosis.base.wait_gpu_stable(gpu_index=0)
        path = args.output / "runs" / run_id / "record.json"
        record = json.loads(path.read_text())
        record["gpu_cleanup"] = cleanup
        record["status"] = ("PASS" if result.returncode == 0 and cleanup["stable"]
            and record["status"] == "PASS_PENDING_CLEANUP" else "INVALID")
        write_json(path, record)
        print(f"补充等待诊断：{run_id} {record['status']}", flush=True)
        if record["status"] != "PASS":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
