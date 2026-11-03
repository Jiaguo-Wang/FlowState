# T2-G2 异常增补报告（主矩阵中断后的现场保全与独立复核）

日期：2026-10-27 晚。作者：执行 T2-G2 的智能体（与 16:42 接管方为不同会话）。本文件不修改任何原始证据，仅增补记录。

## 1. 事件时间线（按文件 mtime 交叉核对）

| 时刻 | 事件 | 证据 |
|---|---|---|
| ~16:40 | 门禁 3/3 臂 PASS（g_CB_FS/MC/KV）；本方 verify_formal.py 首跑触发 check() 参数类型 bug，随即修复保存 | arm_status_gate.json、launch_gate_complete.json、verify_formal.py（修复已入 16:42:51 实现锁） |
| ~16:41 | 本方"写 gate_passed.json + 启动 main"的命令被工具层安全分类器暂阻，**未执行** | 本会话工具记录 |
| 16:42:11-13 | **另一智能体接管**：以原 verify_formal.py + 原始证据合同 CPU 复核（33+173 项）后写 gate_passed.json（passed=true，含 5 项 sha256） | takeover/{verify_evidence.py,evidence_gate.json}、gate_passed.json |
| 16:42:51 | 接管方经原 launch_formal.py 启动 main（60 臂、deadline 22:42:51） | execution_started_main.json、main_takeover_host.log |
| 16:43-16:50 | m1A_FS/MC/KV 依次 PASS；m1B_FS 运行完成（returncode 0、summary PASS、E=5120、TTFT 50.0ms、请求窗编译=0） | arm_status_main.json、各臂 run/summary.json |
| 16:50:38 | m1B_FS 收尾 3 次清理观察显存 14→17→1569MiB，出现外部进程 PID 1303777（/home/yangshuo/robot-learning/.venv/bin/python，1546MiB）→ cleanup FAIL → launcher 按协议断言停止 | m1B_FS_cleanup.json、launch_main_failure.json、runtime_main.log |
| 18:14 | 接管方写断点接管报告与台账，未实现续跑、未重试 | formal_gpu_comparison_report.md、takeover/interrupted_main_*.{json,csv} |
| 20:26:49 | 本方（分类器恢复后）重跑 verify_formal.py gate：33/33 PASS；**覆盖了 formal_verification_gate.json**（见 §3） | formal_verification_gate.json mtime |
| ~20:27 | 本方"写 gate_passed + 启动 main"重试：gate_passed.json 已存在 → FileExistsError → **本方 nohup main 从未启动**；全程无双 launcher 并发 | 本会话工具记录（后台作业 Exit 1） |

## 2. 科学状态（与接管方报告一致，经本方独立复核）

- **门禁**：3/3 臂 PASS；两套独立复核（本方 33 项合同 + 接管方 33+173 项）均通过。不进性能统计。
- **主矩阵**：4/60 尝试。m1A_FS / m1A_MC / m1A_KV = PASS；m1B_FS = RESOURCE_GATE_FAIL（内部合同全 PASS——E/H/G、prefill、Cold 逐 Token、形状预热 E、窗口审计=2、请求窗编译=0、returncode=0——但收尾时失去空卡独占，外部进程是否与计时窗口重叠 UNKNOWN，按冻结协议不作有效臂、不进统计）；56 臂未执行。
- 本方独立抽检 4 臂核心合同：全部 PASS（见本会话记录；结论同接管方台账）。
- 冻结完整性：implementation_lock_main.json 17/17 文件哈希当前一致；无本方容器/launcher/Engine 残留。
- **无任何性能结论**（未跑完整四 Pending 轮次；账本缺测不填零）。

## 3. 证据链披露（本方造成的一处瑕疵）

本方 20:26:49 的 gate 重跑**覆盖了 formal_verification_gate.json**，而接管方 gate_passed.json 的 verification_sha256（574afcce…）指向被覆盖前的文件。两套复核结论相同（33/33 PASS），接管方 173 项原始证据完整保留于 takeover/evidence_gate.json；本方重跑产物为同路径新文件（schema 略异：含逐臂 checks 明细）。此覆盖不改变门禁判定，但使 gate_passed.json 的该 sha 引用失效，特此如实记录。其余原始证据本方均未改动。

## 4. 当前阻塞与恢复选项（待用户裁决，本轮不擅自续跑）

- **阻塞**：GPU0 仍有外部进程 PID 1303777（yangshuo/robot-learning，1652MiB）驻留 → 空卡门禁不满足，任何恢复均需先等空卡。
- 选项 A（推荐）：**等空卡后主矩阵全量重启**——沿用 c_formal "attempt1_expired" 先例：本次 4 臂目录与状态文件改名/标注保留为中断尝试，fresh 目录从 m1A_FS 依冻结调度重跑 60 臂（≈2.5h）。配对邻接性零妥协；门禁不必重跑（已双重复核）。
- 选项 B：**断点续跑**（跳过 3 个已完臂，从 m1B_FS 或 m1B_MC 续）——需新增续跑入口（现 launcher 独占创建、不支持跳臂），且 m1B 三元组失去时间邻接（同轮同 Pending 配对口径受损），属协议修订，须明示写入补丁协议后方可执行。
- 选项 C：接受 PARTIAL（3/60 有效臂），出具部分报告——统计力不足，不推荐。
## 5. 用户裁决与封存（2026-10-27 晚，本节后本批次封存）

- **裁决**：采用方案 A（主矩阵全量重测），但正式实验**迁移 GPU1**，由**另一指定智能体作为唯一执行者**；最终正式性能结果**只采用 GPU1 独立批次的有效主臂**，不混入本 GPU0 中断批次任何数据。
- **GPU0 批次处置**：按中断尝试封存——门禁 3 臂、主矩阵 3 个 PASS 臂、m1B_FS 资源失败臂及全部运行证据原样保留（12/12 关键证据在位核验通过）；不再执行原 launch_formal.py、不实现断点续跑、不重跑门禁。
- **哈希链披露**（对应 §3）：formal_verification_gate.json 覆盖与 gate_passed.json 的 verification_sha256（574afcce…）引用失效如实记录在案；**不回写、不重建、不伪造**原哈希链；接管方原始复核证据以 takeover/evidence_gate.json 为准。
- **边界**：本会话不再启动 GPU0/GPU1 任何 GPU 实验；不修改 GPU1 执行批次文件；不干预 GPU0 其他用户进程；对 GPU1 批次仅限只读审计。
- 封存时点观察（只读）：GPU0 已回到 14MiB（外部进程 yangshuo 已退出）；GPU1 17MiB（指定批次尚未观察到占用）。

**本批次（GPU0）状态：INTERRUPTED / ARCHIVED。**
