# T2-G2 三策略正式 GPU 对照实验报告（formal_gpu1b，GPU1 独立批次）

日期：2026-10-27/28。执行：GPU1 批次唯一执行智能体（用户授权"一次批准，自动完成全部正式实验"及接管指令延续）。

## 0. 判定

**PASS —— 60/60 有效逻辑主臂**（m{1..5}{A..D}_{FS,MC,KV} 全部 PASS，含正确性与资源门禁）。
门禁 3 臂（g_CB_{FS,MC,KV}）3/3 PASS，按协议**严格排除在性能统计之外**。

- 主矩阵执行窗口：21:11:18 → 23:05:21（约 1h54m，deadline 03:11:18 内，协议 deadline=now+6h）。
- 本批次为 GPU1 第二批次（formal_gpu1b）：对封存批次 formal_gpu1 的**最小修正续建**——仅修正编排入口 spawn 重入缺陷（device_entry.py 加 `__main__` 保护 + test_spawn_reentry.py 回归 5/5 PASS，含旧入口 RED 对照）；10 项冻结输入双血缘（formal_gpu1 与 GPU0 封存批次 formal/ 逐字节一致）核验后复制；launch/verify/analyze 仅替换输出目录与容器前缀（替换计数断言）；verify_evidence.py 逐字节复制。
- 血缘边界（用户裁决执行）：GPU0 中断批次与 formal_gpu1 失败批次均已封存，本报告统计**只含本 GPU1b 批次数据**，无任何混合。

## 1. 验证链（全部 PASS）

| 层 | 检查数 | 结果 |
|---|---|---|
| 门禁合同 verify_formal.py gate | 33（3 臂×11） | PASS |
| 门禁原始证据 takeover/verify_evidence.py gate | 180 | PASS → gate_passed.json |
| 主矩阵合同 verify_formal.py main | **660（60 臂×11）** | PASS |
| 主矩阵原始证据 takeover/verify_evidence.py main | **2928** | PASS |

11 项合同/臂：状态、census 16=16、选集==冻结计划、E/G/H 逐格、prefill==len(pt)−E、测量模式 audit==2、形状预热 E 且 fa_hit==E、请求窗 Triton 编译==0、Cold Token 独立重比对、13 项驱逐全落实、TTFT 闭包 <5ms。原始证据复核含逐臂输入 sha、COW 源/slot、FA 保护、编译/审计窗口、device_binding、进程返回码、3 次清理观察、引擎清理、60 生命周期 barrier_id 唯一性。

执行纪律：逐臂空卡门禁（无 compute 进程 + <1000MiB）与逐臂 3 次清理观察（<100MiB + 无进程）全程无触发；独立 Docker/Engine/Triton 生命周期每臂重建；形状预热先于计时，测量窗零新增编译；真实 recurrent-only 驱逐 + FA-KV 保护。

## 2. 三层结果

### 2.1 结构层：实际恢复前沿 E / 缺口 G / Prefill Token（60 臂逐格==冻结计划，12/12 格）

| Pending | FS: E / G / prefill | MC: E / G / prefill | KV: E / G / prefill |
|---|---|---|---|
| C_A (T=4096) | 0 / 4096 / 4128 | 1024 / 3072 / 3104 | 4032 / 64 / 96 |
| C_B (T=5120) | 5120 / 0 / 32 | 1024 / 4096 / 4128 | 2048 / 3072 / 3104 |
| C_C (T=6144) | 6144 / 0 / 32 | 1024 / 5120 / 5152 | 0 / 6144 / 6176 |
| C_D (T=7168) | 7168 / 0 / 32 | 1024 / 6144 / 6176 | 0 / 7168 / 7200 |

模型层 J（冻结口径）：FS 154.35ms vs MC 703.46ms vs KV 627.61ms。

### 2.2 实测层：TTFT 中位（ms，5 轮/格）

| Pending | FS | MC | KV | FS−MC | FS−KV |
|---|---|---|---|---|---|
| C_A | 165.3 | 126.0 | **59.0** | **+39.2**（0/5 负） | **+104.7**（0/5 负） |
| C_B | **42.2** | 163.9 | 127.8 | −124.2（5/5 负） | −84.6（5/5 负） |
| C_C | **36.3** | 209.3 | 247.0 | −172.3（5/5 负） | −210.2（5/5 负） |
| C_D | **48.6** | 250.5 | 289.8 | −201.9（5/5 负） | −241.8（5/5 负） |

Prefill CUDA 中位（ms）：FS 32-token 格 29.8–41.4；基线按 prefill 量线性增长（如 MC C_D 6176 tok→240.7、KV C_D 7200 tok→280.4）。Compute 中位与 TTFT 同构（FS−MC −124.5、FS−KV −117.2 请求级中位）。**负向结果如实保留：C_A 上 FS 输给两基线（E=0 全量重填充）。**

### 2.3 配对统计（请求级 20 对 = 5 轮×4 Pending；轮级四 Pending 等权 5 轮×2 比较器）

- 请求级 TTFT 差中位：FS−MC **−148.9ms** [−217.6, +43.3]，负 15 / 正 5；FS−KV **−148.7ms** [−254.9, +127.2]，负 15 / 正 5（正对恰为 C_A 5/5，两比较器一致）。
- Prefill CUDA 差中位：FS−MC −147.4、FS−KV −146.7 —— 与 TTFT 差几乎相等，**TTFT 优势≈恢复前沿差直接折算的 prefill 计算差**。
- 轮级等权（round_equal_weight.csv，10 行）：FS 69.6–78.4ms vs MC 186.6–188.4 vs KV 175.4–182.5；FS−MC 中位 **−114.1ms**（5/5 轮负）、FS−KV 中位 **−105.7ms**（5/5 轮负）。
- decode 双峰标记（>160ms）：12/60 臂 decode 慢档（FS 8、MC 1、KV 3；mode_mismatch 对 FS-MC 9、FS-KV 7），为已知单发 decode 编译/噪声双峰，TTFT 在首 token 处截断不受其影响，逐 Pending TTFT 差 5/5 一致可证。

## 3. 焦点问题回答

**Q1：FS 在四个 Pending 上的实际恢复成本与 TTFT 是否优于 Marconi-Evict 与 KVFlow？**
分 Pending 分裂：C_B/C_C/C_D 上 FS **全面占优**（TTFT −85 ~ −242ms、5/5 轮全负，prefill 仅 32 token）；C_A 上 FS **全败**（对 MC +39ms、对 KV +105ms，E=0 需 4128 token 重填充）。四 Pending 等权下 FS 净优（轮级 −104 ~ −119ms，五轮全负），因为三格大胜（−85/−172/−202 与 −85/−210/−242）远超单格失利（+39/+105）。

**Q2：优势来自哪些请求与恢复前沿差异？**
全部由恢复前沿结构解释：FS 冻结选集 {C_D_P07168, C_C_P06144, C_B_P05120} 与 C_B/C_C/C_D 前沿逐 token 精确重合（E=T，prefill=32）；MC 选集（JOIN/ROOT 链）只恢复每条 1024-token 根前缀（四格 E=1024 恒定）；KV 选集 {C_AB_P01984, C_AB_P02048, C_A_P04032} 只覆盖 C_A（E=4032）与 C_B（E=2048），C_C/C_D E=0。实测 TTFT 差与 prefill token 差逐格对应（边际约 26–38ms/千 token），请求级 prefill CUDA 差中位（−147ms）≈ TTFT 差中位（−149ms），无残余机制——优势即恢复成本差本身。

**Q3：控制开销是否影响最终收益？**
不影响。控制面在请求窗内近零：TTFT 闭包 <5ms（660/660 合同）；约 11s 的计划应用 RPC 属生命周期外带开销，三策略同量级（FS 11499ms vs MC 11061 / KV 11123 中位），差值 +376~+438ms（远小于其 ±1.1–1.5s 对内波动），且由共享审计段主导（final_heavy_observation ≈2.40s、freshness_preflight ≈1.23s、fa_content_audit ≈0.93s 三者合计 ≈4.6s 三策略几乎相同；FS 略高在 fa_content_postcheck_and_resume_baseline +170~240ms）。**TTFT 净收益不因控制开销折损。**

## 4. 结论与产物

- 判定：**T2-G2 正式 GPU 对照 PASS（60/60）**；模型层 J 排序（FS 154 < KV 628 < MC 703）与实测等权 TTFT 排序一致。
- 结论一句话：FlowState 的精确前沿绑定在 3/4 Pending 上把 TTFT 压至 36–49ms（近乎纯增量解码），代价是 C_A 零覆盖；两基线以恒定小恢复（MC）或跨 Pending 分摊（KV）换取单点优势但等权落后 ~106–114ms/轮。
- 产物（本目录）：analysis_summary.json、pair_analysis.csv、pending_policy_matrix.csv、round_equal_weight.csv、arm_status_main.json、launch_main_complete.json、formal_verification_main.json、takeover/evidence_main.json；门禁侧 gate_passed.json、takeover/evidence_gate.json、spawn_regression.json、formal_precheck.json。
- 上游封存批次（不进统计）：../formal/（GPU0 INTERRUPTED/ARCHIVED）、../formal_gpu1/（spawn 缺陷 0 有效臂）。
