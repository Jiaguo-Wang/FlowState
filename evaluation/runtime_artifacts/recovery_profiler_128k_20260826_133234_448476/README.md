# Recovery Profiler 128K

本目录保存 Step 10D.1 的独立恢复测量。测量对象是服务端内部 recovery/TTFT 路径，不是纯 replay CUDA kernel latency。

每个 trial 均复用 Step 9D 的 fresh/flush、checkpoint 构建、正式 Mamba-only 驱逐、FA 安全验证、流式首 token TTFT 与运行时 H/E/G instrumentation。所有 gap 使用相同服务配置、2 次 warmup 和 12 次正式测量。

本 profiler 不读取 TraceLab policy selection、policy objective 或 policy performance。唯一使用的 TraceLab 协议信息是 recovery model 必须覆盖至 131,072 tokens。

正式 Phi 未修改。`gap_audit.csv` 中 32K 以上的 OldPhi 仅标记为旧模型外推，不是已验证预测。线性与分段线性结果仅作形状诊断。

运行状态：FAIL。
