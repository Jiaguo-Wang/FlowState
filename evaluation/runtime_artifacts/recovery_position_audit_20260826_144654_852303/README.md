# Recovery Cost Context-Position Audit

本目录审计 recovery cost 是否只由 gap G 决定，或还依赖绝对目标位置 T。

Legacy 阶段直接调用 Step 9D 原 profiler implementation；position matrix 阶段复用 Step 10D.1 的正式 runtime recovery 路径。每个 trial 均验证真实 H/E/G、FA-KV、循环状态和树结构。

每个 T 使用自己的 G=0 baseline。所有诊断模型只用于维度审计，不会写回正式 Phi，也不读取任何 policy selection 或 policy performance。

运行状态：PASS。
