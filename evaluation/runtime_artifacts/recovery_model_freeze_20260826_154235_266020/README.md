# 预注册恢复模型选择与独立留出验证

Calibration 参数只来自 Step 10D.2 的 16 个 position-matrix 点与 Step 10D.1 的 4 个 long-gap 点。独立留出 GPU 数据不参与拟合，也不会改变候选公式。

候选模型仅为 M0 gap-only 分段线性、M1 position-aware bilinear、M2 position-aware quadratic。正式 Phi、policy 和 TraceLab protocol 均未修改。

运行状态：PASS。
