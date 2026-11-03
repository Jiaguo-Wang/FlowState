# 简洁能力与缺口清单

| 项目 | 已有/本次复用 | 当前缺口 |
|---|---|---|
| KVFlow ASG/STE | 原policy.py，Ready/AND/OR、UNKNOWN；原14个unittest复用通过 | 不含原系统调度/预取 |
| Fixed/Dynamic/共享 | 原逐Token映射、动态inf、共享min | Strict20无ASG和声明，不能人工补图；新Engine仍须重绑定 |
| Marconi效用 | 原MarconiStylePolicy，Recency+α效率、min-max和排序 | Token代理≠真实GDN FLOPs；累计L≠全部父相对语义 |
| Marconi内存 | 本次读既有Census得到每项51,511,296 bytes | 不能用整池大小；等大小归一化下选集不变 |
| α | 既有预注册1.0 | 无作者bootstrap调参认证，保持适配标签 |
| 共同CPU入口 | interface.py仅转接既有算法；18/18回归 | 声明输入不可当StrictAllocationPlan直接执行 |
| 旧GPU | 24快照旧Marconi；C/K3 FS/ST/WFC；60正式FS/ST/WFC | 不能拼接成已经完成三外部策略严格GPU对照 |
| 新GPU | 原控制器可复用，不开发新Runtime | GPU0有活动进程；本轮不启动；正式多轮协议待审 |

旧数据和生产文件未改。路径、明细和统计口径见final_report.md及source_identity.json。
