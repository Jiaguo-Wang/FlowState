# RQ4-C 正式端到端运行时采集报告

- 状态：RQ4_RUNTIME_FORMAL_READY
- 计划 run：216
- 完成 run：216
- 有效 run：216
- 完成 pending request：864
- correctness：PASS
- source integrity：PASS
- GPU 最终清理：PASS
- recovery latency telemetry：PARTIAL；未用 TTFT 代填。
- 正式统计单位：snapshot；本文件中的 policy 均值仅为 preliminary aggregate。

## Preliminary request-level TTFT

- LRU：requests=288，mean=164.70500180555555 ms，median=170.6718785 ms
- Marconi：requests=288，mean=129.09717030555555 ms，median=81.415031 ms
- FlowState：requests=288，mean=94.02458018402778 ms，median=55.2306155 ms
