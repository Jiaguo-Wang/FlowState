# 最终报告

**判定：NO_MEASURABLE_GAIN。** 保持正确性的本次批量读取修改反而变慢；不是GPU恢复错误，也未达到LEVEL 1、2或3。

1. **原路径为什么慢：** 原十五臂复核确认，完整观察、新鲜度与内容后验为主要成本；FS在线选择约32ms，远小于约10秒控制周期。阶段与嵌套详见 phase1_analysis.md，旧131.904ms为不同Legacy协议，不可比较。
2. **实际修改：** 仅隔离实验包批量读取Recurrent组件与FA内容。原控制、求解、Epoch、K、Phi、逐项驱逐、COW和时间边界不变；完整差异见 implementation.diff 与 batched_payload.py。
3. **正确性：** 59项CPU合同/摘要测试及4项资源门禁测试通过；1次GPU新旧摘要逐调用对照通过；10个性能臂均通过原Native/Cold/FA/组件保护和Oracle检查，共11个独立生命周期，143次真实Recurrent-only驱逐。逐臂证据 verified_arm_inventory.json。跨Workflow物理血缘未另行认证，保持 PROVENANCE_UNVERIFIED。
4. **控制时间：** 五轮批量−原版配对中位数 **+1224.787ms**，批量版0胜、5负，未下降。
5. **请求恢复：** 选集及J全等；E=[0,5120,6144,7168]、G=[4096,0,0,0]、实际Prefill=[4128,32,32,32]。命中的B/C/D各有Native COW，A无Recurrent命中；四请求8 Token输出均与冻结独立Cold相同。TTFT/CUDA实测波动见完整表，不因语义一致宣称时延不变。
6. **完整周期：** 五轮配对中位数 **+1402.973ms**，批量版0胜、5负，未改善。
7. **相对基线：** 预设的“控制改善后才追加三策略”条件未满足，因此没有启动额外15臂；本轮不能声称优于Marconi-Evict Adaptation或KVFlow-inspired。
8. **仍有取证成本：** 全树FA内容SHA、逐组件SHA、前后对照、Journal序列化/fsync、大响应RPC、请求后审计与Oracle均实际保留，未从结果扣除。
9. **必要Runtime成本：** 静止Scope、真实FA前沿、组件/槽位身份及引用/别名/容量账本、严格矩阵、在线选择、Apply-time重验和逐项安全协调仍不可省略。未证明当前内容SHA都可被轻检查替代。
10. **优化机会与限制：** 本次减少复制次数却增加主机重排/暂存，全量FA分组又未减少复制次数；没有单独测得重排/PCIe/SHA的份额。后续若获授权，可基于这两个具体瓶颈评估更合适的布局/读取实现；本任务不继续搜索参数、改安全合同或扩展实验。

## 异常与证据完整性

首对批量臂完成后，新增启动器把历史采样利用率4%误判为当前非空闲。原日志保留，未重跑、未删负结果；只修正收尾判据，启动前门禁不变。4项CPU测试确认仍拒绝真实进程/显存占用。原execution_lock与新execution_amendment分别保存，详见 resource_gate_diagnosis.md。首对原收尾进程字段缺失，后续独立清理证据补充该范围，不能冒称原瞬间字段完整。

全部旧十五臂源文件及原控制源码哈希重新核对不变；新十臂输入逐字节等于冻结来源。原始GPU日志、计时、Cold输出、物理状态与清理均在 arms/；完整五轮差值、分布、阶段归因见 performance_analysis.md，源数据为 performance_results.csv、pending_results.csv、paired_comparison.csv。

本轮完成后停止，不推进其他实验或论文修改。
