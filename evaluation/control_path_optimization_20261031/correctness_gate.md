# 正确性门禁

## CPU及静态门禁：PASS

冻结镜像内无GPU、无网络、源码只读挂载执行 unittest：59项全部通过，原始输出 cpu_tests.log。包含5项新增批量摘要测试及54项生产 Strict Epoch/Strict Control 回归。

新增测试覆盖 BF16/FP16/FP32、连续/非连续布局、空FA区间、乱序/重复FA索引、选择子集摘要、每次调用间内容变化、负数/越界槽位、未完成传输与量化池拒绝；同一真实CPU张量逐字节对照原读取函数。冻结 C/K3 16候选经真实生产 plan_strict 两次结果相等、K=3与13项补集一致。原严格测试覆盖 UNKNOWN、身份/摘要冲突、映射/组件别名、活动引用、陈旧视图、槽位记账及部分驱逐失败的 fail-stop。

优化不引入跨调用快照复用，不改变 scope、physical observe、bind/preflight/reconcile。GPU Smoke 开启 FS_OPT_VERIFY=1：真实每次批量读取额外计算旧摘要并要求相等；全部原 Native/Cold/FA/组件与计时有效性检查保持。Smoke 不进入性能样本。

## GPU门禁

尚待执行，不能由CPU门禁推定已通过。

## GPU正确性Smoke：PASS

arms/smoke_fs/run/summary.json、reconciliation.json 及 gpu_cleanup.json 为原始证据。生产入口实际执行 Marginal Greedy；选集为 C_B_P05120、C_C_P06144、C_D_P07168，J=154.34895561599998模型化ms；13次真实Recurrent-only驱逐全部通过。四个Pending实际E=[0,5120,6144,7168]、G=[4096,0,0,0]；原有Native/COW、Cold逐Token、FA/选中组件保护及Oracle合同全部通过。10次实时读取逐调用新旧SHA相等。单卡生命周期正常退出，资源复查无计算进程；日志保留原生退出CUDA IPC警告，该警告不代替资源证据。

Smoke包含双份读取，控制15416.079ms、完整28425.887ms不作为性能样本。正式配对关闭额外双算，仍保留全部原始检查与审计。

## 全部五轮执行后核验：PASS

十个性能臂与一个Smoke的原始摘要及物理操作独立复核见 verified_arm_inventory.json：全部13项驱逐成功、最终3个驻留、所选组件内容不变、E及Prefill数量相同、逐Token Cold通过、控制及请求测量窗口无新Triton内核。首对一项收尾门禁异常保留且另记，详见 resource_gate_diagnosis.md；后续4项CPU资源门禁测试通过。

最终GPU核验 gpu_cleanup_final.json：GPU0 14MiB、GPU1 17MiB，均0%且无计算进程，无本轮实验容器残留；根盘约50GiB可用。没有重跑、删除或排除首对负结果。
