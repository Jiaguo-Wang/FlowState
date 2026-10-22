# E1 冻结执行协议

仅固定B5-C2四Session、八条历史、目标C、K=2、Qwen3.5-9B、SGLang v0.5.17、TP1、28槽。8输出Token，temperature=0、ignore_eos=True。来源血缘保持PROVENANCE_UNVERIFIED。

先执行一个轻量Exact正确性Smoke及独立Cold。Smoke额外在协调后、Pending前核验全部FA内容和保留组件；不纳入性能样本。任一错误停止，不重试、不补位。

四条件：A=重型Greedy，B=重型Exact，C=轻量Greedy，D=轻量Exact。固定3轮顺序：A B C D；D C B A；C A D B。每臂独立Engine。总计最多14生命周期（Smoke、Cold、12性能臂）。固定三次而非根据结果扩至五次。每轮全部成功且可比才继续，所有失败保留。

每个性能臂在控制计时前执行独立重型参考RPC，核验内容与两种计划，成本另计并报告从该审计开始到请求完成的总Wall。轻量控制重新观察并认证当前Epoch，不能将参考状态当作当前绑定。观察、规划、重验、正式协调全部在同一Scheduler同步窗口内；不使用generation或跨RPC租约。保留生产bind/preflight/逐项postcheck与原Adapter。没有在未覆盖变更入口的前提下使用缓存代际。

重型控制直接调用D4a原apply，重型请求观察保持原逻辑。轻量控制只移除Conv/Temporal和FA数据内容哈希、独立Oracle、Journal；仍完整读取索引与元数据。为了保守保证，暂保留原索引摘要、两次元数据观察、重复生产验证，不开发增量维护器。轻量请求保留原生Match/COW身份、事件计时及Admission元数据检查，但内容哈希与写盘转移到请求完成后；Smoke另保留COW内容对拍。实际新路径计时，不从旧时间减去估计值。

两种路径均有前置内容参考与请求后保留组件/非重算FA内容检查；重型每步多次哈希与轻量元数据安全门禁具有不同观测强度。轻量安全依赖已审计Native原语、完整切片身份、串行窗口与逐项物理后置检查；不承诺检测绕过全部调度入口的静默硬件内存破坏。Smoke独立检查FA完整内容，性能臂不把后验一致性冒充逐次内容认证。

主计时：控制RPC Wall（请求发送到返回）、内部观察阶段（并非独立RPC，明确标为阶段Wall）、Epoch认证、求解、Plan验证、Apply检查、协调；Prefill同流CUDA事件；客户端TTFT、Request Wall；Control-to-Completion（控制RPC开始至流结束）。额外报告前置观察RPC Wall及Audit-inclusive总量。请求后内容对拍不计入请求完成，但单独记录。内层Eager、外层run_batch、COW存在嵌套，不相加。每次事件保留同步边界；首次编译、同步和CPU提交空隙可能影响事件值，不称纯内核耗时。

资源门禁：GPU0无其他计算任务、H100 PCIe、镜像与文件SHA锁、独占容器、只清理本实验容器；每臂3次资源恢复检查。任何正确性、可比性、计时或清理门禁失败即停。无性能优劣筛选，无统计显著性或生产P95主张。
