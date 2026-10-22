# Gate 0：PASS

59项CPU测试通过：54项既有Strict Epoch/Control测试及5项本轮合同测试。新测试遍历两组选集、八个Pending来源，并用全新模拟Node/Slot验证Bind、Preflight、正式reconcile_strict四项驱逐和三个保留项。覆盖引用、别名、组件缺失、状态变更、Admission、FA损坏、部分失败、UNKNOWN、错误预算和Native来源不匹配。首轮测试错误仅是UNKNOWN在构造时已抛错而断言包围过晚；原日志保留，未放宽代码。

实验层必要适配：E1 control副本仅将K2改K3、规划器替换为本目录适配接口、参考策略改Exact/Swap；Exact仍使用B5-D1原枚举，Swap只读取E3冻结选集，按完整Token前缀映射到当前逻辑候选，再调用既有finish/validate_plan。不在Runtime中重算局部替换，不称生产求解器。

E4来源合同原有support>1属于共享案例限定，不是安全条件；本轮允许support>=1的合法专属来源，仍要求唯一最深选中来源、严格边、L≤T、完整Token一致及当前Native来源绑定。预算改K3；复核选集和模型J。Review将驱逐数量5改4；运行路径不跳过引用、组件、池记账、连续隔离、逐项后验、Admission、COW与FA检查。

Bridge仅在独立实验进程接线E1 dispatcher到本目录control，保持其同步控制窗口及Native观测；重型参考全内容、Smoke控制后内容、请求后内容仍按E4执行。原生产代码及全部既有实验文件只读。运行计时和客户端证据保存保持原样。

CPU仅认证接口与模拟绑定，不能复用历史物理映射认证新GPU。每个新Engine必须重新Observe/Bind/Preflight和Native匹配。4个Cold＋8个轻量Smoke全部通过后，才执行24个正式测量臂；顺序见experiment_protocol.json。资源或正确性失败即停，无自动重试。
