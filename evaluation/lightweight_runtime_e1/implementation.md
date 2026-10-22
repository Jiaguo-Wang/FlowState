# E1 最小实现及安全边界

所有新代码在本目录；生产Optimizer、Controller、Adapter及SGLang安装源码均未修改。重型控制直接调用冻结D4a.control.apply，原函数不复制修改。请求观测复制D4a后只在轻量模式分支改变审计载体。

- light_capture.py：复用R2-C0元数据观察、B4 build_edges；保留双读元数据、组件切片身份、全部引用、FA索引摘要、Slot Accounting、真实FA Frontier；去掉张量内容payload读取。
- control.py：独立重型参考RPC；轻量当前Epoch认证、原D1规划、选择来源合同、apply重读、原bind/preflight/StateController.reconcile_strict/RecordedAdapter及每次postcheck。没有mutation_generation，没有跨调用缓存或租约。错误保持Scheduler失败隔离，原始控制响应保存。
- bridge.py：保持真实Req/Match/COW入口；轻量Admission读取元数据对照；每次COW记录真实源/工作槽和CUDA完成。轻量性能期间不读张量内容、不逐事件写盘；Smoke保留COW内容对拍；请求结束后审计历史所选组件和非重算FA内容。Full Metadata快照仍含节点Token与索引摘要，并非真正增量或常数成本观察。
- observation.py：原D4a同流CUDA事件包装，原函数恰好执行一次；轻量模式persist为空操作，事件及同步不取消。
- runner.py：独立Engine、8历史、独立前置参考、实际控制、固定C请求、请求后内容审计、8Token对照。前置参考成本和包含它的整体Wall单独报告，不把移出计时区间等同成本消失。
- review.py、launch.py：Smoke+Cold门禁、四臂3轮固定顺序、逐臂完整性与清理、失败即停。
- test_cpu.py：新轻量合同8项、原生产严格控制28项、D4a来源/失败保存30项。无Torch环境中仅模拟同步接口，元数据输入为冻结真实记录；不以模拟结果声称当前GPU状态已认证。

## 尚保留的重复成本

生产Strict Epoch/Plan重复验证、完整树元数据遍历与索引摘要均保留。它们可能仍占控制时间，但无需改动生产接口即可保证相同安全合同。不采用尚未完整覆盖Runtime变更入口的generation、注册表或验证缓存。

## 内容完整性边界

完整FA张量哈希、Recurrent内容哈希是独立认证，不是Strict Compatibility的判定字段。轻量控制安全依赖Scheduler连续隔离、真实物理身份、原生组件隔离原语与完整逐项物理后置检查；没有能力在无内容读取时发现任意静默内存比特破坏。Smoke在协调前后进行完整内容对照；各性能臂在之前及请求后对保护内容审计。性能路径保留Admission元数据验证；不把已结束窗口当成跨RPC租约。
