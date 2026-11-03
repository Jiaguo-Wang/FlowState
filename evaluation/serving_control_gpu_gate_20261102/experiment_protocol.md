# 两生命周期GPU正确性验证协议

预先固定顺序：A_shadow_validation → B_serving。只用FlowState-Greedy、C/K3、原四Pending顺序；最多两次Engine启动，不重试失败臂，不扩展三策略或性能矩阵。

冻结来源：`p0_control_cycle_gpu/arms/m1_ck3_fs` 的十份原输入、映射、CPU门禁和Cold文件；SHA逐项与 `control_path_separation_20261101/execution_lock_gpu1.json` 比较。模型、Tokenizer、模板、SGLang v0.5.17、TP=1、28槽、K=3不变。

目标GPU：GPU1，UUID `GPU-051c9abc-864a-9f31-c4f7-a737aed997ff`。每臂启动前与结束后均要求目标卡无计算进程、显存<100MiB；保留利用率采样但不把历史非零利用率当成当前占用。磁盘余量>20GiB，原端口空闲；原单卡UUID入口保留。不干预GPU0或其他任务。

镜像：`sha256:1dfce9033670a84c7342a2ae922b7dfbc5cc3ead39160bf3a90847a61bd4f926`；已只读核对SGLang 0.5.17、Torch 2.11.0+cu129、Transformers 5.12.1。

新包相对CPU原型只增加：基线生成/投影/服务端校验打点、客户端持久化RPC父区间及接口校验打点、模式记录、设备入口目标路径。CPU预检另发现原投影在序列化前会遇到list/tuple表示不同；仅将Token和Slot比较改为双方tuple逐元素比较，保留值、长度、引用和全部SHA断言。原CPU包和旧报告不修改。

服务端完整生产者始终执行恰一次。原apply、scope/前沿、Bind/Preflight/每步Postcheck、FA及Recurrent SHA、Native/COW钩子、完整请求后状态及Cold/Oracle均保留。原16条预热、flush、16条重建、四Pending形状预热和正式请求协议不变。

新增计时不改变T0、T3、T4、T5位置。包编码前原始JSON字节数与压缩后字节数由原codec直接记录；线协议实际字节数由持久化response.raw实测。client.call_saved父区间包含服务端控制、传输和原持久化；server编码结束到call_saved结束也是混合尾段，不能标成纯网络耗时。若无法进一步拆分，纯socket传输记UNKNOWN。

只作正确性及计时可观测性验证。两臂各一生命周期，A还含双接口额外校验，禁止据两者差值作性能收益或生产Serving结论。跨Workflow物理血缘仍为PROVENANCE_UNVERIFIED。
