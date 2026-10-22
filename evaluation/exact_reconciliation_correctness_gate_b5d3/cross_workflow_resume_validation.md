# 单目标跨Workflow恢复

本轮目标 `B5C2_B_TURN_003`，来源Checkpoint `B5C3_NODE_18`。来源Session：`nebius-swe-rebench-openhands::chatcmpl-71420ffc5b410547d43ac67939b0441c`；目标Session：`nebius-swe-rebench-openhands::chatcmpl-71b073b89097665d14501919b224c26b`，二者不同。

当前来源最早在历史 `B5C2_A_TURN_001` 完成后的Census中出现；原始记录位于strict/history_*.json，包含实际提交与原生Match接收Token身份。来源记录不是原生Fork事件，不将同任务重复执行称为自然在线分支。首次Census不是精确insert时间戳，但本轮顺序回放及此前空缓存/各步Census给出物化来源边界。

Apply后所选源Node=18，Slot=[3]；Native实际Match Node=18，源Slot=[3]，工作Slot=[19]。实际H/E/G=4096/3712/384。恢复前沿来自执行路径，不只是计划计算。完整输入长度4860，实际Prefill从E到完整输入末端，共1148 Token；模型恢复缺口G只计E到Planning Target/FA frontier，不能把两者混为同一计算长度。

Admission门禁在实际process_input_requests之前核对Token、完整节点、缓存语义、Selected内容、旧FA映射与保护区域内容。协调窗口结束后没有假定跨RPC租约；独占Engine只提交本目标，真实Admission重新核验。

COW原函数返回、目标槽复制后内容等于源槽复制前内容，源槽前后不变；工作槽与全部Selected槽互斥。Prefill/Decode原run_batch均返回。原始证明见strict/native_trace.json、strict/post_request.json及strict/summary.json。

Strict与独立干净Cold采用相同完整Token、temperature=0/max_new_tokens=8/ignore_eos=True；逐Token与停止行为检查见output_correctness.json。Cold原生H/E为0/0。没有将Cold完整prefill和严格恢复做性能比较，也没有执行其他正式Pending。

仅此边及本次恢复获得运行支持，不升级其余EPOCH_ADMISSIBLE边为RESUME_VALIDATED。
