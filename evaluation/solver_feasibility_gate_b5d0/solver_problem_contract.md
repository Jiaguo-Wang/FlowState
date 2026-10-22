# 冻结问题合同

## 目标及边界

固定Epoch、候选C、已知Pending P、T、严格边与Phi：E_p(S)=max({L_c:c∈S且(c,p)准入}∪{0})，J(S)=ΣPhi(T_p−E_p(S),T_p)，F(S)=J(∅)−J(S)=Σ_p max_{c∈S} b_pc。b_pc仅准入边可为正，REJECTED为0；UNKNOWN必须先拒绝形成完整实验输入，不能通过换求解器消除证据缺口。优化约束|S|≤K，K=1–4。

模型沿用position_aware_quadratic_v1，a=37.828150、b=0.345974143、c=−0.156201917，G/T按1024归一化。目标是模型化恢复成本，不是CPU求解时间、GPU TTFT或物理显存效率。所有需求来自冻结的Controlled Pending；没有读取后续请求结果作为新增在线信息。

## 五个固定输入

|快照|候选/请求|共享候选/不同共享域|前缀位置判定与严格边差异|Laminar|一槽一候选且槽互异|
|---|---|---|---:|---|---|
|g001|12/4|1/1|0|True|True|
|g002|16/4|1/1|0|True|True|
|g003|20/4|1/1|0|True|True|
|g004|8/4|1/1|0|True|True|
|b5c3|7/4|4/3|0|True|True|


B4-C的48+64+80+32条和B5-C3的28条共252配对逐项检查。没有新矩阵、候选删减或预算调整。严格矩阵来自Runtime证据，不意味着逐边RESUME_VALIDATED。B4-C四个快照来自受控回放；B5-C3是Trace-derived Controlled Composition。20预算条件只属于5个固定快照，不是20个独立工作负载。

B4-C等大小合同由冻结load及B5-C0逐项核查支持；B5-C4通过实际Conv/Temporal切片核验七候选各51511296字节、一个独占槽。逻辑K并不代表物理池耗尽。Alias或其他联合约束若存在，不能继续使用这个纯基数可行域。

## 单共享与多共享

B5-C0的四组只有一个非单例正支持候选；固定是否选它后，两分支Top-K具有已证明的精确性（exact_decomposition_proof.md）。B5-C3则有四个共享候选、三个不同共享域，不能套用单共享公式。N27支配N21但全部七候选仍在本轮搜索。

Greedy给定一次执行只能不断增加已有选集。B5-C4最优集合K1={N20}、K2={N18,N27}、K3={N22,N24,N27}、K4={N22,N24,N25,N27}，并不嵌套。首步最优不能保证是后续预算最优解的子集；原Greedy没有回退机制。这里只是冻结反例，不声称所有共享结构均使Greedy失败。

证据：B4-C/results.json及evaluate.py；B5-C0/verification.json、exact_decomposition_proof.md；B5-C4/results.json、exact_verification.json、validation.json。完整路径和SHA见integrity.json与report_sources.json。
