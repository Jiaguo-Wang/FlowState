# 理论结构兼容性

推荐候选：M2。

在单次 allocation snapshot 中，每个 pending continuation 的目标位置 T_p 固定。兼容 checkpoint 集合决定最深 executable frontier E_p(S)，因此 G_p(S)=T_p-E_p(S)。

只要 Phi(G,T_p) 对固定 T_p 随 G 单调不减，选择更深 compatible checkpoint 就不会增加恢复成本。相对空集合的恢复收益可写为各 compatible checkpoint 单点收益的最大值，因此仍保持 max-coverage 的单调次模结构。

SUBMODULAR_STRUCTURE_PRESERVED：YES。

本检查仅验证数学兼容性，没有修改 optimizer。
