# 正式位置感知恢复模型

FlowState 的正式恢复成本模型为 `position_aware_quadratic_v1`。该模型由独立制品
`recovery_model_freeze_20260826_154235_266020` 的预注册选择与 held-out GPU 验证确定。

令 `g = G / 1024`、`t = T / 1024`，其中 `G` 是 recovery gap token 数，`T` 是当前
待续请求在本次决策快照中的 planning target token 数。正式模型为：

```text
Phi(G,T) = 37.828150*g + 0.345974143*g*t - 0.156201917*g^2
```

输出单位为毫秒，正式验证域为 `0 <= G <= T <= 131072`。`Phi(0,T)` 精确为零；实现
拒绝负值、`G>T` 和超出验证域的目标位置，不做截断或外推。

## 数学结构

固定 `t` 时，对 `g` 的导数为：

```text
dPhi/dg = 37.828150 + 0.345974143*t - 2*0.156201917*g
```

在 `0 <= g <= t <= 128` 上，导数下界为正，因此成本非负且随 recovery gap 严格增加。

对单个待续请求 `p`，`T_p` 在一次 allocation snapshot 中固定。定义兼容候选 `c` 的
收益：

```text
b_p,c = Phi(T_p,T_p) - Phi(T_p-L_c,T_p)
```

最深兼容 checkpoint 产生最低成本，因此：

```text
F_p(S) = max_{c in S compatible with p} b_p,c
```

原有 monotone max-coverage 与 submodular 结构保持不变，GlobalOptimizer 的贪心逻辑
无需修改；只有恢复成本调用由 `Phi(G)` 升级为显式的 `Phi(G,T_p)`。

历史 WP2 单变量剖面仅通过 `HistoricalRecoveryCostModel` 保留，用于旧制品复现，不能作为
正式 optimizer 的默认目标。
