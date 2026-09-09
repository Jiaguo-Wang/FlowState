# RQ2 冻结恢复模型独立审计

本报告从冻结系数、held-out 逐行预测和实际下游 evaluator 重算，未使用论文草稿作为真值。

canonical formula = 37.828150 g + 0.345974143 g*tau - 0.156201917 g^2
theta1 = 37.828149857892583
theta2 = 0.345974142813189
theta3 = -0.156201917411720

RQ2 metrics reproduction = PASS
RQ2 vs RQ3 coefficient consistency = PASS
Exact OPT objective consistency = PASS
historical artifacts modified = NO

## 重算结果

- held-out MAPE：2.682480422900%
- held-out MAE：31.179582912686 ms
- max relative error：8.179468552928%
- max absolute error：80.726853800266 ms
- 结构网格：147/147，PASS
- Phi(0,T)=0：PASS
- 非负性：PASS
- 固定 T 单调性：PASS
- 有效域：0 <= G <= T <= 131072

## 下游一致性

- OpenHands 冻结快照：168 个，身份一致：True
- AgentX 正式快照：23 个，身份一致：True
- full-precision fit 与部署序列化逐字节相同：False；差异仅为冻结发布精度舍入，数值门限为 5e-07
- RQ4 复用 RQ3 selector/objective：True
- RQ5-A Full 复用 canonical selector/objective：True

最终状态：RQ2_CANONICAL_MODEL_READY
