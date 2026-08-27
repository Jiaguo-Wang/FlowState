# TraceLab 受限预算代表案例

本文件只读取 Step 10E 冻结结果与 Step 10F 审计 artifact，没有重新运行策略、修改模型、重采样或调用 GPU。

## 确定性选择规则

- 只考虑 25%、50%、75% 预算中 FlowState 成本严格低于 Marconi 且 selection 不同的冻结记录。
- 优先使用 N<=24 且 pending<=6 的规模适中案例，再按 absolute reduction 降序、N、pending 数和 snapshot_id 排序。
- 四个 snapshot 不重复；50% 与 75% 案例要求 X>=4，额外 X>=4 案例固定来自 25% 紧预算。
- 该规则只用于解释案例，不改变 Step 10E aggregate、policy selection 或 protocol。

## 案例摘要

| 类别 | Snapshot | X | N | K | Budget | Marconi cost | FlowState cost | Absolute reduction | Relative reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% budget | `c128-small-claude-round-28601` | 2 | 7 | 1 | 25% | 6139.711 ms | 878.081 ms | 5261.630 ms | 85.698% |
| 50% budget | `c128-medium-claude-round-18027` | 4 | 12 | 2 | 50% | 6013.702 ms | 2260.883 ms | 3752.819 ms | 62.404% |
| 75% budget | `c128-medium-claude-round-67076` | 4 | 24 | 3 | 75% | 2299.067 ms | 545.895 ms | 1753.172 ms | 76.256% |
| X>=4 constrained budget | `c128-medium-claude-round-94304` | 4 | 16 | 1 | 25% | 3353.607 ms | 2780.973 ms | 572.634 ms | 17.075% |

## 25% budget：`c128-small-claude-round-28601`

- X=2，N=7，K=1，budget=25%
- Marconi selection：`claude:claude:2a70b8da-34ac-e10e-e7ea-f3182a42d061:run:000001:checkpoint:round:00000000`
- FlowState selection：`claude:claude:3cc6b26c-1e50-f02a-0195-a321ef73822e:run:000005:checkpoint:round:00000057`
- Marconi total cost：6139.711 ms
- FlowState total cost：878.081 ms
- Absolute reduction：5261.630 ms
- Relative reduction：85.698%
- 机制说明：冻结 selection 显示 FlowState 多覆盖 1 个 exact-parent demand，让 1 个 pending 获得更深 E，同时有 1 个 pending 的 E 较低；这是结构性描述，不是因果结论。

| Continuation | T | Marconi E | Marconi G | Marconi Phi | FlowState E | FlowState G | FlowState Phi |
|---|---:|---:|---:|---:|---:|---:|---:|
| `claude:claude:2a70b8da-34ac-e10e-e7ea-f3182a42d061:run:000001:pending:round:00000001` | 21504 | 20899 | 605 | 26.588 ms | 0 | 21504 | 878.081 ms |
| `claude:claude:3cc6b26c-1e50-f02a-0195-a321ef73822e:run:000005:pending:round:00000057` | 108164 | 0 | 108164 | 6113.123 ms | 108164 | 0 | 0.000 ms |

## 50% budget：`c128-medium-claude-round-18027`

- X=4，N=12，K=2，budget=50%
- Marconi selection：`claude:claude:c839369a-dcd7-6cda-5f47-2b33963bc82d:run:000012:checkpoint:round:00000095`, `claude:claude:3697c055-75a9-15b7-76f9-e3c7ea118318:run:000001:checkpoint:round:00000000`
- FlowState selection：`claude:claude:c839369a-dcd7-6cda-5f47-2b33963bc82d:run:000012:checkpoint:round:00000095`, `claude:claude:4c425f1b-635e-78d1-9649-140186150b7a:run:000010:checkpoint:round:00000067`
- Marconi total cost：6013.702 ms
- FlowState total cost：2260.883 ms
- Absolute reduction：3752.819 ms
- Relative reduction：62.404%
- 机制说明：冻结 selection 显示 FlowState 多覆盖 1 个 exact-parent demand，让 1 个 pending 获得更深 E，同时有 1 个 pending 的 E 较低；这是结构性描述，不是因果结论。

| Continuation | T | Marconi E | Marconi G | Marconi Phi | FlowState E | FlowState G | FlowState Phi |
|---|---:|---:|---:|---:|---:|---:|---:|
| `claude:claude:3697c055-75a9-15b7-76f9-e3c7ea118318:run:000001:pending:round:00000001` | 28284 | 26144 | 2140 | 98.344 ms | 0 | 28284 | 1189.637 ms |
| `claude:claude:4c425f1b-635e-78d1-9649-140186150b7a:run:000010:pending:round:00000067` | 90767 | 0 | 90767 | 4844.112 ms | 90767 | 0 | 0.000 ms |
| `claude:claude:c839369a-dcd7-6cda-5f47-2b33963bc82d:run:000012:pending:round:00000095` | 118093 | 118093 | 0 | 0.000 ms | 118093 | 0 | 0.000 ms |
| `claude:claude:cb3adab6-0c60-380d-3403-c4a7905333ea:run:000001:pending:round:00000001` | 25750 | 0 | 25750 | 1071.247 ms | 0 | 25750 | 1071.247 ms |

## 75% budget：`c128-medium-claude-round-67076`

- X=4，N=24，K=3，budget=75%
- Marconi selection：`claude:claude:70a26fe3-480b-7608-82fa-33ebaef9a10c:run:000004:checkpoint:round:00000009`, `claude:claude:418037d3-5ff1-71e4-2f2f-1ee569fae523:run:000001:checkpoint:round:00000000`, `claude:claude:418037d3-5ff1-71e4-2f2f-1ee569fae523:run:000001:checkpoint:round:00000003`
- FlowState selection：`claude:claude:70a26fe3-480b-7608-82fa-33ebaef9a10c:run:000004:checkpoint:round:00000015`, `claude:claude:a8e981b6-991f-311e-6e2d-453adda9f0de:run:000041:checkpoint:round:00000255`, `claude:claude:05f731c7-2612-7142-5a1a-4f90101cfb4e:run:000001:checkpoint:round:00000010`
- Marconi total cost：2299.067 ms
- FlowState total cost：545.895 ms
- Absolute reduction：1753.172 ms
- Relative reduction：76.256%
- 机制说明：冻结 selection 显示 FlowState 多覆盖 3 个 exact-parent demand，让 3 个 pending 获得更深 E，少保留 1 个无 E 增量的冗余 checkpoint，同时有 1 个 pending 的 E 较低；这是结构性描述，不是因果结论。

| Continuation | T | Marconi E | Marconi G | Marconi Phi | FlowState E | FlowState G | FlowState Phi |
|---|---:|---:|---:|---:|---:|---:|---:|
| `claude:claude:05f731c7-2612-7142-5a1a-4f90101cfb4e:run:000001:pending:round:00000010` | 18941 | 0 | 18941 | 764.639 ms | 18941 | 0 | 0.000 ms |
| `claude:claude:418037d3-5ff1-71e4-2f2f-1ee569fae523:run:000001:pending:round:00000003` | 13839 | 13839 | 0 | 0.000 ms | 0 | 13839 | 545.895 ms |
| `claude:claude:70a26fe3-480b-7608-82fa-33ebaef9a10c:run:000004:pending:round:00000015` | 32361 | 27178 | 5183 | 242.807 ms | 32361 | 0 | 0.000 ms |
| `claude:claude:a8e981b6-991f-311e-6e2d-453adda9f0de:run:000041:pending:round:00000255` | 30428 | 0 | 30428 | 1291.621 ms | 30428 | 0 | 0.000 ms |

## X>=4 constrained budget：`c128-medium-claude-round-94304`

- X=4，N=16，K=1，budget=25%
- Marconi selection：`claude:claude:a6ef802a-b8c5-31e2-813f-7b9d81661e2a:run:000001:checkpoint:round:00000000`
- FlowState selection：`claude:claude:13843e27-0b10-704c-5850-125d97ce1a13:run:000003:checkpoint:round:00000009`
- Marconi total cost：3353.607 ms
- FlowState total cost：2780.973 ms
- Absolute reduction：572.634 ms
- Relative reduction：17.075%
- 机制说明：冻结 selection 显示 FlowState 多覆盖 1 个 exact-parent demand，让 1 个 pending 获得更深 E，同时有 1 个 pending 的 E 较低；这是结构性描述，不是因果结论。

| Continuation | T | Marconi E | Marconi G | Marconi Phi | FlowState E | FlowState G | FlowState Phi |
|---|---:|---:|---:|---:|---:|---:|---:|
| `claude:claude:13843e27-0b10-704c-5850-125d97ce1a13:run:000003:pending:round:00000009` | 31246 | 0 | 31246 | 1330.970 ms | 31246 | 0 | 0.000 ms |
| `claude:claude:688e4151-3679-7ee2-226e-a08de7748fbb:run:000001:pending:round:00000001` | 21859 | 0 | 21859 | 893.981 ms | 0 | 21859 | 893.981 ms |
| `claude:claude:6cfa7c68-1703-5556-f6ba-03a378008fdc:run:000004:pending:round:00000010` | 26985 | 0 | 26985 | 1128.656 ms | 0 | 26985 | 1128.656 ms |
| `claude:claude:a6ef802a-b8c5-31e2-813f-7b9d81661e2a:run:000001:pending:round:00000000` | 18797 | 18797 | 0 | 0.000 ms | 0 | 18797 | 758.336 ms |

## 原 Step 10F 的 100% 案例角色

以下原案例不删除，但统一标记为 **demand-sufficient sanity examples**，不作为 main benefit examples：

- `c128-medium-claude-round-255147`，budget=100%，K=X
- `c128-medium-claude-round-182500`，budget=100%，K=X
- `c128-medium-codex-round-333524`，budget=100%，K=X

这些案例验证预算足以覆盖全部 distinct exact-parent demands 时 FlowState 应得到 G=0；它们不代表受限预算下的核心收益。
