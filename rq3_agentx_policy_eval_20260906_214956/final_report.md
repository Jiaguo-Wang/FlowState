# Step 13G-C AgentX Formal Policy Evaluation

- 状态：`AGENTX_RQ3_EVAL_READY`
- snapshots：`23`
- cases：`69`
- formal max d_t(c)：`2`
- 使用 B2 raw compatibility：`否`
- shared candidates：`31`
- Exact tractable：`48`
- FlowState == Exact：`48`
- determinism：`通过`
- source integrity：`通过`

## 各预算 FlowState 相对基线

### 25%

- 相对 LRU：mean reduction=0.1385753412674689，95% CI=[0.05373256382006086, 0.2362878256269457]，win/tie/loss=11/12/0
- 相对 LFU：mean reduction=0.08595121372396097，95% CI=[0.02065979917953812, 0.17074200612040955]，win/tie/loss=7/16/0
- 相对 Marconi：mean reduction=0.05002254566552837，95% CI=[0.004418757343063213, 0.1248742382254421]，win/tie/loss=12/11/0

### 50%

- 相对 LRU：mean reduction=0.05179697821554721，95% CI=[0.0, 0.13552049990653203]，win/tie/loss=3/20/0
- 相对 LFU：mean reduction=0.02252403116497459，95% CI=[0.0014327934364252095, 0.05428613727860372]，win/tie/loss=4/19/0
- 相对 Marconi：mean reduction=0.0025111346326042597，95% CI=[0.0, 0.00652674483159455]，win/tie/loss=3/20/0

### 75%

- 相对 LRU：mean reduction=0.03390059581957585，95% CI=[0.0, 0.10104377001068147]，win/tie/loss=2/21/0
- 相对 LFU：mean reduction=0.003585374781463681，95% CI=[0.0, 0.010756124344391044]，win/tie/loss=1/22/0
- 相对 Marconi：mean reduction=0.0006580174480460853，95% CI=[0.0, 0.0019740523441382557]，win/tie/loss=1/22/0

## AgentX 跨 pending 结论

存在 OpenHands 正式快照中没有出现的跨 pending 边际依赖

- artifact root：`rq3_agentx_policy_eval_20260906_214956`
