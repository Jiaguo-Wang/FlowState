# Task A6 接管核查

本文件是接管进度，不是已完成的策略比较报告。

## 已完成

- 初始 Git 状态：没有已跟踪文件修改；已有多组未跟踪研究目录，保持不变。目标 A6 目录原先不存在，本轮新建。
- 16 组输入的 32 个 token 文件摘要均匹配冻结 manifest，长度、真实 LCP 和 token 类型核验通过。处理顺序首先 lmcache_10，再按案例 ID；这只是身份核对顺序，未擅自替代原 A6 的样本规则。
- PCB、LDR、WFC、common.py 均匹配 implementation_lock.json。
- A5 input_identity.json 指向的两输入摘要匹配本次 lmcache_10。A5 runtime_evidence.json 记录 PASS，报告确认 C0=9984 的物化和两请求恢复。未运行新的 Runtime 验证。
- 来源摘要见 source_integrity_before.json，逐组身份见 input_identity_check.json；复现入口为 preflight.py，只读来源并写本目录。

## 策略规则核对

|策略|真实规则|来源|
|---|---|---|
|FlowState|每轮按总成本下降选最大边际项；1e-9 ms 比较容差，平手 checkpoint ID；最大收益不正停止|flowstate/optimizer.py:89–147|
|PCB|逐请求按 Phi 选最大正收益项，平手 ID；推荐列表按收益、候选 ID、请求 ID 排序，去重直到 K|evaluation/p1_strong_baseline_validation/pcb.py:5–22|
|LDR|逐请求推荐最深正位置项；推荐列表按位置、候选 ID、请求 ID 排序，去重直到 K|evaluation/p1_strong_baseline_validation/ldr.py:4–17|
|WFC|动态 E；按取得正前沿推进的请求数、总 token 推进、候选 ID 排序；覆盖数为零停止|evaluation/p1_strong_baseline_validation/wfc.py:4–21|
|StandaloneTopK|静态求 F({c})，过滤不超过 1e-9 ms 的项，再按收益与候选 ID 取 K，返回 ID 排序集合|evaluation/rq3_sanity_structure_audit.py:1116–1131|
|Exact OPT|须对原规则确定的同一候选与预算求精确最小成本；当前尚未确定候选全集，未执行|待原 A6 规则|

Phi 参数与域来自 evaluation/FORMAL_RECOVERY_MODEL.md 和 flowstate/recovery_model.py:39–70，0≤G≤T≤131072，不重新拟合。

## 独立矩阵适配边界

flowstate/state_catalog.py:60–73 使用 workflow_id/lineage_path，不能认证这些跨会话输入。flowstate/executable_state.py:17–22 调用该谓词。因此不能直接虚构共同 workflow 或 lineage 使原 select 接受这些边。

后续必须用完整冻结整数数组判断 prefix，摘要只用于身份与索引，不能仅靠摘要取代实际比较。TOKEN_COMPATIBLE(c,p) 要求 c 的源输入前 L 个 token 与 p[:L] 相同且 L≤len(p)；用于目标函数还必须满足 L≤T_p。T_p 的规则须来自原任务，不能默认为完整输入长度或者 A5 的 FA frontier。

可独立实现矩阵上的 E=max L、G=T−E、C=sum Phi，保持原 greedy 比较和停止条件；PCB/LDR/WFC 的冻结接口已经接受明确 compatible ID 集合，可原样调用。适配一致性必须另行检验，不得把矩阵选择结果宣称为核心 Runtime 直接可执行。

## 证据等级

- 16 组冻结输入有内容级 exact prefix 证据。
- lmcache_10、C0=9984 有 A5 历史恢复认证；C5 generation 仍 UNKNOWN。A5 证据内 gate.unknown_conditions 还保留 C7_resume_certified 字样，而 gate.runtime_resume_certified 为 true；应引用各阶段恢复记录与报告，不能把混合字段笼统称为全部无条件通过。
- 新增深层 checkpoint 的位置和 eligibility 未获授权定义；不得由 token 长度推断已物化。
- A5 的物理池配置和恢复成功不能证明任何情景 K 的物理容量竞争，也不能证明全部拟议候选同时驻留。

## 当前唯一阻塞

用户要求沿用原 A6，但当前会话未提供其候选、planning target、预算及其余样本规则。已搜索附件文本、研究目录和既有 A5 报告，未找到原 A6 全文或输出。已请求补充；暂不生成任意候选、收益矩阵或性能结果。

缺少规则时，只有一个共同 C0 的情景与含分支深层状态的情景会得到不同的 K_min 与竞争结论；自行选其中一种会实质重新设计实验，违反续跑要求。补充原规则后从此进度继续，无需重新准备 16 组数据或重跑 A5。
