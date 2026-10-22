# Marconi 源码审计（Phase B5-A）

对象：`raw/marconi_repo/` @ `08016617b1524e6bf6ac29b680641cc945bda7f0`（2025-03-05, "add license"，仓库最新提交）
论文：MLSys 2025《Marconi: Prefix Caching for the Era of Hybrid LLMs》
审计方式：只读；论文文本与全部关键源文件逐行核对。所有行号对应该 commit 的文件内容。

核心背景：Marconi 的官方 artifact 是**CPU 上的 trace 驱动模拟器**（`artifact_evaluation.md` 明言），不是可服务的系统。radix tree 节点携带 `HybridStates`（mamba_states + kv_cache），驱逐策略分 V1（SGLang+ LRU 叶子）/ V2（Marconi 效用驱逐）/ V3（离线静态-α oracle）三档。

---

## 审计项 1：准入条件（论文 §4.1 Cache Admission）

**论文机制**：judicious admission——两类准入：
1. **分支点状态**（branch-point state）：请求命中部分前缀时，在分叉处投机插入（speculative insertion）并缓存该点状态；
2. **最后解码 token 状态**（last-decoded-token state）：解码结束时在序列末尾 checkpoint 状态。
每序列 SSM 状态数上限 2。

**源码证据**（`radix_cache_hybrid.py`）：
- `match_prefix` L220-263：`branchoff_required = prefix_len > len(prefix_token_ids)`（L247）——命中前缀超出既有已缓存前缀时要求补分支点状态；
- `_split_node` L322-343：分裂产生的新（分支）节点获得 `state_at_branchoff`；
- `insert` L163-165：`bytes_needed` 计算中 `num_extra_mamba_states = 2 if branchoff else ...`——分支场景按 2 个额外 SSM 状态预留（对应论文"每序列 ≤2 SSM 状态"）；
- 最后解码 token 状态：模拟器以 `insert` 完整 token 序列（含生成部分）体现（`policy_exploration.py` `run_trace_with_config` L53-156 的 match→insert 循环）。

**与 FlowState 固定候选集的差异**：
- Marconi 的准入是**运行时逐请求决策**（是否分叉、是否缓存末尾状态、留多少状态），依赖 match 时刻的投机插入；
- FlowState 的 `CheckpointCandidate` 集合由严格纪元兼容矩阵（EPOCH_ADMISSIBLE 边）**预先固定**，无运行时准入自由度、无投机插入、无分支点推断。适配时 Marconi 的准入层整体不可保留（详见 adaptation_contract.md）。

---

## 审计项 2：可驱逐候选集

**论文机制**（§4.2）：驱逐发生在叶子与"单孩子"节点；中间节点驱逐将 KV 合并向孩子、释放 SSM 状态。

**源码证据**：
- `_collect_leaf_and_single_child_nodes` L383-394：候选 = 叶子 + 孩子数 ≤1 的节点；
- `evict_v2` L503-559：**每次驱逐迭代重新收集候选**（L509 附近在循环内调用收集函数），驱逐一个节点后重新计算（树形已变化）；
- `_delete_leaf` L409-415：叶子删除；
- `_evict_intermediate_node` L417-434：中间节点驱逐 = KV 并入孩子（代码注释含对 KV 张量操纵的 NOTE），时间戳取 `max(parent, child)`，孩子的 `hybrid_states` 保留；
- 释放量差异：叶子驱逐释放 mamba+attn（L555 附近），中间节点驱逐仅释放 mamba（L558-559）。

**与 FlowState 的差异**：FlowState 驱逐对象是固定候选集内的 checkpoint 槽位（逻辑预算 K），无树结构、无合并语义、无"叶子/中间"之分。适配后"候选集"= 全部 CheckpointCandidate，驱逐粒度 = 整个候选（等价 Marconi 的叶子驱逐）。

---

## 审计项 3：FLOPs 节省与内存占用（论文 §4.2；§5.3）

**论文机制**：驱逐效用含 FLOP 效率项 = 该条目节省的 FLOPs / 其占用字节数；节省相对父节点计算。

**源码证据**（`utils.py`）：
- `get_attn_flops(l,d) = 8·l·d² + 4·l²·d`（L46）；
- `get_mlp_flops(l,d) = 16·l·d²`（L65）；
- `get_mamba1_flops` L104-107：注释明言**投稿版用错公式 `10*l`，camera-ready 更正为 `10*l*d`；当前代码 = 更正后版本**（论文-代码演进点，引用公式时须用当前代码版）；
- `get_kvs_size = 2·l·d·2` 字节（L121）；`get_mamba_state_size` L124-145（`d·n·2 + (expand·d+2n)·conv_kernel·2`，按字段展开）；
- `get_flops_efficiency = flops_savings / bytes`（L163-168）。

**`evict_v2` 中的相对父节点节省**（L503-559）：
- mamba 项按 `seqlen_child` 计算、attention 项按 `seqlen_total − seqlen_parent`、MLP 项应为 total−parent；
- **发现代码缺陷（L526）**：`flops_savings_mlp = self.num_mlp_layers * (get_mlp_flops(seqlen_total, self.d) - get_attn_flops(seqlen_parent, self.d))`——MLP 节省项中混入 `get_attn_flops`（应为 `get_mlp_flops(seqlen_parent,...)`）。两处 `_normalize` 后进入效用函数，属实现笔误而非论文意图。适配复现时应按论文语义修正，并在此备案。

**模型配置**（`policy_exploration.py` L198-202）：模拟采用 NVIDIA Attention-Mamba2 Hybrid 7B 配置：24 SSM / 4 Attn / 28 MLP 层，d=4096，n=128。适配到 FlowState 时应改用 Qwen3.5-9B 混合架构的真实层数配置（另行从 FlowState 运行时配置取，不属于本仓库信息）。

**与 FlowState 的差异**：FlowState 价值信号是纪元结构信息（EPOCH_ADMISSIBLE、续接深度等），无 FLOPs 模型。适配 Marconi 时 FLOPs/字节公式**可以完整保留**（纯函数，只依赖模型配置与 token 长度）。

---

## 审计项 4：Recency 计算与分数归一化

**源码证据**：
- `evict_v2` L535：`recency = 1 / (current_ts − node.ts)`，随后 `_normalize`；
- `utils.py` `_normalize` L11-19：min-max 归一化，**退化情形（全相等）返回全 1**；
- 效用（L543-546）：`utility = eff_weight * eff_score + recency_score`（即论文 S(n)=recency+α·flop_efficiency 的代码形态，α 即 `evict_policy_version=2` 时的 eff_weight）；
- 驱逐最小效用者（L551）。
- 时间戳更新语义对比：V1 在 `_insert_helper` L269-273 对**全部祖先**更新时间戳；V2/V3 在 `match_prefix` L256-261 仅更新**单节点**（论文宣称的降开销改动，代码一致）。

**与 FlowState 的差异**：FlowState 候选的"命中"= 续接被实际恢复（recovery 事件），时间源 = 严格纪元边界事件序。recency 公式与归一化可平移；"当前时间戳"必须换成纪元序号/事件号（无墙钟语义），需在适配契约中显式定义。

---

## 审计项 5：参数选择与历史回放（论文 §4.3 Implementation details）

**论文机制**：α 无法先验，靠回放历史请求调参；首次驱逐前积累 multiplier×（5-15×）请求量；CPU 并行网格回放。

**源码证据**：
- `radix_cache_hybrid.py` `evict` L465-482：驱逐时记录 `num_reqs_before_eviction`，置 `bootstrap_window_size = multiplier × num_reqs`；`insert` L147-155 在窗口内触发 bootstrap 调参；
- `config_tuner.py`：
  - `replay_trace` L13-48 回放使用 **`evict_policy_version=3`（oracle 档）**（L80 附近）；
  - `tune_config` L58-118：α 网格 {0, 0.1, ..., 2.0} 共 21 档（L61）；快照 pickle 拷贝（L78）；`ProcessPoolExecutor` CPU 并行（L84）；
  - **论文-代码不一致（L110）**：选择最优 α 的依据是 `results[k][2] = total_flops_savings`，而论文 §4.3 表述为最大化**命中率**。两者在相同轨迹上不一定同序。引用或复现调参协议时必须注明此差异；
- `policy_exploration.py`：`bootstrap_multiplier` 默认 5、lmsys 负载设 15（L188-194）；V3 网格扫描 L268-316 中 **`v3_max_hit_rate` 与 `v3_max_flops_saved` 对 α 分别独立取最大（L314-315）**——oracle 上界按指标分别取最优 α，属乐观上界构造，对比时须知。

**与 FlowState 的差异/适配**：FlowState 严格纪元下"回放历史"= 对已完成纪元的确定性事件序列重放（FlowState 评测本身具备确定性轨迹），α 网格回放协议可平移；但 bootstrap 触发条件需改写为"首次逐出候选前积累 multiplier×N 个纪元事件"。

---

## 审计项 6：树节点删除/合并的副作用

**源码证据**：
- `_delete_leaf` L409-415：标准摘除；
- `_evict_intermediate_node` L417-434：合并后孩子承载全部 KV；节点 `hybrid_states` 的去留：代码保留孩子的 states（父的释放）；时间戳 max 合并；代码注释含对底层 KV 张量操纵方式的 NOTE（模拟器内直接拼接索引）；
- `_insert_helper` L268-320：每次插入**至多一次分裂**（注释明言），控制树形态；
- `policy_exploration.py` L25：`sys.setrecursionlimit(1500)`——树深可达千级，侧面说明长轨迹下树的深度压力。

**与 FlowState 的差异**：无对应机制（无树）。适配时不存在该类副作用；这是 Marconi 不可平移部分之一。

---

## 审计项 7：Hybrid 状态联合管理依赖（运行时前提）

**源码证据**：
- 节点 `HybridStates = mamba_states + kv_cache` 双载荷贯穿全部路径（insert/match/evict）；
- `insert` L163-169：`bytes_needed` 把分支额外 SSM 状态计入；**插入前先驱逐腾空间**（evict-before-insert）；
- `artifact_evaluation.md`：端到端评估为 **trace 驱动 CPU 模拟**（Llama-2-7b tokenizer 的轨迹，托管于 Google Drive；部分图因"代码库早期版本"无法精确复现——论文作者自述）；
- vLLM+ 基线：`policy_exploration.py` L213-228 经 `RadixCacheVLLM`（admission="all"，block_size=32），即论文中的 vLLM+ 对照。

**结论（运行时前提）**：Marconi 官方代码**不包含真实 GPU 服务路径**；任何"Marconi 端到端 GPU 对比"都是再实现而非复用其 artifact。FlowState 侧的算法层对比因此不受影响，但论文引用其数字时须注意口径（模拟口径）。

---

## 论文-代码差异汇总表

| # | 位置 | 差异 | 处置建议 |
|---|------|------|----------|
| D1 | `radix_cache_hybrid.py:526` | MLP 节省项误用 `get_attn_flops(parent)` | 适配实现按论文语义修正，备案 |
| D2 | `config_tuner.py:110` | 调参选择依据为 FLOPs 节省，论文 §4.3 称命中率 | 复现调参协议时二选一并注明 |
| D3 | `utils.py:104-107` | mamba1 FLOPs 公式投稿版错误、camera-ready 更正 | 引用更正后公式 `10·l·d` |
| D4 | `policy_exploration.py:314-315` | V3 oracle 对两个指标分别独立取 α 最大值 | 对比口径注明"乐观上界" |

## UNVERIFIED 清单

- 论文实验所用轨迹文件本体（Google Drive 链接未下载，网络与配额约束）；
- artifact 之外是否存在可运行的 GPU 服务版 Marconi（论文未提供，仓库无此代码）→ 视为不存在于公开 artifact；
- `radix_cache_vllm.py` 与论文 vLLM+ 数字的精确对应关系（该文件仅通读，未逐行核对 FLOPs 记账）。
