# KVFlow 源码审计（Phase B5-A）

对象：`raw/kvflow_repo/` @ `7ef897eaedc16626a281ccfe0d453fae80251e8f`（"Update README", 2026-03-12）
论文：NeurIPS 2025《KVFlow: Efficient prefix caching for accelerating LLM-based multi-agent workflows》（arXiv 2507.07400，仅 v1）
审计方式：只读；论文 §2-§4 与 sglang 侧及 SScheduler 侧源码逐行核对。

**版本归属前提（影响全部结论的解读）**：
- 仓库仅 2 个提交（[Init] 2026-02-18、Update README 2026-03-12），均晚于论文（2025-07），历史被压缩，**论文时点代码状态不可恢复**；
- README.md:3 明示仓库同时承载 KVFlow（NeurIPS'25）与 **ScaleSim**（Preprint）两个项目；SScheduler 的 SpaceManager（移动速度/交互距离语义）、`example_policies.py`、`hold_priority` 字段族等带明显的 ScaleSim 期痕迹；
- 因此本文所有结论的表述均为"**当前 HEAD 的可证实事实**"；凡推断论文版本行为的，一律标注 UNVERIFIED，不断言原算法失效。

---

## 审计项 1：SScheduler 的工作流元数据产生与传播

**链路（当前 HEAD 完整可追）**：
1. `SScheduler/Scheduler.py:202-233` `do_schedule` → `collect_and_merge`（:129-165）：向各注册 Manager 调 `get_agent_n_timesteps(steps)`（`BaseManager.py:118-142`，返回 `{timestep: set(agent_id)}`）；
2. `SScheduler/policy/base_policy.py:37-72` `DefaultPolicy.integrate_agents`：跨 Manager 对同一 agent 取**最小 timestep**，产出 `{agent_id(str): timestep(int)}`（a2t）与 `{timestep: [agent_id]}`（t2a）；
3. `Scheduler.py:167-200` `send_result`：`requests.post("http://localhost:8001/v1/update")`（**:188，端口硬编码**）。

**Manager 实现**：
- `PlanManager.py:33-71`：输入 plan 持续时间（int），`timestep = (obj-1)//timestep_ticks`；
- `SpaceManager.py:34-219`：**仿真域语义**（moving_speed/agent_place/distance→到达时间取 min）——ScaleSim 项目代码，与 LLM workflow 无关；
- `example_policies.py:2`：`from SScheduler.Scheduler import SchedulerPolicy`——**该符号在当前 Scheduler.py 中不存在**，且其 `integrate_agents` 签名与 `Scheduler.py:161` 的调用（4 键 manager_data dict）不符 → 不兼容残留；
- **仓库内不存在任何真实 agent 框架（LangGraph/AutoGen 等）的 workflow 解析器或 DAG→step 转换器**；`benchmark/` 下客户端亦无 `agent_id` 使用者。论文的 "Agent Step Graph"（节点聚合 max/min+1 等构造）**未在当前 HEAD 中找到对应实现，UNVERIFIED**（推测在论文实验的客户端代码中，未随仓库发布）。

## 审计项 2：Agent Manager 中的 Steps-to-execution 更新

**服务端接收链**：
`http_server.py:1084-1097` `/v1/update`（`UpdateAgentTimestepReq`）→ `tokenizer_manager.py:2107-2110`（control 通道发往 scheduler）→ `scheduler.py:2827-2833` `agent_manager.update_agent_timestep(agent_data, timestep_data)`（`agent_manager.py:21-29`：整体替换两 dict，`update_version` 递增）→ `scheduler.py:2834` `tree_cache._update_leaf_node_timestep()`（`lora_hiradix_cache.py:933-947`：每个**叶子** `hold_priority` 置 1000 后对 `update_dict_agent` 中出现于 `leaf.agents` 的 agent 取 **min**）。

**数值语义**：`update_dict_agent[agent_id]` = 该 agent 距下次执行的步数（由外部 SScheduler 声明）；消费侧 clamp：`get_agent_hold_priority = min(hold_step+1, 值)`（`agent_manager.py:43`），未知 agent → `hold_step+1`（:39），非法 `'-1'` → `hold_step+2`（:36）。**数值越小 = 越临近执行 = 越不可驱逐**。

## 审计项 3：Agent 与 Radix Cache Node 的关联

- 请求携带 `agent_id`：`entrypoints/openai/protocol.py:233`（ChatCompletionRequest，`Optional[Union[List[str], str]]`）→ `serving_chat.py:143` → `managers/io_struct.py:132/485`（GenerateReqInput）→ `:540`（TokenizedGenerateReqInput）→ `managers/schedule_batch.py:438/621`（`Req.agent_id`）。即论文 §3.3 的 client 侧 ID 命名空间：**agent 身份由客户端逐请求显式声明**；
- 关联写入：请求缓存完成/未完成时（`scheduler_output_processor_mixin.py:92-98, 263`）调用 `_update_leaf_node_priority(req, req.last_node)`（`lora_radix_cache.py:618-624`）——**只对 last_node（该请求最深节点）打标** `n.agents[agent_id] = str(agent_id)`；
- `_update_agent_to_last_nodes`（`lora_radix_cache.py:591-616`）：维护每 agent 的"最后节点"集合，含祖先/后代去重（新节点更深则移除祖先条目 :601-605；新节点是既有条目祖先则不加 :607-614）；
- 继承：`insert`/`_split_node` 新节点 `agents = copy.deepcopy(父.agents)`（`lora_radix_cache.py:426/468`；`lora_hiradix_cache.py:747/788/856`；`radix_cache.py:469/511`；`hiradix_cache.py:743/781/849`）——前缀共享时子节点继承全部父 agent 标签；
- 注意：`agents` 值类型为 `str`（`lora_radix_cache.py:84`，AgentInfo 类版本被注释 :625-638），与 `__lt__` 死分支中 `.get_priority()` 调用（:126）类型矛盾——若该分支可达将抛 AttributeError（见审计项 6）。

## 审计项 4：共享前缀节点的 Priority 聚合

当前 HEAD 存在**三种方向并存**的聚合写法（均为事实记录）：
1. **叶子按 agent 取 min**：`_update_leaf_node_timestep`（`lora_hiradix_cache.py:939-943`）与 `get_agents_hold_priority`（`agent_manager.py:45-53`，变量名 `max_priority` 但逻辑为取 min）——对应论文 §3.1 "共享节点取 children 最小 priority"的表述（min = 最临近执行的 agent 主导保护）；
2. **驱逐时父节点继承 max**：`_evict_backuped`/`_evict_regular`（`lora_hiradix_cache.py:334/344`）`node.parent.hold_priority = max(parent, child)`——孩子被驱逐后，父节点（即将成为新叶子）继承**最远**复用距离，使其更易被继续驱逐；
3. **内部节点按需计算**：`get_node_hold_priority`（`agent_manager.py:55-65`）以 `update_version` 缓存，过期则从 `node.agents` 现算 min。

方向 1 与 2 服务于不同事件（元数据刷新 vs 驱逐传播），非同一计算的两处实现；论文是否同时含两者 UNVERIFIED（无历史可查）。

## 审计项 5：实际 Eviction 调用链（按缓存类别分列）

**(a) LoRAHiRadixCache（分层+LoRA，论文机制的主载体）** `lora_hiradix_cache.py`：
- `evict` :269-277：`enable_holding=True` 时 `steps` 自 `hold_step-1`（= `evict_pri_level-1`，默认 4-1=3）递减至 1 逐档调用 `_evict_helper(ignore_holding=False, steps)`；不足则 `_evict_helper(ignore_holding=True)`；`enable_holding=False`（**默认，`server_args.py:389`**）直接跳到 ignore_holding 分支；
- `_evict_helper` :279-328：候选 = `_collect_leaves_device()`（:877-900，**DFS 栈序**）；`heapq.heapify` 被注释（**:281**）、`x = leaves.pop()`（**:287**）——**候选之间没有任何优先级排序**；逐节点判断 `lock_ref>0 or loading` 跳过（:289），`ignore_holding=False` 时 `get_node_hold_priority(x) < steps` 则**保护**（:292-295）；`write_back` 策略下先写宿主内存再驱逐（:297-306, :322-327 阻塞等写回完成）；
- 即：**优先级在当前 HEAD 的驱逐中是"阈值式保持"（hold）而非"排序式选择"**；排序语义（论文 §3.1 按 steps 排序驱逐）未在此路径出现。

**(b) evict_host（宿主层）** :349-374：`heapq` + `__lt__` → **纯 LRU**（`last_access_time`），仅驱逐已在设备层驱逐过的节点。

**(c) 非分层 LoRARadixCache.evict**（`lora_radix_cache.py:317-340`）与 **base RadixCache.evict**：`heapq` + `__lt__` → **纯 LRU**。

## 审计项 6：`TreeNode.__lt__` 专项核查（任务点名项）

**源码事实（两处并列，逐行核对）**：
- `python/sglang/srt/mem_cache/radix_cache.py:140-159`：`__lt__` 第 **141** 行 `return self.last_access_time < other.last_access_time` 为**无条件早退**；其后 142-159 行的 agent priority 比较（含 `get_agents_hold_priority`、`ignore_holding`、agent 优先级回退 max）**不可达**；且 `self.agents: dict[str, str] = {}`（:105）与死分支内 `self.agents[...].get_priority()`（:147）类型矛盾；
- `python/sglang/srt/mem_cache/lora_radix_cache.py:119-138`：第 **120** 行同样无条件早退；121-138 行同构死分支；`agents: dict[str,str]`（:84）同样矛盾。

**该早退对运行路径的实际影响（可证实部分）**：
- 凡以 `heapq + __lt__` 驱动的路径——(b)(c) 两类——在当前 HEAD **退化为纯 LRU**，agent priority 不参与；
- 但**分层路径 (a) 的优先级保持不经过 `__lt__`**：`_evict_helper` 直接调 `get_node_hold_priority` 做阈值判断（:292-295），该机制在 `--enable-holding` 下可独立于 `__lt__` 生效；
- `evict_host` 与非分层 `evict` 中的 LRU 行为本身与 `__lt__` 早退**自洽**（无崩溃），死分支内类型矛盾因此从未在运行中暴露。

**论文版本是否使用此调用路径：UNVERIFIED**。仓库历史被压缩为 2 个 2026 年提交，无法证明论文（2025-07）实验代码中 `__lt__` 是否含此早退。**结论仅限：当前 HEAD 的非分层路径不含 priority 排序，分层路径的 priority 以阈值保持形式存在且依赖 `--enable-holding` 显式开启（默认关闭）。不对论文原算法是否失效作任何断言。**

## 审计项 7：Fixed Prompt 与动态 Suffix 处理

- **服务端无显式 fixed-prompt 原语**：全库检索 `fixed_prompt`/`fixed prompt` 无命中；节点上没有"固定提示边界"标记字段；
- 论文 §3.1 "varying suffix 总是最高驱逐优先级"的显式机制（suffix 节点无条件先驱逐）在当前 HEAD **未见对应实现，UNVERIFIED**；
- 现存最接近的语义：agent 只标注 last_node（请求最深节点，含解码输出，即"最新 suffix 端"），未标注者 `hold_priority` 缺省 1000（`lora_hiradix_cache.py:939`）——无标签节点事实上最易被驱逐，与"suffix 最先驱逐"有**弱等价**效果，但这是缺省行为而非显式机制，归属推断 UNVERIFIED；
- fixed prompt 的识别在论文 §3.3 属 workflow 声明侧（Agent Step Graph 的节点即 agent 固定提示）→ 对应代码应在未发布的客户端/SScheduler 集成层。

## 审计项 8：KV Prefetch 及其 Runtime 前提

**主动预取（论文 §3.2 overlapped prefetch 的当前 HEAD 形态）**：
- 触发：`scheduler.py:2848-2849`——处理完 `/v1/update` 后（`disable_prefetch=False` 默认）调 `DispatchLoadTasks(load_ahead_step)`；
- `mixin:747-810`：`for step in 0..p_step`（`p_step = load_ahead_step`，默认 2，`server_args.py:387`）：取 `update_dict_timestep[step]` 的 agents →
  - LoRA 权重预取：`prefetch_lora_timesteps(lora_name, priority=step)`（`:778`，受 `disable_lr_pf` 开关；`lora_name = f"lora{agent_id}"`（:766/773）——**agent_id 兼作 LoRA 名**，两个命名空间绑定）；
  - KV 预取：对该 agent 的 `agent_to_last_nodes` 中 `evicted 且非 loading` 的节点 `load_back(node, priority=step+1, check_reserve=True)`（`:794-799`）；空间不足（返回 None）则中止本轮；
- `load_back`（`lora_hiradix_cache.py:376-458`）：`check_reserve` 检查 available+evictable（:412-416）；装载后节点 `loading=True`、`hold_priority=priority`（:447, 453）。

**被动加载（状态感知调度，节点三态）**：
- 请求入队（`scheduler.py:1865-1873`）：`get_node_chain_status(req.last_host_node)`（`lora_hiradix_cache.py:920-931`，REQ_IS_EVICTED/LOADING/READY，:32-34）：EVICTED → `load_back(priority=0)` 并 defer；LOADING → defer（跳过本批）；
- PrefillAdmission：`host_hit_length>0` → `init_load_back`（`schedule_policy.py:521-527` → `lora_hiradix_cache.py:460-482`）；
- 存储层（NVMe 等）prefetch 函数族全部标注 `# NO USE`（`lora_hiradix_cache.py:495-714`）——当前 HEAD 未启用。

**Runtime 前提汇总**：分层 KV（host backup/write_back 策略）、外部 SScheduler 持续 POST `/v1/update`、请求携带 agent_id、LoRA-per-agent 命名约定。四者缺一则预取链路不成立。

## 审计项 9：专属参数与版本归属

`server_args.py:387-394`（README:33 同列）：`load_ahead_step=2`（"Step to load ahead when prefetching"）、`evict_pri_level=4`（"Priority level when evicting from leaf nodes"，即 `AgentManager.hold_step`，`scheduler.py:267`）、`enable_holding=False`、`enable_interrupt=False`、`disable_prefetch=False`、`disable_lr_pf=False`、`disable_kv_pf=False`。
其中 `enable_holding`/`enable_interrupt`/`disable_lr_pf` 属 KVFlow/ScaleSim 混合期新增；**默认配置下（不开 --enable-holding）当前 HEAD 的设备层驱逐不消费任何 priority**——该默认组合属于哪个项目的意图 UNVERIFIED。

## UNVERIFIED 清单（汇总）

1. 论文时点（2025-07）的代码状态：不可恢复（历史压缩），`__lt__` 早退、`# NO USE` 标注、SpaceManager 混入等均无法断代；
2. Agent Step Graph（workflow DAG→steps 聚合构造）的服务端/客户端实现：仓库内不存在；
3. varying-suffix 显式最高驱逐优先级：仓库内不存在；
4. 基准 SGLang 版本的精确 commit（论文 §3.3 自述基于 SGLang；仓库为深度改造版，pyproject 版本号不可直接对应上游 tag）；
5. 论文实验的客户端（发送 agent_id + 驱动 SScheduler 的 benchmark 代码）：未随仓库发布。
