# P1-B：完整 Agent 轨迹可支撑最小严格重放，原生 Qwen Agent 闭环尚未接通

审查日期：2026-10-29。**最终判定：PARTIAL。**

推荐的最小实现是：复用已有 OpenHands 同一 Coding 任务的 A/B 两条真实执行轨迹、完整消息及冻结 Qwen Token 数组，增加客户端 Prepared-Pending 边界和来源事件账本，连接已经验证的 Strict Epoch、物理观察及 Native Resume 路径。它应命名为 **Trace-derived Controlled Composition / Token-history Replay**，不能称为原生 Qwen Coding Agent、自然同时 Ready 或原生 Fork。

若“真实 Agentic”要求 Qwen 自己生成工具调用、实际执行工具并继续推理，则仍需接入 Coding Agent。现有 Agent 源码提供接口基础，但本轮没有确认任何本地 Qwen 工具闭环或真实多分支执行环境已就绪。因此不判 READY；同时，完整真实消息与既有物化/恢复接口均存在，也不判为没有可信原型路径。

本轮只读取源码、消息和已有结果，进行 CPU 哈希、因果顺序和 Token 数组检查。没有启动 GPU、模型服务、Docker、Agent 或工具，没有安装依赖、重新分词、拉取模型或修改冻结文件；P0-4 路径未改动。

## 1. 三类来源必须分别命名

|来源|现有证据|当前资格|
|---|---|---|
|原生 Qwen Agent 执行|应由 Qwen 输出 action/tool call，真实工具结果进入下一请求；本地有 SWE-agent/Captain-Agent/Magentic-One 源码，但没有本次所需的 Qwen 完整运行证据|尚未接通，不能用历史 Tool 文本代替工具执行|
|真实轨迹驱动的 Qwen 重放|OpenHands 有完整 System/User/Assistant/Tool；B5-C2 已保存目标模型完整输入，B5-C3 已实际回放及物化，B5-D3 已完成一个目标的 Native Resume/Cold 对照|推荐的最小实现路径；历史 Assistant/Tool 属于原轨迹，Qwen 回放生成结果不替换原轨迹历史|
|人工受控共享分支|Synthetic A/B/C、SGLang fork 示例等可有显式拓扑，但不代表真实 Coding 轨迹原生分支|不作为本轮推荐来源；不得为满足共享或策略胜负构造新分支|

同任务两次独立执行可以有相同 Token 前缀，但没有因此获得共同原生父请求。真实继承事件、Token 前缀相同、Recurrent 状态共享是三个不同层次，均需各自证据。

## 2. 完整消息与可重复输入：已有直接证据

### 2.1 来源和最小固定实例

来源：`/home/wjg/data/agentic_coding_trajectories/sessions.parquet`。本轮读取 Parquet 元数据和 source_dataset 列：15,000 条总记录，其中 OpenHands 来源 `nebius-swe-rebench-openhands` 为 5,000 条；其余两来源各 5,000，不能混称全部为 OpenHands。

本轮重新计算约311 MiB文件的完整 SHA-256，与 B5-C2 来源锁一致：

`3dd8ec3546cf771ce4ab2ac6c51ccefdd621197fa997a2cefb430b50df808fb6`

推荐直接使用 B5-C2 已固定的首个任务 `0b01001001__spectree-64` 的 A/B，不重新搜索来源：

|项目|A|B|
|---|---|---|
|Session|`nebius-swe-rebench-openhands::chatcmpl-71420ffc5b410547d43ac67939b0441c`|`nebius-swe-rebench-openhands::chatcmpl-71b073b89097665d14501919b224c26b`|
|原始消息总数|97|99|
|零基Parquet行|10918|12519|
|前两次历史输入长度|3728、4182|3728、4150|
|第三次 Pending 输入长度|4862|4860|
|决策输入消息|索引0–5|索引0–5|

两条轨迹均含完整角色与文本、tool_calls_json、tool_call_id。本轮逐条比对原 Parquet 的索引0–5与保存的 A/B_source_history.json，完全一致。角色顺序是 System→User→Assistant→Tool→Assistant→Tool；A 的工具名包括 think、str_replace_editor，历史文本是实际记录，而不是本轮生成。原始 recorded_model 均为空；原生产模型/Tokenizer、墙钟和 Ready 事件仍为 UNKNOWN。

这个推荐是一项任务的两次真实执行的受控组合，不是一次自然运行中的两个并发分支。单条串行轨迹在任一模型调用边界通常只有一个下一请求，不能仅靠增加逻辑 ID 得到2个 Pending。

### 2.2 Qwen Token 身份与共享内容

B5-C2 的12份完整Token数组均仍存在。本轮核对全部长度、文件SHA和规范化Token数组SHA，全部通过；目标Tokenizer/模型config清单的4个带摘要条目也与本地文件一致。本轮未生成新的Token数组。

- A1 与 B1 的完整3728-Token数组相同。
- A3/B3 的共同 Token 前缀长度为3765；在先前四 Session 原型中，实际 A/B 共享 Recurrent 位置为3712，**3765不是物化位置**。
- 四Session公共LCP为2438，其中完整System渲染前缀是2428；不能把2438全称System，也不能假设只运行A/B仍会出现四Session时的2432节点。

输入构造代码：`openhands_single_workflow_smoke.py::normalize_message/_parse_tool_calls` 保留角色、内容、工具调用及调用ID；`rq3_openhands_neutral_collector.py::materialize_group_requests`（236起）在第3个Assistant标记之前停止，仅将之前消息交给模板，不读取第3次输出。

冻结输入使用 Qwen3.5-9B、Transformers 5.12.1、Tokenizers 0.22.2、SGLang 0.5.17 镜像。模板摘要为：

`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`

构造参数为 tokenize=True、add_generation_prompt=True；tools 列表未额外注入，使用原System工具说明和历史tool_calls；enable_thinking沿用冻结模板默认值，无Token截断。宿主 uv312 是 Transformers 4.57.6，**不得用宿主版本悄悄重建并替换正式数组**。需要重渲染时应另用既有冻结镜像CPU路径并核对全数组；这不是本轮执行内容。

### 2.3 因果性边界

本轮固定输入只用第三轮之前的真实历史，不把未来Assistant、未来Tool或后续请求作为当前决策输入。现有 causal_checks.json 已保存工具调用→结果的先后索引；B5-C2四来源合计8条结果有对应的先前tool_call_id。消息顺序可以核验，原始墙钟/自然Ready不能恢复。

新采集必须另外记录本次重放中的准备、提交、接收和完成时间。不能把原消息索引或事后看到的完整轨迹当作原应用的时间戳，更不能把“冻结文件里存在第三次输入”改名为 agent_native_ready。

## 3. 本地 Agent 与工具执行能力

|现有入口|源码能够做什么|尚缺什么|
|---|---|---|
|SWE-agent：`/home/wjg/code/TraceElephant/code/agent_system/SWE-agent/`|`sweagent/agent/agents.py::step/run` 调用模型、执行动作、追加真实observation；约961行经 `_env.communicate` 执行工具。`models.py::LiteLLMModel` 约740行支持messages/tools和api_base，存在指向本地兼容API的接口|没有现成Qwen3.5运行证据；需确认依赖、SWEEnv/任务仓库、工具解析、模型client及服务端真实Token采集。单Agent循环本身不提供2–4个同时Ready请求|
|Captain-Agent：同目录 `Captain-Agent/scripts/run_assistantbench.py`|有真实 initiate_chat、动态Agent组和代码执行配置；并非纯回放|需其autogen/模型client/工具依赖；当前任务入口涉及检索器等额外组件，不能认定为轻量Qwen Coding运行环境；groupchat逐speaker不等于同源并发fork|
|Magentic-One：`run_assistant_bench.py`|构造OpenAI-compatible client、Coder、CodeExecutorAgent、FileSurfer/WebSurfer、MagenticOneGroupChat；可生成真正多角色工具活动|依赖及本地Qwen兼容性未就绪；外层run_stream不是同时Ready或继承前缀证明。其logged_create先await原调用再保存日志，不提供准备后、dispatch前的门禁|
|TraceElephant `code/llm_api_middleware/middleware.py`|已有OpenAI-compatible请求/响应原文持久化，支持流式/非流式|不保存实际Runtime Token、原生父子事件、Prepared-Pending集合或Node/Slot/组件；不能仅加代理就满足Strict合同|
|SGLang Tree-of-Thought归档|`evaluation/sota_baseline_audit_b5a/raw/kvflow_repo/benchmark/tree_of_thought_deep/bench_sglang.py`有真实模型多阶段和s.fork；归档interpreter复制context并发分支|是算术/推理benchmark，不执行Coding工具；归档来自KVFlow分支，非已认证当前SGLang源码，不能直接移植为现成可用路径|

本轮只查询发行包元数据，没有导入运行Agent：uv312中未安装OpenHands、Qwen-Agent、LiteLLM、smolagents、langgraph、autogen-agentchat，SGLang也不在宿主该环境内。检查uvgood、torch-base、torch-base-310的site-packages同样未见相应Agent包。Magentic-One虽有名为`.audit-autogen075`的环境目录，其Python发行包查询中autogen-agentchat/core/ext、openai、litellm和浏览器依赖均未安装。不能由目录名推断依赖已就绪；未扫描所有可能外部环境，其他环境能力为UNKNOWN。

旧`p3_native_instrumentation_feasibility/agent_event_capability.md`引用的`/tmp/audit_sglang/sglang-v0517-tag/`当前已不存在；本轮没有把历史临时目录的可运行性沿用为当前事实。仅保留现存归档的源码机制参考。

**是否必须接Coding Agent：**

- 为“完整真实轨迹→固定Qwen输入→可物化状态→真实Resume”的受控原型，不需要先安装Coding Agent。
- 为“原生Qwen工具闭环、自然Ready或真实Fork”的主张，必须实际接入并运行Agent。建议后续优先评估已有SWE-agent的模型/工具边界，先取得一个合法工具循环；多Agent和并发边界再单独核验。本轮不承诺少量安装就能完成，也不把Agent框架源码称作已部署系统。

## 4. 最小执行与采集合同

建议由独立的来源/消息适配器、客户端Prepared-Pending gate和现有Runtime bridge组成，不修改生产Optimizer、Controller、SGLang。以下字段必须按真实事件保存，而不是仅存Hash。

|对象|必须持久化的数据|来源/验证位置|
|---|---|---|
|来源|原始messages及索引、角色顺序、tool_calls/结果、Session/task/原始记录摘要、消息→请求映射|Parquet截断历史与source_selection_lock；缺原生产时间/模型标UNKNOWN|
|输入配置|目标模型权重/config身份、Tokenizer文件/版本、实际Chat Template、tools schema/参数、normalization、thinking/generation参数|复用C2配置锁；新原生Agent增加的tools必须另锁，不可沿用“未传tools”的旧身份|
|实际请求|完整input_token_ids、计数、摘要、完整生成Token及停止原因|客户端提交数组与Runtime收到的Token逐项/摘要核对；不能只靠文本重新分词认证收到的输入|
|因果身份|run/workflow/request/branch、previous_request_id、真实parent/fork事件或显式无该事件|A/B为独立根Session，不填共同fork_parent；source relation标same-task-independent-executions。Native版本必须在实际fork/调用处记录|
|时刻|prepared_ns、dispatch_ns、runtime_receive_ns、complete_ns，时钟类型/作用域；原轨迹时间另列|用monotonic时钟；同宿主跨进程可衔接，同步条件需说明；原source时间未知不补写|
|Pending集合|完整输入已封存、依赖完成、尚未发送/取消，当前成员列表及冻结摘要|新增客户端gate，在发送前记录。不得先把请求放进Scheduler waiting_queue再声称完全静止|
|候选|当前全树已驻留Node的完整Token prefix、L、FA路径、Slot及Conv/Temporal切片和驻留、引用/所有权/别名、候选外状态|复用R2-C0 PhysicalView与B5-C3全树观察；按当前物理对象建立唯一候选，另存多来源映射|
|Strict关系|每个candidate×pending的L≤T、实际Token前缀、同配置/同边界、组件与物理身份条件，三态及原因|复用B4-B0 contracts.build_edges和StrictAllocationEpoch；UNKNOWN fail-closed，不用谱系兜底|
|规划/应用|逻辑K、选集与补集、计划摘要、当前重新绑定、Preflight、逐驱逐Postcheck及失败账本|现有strict_control；检查与控制处于同一Scheduler同步窗口，不跨RPC假设租约|
|恢复结果|Admission前状态、实际Native H/E/Node/source Slot、COW返回及工作槽、实际extend区间、输出与Cold、源组件保护|复用R2-C1b2/B5-D3路径；新增工作状态和正常Cache副作用与源损坏分开记录|

候选须按**真实物理前缀状态**建模。A1/B1输入完全相同，可能共享同一Node；不能因为有两个历史Request ID就计为两份可独立保留/驱逐状态。`strict_control.bind`拒绝多个候选映射同一Node、槽/组件别名及候选域外Recurrent状态。复用B5-C3“全树唯一Node→候选”模式，并保存该物理状态关联的历史请求集合；不能随意删掉候选外驻留祖先来凑预算。

## 5. 已有 Runtime 证据与八项正确性问题

本轮重读B5-C3原始账本：8条历史均completed，提交摘要与Native接收摘要全部一致；Strict文件保存7个候选、4个Prepared Pending、28关系。已有13 EPOCH_ADMISSIBLE、15 REJECTED、0 UNKNOWN，对应当前文件记录，不是本轮新采集。

B5-D3的原始output_correctness.json记录8 Token Strict/Cold逐项相等；已审阅状态保护报告。该案例只能证明其独立生命周期、计划和目标B，不能直接外推到只运行A/B的新边界。

|问题|已有可支持的判断|新原型仍需验证|
|---|---|---|
|1. 祖先是否同Token前缀|A/B完整第一输入相同，第三输入LCP=3765；这是实际数组比较|无真实共同fork事件；不能由工作流标签推断共享，也不保证物化位置=3765|
|2. 指定Pending能否Native复用|B5-D3中B从共享L=3712恢复，Native H/E/G=4096/3712/384，实际来源Node18、COW完成、8 Token Cold相同|新Engine重新观测与绑定，不能硬编码Node18/Slot3；新A/B-only状态不保证等价|
|3. 分配时状态已物化有效|B5-C3实际7项同边界驻留，Conv/Temporal/引用/记账检查已有|新运行必须检查全部原生状态，不插入预设中间切点、不事后搬回被驱逐状态|
|4. Pending已经准备|旧C2有完整客户端输入，B4-B0有controlled_input_ready_ns|新原型增加逐请求准备→发送门禁日志；旧轨迹没有自然Ready墙钟|
|5. 未来信息泄漏|冻结构造器在第3次Assistant内容前停止；原工具结果顺序可核验|Native Agent必须在真实依赖完成后准备；不能用未来工具返回填充本次输入|
|6. 逻辑/物理别名|相同A1/B1输入是必须处理的共享来源；Strict路径会拒绝重复物理绑定|多请求归属一个物理候选，当前Node/Slot/组件与全树校验；不按Request数计物理预算|
|7. Strict/Reconciliation|既有接口和B5-D3五项Recurrent-only驱逐、Selected/FA保护已有功能证据|新增源适配与Prepared gate尚未CPU回归；新Epoch和所有物理门禁仍需GPU实测|
|8. 公平策略对照|已验证独立Engine、固定历史及输入、新鲜绑定和Cold机制|最小原型先只验证一个保留计划；后续各策略独立重建并核对语义状态等价。不得顺序污染后复用|

EPOCH_ADMISSIBLE、RECONCILIATION_VALIDATED、RESUME_VALIDATED分别记录。PROVENANCE_UNVERIFIED保持不变；来源请求最早Census关联不等于逐张量原生创建血缘。

H为Attention-side frontier，T=min(当前协议anchor/完整输入长度，H)，E为选集支持的最大合法L，G=T−E；实际prefill_tokens=len(input_ids)−Native E。B5-D3中B的输入4860、E3712，实际Prefill=1148，而G=384。不能混用这两个区间，也不能把模型化ms当作GPU时间。

## 6. 推荐最小规模和模块边界

**第一选择：A/B同一真实任务的两Session受控重放。** 4条历史输入，2个第三轮Pending，全部既有数组与截断消息复用。原消息97/99条保留来源归档，但本次构造仅消费每条的前6条消息。

- 初始目标为一个同边界、至少一个两Pending共享的驻留候选；新候选数及位置由Runtime自然观察，CPU阶段为UNKNOWN。
- 预先约定预算规则：n≥2时K=1，保持工作/COW空间；n<2则报告没有非平凡预算竞争，不凑状态。K只表示可控唯一Recurrent候选的逻辑保留数量，不称HBM不足。
- 目标Pending固定为B。若当前合法选集中没有能支持B恢复的共享状态，报告真实结果，不换来源/候选/预算追求成功。
- 不恢复已驱逐状态、不强制设置3712或3765、不为了出现多层共享加入C/D。四来源原型只作已有机制证据，不作为失败补位。

需要的实验层小模块（本轮均未实现）：

1. **来源/消息适配**：复用C2输入与neutral collector规范化函数，导出request到来源消息范围的完整映射和明确父子字段。现有输入已完整，无需新生成器。
2. **Prepared-Pending gate及事件账本**：输入构造完成后封存，记录所有成员，禁止客户端发送；控制阶段结束后仅派发预定目标。该gate是披露的受控等待，不声称自然并发。
3. **窄范围Runtime接线**：复用B5-C3观察和R2-C1b2/B5-D3恢复合同，将固定工作负载路径/全局文件读取改为独立实验输入；仍用同一Strict Epoch/Controller/Adapter。现有脚本包含固定目录、label数量、K等实验参数，不能直接拷命令就宣称任意Agent通用接入。

原生Qwen Agent属于独立的后续接入条件：优先现有SWE-agent模型query和工具执行边界，先验证一次真实tool call→真实结果→下一请求。应保存实际tools schema、请求Token和完整模型输出，不能沿用8 Token正确性Smoke当完整Agent生成协议。若需要大型依赖/任务镜像/仓库恢复，超出本轮及“少量采集适配”的批准范围，应另行审查。

## 7. 后续最小 GPU 验证方案（仅方案，不执行）

1. 完成上述三个实验层模块的CPU合同检查：消息截断、tool_call_id因果性、数组身份、Prepared成员与发送隔离、来源多对一物理归属、UNKNOWN/别名/陈旧映射拒绝、spawn入口保护。不得降低既有严格条件。
2. 未来另获GPU授权后，重新检查单卡独占资源，固定已有Qwen3.5-9B/SGLang0.5.17/TP=1/28槽，独立Engine按A1→B1→A2→B2重放。新输出只记录，不加入冻结历史。
3. 两个Pending均已准备且尚未发送；在单一Scheduler同步边界读取全树、FA/T、唯一物理候选、组件、引用和全部三态关系。先完整保存不可变Epoch；没有合法共享或n<2，原样报告，不补位。
4. 若完整合同通过，使用现有正式规划/协调路径按预先约定K=1应用计划，重验物理补集和FA保护；只执行固定B一个Pending。实际Match不符合计划时失败停止。这里测试一个合法计划，不作策略性能比较。
5. 若B确实从预期合法源恢复，检查COW完成、源组件不变、实际Prefill及输出；独立干净Cold用相同输入和既有8 Token确定性协议做正确性对照。最多一个Strict生命周期加一个Cold；不顺序派发另一个Pending作为同边界证据。

真实Agent版若将来实施，先采集一次Qwen原生工具执行及Prepared事件。策略对照再使用冻结的实际采集输入、在独立等价Engine重建；不同策略若触发Agent后续轨迹分化，应另行定义在线比较，而不是混入同输入恢复对照。

## 8. 存储、硬件与工程阻塞

### 8.1 已有数据和预算依据

|项目|本轮实际读取/计算|含义|
|---|---|---|
|根盘剩余|约5.6 GiB（本轮开始时）|只足够小规模有上限采集；不是独占存储承诺，不自动清理他人或P0-4数据|
|来源Parquet|约311 MiB，已存在|不复制全数据；新目录仅保存选中消息与Token及来源摘要|
|C2整个CPU目录|当前约1.10 MiB|12份数组、消息、渲染、offset等；两Session通常更小，不能因此估计GPU内容审计也同样小|
|C3当前非Triton证据|14.93 MiB，8历史/4Pending|说明一个受控capture不必复制完整模型，但不代表新运行日志必然等大|
|B5-D3当前目录|36.30 MiB非Triton＋22.27 MiB缓存=58.57 MiB|一个Strict及Cold功能验证量级，沿用完整日志时会受重复审计影响|
|R2-C1b2当前目录|41.98 MiB非Triton＋22.05 MiB缓存=64.03 MiB|另一个独立Strict/Cold实例的实际磁盘依据|
|已有Qwen权重|索引total_size约17.98 GiB，4个shard，已存在|不下载、不复制；目标模型只读挂载；本轮只读索引与config，未重新核验全部权重内容SHA|
|Recurrent池|49.125 MiB/候选逻辑单元；28槽约1.343 GiB|配置层估算，不含工作槽、FA-KV、模型权重和运行时workspace，不是GPU总峰值|

建议最小Strict/Cold两生命周期预留**0.5–1 GiB磁盘余量**，依据上述约60 MiB实例增加日志、审计和编译缓存余量；这是工程预估，不是实测上限。每阶段写入前重新核对剩余空间，空间不足时停止而不是省略必要证据。本轮没有为本报告复制模型或原轨迹。

硬件沿用单H100 PCIe约80 GiB设备；模型与FA/工作区加总峰值本轮未测，标UNKNOWN。28槽有空闲不能证明真实容量竞争。不得拿权重18 GiB＋Recurrent1.34 GiB直接认定Engine总峰值为19.34 GiB。

### 8.2 尚未闭合的工程问题

- 实际Qwen Coding Agent依赖、任务环境与工具协议未就绪；源码可用不等于部署完成。
- 原始OpenHands没有原模型和自然Ready/Fork墙钟；新重放能记录自己的时间，不能补回原生产事件。
- 两Session均来自独立执行；共享是内容和状态证据，不是共同父工作流血缘。
- 新两Session自然物化结果未知；已有四Session的Node/Slot和候选数不能移植。
- 物理候选必须按完整全树唯一状态计数；重复历史输入、祖先状态、非目标驻留及COW空间不可忽略。
- `/tmp`内旧SGLang源码已消失，不能沿用失效路径的能力认证；当前冻结镜像及源摘要在未来GPU启动前必须重新核验。
- 后续策略若用Native Agent重新生成历史，各臂可能不同；应先固定真实采集输入并声明受控策略重放范围。

## 9. 证据与本轮核验记录

直接来源和调用链：

- `evaluation/multishare_cpu_input_gate_b5c2/{source_selection_lock.json,A_source_history.json,B_source_history.json,requests.json,*_tokens.json,causal_checks.json,input_configuration_lock.json,verification.json}`：消息来源、请求数组、工具顺序与目标配置；本轮检查原Parquet A/B截断与12份数组双摘要。
- `evaluation/openhands_single_workflow_smoke.py:66–110`、`rq3_openhands_neutral_collector.py:236–303`、`multishare_cpu_input_gate_b5c2/tokenize_cpu.py`：规范化、截断与固定模板参数。
- `evaluation/strict_compatibility_b4b0/{capture.py,contracts.py}`：Prepared时间、先冻结后核对接收身份、完整三态关系与未知拒绝机制；不能原样绕过其g001–g006冻结范围，新增适配应独立。
- `evaluation/multishare_runtime_gate_b5c3/{runner.py,bridge.py,historical_replay_log.json,strict_epoch_snapshot.json,physical_state_census.json}`：8条历史实际接收身份、全树唯一物理候选和同期严格边。
- `flowstate/{strict_epoch.py,strict_control.py}`、`evaluation/runtime_control_closure_r2c0/bridge.py`：Token/配置/epoch校验、唯一物理绑定、连续窗口、组件身份及逐项后置检查。
- `evaluation/exact_reconciliation_correctness_gate_b5d3/{cross_workflow_resume_validation.md,output_correctness.json,state_integrity_report.md}`：同一固定来源中目标B的真实匹配、COW及8 Token Cold结果。
- `evaluation/runtime_control_closure_r2c1b2/{req_api_audit.md,observation.py,output_correctness.json}`：真实Req API、执行区间观测和输出合同。
- `/home/wjg/code/TraceElephant/code/agent_system/{SWE-agent,Captain-Agent,Magentic-One}/`与`code/llm_api_middleware/`：只读的Agent/工具及API日志接入点；未运行这些Agent。

关键文件本轮SHA-256：

|文件|SHA-256|
|---|---|
|C2/source_selection_lock.json|`12865cd0f523a55c871030110742a09c80c0846e83314b8a05134c6ad644399e`|
|C2/input_configuration_lock.json|`20c150de9ad4513eb865f02ef0edc3e6e6cc4cb81e7fbc5422372fed602ac461`|
|C3/strict_epoch_snapshot.json|`3b41f232b4ee4d79aab081643c6c2adad1671d0364841617e849f52b6c16ac7a`|
|B5-D3/output_correctness.json|`095e64f3419635b289512d739fb950041771303442b21be4a015e6e8ddf56b35`|

**裁决适用范围：PARTIAL。** 完整真实历史、固定目标输入和严格物化/恢复组件已具备，可进入独立的两Pending受控重放适配；Prepared事件、唯一物理候选归属和新边界认证仍需补齐。原生Qwen Agent在线多轮工具闭环及原生Fork尚未认证，不能通过重命名旧重放取得READY。完成本审查后停止，不自动运行GPU或实现接线。
