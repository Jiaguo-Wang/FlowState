# Task 0：实验环境与模型路径只读门禁（2026-10-29）

## 0. 最终结论：**READY**

四项检查全部通过：根盘 72G 余量充足（预计 P0-4 新增产物 ~27–35G）；两个模型符号链接在宿主机与**正式容器挂载方式下**均全分片实读通过；GPU 双卡空闲；冻结镜像逐位匹配。可进入 Task 1 与后续 GPU 实验。

---

## 1. 磁盘状态

| 文件系统 | 容量 | 已用 | 可用 | 用途 |
|---|---:|---:|---:|---|
| `/dev/sda2`（`/` 与 `/home`） | 1759G | 1599G | **72G** | FlowState 仓库、实验产物、两个模型 symlink 头 |
| `/dev/sdb1`（`/data1`） | 14784G | 10532G | **3507G** | docker 镜像/容器、共享模型库实体 |

**实验输出目录所在文件系统**：`evaluation/` 在 sda2；正式臂的 `/b5out` 是指向 `evaluation/<实验>/<臂>/` 的 bind mount（证据：`formal_gpu1b/g_CB_FS_command.json` 的 `--mount type=bind,src=.../g_CB_FS,dst=/b5out`），故 P0-4 产物落在 **sda2**。

**P0-4 新增存储估算**（依据冻结实测，非推测）：
- T2-G2 formal_gpu1b：60 臂共 27G → **~0.45G/臂**（每臂=完整 docker 生命周期+重置协议+1 请求）；
- P0-4：3 策略 × 5 轮 = 15 个 docker 生命周期，每周期含重置双份 replay + apply + 4 Pending 请求，输出 ≈ 4 臂当量 → 15 × 4 × 0.45 ≈ **27G**，加上重置/取证余量按 **~35G** 上界规划；
- 72G − 35G = 剩余 ≥37G 缓冲。结论：**充足**。

注意事项（不改结论）：sda2 为多用户共享盘且此前观测到其他用户持续增长（曾一天内吃掉数十 G）。**实验启动前与每轮之间必须 `df -B1G /` 复查**，低于 20G 时暂停并协调。

---

## 2. 模型符号链接（宿主侧 + 容器侧双重核验）

### 2.1 真实解析路径

| 路径 | 链接链 | 终点 |
|---|---|---|
| `/home/wjg/models/qwen3.5-9b` | 单跳 symlink | `/data1/models/Qwen3.5-9B`（真实目录） |
| `/home/wjg/models/qwen38-27b` | **双重 symlink** | `/data1/models/Qwen3.8-27B` → `/data1/models/.cache/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master` |

证据：`readlink`/`readlink -f`/`os.path.realpath` 三种方式一致（qwen38-27b 的双重链为本次检查新确认的事实——共享库条目本身指向 modelscope 快照）。

### 2.2 宿主侧全分片实读（Python 逐分片 `open`+实读 16 字节，非 du/find）

| 模型 | index 声明分片 | 全部实读 | 权重合计 | 附带文件 |
|---|---:|---|---:|---|
| Qwen3.5-9B（qwen3_5） | 4/4 | **PASS** | 19.31 GB | tokenizer_config/tokenizer OK；generation_config **本就缺失**（替换前旧副本亦无，60/60 正式臂照常运行，非回归） |
| Qwen3.8-27B（qwen3_5） | 18/18 | **PASS** | 55.56 GB | tokenizer_config/tokenizer/generation_config 全 OK |

### 2.3 一致性验证的诚实边界

替换 symlink 前完成的核对为：**两副本分片清单逐字节同大小（4/4 与 18/18 全部分片）+ config.json/tokenizer_config.json md5 一致 + 首分片头/中/尾 3×64MB 抽样 md5 一致**。抽样哈希不等于全权重逐字节一致性验证；分片字节数完全相同 + 配置 md5 相同 + 抽样命中，是"同一 hub 版本"的强证据而非密码学证明。若需逐字节证明，可后续跑全量 md5（9B ~1 分钟、27B ~3 分钟，本轮未做）。

---

## 3. Docker 实际访问（决定性测试）

### 3.1 正式 Runner 的真实挂载参数（读冻结命令记录，非模板推测）

`formal_gpu1b/g_CB_FS_command.json`（T2-G2 正式臂）实际参数：

```
docker run --rm --pull never --gpus device=GPU-051c9abc-... --network host --shm-size 32g
  --mount type=bind,src=/home/wjg/code/FlowState,dst=/home/wjg/code/FlowState,readonly
  --mount type=bind,src=/home/wjg/models/qwen3.5-9b,dst=/home/wjg/models/qwen3.5-9b,readonly
  --mount type=bind,src=.../g_CB_FS,dst=/b5out
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 ...
```

即模型按**符号链接原路径** bind mount——docker 在创建挂载时于宿主侧解析 symlink，容器内该路径成为解析后目标的真实目录挂载。

### 3.2 最小只读容器测试（同镜像 + 同挂载方式，不挂 GPU、不加载权重）

用 `lmsysorg/sglang:v0.5.17-cu129-runtime` + 与正式臂相同的两条模型 bind mount（`--network none`、无 `--gpus`），容器内 Python 逐分片实读：

| 模型 | 容器内 realpath | 分片实读 | 合计 |
|---|---|---|---:|
| qwen3.5-9b | `/home/wjg/models/qwen3.5-9b`（挂载点即真实目录） | 4/4 **PASS** | 19.31 GB |
| qwen38-27b | `/home/wjg/models/qwen38-27b`（同上） | 18/18 **PASS** | 55.56 GB |

**结论**：symlink 在正式挂载方式下正确解析（宿主侧解析、容器内无悬空链），分片合计与宿主侧逐位一致。spec 提醒的"不能假设所有 bind mount 配置下自动正确解析"已用实测排除：若未来有 Runner 改为挂载**父目录** `/home/wjg/models`（symlink 在容器内遍历），目标 `/data1/...` 容器内不可见会悬空——现有正式 Runner 均为按模型路径直挂，不受影响。

---

## 4. GPU 与实验依赖

| 检查 | 结果 | 证据 |
|---|---|---|
| GPU | 2× H100 PCIe 均**空闲**（GPU0 14MiB/GPU1 17MiB，0% util，零计算进程） | `nvidia-smi --query-gpu/--query-compute-apps` |
| P0-4 目标卡 | GPU1（`GPU-051c9abc-…`，T2-G2 同卡）当前空闲 | 同上 |
| 冻结镜像 | `lmsysorg/sglang:v0.5.17-cu129-runtime` = `sha256:1dfce9033670a84c…`，与 G1/T2-G2 冻结记录 `sha256:1dfce…` 逐位一致 | `docker inspect` |
| 在跑容器 | 仅 cs336/tempo 监控栈（4 个，与实验无关） | `docker ps` |
| 未做 | 未终止任何进程、未重启 docker、未拉取镜像、未加载模型、未启动 GPU 推理 | — |

---

## 5. 证据索引

| 检查 | 命令/文件 | 结果 |
|---|---|---|
| 磁盘余量 | `df -B1G / /data1` | 72G / 3507G |
| 输出文件系统 | `g_CB_FS_command.json` 第三条 mount | sda2（`/b5out`→仓库内臂目录） |
| 产物估算 | formal_gpu1b 27G/60 臂实测 | ~0.45G/臂 → P0-4 ~27–35G |
| 链接解析 | `readlink -f` + `os.path.realpath` | 9B 单跳 / 27B 双跳，均落到 /data1 实体 |
| 宿主分片实读 | Python `open`+read16 循环 | 4/4、18/18 PASS |
| 容器分片实读 | 同镜像同 mount 的 `docker run --rm` | 4/4、18/18 PASS，合计逐位同宿主 |
| 挂载真实性 | `formal_gpu1b/g_CB_FS_command.json` | 按模型路径直挂 readonly |
| GPU/镜像/容器 | `nvidia-smi`、`docker inspect`、`docker ps` | 双卡空闲、sha 一致、无实验容器 |

## 6. 结论与放行

**READY**——模型可由实际实验容器按正式挂载方式读取、磁盘余量充足、GPU 与冻结镜像就绪。允许进入 Task 1 与后续 GPU 实验。

两条随行纪律（不阻塞）：①每次实验启动前与轮间复查 `df -B1G /`（共享盘有并发增长史，低于 20G 暂停）；②若未来任何 Runner 改为挂载模型父目录而非模型路径本身，必须重跑本节 3.2 的容器内解析测试。
