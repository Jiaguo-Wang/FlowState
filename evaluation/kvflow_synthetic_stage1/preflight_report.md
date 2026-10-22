# 预检查与固定协议

- 工作负载：按KVFlow论文§4.1机制独立重建，10个Agent顺序调用、随机Token固定前缀及变化后缀。不是官方数据集或官方性能复现。
- 已检查本地官方提交7ef897eaedc16626a281ccfe0d453fae80251e8f的benchmark/test；未找到可直接复现论文10-Agent固定/动态生成配置的入口，未使用通用HiCache脚本冒充对应实现。
- 种子20261021；4096固定Token、32动态Token；两轮预先声明的完整工作流输入；两次固定提示预热；预算3与5。
- Qwen3.5-9B词表来自本地tokenizer.json，排除special added tokens。直接输入Token IDs，不做Chat Template渲染。配置及模板摘要仍保存并核对。
- uv312的Transformers导入因huggingface-hub版本冲突失败。未安装或升级依赖；生成器改用原Token词表直接采样，符合论文随机Token机制。Runtime使用既有锁定Docker镜像及原依赖。
- 16项新增CPU合同检查通过；KVFlow策略原实现未修改。CPU候选为PROPOSED，STE为0至9；只有第一调用Ready，其他输入预先知道，但要等待顺序依赖，不称为十个同时Ready Pending。
- GPU0型号H100 PCIe，预启动无计算进程，显存14 MiB；详见strict_gpu_before.json。
- 使用原SGLang v0.5.17配置、TP=1、28 Recurrent Slots。Runtime只尝试K3物理协调及Agent00恢复，再做独立Cold对照。不是容量压力或性能Benchmark。
- 本轮必须自然出现固定前缀末端；候选域包含未知或额外无法归类状态时即停止，不修改缓存插入或删减候选凑结果。
- 观察、Bind、Preflight、Reconciliation、逐项Postcheck复用已有安全代码。新实验Hook仅在独立子进程内替换实验入口，不修改SGLang磁盘源码。
- 请求恢复复用D3的Admission、Native Match、COW内容核验、FA保护及8 Token Cold比较协议。

来源：https://arxiv.org/html/2507.07400#S4.SS1 ，https://github.com/PanZaifeng/KVFlow 。
