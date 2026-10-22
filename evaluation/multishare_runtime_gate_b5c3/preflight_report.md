# CPU与资源预检查

PASS：B5-C2全部摘要、固定12份输入、八请求顺序及既有R2观察源码一致。原模型权重/配置按既有来源锁检查。GPU0为H100 PCIe、无计算进程，显存低于1000 MiB；镜像ID与B4/C2一致。

真实SGLang源码、版本及模板在runner创建Engine之前逐项核验；未通过则不创建Engine。历史参数沿用temperature=0、max_new_tokens=1、ignore_eos=True。

使用R2-C0已验证scope/observe和B1-C原始match/COW只读记录；FA直接只读遍历，前后完整物理视图与组件内容复核。没有调用R2控制计划、bind、reconcile或驱逐入口。新适配只负责全树枚举和证据序列化。
