# GPU1设备绑定CPU预检

READY：冻结tp_size=1，28槽；输入、配置、协议和选集逐字节一致。

Docker --gpus、CUDA_VISIBLE_DEVICES和所有nvidia-smi检查均指定GPU1 UUID。容器唯一可见CUDA设备为逻辑0，加载模型前校验UUID。原Runner、Bridge、Control不变，其torch.cuda当前设备操作均位于单卡可见域。无GPU0物理索引操作。控制端口50087、容器名独立。

GPU可见性运行证据逐臂保存在device_binding.json；CPU预检不冒称已通过GPU运行。旧GPU0产物不复制进统计。
