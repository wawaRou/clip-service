# 测试

通过配置加载和公开服务/HTTP 接口验证行为。视频输入使用外部摄像头连接替身，输出
真实图像像素以覆盖 JPEG 编码与缓存；模型接入后的确定性测试替换图像编码边界，
模型测试会现场生成小型随机 CLIP 权重，验证 CPU 和可用 GPU 的编码；真实预训练模型
及 Frigate 流另行实测。测试不依赖用户真实摄像头，也不按内部私有方法组织。

运行单文件：`uv run --locked --extra mac pytest tests/test_startup.py`。
完整回归：`uv run --locked --extra mac pytest`。Thor / Orin 使用对应 extra。
不安装推理依赖也能运行多数测试，但真实模型加载测试会跳过，不能据此称完整回归通过。

CUDA 测试在 Mac 上跳过，MPS 测试在 Jetson 上跳过，这是平台差异；应单独记录
跳过原因。pytest 不需要下载 Hugging Face 预训练模型。

帧驱动调度的定向回归：
`uv run --locked --extra thor pytest tests/test_scheduler.py tests/test_reader_settings.py tests/test_frame_driven.py tests/test_multi_camera.py`。
覆盖到帧唤醒、突发帧顺序、多路轮转、限频、队列丢帧、断流恢复和会话状态隔离。

`tests/test_reload.py` 与 `tests/test_recovery.py` 还覆盖摄像头删除、禁用或服务重启后
的已完成回执重试，确认返回持久化结果、拒绝冲突且不重新应用基准。

真实硬件与 Frigate 流验收通过
`uv run --locked --extra mac python tests/manual_native.py --help` 查看参数；它需要
相应平台 extra、本地 CLIP 模型和可用转发流，不属于普通 pytest 自动回归。
脚本仅保留 JSON 报告，临时配置、候选图片和服务进程在结束时清理。
