# 测试

通过配置加载和公开服务/HTTP 接口验证行为。视频输入使用外部摄像头连接替身，输出
真实图像像素以覆盖 JPEG 编码与缓存；模型接入后的确定性测试替换图像编码边界，
硬件推理另行实测。测试不依赖用户真实摄像头，也不按内部私有方法组织。

运行单文件：`uv run --locked --extra mac pytest tests/test_startup.py`。
完整回归：`uv run --locked --extra mac pytest`。Thor / Orin 使用对应 extra。
不安装推理依赖也能运行多数测试，但真实模型加载测试会跳过，不能据此称完整回归通过。

真实硬件与 Frigate 流验收通过 `python tests/manual_native.py --help` 查看参数；它需要
相应平台 extra、本地 CLIP 模型和可用转发流，不属于普通 pytest 自动回归。
脚本仅保留 JSON 报告，临时配置、候选图片和服务进程在结束时清理。
