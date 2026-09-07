# 测试

通过配置加载和公开服务/HTTP 接口验证行为。视频输入使用外部摄像头连接替身，输出
真实图像像素以覆盖 JPEG 编码与缓存；模型接入后的确定性测试替换图像编码边界，
硬件推理另行实测。测试不依赖用户真实摄像头，也不按内部私有方法组织。

运行单文件：`uv run --locked pytest tests/test_startup.py`。
完整回归：`uv run --locked pytest`。
