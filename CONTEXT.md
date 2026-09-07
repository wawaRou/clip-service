# CLIP 视觉变化检测

本项目处理经 Frigate 接入的摄像头画面，识别需要进一步关注的视觉变化。

## Language

**摄像头（Camera）**：提供某个观察视角的设备。在本项目中，所有摄像头画面均来自 Frigate。

**Frigate**：统一接入本项目所有摄像头的视频系统。

**检测会话（Episode）**：外部 Agent Server 为某台摄像头开启的一段视觉变化检测任务。

**基准图（Baseline）**：由外部 Agent Server 指定、作为该摄像头画面比较参照的图像。

**变化候选（Candidate）**：相对基准图的视觉变化线索，供外部 Agent Server 决定是否进一步调用 VLM。候选不代表已确认的语义事件。

**回执（Acknowledgement）**：外部 Agent Server 对变化候选的处理确认，可以同时指定后续比较使用的新基准图。
