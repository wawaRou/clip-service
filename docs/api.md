# Agent Server 接口

默认地址为 `http://127.0.0.1:18080`。摄像头标识来自 TOML 的 cameras 表；所有带
会话的请求均使用 Agent Server 自行生成的非空 episode_id。沿用旧项目的 HTTP/SSE
协议，CLIP 服务只生成变化候选，不调用 VLM。

| 操作 | 方法及路径 |
| --- | --- |
| 查询状态 | GET /api/v1/health |
| 订阅所有摄像头事件 | GET /api/v1/events |
| 开始会话 | POST /api/v1/cameras/{camera}/episodes |
| 设置基准 | PUT /api/v1/cameras/{camera}/baseline |
| 提取时间窗图片 | POST /api/v1/cameras/{camera}/frame-windows |
| 查询候选 | GET /api/v1/candidates/{candidate_id} |
| 获取候选图片 | GET /api/v1/candidates/{candidate_id}/frames/{index} |
| 回执 | POST /api/v1/candidates/{candidate_id}/ack |
| 结束会话 | DELETE /api/v1/cameras/{camera}/episodes/{episode_id} |

先订阅 SSE，再开始会话和设置基准。例如摄像头 front_door：

```bash
curl -N http://127.0.0.1:18080/api/v1/events
```

另一个终端：

```bash
curl -H 'Content-Type: application/json' -d '{"episode_id":"motion-1"}' \
  http://127.0.0.1:18080/api/v1/cameras/front_door/episodes
curl -X PUT -H 'Content-Type: application/json' \
  -d '{"episode_id":"motion-1","source":{"type":"latest"}}' \
  http://127.0.0.1:18080/api/v1/cameras/front_door/baseline
```

基准 source 支持最新有效帧 latest、指定候选图片 candidate（candidate_id、frame_index），
以及 jpeg_base64（data）。候选来源必须属于同一摄像头和会话，陈旧最新帧返回 503。

SSE 的 candidate_change 数据包含 camera、episode_id、candidate_id、相似度和五张
图片的 frames 列表。通知发出时，列表中的 url 已可通过 HTTP 读取。Agent Server 获取
图片、决定是否触发 VLM 后向候选的 ack 路径提交 episode_id、布尔 triggered_vlm，
以及可选 baseline（格式与 source 相同）。省略 baseline 将沿用原基准继续检测。

时间窗请求包含 episode_id、window_start、window_end、frame_count（1—9）。返回均匀
目标时刻附近的 JPEG base64；单帧取窗口中点，缺失任一目标帧返回 503。时间戳为
CLIP 主机收到画面的 Unix 秒数，不代表摄像头采集时间；跨机器调用应使用一致时钟，
并考虑 Frigate 转发与网络延迟。

默认确认时间为 0.3 秒、五帧间隔为 0.075 秒，均可通过 detection 配置覆盖。这些是
可运行的初始值，仍须用真实画面验证误报和延迟。模型不可用时基准请求返回 503，
健康信息包含该路 inference_error；修复模型环境后可重试设置基准。
