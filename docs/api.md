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
| 查询当前未处理候选 | GET /api/v1/candidates?status=pending |
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

## 检测频率和处理延迟

所有摄像头共享一份模型。detection.inference_fps 是每路默认目标频率，单路可覆盖；
服务按各路固定周期处理最新且未处理的有效帧。ring_max_fps 只控制历史缓存采样上限，
推理直接读取最新解码帧；源视频必须提供足够新帧。默认目标为 10 FPS，100 FPS 未经实测，
不是默认能力。

健康响应中每路 inference 字段提供：

- target_fps：配置的目标检测频率。
- actual_fps、measurement_seconds：最近最多 5 秒内完成的检测次数除以窗口时长；
  启动不足 5 秒时使用已运行时长。暂停检测和等待回执也包含在该时间窗口中。
- frames_processed：该路累计完成的检测编码次数，不含基准图编码。
- last_processing_seconds：最近一次检测编码从开始到完成的耗时，包括预处理、模型
  推理及输出特征处理。
- last_frame_latency_seconds：最近一次检测从本机收到帧到编码完成的耗时；使用单调
  时钟测量，不包含摄像头采集和 Frigate 转发延迟，也不是候选通知的完整延迟。

算力不足时跳过过期检测周期，公平处理各路；不会积压历史帧。目标 FPS 相同不代表
所有输入条件下实际 FPS 必然相等，应结合断流、帧年龄、会话状态和处理耗时判断。

## 断线、超时与重启

SSE 只广播当前在线事件，不保存历史事件。Agent Server 连接或重连时先建立 SSE，
再查询 `GET /api/v1/candidates?status=pending`，以 candidate_id 去重并处理返回的
`candidates` 列表。这样可找回断线期间仍未过期的候选，期间也可能同时收到相同 SSE
通知。省略 status 时同样只查询当前未处理候选。

候选的 ack_deadline_at 是回执截止时间，expires_at 是图片保留截止时间。默认回执等待
60 秒，图片保留 24 小时，两者分别配置；等待期限或图片保留期限先到时，候选过期，
服务保留原基准继续检测，所以同一画面变化可能再次产生候选。图片仍在保留期内时，
旧候选可查询为 expired，迟到回执返回冲突，不会更新基准或清除新候选。

已成功回执的候选在记录保留期内可以安全重试，相同 episode_id 和 triggered_vlm 返回
已保存的结果，baseline 不再解析或应用；冲突的处理结果返回 409。并发编码期间若会话
或基准已改变，请求会返回状态冲突，Agent Server 应重新查询当前状态，而非假定写入成功。

CLIP 重启后自动恢复拉流与缓存，旧 pending 候选标记为 cancelled，不恢复会话和基准。
Agent Server 根据健康信息重新开启会话、设置基准，再开始处理候选。候选存储错误会
使请求失败或健康状态降级；后台会重试到期清理，不把持久化失败报告为成功。
