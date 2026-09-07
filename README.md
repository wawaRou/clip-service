# CLIP Service

使用 uv 管理的跨平台 CLIP 视觉变化检测服务。当前已实现 TOML 配置、Frigate
视频读取、检测会话、候选图片、HTTP 与 SSE 通知，以及多路共享模型的目标 FPS
调度和运行指标。异常恢复和动态重载按后续任务完善。

## 目录

```text
src/clip_service/   Python 包与命令行入口
tests/            配置与服务接口测试
docs/             平台部署与开发约定
pyproject.toml    项目、平台依赖、工具配置
uv.lock           依赖锁文件（提交到 Git）
```

## 初始化与运行

基础开发环境不安装 PyTorch，可以运行拉流和健康查询：

```bash
uv sync --locked --python 3.12
uv run --locked clip-service --help
uv run --locked clip-service --version
uv run --locked python -m clip_service --version
```

复制 `config.example.toml` 为 `config.local.toml`，填写 Frigate 转发地址和所选流：

```bash
cp config.example.toml config.local.toml
uv run --locked clip-service --config config.local.toml
curl http://127.0.0.1:18080/api/v1/health
```

Frigate 需要配置对应 go2rtc 转发流，默认 RTSP 端口 8554 必须能从服务所在机器访问。
每路独立拉流并维护短时 JPEG 缓存；某路断流会自动重连，健康信息包含连接状态、
缓存帧数和帧年龄。按 Ctrl+C 关闭服务。

配置以全局默认值为基础，`cameras` 中可覆盖检测参数。空摄像头列表合法，不会连接
示例摄像头。未知字段和非法值会在启动时被拒绝；相对模型及数据路径以配置文件所在
目录为基准。机器配置和凭据存放在已忽略的 `config.local.toml` 或环境变量中。

部署环境变量优先于 TOML，TOML 优先于内置默认值：`CLIP_MODEL_PATH`、`CLIP_DATA_DIR`、
`CLIP_DEVICE`、`CLIP_FRIGATE_URL`、`CLIP_FRIGATE_USERNAME`、`CLIP_FRIGATE_PASSWORD`。
RTSP 用户名和密码单独配置，禁止直接嵌在转发地址中；运行日志只报告摄像头标识与
概括错误。原生解码器默认静音，主动开启其诊断日志时需要自行保护其中的流地址。

需要推理依赖时，按机器选择一个 extra。每次运行继续传入相同 extra，避免 uv
同步时移除推理依赖。平台及 Python 版本必须按下表选择，extra 不负责自动识别硬件。

| 平台 | Python | 安装命令 | 运行命令 |
| --- | --- | --- | --- |
| Apple Silicon Mac | 3.12 | `uv sync --locked --python 3.12 --extra mac` | `uv run --locked --extra mac clip-service --help` |
| Jetson Thor | 3.12 | `uv sync --locked --python /usr/bin/python3.12 --extra thor` | `uv run --locked --extra thor clip-service --help` |
| Jetson Orin | 3.10 | `uv sync --locked --python /usr/bin/python3.10 --extra orin` | `uv run --locked --extra orin clip-service --help` |

三组 extra 互斥，不使用 `--all-extras`。未提交统一的 `.python-version`，避免 Mac / Thor
的默认 Python 版本覆盖 Orin 的 3.10。Jetson 使用系统解释器创建隔离的 `.venv`，
不启用 `--system-site-packages`，不向系统 Python 安装包。

## 开发检查

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright
uv run --locked pytest
uv build
```

新增公共业务依赖使用 `uv add`；开发工具使用 `uv add --group dev`。
更新依赖后提交 `pyproject.toml` 与 `uv.lock`。推理依赖由平台 extra 统一管理，
不在 `uv sync` 后另外手工覆盖 torch。

详细版本、系统库要求和验证状态见 [平台部署说明](docs/platforms.md)。

检测前在配置中指定本地 CLIP 模型路径，并使用对应平台 extra 启动服务。模型在首次
设置基准图时加载；未开启会话或未设置基准时只拉流缓存，等待候选回执时暂停该路检测。
完整交互见 [Agent Server 接口说明](docs/api.md)。
