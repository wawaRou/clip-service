# CLIP Service

使用 uv 管理的跨平台 CLIP 视觉变化检测服务。当前仅完成项目骨架，尚未迁移
`test_clip` 的视频读取、HTTP API、模型编码与候选检测功能。

## 目录

```text
src/clip_service/   Python 包与命令行入口
tests/            测试目录（迁移时补充）
docs/             平台部署与开发约定
pyproject.toml    项目、平台依赖、工具配置
uv.lock           依赖锁文件（提交到 Git）
```

## 初始化与运行

基础开发环境不安装 PyTorch，适合先整理业务代码：

```bash
uv sync --locked --python 3.12
uv run --locked clip-service --help
uv run --locked clip-service --version
uv run --locked python -m clip_service --version
```

入口目前只提供帮助与版本信息，不启动服务。

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
uv build
```

新增公共业务依赖使用 `uv add`；开发工具使用 `uv add --group dev`。
更新依赖后提交 `pyproject.toml` 与 `uv.lock`。推理依赖由平台 extra 统一管理，
不在 `uv sync` 后另外手工覆盖 torch。

详细版本、系统库要求和验证状态见 [平台部署说明](docs/platforms.md)。
