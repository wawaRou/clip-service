# CLIP Service

使用 uv 管理的跨平台 CLIP 视觉变化检测服务。当前已实现 TOML 配置、Frigate
视频读取、检测会话、候选图片、HTTP 与 SSE 通知，以及多路共享模型的目标 FPS
调度和运行指标、断线对账、回执超时、重启恢复，以及运行中增删改摄像头。

## 目录

```text
src/clip_service/   Python 包与命令行入口
tests/            配置与服务接口测试
docs/             平台部署与开发约定
pyproject.toml    项目、平台依赖、工具配置
uv.lock           依赖锁文件（提交到 Git）
```

## 初始化与运行

先按下方平台表安装对应 extra。以下以 Mac 为例，复制完整示例，修改 Frigate 地址、
摄像头标识及流名，并确认所选模型已经缓存在本机：

```bash
uv sync --locked --python 3.12 --extra mac
cp config.example.toml config.local.toml
# 编辑 config.local.toml 后启动：
uv run --locked --extra mac clip-service --config config.local.toml
```

另一个终端查询状态：

```bash
curl http://127.0.0.1:18080/api/v1/health
```

Thor / Orin 使用下表对应的解释器与 extra。每次运行都保留 extra；省略它可能使 uv
同步时移除 PyTorch。只开发拉流和 HTTP 功能时，可以不选 extra 安装基础依赖，
但此环境无法进行模型推理。

Frigate 需要配置对应 go2rtc 转发流，默认 RTSP 端口 8554 必须能从服务所在机器访问。
每路独立拉流，检测激活时，解码后的 BGR 帧进入最多 6 帧的推理队列并唤醒共享推理线程；
各路按就绪状态公平轮转，并保留各自的推理频率上限。另按采样率编码短时 JPEG 缓存；
某路断流会自动重连，健康信息包含连接状态、
缓存帧数和帧年龄。按 Ctrl+C 关闭服务。

当前使用 OpenCV 的 FFmpeg 后端软件解码，尚未启用 GPU 硬件解码。
HTTP/SSE 内置默认只监听本机；示例配置使用 `0.0.0.0` 监听所有 IPv4 接口。
服务不提供认证或 TLS；跨机器部署应限制在可信网络，
并通过访问控制或带认证的反向代理保护接口。

`detection.stable_seconds` 是唯一的持续变化窗口，默认 1 秒；确认后从这段窗口内
均匀选取 5 张图片给上层。旧配置须删除 `candidate_frame_interval`，只保留窗口时长。

配置以全局默认值为基础，`cameras` 中可覆盖检测参数。删除全部 `[cameras.<标识>]`
表即可空摄像头启动；示例中的 `front_door` 若保留则会实际尝试连接。未知字段和非法值
会在启动时被拒绝；相对模型及数据路径以配置文件所在目录为基准。机器配置和凭据存放在已忽略的 `config.local.toml` 或环境变量中。

部署环境变量优先于 TOML，TOML 优先于内置默认值：`CLIP_MODEL_PATH`、`CLIP_DATA_DIR`、
`CLIP_DEVICE`、`CLIP_FRIGATE_URL`、`CLIP_FRIGATE_USERNAME`、`CLIP_FRIGATE_PASSWORD`。
RTSP 用户名和密码单独配置，禁止直接嵌在转发地址中；运行日志只报告摄像头标识与
概括错误。原生解码器默认静音，主动开启其诊断日志时需要自行保护其中的流地址。

修改启动时使用的 TOML 后，显式重载摄像头配置：

```bash
curl -X POST http://127.0.0.1:18080/api/v1/config/reload
```

配置错误时保留旧配置。普通检测参数更新保留会话与候选；换流、禁用或删除摄像头
会结束该路会话，并发送 `episode_ended` 事件。摄像头数量没有固定上限，新增后由
Agent Server 开启会话并设置基准。模型、监听地址、数据目录与保留时间的修改需要
重启。重载响应列出应用的变更及每路当前状态，流是否可用还需查看健康信息。

需要推理依赖时，按机器选择一个 extra。每次运行继续传入相同 extra，避免 uv
同步时移除推理依赖。平台及 Python 版本必须按下表选择，extra 不负责自动识别硬件。

| 平台 | Python | 安装命令 | 运行命令 |
| --- | --- | --- | --- |
| Apple Silicon Mac | 3.12 | `uv sync --locked --python 3.12 --extra mac` | `uv run --locked --extra mac clip-service --config config.local.toml` |
| Jetson Thor | 3.12 | 先执行下方 wheel 准备命令，再 `uv sync --locked --python /usr/bin/python3.12 --extra thor` | `uv run --locked --extra thor clip-service --config config.local.toml` |
| Jetson Orin | 3.10 | `uv sync --locked --python /usr/bin/python3.10 --extra orin` | `uv run --locked --extra orin clip-service --config config.local.toml` |

三组 extra 互斥，不使用 `--all-extras`。未提交统一的 `.python-version`，避免 Mac / Thor
的默认 Python 版本覆盖 Orin 的 3.10。Jetson 使用系统解释器创建隔离的 `.venv`，
不启用 `--system-site-packages`，不向系统 Python 安装包。

## 开发检查

```bash
uv run --locked --extra mac ruff check .
uv run --locked --extra mac ruff format --check .
uv run --locked --extra mac pyright
uv run --locked --extra mac pytest
uv build
```

上述完整检查以 Mac 为例；其他机器替换为对应 extra，保留真实模型测试所需的推理依赖。

默认安装的 `dev` 依赖组包含实验绘图所需的 Matplotlib，Pyright 同时检查服务源码和
`docs/validation/plot_thor_capacity.py`。VS Code 应选择本项目 `.venv` 的 Python。
Thor 首次安装（或删除本地 `vendor/*.whl` 后）先执行：

```bash
/usr/bin/python3 scripts/repair_thor_wheel.py
```

脚本校验官方 cuSPARSELt 0.8.1 原包并修正平台元数据，不改变 CUDA 二进制。
生成文件不提交 Git；新机器必须先准备，再执行 uv。详见
[Thor wheel 修复](docs/platforms.md#thor-wheel-元数据修复)。

生产部署需在同步和运行时都保留 `--no-dev`，排除开发工具和绘图依赖；Thor 示例：

```bash
uv sync --locked --extra thor --no-dev
uv run --locked --extra thor --no-dev clip-service --config config.local.toml
```

其他平台替换对应 extra。省略运行命令中的 `--no-dev` 会重新同步默认开发依赖。

新增公共业务依赖使用 `uv add`；开发工具使用 `uv add --group dev`。
更新依赖后提交 `pyproject.toml` 与 `uv.lock`。推理依赖由平台 extra 统一管理，
不在 `uv sync` 后另外手工覆盖 torch。

详细版本、系统库要求和验证状态见 [平台部署说明](docs/platforms.md)。

完整可配置项及默认值见 [config.example.toml](config.example.toml)。默认通过
`model.id = "openai/clip-vit-base-patch16"` 从 Hugging Face 本地缓存读取模型，遵循
`HF_HOME` / `HF_HUB_CACHE`；也可设置 `model.path` 指定本地模型目录（优先于 ID）。
缓存缺失或模型文件不完整时，首次加载直接失败，绝不联网下载。模型在首次
设置基准图时加载，失败时该请求返回 503；服务仍可启动并提供健康查询。
未开启会话或未设置基准时只拉流缓存，等待候选回执时暂停该路检测。
完整交互见 [Agent Server 接口说明](docs/api.md)。
原生环境验收、性能测量与未验证项目见 [运行验收](docs/validation.md)。

## 推理精度

通过 TOML 的 `model.precision` 选择，默认 `fp32`。例如 Thor：

```toml
[model]
id = "openai/clip-vit-base-patch16"
device = "cuda"
precision = "tf32"
```

| 配置值 | 支持后端 | 行为 |
| --- | --- | --- |
| `fp32` | CPU、MPS、CUDA | FP32 权重和输入；CUDA 矩阵乘法及卷积禁用 TF32 |
| `tf32` | CUDA | 权重和输入仍为 FP32，允许矩阵乘法及卷积使用 TF32 |
| `fp16` | CUDA、MPS | 权重和输入使用 FP16；输出特征转换为 FP32 后归一化 |

精度由所有摄像头共享，修改后需要重启；重载接口会返回 `restart_required`。
`device = "auto"` 仍按原设备选择规则解析，不会为了精度改选后端；组合不支持时明确
报错，不自动降级。健康接口的 `model.precision` 显示选择值，加载后的 `weight_dtype`
显示实际权重类型（加载前为 null）。CUDA 精度开关是进程级设置，服务只持有一份共享模型。

TF32、FP16 可能改变相似度，调整后应观察阈值附近的候选判断；目标 FPS 仍由
`detection.inference_fps` 单独控制。原生验收脚本可传 `--precision tf32` 或
`--precision fp16` 复测完整流程。

## 候选文件存储

`data_dir = "data"` 相对于配置文件所在目录；配置位于项目根目录时，候选保存在
`data/candidates/<候选ID>/`。每个候选包含 5 张 JPEG 和记录时间、相似度及处理状态的
`manifest.json`，用于 Agent 获取图片、查询状态和重试回执。

`retention_hours = 24.0` 从候选创建时开始计算图片及记录的保留期限。到期后接口不再
提供候选，后台清理已结束的记录；仍处于 pending 状态时暂缓物理删除，等待状态结束。
回执等待时间单独由 `detection.ack_timeout_seconds` 控制，默认 60 秒。
短时视频帧缓存位于内存，由 `ring_seconds` 等参数控制；本服务不保存连续录像。
