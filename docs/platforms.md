# 原生平台环境

所有平台直接使用 uv 虚拟环境，不使用容器。uv 锁定 Python 包及其来源；JetPack、
驱动、CUDA 和系统共享库由机器环境提供，不属于 `uv.lock` 的管理范围。

## 版本基准

| 平台 | 系统 | Python | torch | torchvision | 状态 |
| --- | --- | --- | --- | --- | --- |
| Mac / MPS | Apple Silicon macOS | 3.12 | 2.13.0 / PyPI | 无需安装 | 2026-09-07 历史版本闭环通过；当前帧驱动实现待 MPS 复验 |
| Thor / CUDA | L4T R39.2 / JetPack 7 系列 / CUDA 13.2 | 3.12 | 2.14.0+cu132 | 0.29.0+cu132 | 原生 CLIP 与 Frigate 单路、三路逻辑摄像头闭环通过 |
| Orin 64GB / CUDA | L4T 36.4.4 / JetPack 6.2.1 / CUDA 12.6 | 3.10 | 2.8.0 | 0.23.0 | 用户指定暂定固定，上机验证待完成 |

公共业务依赖已纳入同一份锁文件，Mac 使用 transformers 的 PIL 图像预处理器。
共享源码保持 Python 3.10 兼容。锁定目标限制为 macOS ARM64 / Python 3.12，
以及 Linux ARM64 / Python 3.10、3.12。平台 extra 与解释器组合应遵循 README 表格。

## 依赖来源

- Thor：[PyTorch cu132](https://download.pytorch.org/whl/cu132)。
- Orin：[Jetson AI Lab jp6/cu126](https://pypi.jetson-ai-lab.io/jp6/cu126)。
- Mac 与公共依赖：PyPI。

两个专用索引均设置 `explicit = true`，通过 `tool.uv.sources` 绑定对应平台的
`torch`、`torchvision`，避免其他依赖意外从专用源安装。

Orin 的相同版本号可能存在其他平台构建，必须保留 Jetson 专用来源。
参考：[PyTorch 维护者对 Thor / Orin 二进制支持的说明](https://discuss.pytorch.org/t/does-torch-now-offically-supported-nvidia-jetson/224665)。

## 系统库与验证

Orin 的 torch 2.8.0 原生安装可能需要额外提供 cuSPARSELt 等共享库。
应检查目标机实际缺失的库后处理，不能把 Python 包安装成功视为 GPU 验证通过。
参考：[NVIDIA JetPack 6.2.1 安装说明](https://forums.developer.nvidia.com/t/torch-2-8-wheel-for-jetpack-6-2-1/347015)。

安装对应 extra 后，先运行以下最小验证，将 `thor` 替换为当前平台：

```bash
uv run --locked --extra thor python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
device = "mps" if torch.backends.mps.is_available() else "cuda"
if device == "cuda":
    assert torch.cuda.is_available(), "CUDA unavailable"
    print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(32, 32, device=device)
assert torch.isfinite((x @ x.T).cpu()).all()
print("GPU matrix operation passed:", device)
PY
```

真实 Frigate 流的单路、三路集成验收命令和实测边界见 [运行验收](validation.md)。

`model.precision` 默认 `fp32`，可以显式选择 `tf32` 或 `fp16`，后端约束见 README。
CUDA 使用 PyTorch 2.9+ 的 `fp32_precision` 接口；Orin 固定的 PyTorch 2.8 使用
`allow_tf32` 接口，同一次运行不混用新旧接口。Orin 仍待实机验证。

## Thor wheel 元数据修复

2026-09-08，本机 uv 0.12.10 对锁定的 `nvidia-cusparselt-cu13==0.8.1` 报告平台标签
不匹配：下载文件名为 `manylinux2014_aarch64.whl`，但包内 `WHEEL` 的标签为
`py3-none-manylinux2014_sbsa`。`uv sync --locked --extra thor --check` 因此返回非零，
普通同步会重复安装同一版本。这是官方 0.8.1 原包的问题。

现在 Thor extra 使用项目相对路径 `vendor/` 下的修正版 wheel。首次同步前运行
`/usr/bin/python3 scripts/repair_thor_wheel.py`，只需标准库和网络，无需先创建虚拟环境。
脚本从 NVIDIA 下载官方原包，强制校验 SHA256
`4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f`，
仅改变 `WHEEL` 的平台标签和重新生成的 `RECORD`；所有其他成员内容保持不变。
ZIP 使用无压缩存储，以便跨 zlib 版本重建相同制品，代价是本地 wheel 较大。
修正版不是 NVIDIA 官方发行物，版本仍为 0.8.1，来源和新哈希记录在 `uv.lock`。

生成的二进制不提交 Git。新克隆、离线部署和构建锁文件前须准备该文件；离线部署应
连同 `vendor/` 制品一起传输。Mac/Orin 不安装该包，但重新解析整个锁文件时也可能
需要本地制品。普通运行不需要反复执行准备脚本。不修改 uv 缓存或已安装的包元数据。
系统驱动、PyTorch cu132、CUDA 二进制和 vLLM 环境均不更换。

2026-09-08 修复验证：现有环境连续同步及 `--check --offline` 通过，新建独立虚拟
环境安装及检查通过；TF32、FP16 从默认 Hugging Face 缓存加载真实 CLIP，输出有限且
归一化的 512 维特征。全套测试 123 通过、1 跳过（无 MPS 硬件）；修复脚本另覆盖
原包哈希拒绝、成员内容保留、RECORD 校验和及重复生成不改变文件时间戳。
这次未重跑 Frigate 容量实验。修正版 wheel SHA256：
`e10d2e9eb418c133691964732c38b336baa48fb2bc3c7b9695fd0cc2fb5dd54e`。
