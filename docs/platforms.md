# 原生平台环境

所有平台直接使用 uv 虚拟环境，不使用容器。uv 锁定 Python 包及其来源；JetPack、
驱动、CUDA 和系统共享库由机器环境提供，不属于 `uv.lock` 的管理范围。

## 版本基准

| 平台 | 系统 | Python | torch | torchvision | 状态 |
| --- | --- | --- | --- | --- | --- |
| Mac / MPS | Apple Silicon macOS | 3.12 | 2.13.0 / PyPI | 暂未声明 | torch 版本沿用旧项目；新环境推理待验证 |
| Thor / CUDA | L4T R39.2 / JetPack 7 系列 / CUDA 13.2 | 3.12 | 2.14.0+cu132 | 0.29.0+cu132 | 用户已实测通过此推理组合；项目集成待验证 |
| Orin 64GB / CUDA | L4T 36.4.4 / JetPack 6.2.1 / CUDA 12.6 | 3.10 | 2.8.0 | 0.23.0 | 用户指定暂定固定，上机验证待完成 |

Mac 尚未迁移旧项目的 transformers 等业务依赖；迁移时按实际使用添加并验证。
共享源码保持 Python 3.10 兼容。目前锁定目标限制为 macOS ARM64 / Python 3.12，
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

业务迁移后还需验证真实 CLIP 模型编码、图像预处理与视频读取。
