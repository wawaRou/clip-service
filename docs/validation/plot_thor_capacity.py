"""Render the recorded Thor experiment; no inference or network requests are run.

uv run --locked --extra thor python docs/validation/plot_thor_capacity.py

Matplotlib is a development dependency. Use the matching platform extra.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent
COLORS = {"tf32": "#167D8D", "fp16": "#D96B50"}
TEXT = "#243039"
MUTED = "#5C6872"


def configure_fonts():
    regular = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    bold = regular.with_name("NotoSansCJK-Bold.ttc")
    if regular.exists():
        font_manager.fontManager.addfont(regular)
        if bold.exists():
            font_manager.fontManager.addfont(bold)
        family = font_manager.FontProperties(fname=regular).get_name()
    else:
        family = "sans-serif"
    plt.rcParams.update(
        {
            "font.family": [family, "DejaVu Sans"],
            "font.size": 11,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#C7CFD4",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def canvas(title, subtitle, notes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8))
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.27, top=0.735, wspace=0.27)
    fig.text(0.075, 0.93, title, fontsize=21, weight="bold")
    fig.text(0.075, 0.875, subtitle, fontsize=11, color=MUTED)
    fig.legend(
        handles=[Patch(color=color, label=name.upper()) for name, color in COLORS.items()],
        loc="upper left",
        bbox_to_anchor=(0.07, 0.835),
        ncol=2,
        frameon=False,
    )
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6EBEE", linewidth=0.8)
        ax.tick_params(axis="both", length=0, pad=8)
    for index, note in enumerate(notes):
        fig.text(0.075, 0.15 - index * 0.043, note, fontsize=10, color=MUTED)
    return fig, axes


def grouped_bars(ax, counts, values, digits=1):
    x = np.arange(len(counts))
    for index, (precision, color) in enumerate(COLORS.items()):
        bars = ax.bar(x + (index - 0.5) * 0.34, values[precision], width=0.3, color=color)
        ax.bar_label(bars, fmt=f"%.{digits}f", padding=5, fontsize=10, color=TEXT)
    ax.set_xticks(x, [f"{count} 路" for count in counts])
    ax.set_xlabel("同时激活并持续推理的摄像头")


def model_chart(base):
    fig, axes = canvas(
        "01  模型吞吐：GPU 前向与完整编码",
        "Thor · CLIP ViT-B/16 · batch=1 · vLLM 驻留但未生成",
        [
            "每项预热 80 次，测 3 轮 × 3 秒；柱高为 FPS 中位数，误差线为三轮最小至最大值。",
            "完整编码包含 BGR→RGB、图像预处理、GPU 前向、CPU 特征输出与归一化；"
            "两项均不含视频解码。",
            "GPU 前向吞吐不能直接除以 25 作为摄像头容量。数据：2026-09-08 本机实测。",
        ],
    )
    for ax, key, title in zip(
        axes, ["gpu_forward", "encode_bgr"], ["仅 GPU 图像前向", "BGR 帧 → 归一化特征"]
    ):
        for index, (precision, color) in enumerate(COLORS.items()):
            result = base["precisions"][precision][key]
            value = result["median_fps"]
            rounds = [r["fps"] for r in result["rounds"]]
            ax.bar(index, value, color=color, width=0.52)
            ax.errorbar(
                index,
                value,
                yerr=[[value - min(rounds)], [max(rounds) - value]],
                fmt="none",
                ecolor=TEXT,
                capsize=5,
            )
            ax.text(index, value + 0.07 * value, f"{value:.1f}", ha="center", fontsize=15)
        ax.set_title(title, loc="left", fontsize=14, pad=15)
        ax.set_xticks([0, 1], ["TF32", "FP16"])
        ax.set_ylabel("吞吐量 (FPS)")
        ax.set_ylim(0, 450 if key == "gpu_forward" else 150)
        ax.set_xlim(-0.7, 1.7)
    return fig


def throughput_chart(base, active):
    fig, axes = canvas(
        "02  摄像头数量增加后，每路还能跑多少帧？",
        "720p 真实 RTSP（约 25 FPS）· 推理上限 30 FPS · 6 帧队列 · 解码与 JPEG 缓存开启",
        [
            "点位为各路平均推理 FPS；每档约 25 秒，摄像头均独立订阅同一条真实视频流。",
            "右图：VLM 单并发、连续五图请求、返回 JSON；混跑 3 路时两种精度均低于每路 25 FPS。",
            "FP16 单路混跑捕获约 864 ms 慢推理、丢 17 帧，因此该点为 24.32 FPS，未剔除。",
        ],
    )
    for ax, resident in zip(axes, [True, False]):
        counts = [2, 3, 4, 6] if resident else [1, 2, 3]
        series = {}
        for precision in COLORS:
            phases = (
                base["precisions"][precision]["live"]
                if resident
                else [active[(precision, n)] for n in counts]
            )
            series[precision] = [
                sum(c["inference_fps"] for c in p["cameras"].values()) / p["logical_readers"]
                for p in phases
            ]
        for precision, color in COLORS.items():
            ys = series[precision]
            other = series["fp16" if precision == "tf32" else "tf32"]
            ax.plot(counts, ys, "o-", color=color, linewidth=2.3, markersize=7)
            for x, y, other_y in zip(counts, ys, other):
                ax.annotate(
                    f"{y:.2f}",
                    (x, y),
                    xytext=(0, 13 if y >= other_y else -21),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=10,
                )
        ax.axhline(25, color="#677580", linestyle="--", linewidth=1.1)
        ax.set_title(
            "vLLM 仅驻留" if resident else "vLLM 持续处理五图 JSON 请求",
            loc="left",
            fontsize=14,
            pad=15,
        )
        ax.set_xticks(counts)
        ax.set_xlim(min(counts) - 0.35, max(counts) + 0.35)
        ax.set_ylim(0, 30)
        ax.set_ylabel("每路平均推理 FPS")
        ax.set_xlabel("同时激活的摄像头路数")
        ax.text(
            0.98,
            0.05,
            "虚线：25 FPS 目标",
            transform=ax.transAxes,
            ha="right",
            fontsize=10,
            color=MUTED,
        )
    return fig


def impact_chart(mixed, active):
    alone = next(p for p in mixed["phases"] if not p["clip_active"])
    reference = alone["vlm_summary"]["request_seconds"]["p50"]
    fig, axes = canvas(
        "03  混跑代价：VLM 响应与 CLIP 丢帧",
        "五张实时图片 / 请求 · 单并发持续发送 · 关闭思考 · JSON 输出上限 256 tokens",
        [
            "左图为 VLM 完整响应耗时中位数，每档 6–7 个完成请求；"
            "输出长度随画面变化，非固定 token 对照。",
            "右图为约 25 秒测量窗口内所有摄像头合计丢弃的队列帧数，不是网络丢包数。",
            "首次 VLM 请求约 67.97 秒，已作为冷启动单独记录，不计入稳态图表。",
        ],
    )
    counts = [1, 2, 3]
    times = {
        p: [active[(p, n)]["vlm_summary"]["request_seconds"]["p50"] for n in counts] for p in COLORS
    }
    drops = {
        p: [sum(c["dropped_frames"] for c in active[(p, n)]["cameras"].values()) for n in counts]
        for p in COLORS
    }
    grouped_bars(axes[0], counts, times, digits=2)
    axes[0].axhline(reference, color="#677580", linestyle="--", linewidth=1.2)
    axes[0].text(
        0.02,
        0.93,
        f"虚线：仅 VLM，{reference:.2f} 秒",
        transform=axes[0].transAxes,
        color=MUTED,
        fontsize=10,
    )
    axes[0].set_title("VLM 响应时间", loc="left", fontsize=14, pad=15)
    axes[0].set_ylabel("响应耗时中位数 (秒)")
    axes[0].set_ylim(0, 5.5)
    grouped_bars(axes[1], counts, drops, digits=0)
    axes[1].set_title("CLIP 队列丢帧", loc="left", fontsize=14, pad=15)
    axes[1].set_ylabel("所有路合计丢帧数 (帧)")
    axes[1].set_ylim(0, 500)
    return fig


def resources_chart(active):
    fig, axes = canvas(
        "04  混跑资源：GPU 忙碌程度与可用内存",
        "GPU 为全局采样利用率，无法按进程拆分；内存为系统 MemAvailable，不是独立显存池",
        [
            "nvidia-smi 进程口径：vLLM 约 82.58 GiB；CLIP 约 0.94 GiB（TF32）/ 0.63 GiB（FP16）。",
            "混跑 GPU 采样峰值为 98%；系统可用内存最低约 17.76 GiB，未出现内存不足。",
            "阶段顺序执行、时钟未锁定，缓存未重置；内存曲线不能用于孤立比较两种精度的内存需求。",
        ],
    )
    counts = [1, 2, 3]
    utilization = {}
    available = {}
    for precision in COLORS:
        samples = [active[(precision, n)]["machine_samples"] for n in counts]
        utilization[precision] = [
            np.mean([s["gpu"]["utilization_percent"] for s in group]) for group in samples
        ]
        available[precision] = [
            min(s["available_memory_mib"] for s in group) / 1024 for group in samples
        ]
    grouped_bars(axes[0], counts, utilization)
    axes[0].set_title("GPU 利用率采样均值", loc="left", fontsize=14, pad=15)
    axes[0].set_ylabel("GPU 利用率 (%)")
    axes[0].set_ylim(0, 110)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    grouped_bars(axes[1], counts, available, digits=2)
    axes[1].set_title("系统可用内存最低值", loc="left", fontsize=14, pad=15)
    axes[1].set_ylabel("MemAvailable (GiB)")
    axes[1].set_ylim(0, 24)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "2026-09-08-thor-figures")
    args = parser.parse_args()
    configure_fonts()
    base = json.loads((ROOT / "2026-09-08-thor-precision-capacity.json").read_text())
    mixed = json.loads((ROOT / "2026-09-08-thor-clip-vlm-capacity.json").read_text())
    active = {
        (p["precision"], p["logical_readers"]): p for p in mixed["phases"] if p["clip_active"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        ("01-model-throughput", model_chart(base)),
        ("02-camera-throughput", throughput_chart(base, active)),
        ("03-vlm-impact", impact_chart(mixed, active)),
        ("04-resource-usage", resources_chart(active)),
    ]
    pdf_path = args.output_dir / "thor-clip-experiment.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Thor CLIP 与 VLM 实验图表"}) as pdf:
        for name, fig in figures:
            fig.savefig(args.output_dir / f"{name}.png", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Generated {len(figures)} PNG charts and {pdf_path}")


if __name__ == "__main__":
    main()
