"""生成报告用图：分布、训练曲线、混淆矩阵、消融对比。

用法：
    py scripts/make_figures.py                 # 读 artifacts/ 与 configs/default.yaml
    py scripts/make_figures.py --tag main --data data/processed/mimic_ed.csv

产物：artifacts/figures/fig1~fig5.png（200 dpi，可直接插进报告）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from er_triage.config import load_config, resolve_path  # noqa: E402
from er_triage.schema import ESI_LABELS, N_CLASSES  # noqa: E402

# 中文字体：Windows 自带，按优先级挑第一个可用的
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

COLOR_MAIN = "#2f6f9f"
COLOR_WARN = "#c0504d"
COLOR_MUTED = "#8c8c8c"


def fig_label_distribution(df: pd.DataFrame, target_col: str, out: Path) -> Path:
    """图 1：ESI 类别分布，说明类别不平衡。"""
    counts = df[target_col].value_counts().reindex(range(1, N_CLASSES + 1), fill_value=0)
    ratios = counts / max(counts.sum(), 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar([f"ESI {i}" for i in counts.index], counts.values,
                  color=[COLOR_WARN, "#e08b4a", COLOR_MAIN, "#6a9fd0", COLOR_MUTED])
    for bar, ratio in zip(bars, ratios.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(bar.get_height())}\n{ratio:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("样本数")
    ax.set_title(f"ESI 类别分布（n={len(df)}）", fontsize=12)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_training_curve(metrics_path: Path, out: Path) -> Path | None:
    """图 2：训练损失与验证集 QWK 随 epoch 变化。"""
    if not metrics_path.exists():
        print(f"跳过训练曲线：缺少 {metrics_path.name}")
        return None
    history = json.loads(metrics_path.read_text(encoding="utf-8")).get("history", [])
    if not history:
        return None
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    qwk = [h["qwk"] for h in history]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    ax1.plot(epochs, losses, "o-", color=COLOR_WARN, label="训练损失")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("训练损失", color=COLOR_WARN)
    ax1.tick_params(axis="y", labelcolor=COLOR_WARN)

    ax2 = ax1.twinx()
    ax2.plot(epochs, qwk, "s-", color=COLOR_MAIN, label="验证集 QWK")
    ax2.set_ylabel("验证集 QWK", color=COLOR_MAIN)
    ax2.tick_params(axis="y", labelcolor=COLOR_MAIN)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
    ax1.set_title("训练过程：损失下降与 QWK 收敛", fontsize=12)
    ax1.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_confusion(confusion_path: Path, out: Path, tag: str) -> Path | None:
    """图 3：混淆矩阵热力图，行=真实，列=预测；右上方为分诊不足（危险方向）。"""
    if not confusion_path.exists():
        print(f"跳过混淆矩阵：缺少 {confusion_path.name}")
        return None
    cm = np.loadtxt(confusion_path, delimiter=",", dtype=int)
    row_sum = cm.sum(axis=1, keepdims=True)
    ratio = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(ratio, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES), [f"预测 {i}" for i in range(1, N_CLASSES + 1)])
    ax.set_yticks(range(N_CLASSES), [f"真实 {i}" for i in range(1, N_CLASSES + 1)])
    ax.set_xlabel("模型预测")
    ax.set_ylabel("临床真实分级")
    ax.grid(False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{cm[i, j]}\n{ratio[i, j]:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if ratio[i, j] > 0.55 else "#222222")

    # 分诊不足区域：预测等级数值更大（更轻），左上三角之外
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=COLOR_WARN, lw=1.4, ls="--"))
    ax.set_title(f"混淆矩阵（{tag}）虚线框=分诊不足（漏诊方向）", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="行归一化比例")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_ablation(summary_csv: Path, out: Path) -> Path | None:
    """图 4：消融对比，双面板（QWK 越高越好 / 危重降级率越低越好）。"""
    if not summary_csv.exists():
        print(f"跳过消融图：缺少 {summary_csv.name}")
        return None
    df = pd.read_csv(summary_csv)
    if df.empty:
        return None
    # 按 QWK 排序，报告里读起来更直观；main 用高亮色区分
    df = df.sort_values("qwk", ascending=False).reset_index(drop=True)

    labels = df["experiment"].tolist()
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    panels = [
        ("qwk", "QWK（越高越好）", False),
        ("critical_under_triage", "危重降级率（越低越好）", True),
    ]
    for ax, (col, title, as_pct) in zip(axes, panels):
        values = df[col].to_numpy(dtype=float)
        base = COLOR_WARN if as_pct else COLOR_MAIN
        colors = ["#7b1f1c" if name == "深度-main" else base for name in labels]
        ax.barh(y, values, color=colors,
                edgecolor=["black" if name == "深度-main" else "none" for name in labels],
                linewidth=1.2)
        for yi, value in zip(y, values):
            text = f"{value:.2%}" if as_pct else f"{value:.4f}"
            ax.text(value, yi, f" {text}", va="center", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, values.max() * 1.22 if values.max() > 0 else 1)
    # sharey 的轴只翻一次，翻两次会互相抵消
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    axes[0].set_yticks(y, labels)
    fig.suptitle("模型与消融对比（黑框=本方法）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_text_backend(artifacts: Path, out: Path) -> Path | None:
    """图 5：同子集下 hash 与 Bio_ClinicalBERT 两种文本后端对照。"""
    paths = {"hash": artifacts / "subset_hash_metrics.json",
             "BERT": artifacts / "subset_hf_metrics.json"}
    tests = {}
    for name, path in paths.items():
        if path.exists():
            tests[name] = json.loads(path.read_text(encoding="utf-8")).get("test", {})
    if len(tests) < 2:
        print("跳过文本后端对照：缺少 subset_hash / subset_hf 指标")
        return None

    names = list(tests)
    colors = [COLOR_MUTED, COLOR_MAIN]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    ax = axes[0]
    x = np.arange(2)
    width = 0.34
    for i, (name, color) in enumerate(zip(names, colors)):
        values = [tests[name]["qwk"], tests[name]["macro_f1"]]
        bars = ax.bar(x + (i - 0.5) * width, values, width * 0.92, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ["QWK", "Macro-F1"])
    # 从 0 起画：截断纵轴会把 0.01 的差异放大成视觉上的两倍
    ax.set_ylim(0, 1.0)
    ax.set_title("分得准：有序一致性更高不代表更安全", fontsize=11)

    ax = axes[1]
    values = [tests[name]["critical_under_triage"] for name in names]
    bars = ax.bar(names, values, 0.5, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:.2%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("危重降级率")
    ax.set_ylim(0, max(values) * 1.35)
    ax.set_title("分得安全：BERT 后端危重降级率更低", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("文本后端对照（12k 子集，同结构同超参）", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成报告用图")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out-dir", default="artifacts/figures")
    parser.add_argument("--tag", default="main", help="混淆矩阵与训练曲线对应的实验标签")
    parser.add_argument("--data", default=None, help="用于类别分布的数据文件，默认取配置里的 train_file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts = resolve_path(args.artifacts)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_path(args.data) if args.data else resolve_path(cfg["data"]["train_file"])
    made: list[Path] = []

    if data_path.exists():
        df = pd.read_csv(data_path)
        made.append(fig_label_distribution(df, cfg["data"]["target_col"],
                                           out_dir / "fig1_label_distribution.png"))
    else:
        print(f"跳过类别分布：缺少数据 {data_path}")

    curve = fig_training_curve(artifacts / f"{args.tag}_metrics.json", out_dir / "fig2_training_curve.png")
    conf = fig_confusion(artifacts / f"{args.tag}_confusion.csv", out_dir / "fig3_confusion_matrix.png", args.tag)
    abl = fig_ablation(artifacts / "summary.csv", out_dir / "fig4_ablation.png")
    backend = fig_text_backend(artifacts, out_dir / "fig5_text_backend.png")
    made += [p for p in (curve, conf, abl, backend) if p is not None]

    print("已生成：")
    for path in made:
        print(f"  {path}")
    print(f"图例说明：{ESI_LABELS[1]} ... {ESI_LABELS[5]}")


if __name__ == "__main__":
    main()
