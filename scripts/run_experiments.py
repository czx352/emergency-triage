"""一键跑完「基线 + 主模型 + 消融」，并汇总成报告用的两张表。

用法：
    py scripts/run_experiments.py                 # 正式实验（读 configs/default.yaml）
    py scripts/run_experiments.py --quick         # 小样本 + 少轮次，先验证流程
    py scripts/run_experiments.py --resume        # 已有指标文件的实验自动跳过
    py scripts/run_experiments.py --config configs/mimic.yaml

产物：
    artifacts/summary.md   报告可直接粘贴的 Markdown 表
    artifacts/summary.csv  便于绘图或贴进 Excel
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from er_triage.config import resolve_path  # noqa: E402

# (实验名, 训练参数)
EXPERIMENTS: list[tuple[str, list[str]]] = [
    ("main", []),
    ("ablate_head_ce", ["--head", "ce", "--loss", "ce"]),
    ("ablate_concat", ["--fusion", "concat"]),
    ("ablate_cost", ["--head", "ce", "--loss", "cost_ce"]),
    ("ablate_no_text", ["--no-text"]),
]

BASELINE_NAMES = {
    "structured_hgb": "基线-结构化(HGB)",
    "text_tfidf_lr": "基线-文本(TFIDF+LR)",
    "fusion_baseline": "基线-简单拼接",
}

FIELDS = ["qwk", "macro_f1", "critical_under_triage", "cost_weighted_error"]


def run(cmd: list[str], title: str) -> None:
    print(f"\n>>> {title}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"实验失败: {title}")
    tail = [line for line in proc.stdout.splitlines() if line.strip()][-3:]
    print("\n".join(tail))


def load_rows(artifacts: Path) -> list[dict]:
    rows: list[dict] = []
    baseline_path = artifacts / "baseline_metrics.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        for key, label in BASELINE_NAMES.items():
            if key in baseline:
                rows.append({"experiment": label, **{f: baseline[key][f] for f in FIELDS}})
    for tag, label in [(t, f"深度-{t}") for t, _ in EXPERIMENTS]:
        path = artifacts / f"{tag}_metrics.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"experiment": label, **{f: data["test"][f] for f in FIELDS}})
    return rows


def write_summary(artifacts: Path, rows: list[dict]) -> None:
    header = ["实验", "QWK", "Macro-F1", "危重降级率", "代价加权误差"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['qwk']:.4f} | {r['macro_f1']:.4f} "
            f"| {r['critical_under_triage']:.2%} | {r['cost_weighted_error']:.3f} |"
        )
    (artifacts / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (artifacts / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["experiment", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="跑完基线、主模型与消融并汇总")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true", help="小样本 + 6 轮，验证流程用")
    parser.add_argument("--resume", action="store_true", help="已有指标文件的实验跳过")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    limit = args.limit if args.limit is not None else (5000 if args.quick else None)
    epochs = args.epochs if args.epochs is not None else (6 if args.quick else None)
    common = [] if not limit else ["--limit", str(limit)]
    # 小样本时默认 lr 太小，几十步内学不动会退化成单类预测，这里显式放大学习率
    if args.lr:
        lr_args = ["--lr", str(args.lr)]
    elif args.quick:
        lr_args = ["--lr", "0.001"]
    else:
        lr_args = []

    artifacts = resolve_path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    baseline_path = artifacts / "baseline_metrics.json"
    if not args.skip_baseline and not (args.resume and baseline_path.exists()):
        run([sys.executable, "scripts/train_baseline.py", "--config", args.config, *common], "基线实验")

    for tag, extra in EXPERIMENTS:
        metrics_path = artifacts / f"{tag}_metrics.json"
        if args.resume and metrics_path.exists():
            print(f"\n>>> 跳过（已存在）: {tag}")
            continue
        cmd = [sys.executable, "-m", "er_triage.train", "--config", args.config,
               "--tag", tag, *extra, *common, *lr_args]
        if epochs:
            cmd += ["--epochs", str(epochs)]
        run(cmd, f"{tag}")

    rows = load_rows(artifacts)
    write_summary(artifacts, rows)
    print(f"\n汇总已写入 {artifacts / 'summary.md'} 和 summary.csv")


if __name__ == "__main__":
    main()
