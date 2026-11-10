"""训练与评估入口。

快速开始：
    py -m er_triage.train --config configs/default.yaml --epochs 3

消融实验（每次改一个开关，产出独立指标文件）：
    py -m er_triage.train --tag no_text --fusion concat ...
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .data.dataset import TriageDataset
from .data.features import FeatureSpec, load_frame, split_by_patient, targets_to_index
from .metrics import confusion, evaluate
from .models.fusion import build_model, probs_from_logits
from .models.losses import build_loss
from .models.text import TextProcessor
from .schema import N_CLASSES, TEXT_COL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loader(x_num, x_text, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TriageDataset(x_num, x_text, y)
    # drop_last 避免 BatchNorm 遇到 batch=1
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle and len(ds) > batch_size)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, head: str):
    model.eval()
    ys, preds, probs = [], [], []
    for batch in loader:
        x_num, x_text, y = (t.to(device) for t in batch)
        logits = model(x_num, x_text)
        p = probs_from_logits(logits, head)
        ys.append(y.cpu().numpy())
        preds.append(p.argmax(dim=1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def run_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for x_num, x_text, y in loader:
        x_num, x_text, y = x_num.to(device), x_text.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_num, x_text), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += float(loss.item()) * len(y)
        count += len(y)
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练急诊分诊模型")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 条样本，便于冒烟")
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-backend", default=None, choices=["hash", "hf"])
    parser.add_argument("--fusion", default=None, choices=["gated", "concat"])
    parser.add_argument("--head", default=None, choices=["coral", "ce"])
    parser.add_argument("--loss", default=None, choices=["coral", "ce", "cost_ce"])
    parser.add_argument("--tag", default="base", help="实验标签，用于产物文件命名")
    parser.add_argument("--no-text", action="store_true", help="消融：主诉置空，只保留结构化特征")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.device:
        cfg["train"]["device"] = args.device
    if args.text_backend:
        cfg["model"]["text_backend"] = args.text_backend
    if args.fusion:
        cfg["model"]["fusion"] = args.fusion
    if args.head:
        cfg["model"]["head"] = args.head
    if args.loss:
        cfg["train"]["loss"] = args.loss

    head, loss_name = cfg["model"]["head"], cfg["train"]["loss"]
    if head == "coral" and loss_name != "coral":
        raise SystemExit("head=coral 时必须使用 loss=coral；如需代价敏感请设 head=ce loss=cost_ce")

    set_seed(int(cfg["seed"]))
    device = pick_device(cfg["train"]["device"])

    df = load_frame(resolve_path(cfg["data"]["train_file"]))
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    train_df, val_df, test_df = split_by_patient(
        df, cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], int(cfg["seed"]),
        cfg["data"]["patient_col"],
    )
    print(f"样本数 train={len(train_df)} val={len(val_df)} test={len(test_df)} | device={device}")

    spec = FeatureSpec().fit(train_df, cfg["data"]["numeric_cols"])
    processor = TextProcessor(
        backend=cfg["model"]["text_backend"],
        max_len=int(cfg["model"]["text_max_len"]),
        vocab_size=int(cfg["model"]["text_vocab_size"]),
        model_name=cfg["model"]["hf_model_name"],
    )

    def arrays(frame):
        x_num = spec.transform(frame)
        texts = [""] * len(frame) if args.no_text else frame[TEXT_COL].tolist()
        x_text = processor.encode(texts)
        y = targets_to_index(frame[cfg["data"]["target_col"]])
        return x_num, x_text, y

    x_tr, t_tr, y_tr = arrays(train_df)
    x_va, t_va, y_va = arrays(val_df)
    x_te, t_te, y_te = arrays(test_df)

    bs = int(cfg["train"]["batch_size"])
    train_loader = make_loader(x_tr, t_tr, y_tr, bs, shuffle=True)
    val_loader = make_loader(x_va, t_va, y_va, bs, shuffle=False)
    test_loader = make_loader(x_te, t_te, y_te, bs, shuffle=False)

    text_encoder = processor.build_encoder(int(cfg["model"]["text_dim"]), float(cfg["model"]["dropout"]))
    model = build_model(cfg, n_numeric=spec.n_features, text_encoder=text_encoder).to(device)
    criterion = build_loss(loss_name, cfg["train"]["cost_matrix"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_qwk, best_state, history, wait = -2.0, None, [], 0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, _ = predict(model, val_loader, device, head)
        val = evaluate(y_true, y_pred, cfg["train"]["cost_matrix"])
        scheduler.step(val["qwk"])
        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        print(f"epoch {epoch:02d} loss={train_loss:.4f} val_qwk={val['qwk']:.4f} "
              f"val_f1={val['macro_f1']:.4f} val_critical_under={val['critical_under_triage']:.3f}")
        if val["qwk"] > best_qwk:
            best_qwk, best_state, wait = val["qwk"], {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= int(cfg["train"]["patience"]):
                print("验证集 QWK 未提升，提前停止")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, probs = predict(model, test_loader, device, head)
    test_metrics = evaluate(y_true, y_pred, cfg["train"]["cost_matrix"])
    cm = confusion(y_true, y_pred)

    model_version = f"{cfg['serve']['model_version']}-{args.tag}"
    artifact = {
        "state_dict": model.state_dict(),
        "config": cfg,
        "feature_spec": spec.to_dict(),
        "text_state": processor.state(),
        "n_numeric": spec.n_features,
        "head": head,
        "fusion": cfg["model"]["fusion"],
        "model_version": model_version,
    }
    ckpt_path = out_dir / f"{args.tag}.pt"
    torch.save(artifact, ckpt_path)

    metrics_path = out_dir / f"{args.tag}_metrics.json"
    metrics_path.write_text(json.dumps({
        "tag": args.tag,
        "model_version": model_version,
        "used_numeric_cols": spec.used_cols,
        "test": test_metrics,
        "history": history,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    cm_path = out_dir / f"{args.tag}_confusion.csv"
    np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

    print("\n=== 测试集 ===")
    for key in ["macro_f1", "qwk", "accuracy", "critical_under_triage", "under_triage",
                "over_triage", "cost_weighted_error"]:
        print(f"{key:>24}: {test_metrics[key]:.4f}")
    print(f"混淆矩阵（行=真实 1-5，列=预测 1-5）：\n{cm}")
    print(f"模型已保存: {ckpt_path}\n指标已保存: {metrics_path}")
    _ = N_CLASSES


if __name__ == "__main__":
    main()
