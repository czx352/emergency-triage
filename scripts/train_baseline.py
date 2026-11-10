"""基线实验：不依赖深度模型，用于回答「深度双塔到底带来了多少增益」。

1) 结构化基线：生命体征 + 人口学 -> HistGradientBoosting（sklearn 自带，无需额外依赖）
2) 文本基线：主诉 TF-IDF -> 逻辑回归
3) 简单拼接：把 TF-IDF 的类别概率与结构化特征拼起来再训一次

用法：
    py scripts/train_baseline.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from er_triage.config import load_config, resolve_path  # noqa: E402
from er_triage.data.features import FeatureSpec, load_frame, split_by_patient, targets_to_index  # noqa: E402
from er_triage.metrics import confusion, evaluate  # noqa: E402
from er_triage.schema import TEXT_COL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="结构化/文本基线")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_frame(resolve_path(cfg["data"]["train_file"]))
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    train_df, val_df, test_df = split_by_patient(
        df, cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], int(cfg["seed"]),
        cfg["data"]["patient_col"],
    )

    spec = FeatureSpec().fit(train_df, cfg["data"]["numeric_cols"])
    x_tr, x_va, x_te = (spec.transform(f) for f in (train_df, val_df, test_df))
    y_tr, y_va, y_te = (targets_to_index(f[cfg["data"]["target_col"]]) for f in (train_df, val_df, test_df))

    results = {}
    cost = cfg["train"]["cost_matrix"]

    # 1) 结构化基线
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=int(cfg["seed"]))
    gb.fit(np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]))
    pred = gb.predict(x_te)
    results["structured_hgb"] = evaluate(y_te, pred, cost)
    print("结构化基线:", {k: round(v, 4) for k, v in results["structured_hgb"].items()
                        if isinstance(v, float)})

    # 2) 文本基线（TF-IDF + LR）
    text_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
        LogisticRegression(max_iter=2000, C=2.0),
    )
    text_model.fit(train_df[TEXT_COL].astype(str), y_tr)
    pred_text = text_model.predict(test_df[TEXT_COL].astype(str))
    results["text_tfidf_lr"] = evaluate(y_te, pred_text, cost)
    print("文本基线:", {k: round(v, 4) for k, v in results["text_tfidf_lr"].items()
                      if isinstance(v, float)})

    # 3) 融合基线：文本概率 + 结构化特征
    prob_tr = text_model.predict_proba(train_df[TEXT_COL].astype(str))
    prob_te = text_model.predict_proba(test_df[TEXT_COL].astype(str))
    gb_fusion = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=int(cfg["seed"]))
    gb_fusion.fit(np.hstack([x_tr, prob_tr]), y_tr)
    pred_fusion = gb_fusion.predict(np.hstack([x_te, prob_te]))
    results["fusion_baseline"] = evaluate(y_te, pred_fusion, cost)
    print("融合基线:", {k: round(v, 4) for k, v in results["fusion_baseline"].items()
                      if isinstance(v, float)})

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savetxt(out_dir / "baseline_fusion_confusion.csv", confusion(y_te, pred_fusion), fmt="%d", delimiter=",")
    print(f"已保存 -> {out_dir / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
