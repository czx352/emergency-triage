"""合成急诊分诊数据。

用途：在 MIMIC-IV-ED 审批未下来前，先把「数据 -> 训练 -> 评估 -> 服务」全链路跑通。
数据按 ESI 分布与生理参数区间构造，并注入标签噪声与缺失值，使指标不会虚高。

注意：合成数据只能验证管线，结论不可外推到临床，报告必须写明。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..config import resolve_path
from ..schema import CLIP_RANGES

# 各 ESI 级的生理参数 (均值, 标准差)，依据公开文献的典型分布设定
LEVEL_VITALS: dict[int, dict[str, tuple[float, float]]] = {
    1: {"hr": (128, 20), "sbp": (82, 16), "dbp": (50, 12), "rr": (30, 7),
        "temp_c": (36.6, 1.1), "spo2": (86, 7), "pain": (7.5, 3.0), "gcs": (10, 3.5), "age": (64, 19)},
    2: {"hr": (112, 17), "sbp": (102, 18), "dbp": (63, 13), "rr": (24, 6),
        "temp_c": (37.2, 0.9), "spo2": (92, 4), "pain": (6.8, 2.4), "gcs": (14, 1.6), "age": (61, 18)},
    3: {"hr": (94, 15), "sbp": (123, 18), "dbp": (76, 12), "rr": (19, 4),
        "temp_c": (37.6, 0.8), "spo2": (96, 2.5), "pain": (5.4, 2.3), "gcs": (15, 0.4), "age": (49, 18)},
    4: {"hr": (84, 13), "sbp": (131, 16), "dbp": (80, 11), "rr": (17, 3),
        "temp_c": (37.1, 0.7), "spo2": (98, 1.6), "pain": (3.6, 2.0), "gcs": (15, 0.1), "age": (38, 16)},
    5: {"hr": (76, 11), "sbp": (126, 15), "dbp": (78, 10), "rr": (16, 3),
        "temp_c": (36.8, 0.6), "spo2": (98, 1.2), "pain": (2.1, 1.7), "gcs": (15, 0.05), "age": (31, 13)},
}

# 主诉模板：按分级给出典型表述
COMPLAINTS: dict[int, list[str]] = {
    1: ["cardiac arrest", "unresponsive, no pulse", "severe respiratory distress",
        "massive trauma after fall", "active seizure, not breathing", "gunshot wound to chest"],
    2: ["chest pain radiating to arm", "severe shortness of breath", "altered mental status",
        "syncope episode", "severe abdominal pain with vomiting", "suspected stroke, facial droop"],
    3: ["abdominal pain for 2 days", "fever and chills", "vomiting and diarrhea",
        "dizziness on standing", "flank pain, burning urination", "headache with photophobia"],
    4: ["sore throat", "ankle sprain after sports", "rash on forearm", "ear pain",
        "lower back pain after lifting", "laceration on hand, bleeding stopped"],
    5: ["medication refill request", "mild cold symptoms", "prescription renewal",
        "itchy eyes, seasonal allergy", "minor bruise, no pain", "asking for sick note"],
}

# 修饰语，增加文本多样性，避免模型只靠关键词记忆
MODIFIERS = ["for 2 days", "worse at night", "after eating", "since this morning",
             "no relief with acetaminophen", "recurrent episode", "gradual onset"]

ESI_PROBS = [0.02, 0.09, 0.39, 0.35, 0.15]  # 近似真实急诊金字塔分布


def _sample_level(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.choice(np.arange(1, 6), size=size, p=ESI_PROBS)


def _vitals_for_level(rng: np.random.Generator, levels: np.ndarray) -> pd.DataFrame:
    cols = ["hr", "sbp", "dbp", "rr", "temp_c", "spo2", "pain", "gcs", "age"]
    out = {c: np.zeros(len(levels)) for c in cols}
    for lvl in np.unique(levels):
        mask = levels == lvl
        for c in cols:
            mean, std = LEVEL_VITALS[int(lvl)][c]
            out[c][mask] = rng.normal(mean, std, size=mask.sum())
    df = pd.DataFrame(out)
    for c, (lo, hi) in CLIP_RANGES.items():
        if c in df:
            df[c] = df[c].clip(lo, hi)
    df["age"] = df["age"].round(0)
    df["pain"] = df["pain"].round(0)
    df["gcs"] = df["gcs"].round(0)
    df["spo2"] = df["spo2"].round(0)
    return df


def _complaints(rng: np.random.Generator, levels: np.ndarray, noise: float) -> list[str]:
    """按分级抽主诉，并以 noise 概率串到相邻分级，模拟真实标注不一致。"""
    texts = []
    for lvl in levels:
        draw = int(lvl)
        if rng.random() < noise:
            draw = int(np.clip(lvl + rng.choice([-1, 1]), 1, 5))
        text = rng.choice(COMPLAINTS[draw])
        if rng.random() < 0.45:
            text = f"{text}, {rng.choice(MODIFIERS)}"
        texts.append(str(text))
    return texts


def generate(n: int = 40000, seed: int = 42, label_noise: float = 0.12,
             missing_rate: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    levels = _sample_level(rng, n)
    df = _vitals_for_level(rng, levels)
    df.insert(0, "esi", levels)
    df.insert(1, "chief_complaint", _complaints(rng, levels, label_noise))
    df.insert(2, "sex", rng.choice(["M", "F"], size=n, p=[0.49, 0.51]))
    df.insert(3, "arrival_mode", rng.choice(
        ["ambulance", "walk_in", "wheelchair", "police"], size=n, p=[0.28, 0.55, 0.12, 0.05]))

    # 同一患者可有多次就诊，用于验证“按患者划分”防泄漏
    n_patients = max(2, n // 2)
    visit_owner = rng.integers(0, n_patients, size=n)
    df.insert(0, "subject_id", [f"P{i:06d}" for i in visit_owner])
    df.insert(1, "stay_id", [f"S{i:07d}" for i in range(n)])

    # 随机缺失，模拟真实采集不全
    for col in ["hr", "sbp", "dbp", "rr", "temp_c", "spo2", "pain", "gcs"]:
        mask = rng.random(n) < missing_rate
        df.loc[mask, col] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="生成合成急诊分诊数据")
    parser.add_argument("--n", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-noise", type=float, default=0.12)
    parser.add_argument("--out", default="data/processed/synthetic.csv")
    args = parser.parse_args()

    df = generate(args.n, args.seed, args.label_noise)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"已写入 {out_path}  形状={df.shape}")
    print("ESI 分布：")
    print(df["esi"].value_counts(normalize=True).sort_index().round(3).to_string())


if __name__ == "__main__":
    main()
