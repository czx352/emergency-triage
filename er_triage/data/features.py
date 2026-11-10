"""特征工程：清洗、缺失值处理、标准化、按患者划分。

标准化统计量只在训练集上拟合，避免验证/测试集泄漏。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema import CLIP_RANGES, NUMERIC_COLS, PATIENT_COL, TARGET_COL, CLASS_INDEX_OFFSET


@dataclass
class FeatureSpec:
    """数值特征的拟合结果，随模型一起保存，保证线上与离线特征一致。"""
    used_cols: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    missing_mean: float = 0.0
    missing_std: float = 1.0

    @property
    def n_features(self) -> int:
        return len(self.used_cols) + 1  # +1 为缺失数量特征

    def fit(self, df: pd.DataFrame, numeric_cols: list[str] | None = None) -> "FeatureSpec":
        cols = numeric_cols or NUMERIC_COLS
        self.used_cols = [
            c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        raw = self._to_numeric(df)
        self.medians = {c: float(raw[c].median()) for c in self.used_cols}
        filled = raw.fillna(self.medians)
        self.means = {c: float(filled[c].mean()) for c in self.used_cols}
        self.stds = {c: float(filled[c].std(ddof=0)) or 1.0 for c in self.used_cols}
        miss = raw.isna().sum(axis=1).astype(float)
        self.missing_mean = float(miss.mean())
        self.missing_std = float(miss.std(ddof=0)) or 1.0
        return self

    def _to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.used_cols:
            values = pd.to_numeric(df[c], errors="coerce")
            lo, hi = CLIP_RANGES.get(c, (float("-inf"), float("inf")))
            out[c] = values.clip(lo, hi)
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._to_numeric(df)
        miss = raw.isna().sum(axis=1).astype(float)
        filled = raw.fillna(self.medians)
        blocks = [
            ((filled[c] - self.means[c]) / self.stds[c]).to_numpy(dtype=np.float32)
            for c in self.used_cols
        ]
        blocks.append(((miss - self.missing_mean) / self.missing_std).to_numpy(dtype=np.float32))
        return np.stack(blocks, axis=1).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "used_cols": self.used_cols,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "missing_mean": self.missing_mean,
            "missing_std": self.missing_std,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        return cls(**data)


def split_by_patient(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    patient_col: str = PATIENT_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按患者划分，避免同一患者同时出现在训练与测试集造成泄漏。"""
    if patient_col in df.columns and df[patient_col].nunique() > 2:
        ids = df[patient_col].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
        n_test = max(1, int(len(ids) * test_ratio))
        n_val = max(1, int(len(ids) * val_ratio))
        test_ids, val_ids = set(ids[:n_test]), set(ids[n_test:n_test + n_val])
        test = df[df[patient_col].isin(test_ids)]
        val = df[df[patient_col].isin(val_ids)]
        train = df[~df[patient_col].isin(test_ids | val_ids)]
    else:
        shuffled = df.sample(frac=1.0, random_state=seed)
        n_test = max(1, int(len(df) * test_ratio))
        n_val = max(1, int(len(df) * val_ratio))
        test = shuffled.iloc[:n_test]
        val = shuffled.iloc[n_test:n_test + n_val]
        train = shuffled.iloc[n_test + n_val:]
    return train.copy(), val.copy(), test.copy()


def targets_to_index(series: pd.Series) -> np.ndarray:
    """ESI 1-5 -> 0-4 的类别下标。"""
    return (pd.to_numeric(series, errors="coerce") - CLASS_INDEX_OFFSET).to_numpy(dtype=np.int64, copy=True)


def load_frame(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"数据缺少目标列 {TARGET_COL}: {path}")
    df = df[df[TARGET_COL].between(1, 5)].reset_index(drop=True)
    return df
