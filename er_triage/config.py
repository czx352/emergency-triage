"""配置加载：YAML + 默认值深合并，路径统一相对项目根目录。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "data": {
        "processed_dir": "data/processed",
        "raw_dir": "data/raw",
        "train_file": "data/processed/synthetic.csv",
        "text_col": "chief_complaint",
        "target_col": "esi",
        "patient_col": "subject_id",
        "numeric_cols": ["age", "hr", "sbp", "dbp", "rr", "temp_c", "spo2", "pain", "gcs"],
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "n_folds": 5,
    },
    "model": {
        "text_backend": "hash",
        "hf_model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "text_max_len": 24,
        "text_vocab_size": 32768,
        "text_dim": 128,
        "vitals_hidden": [128, 64],
        "fusion": "gated",
        "head": "coral",
        "dropout": 0.2,
    },
    "train": {
        "epochs": 8,
        "batch_size": 64,
        "lr": 2e-4,
        "weight_decay": 0.01,
        "device": "auto",
        "patience": 2,
        "loss": "coral",
        "cost_matrix": [
            [0, 1, 2, 4, 8],
            [1, 0, 1, 3, 6],
            [2, 1, 0, 1, 3],
            [4, 3, 1, 0, 1],
            [8, 6, 3, 1, 0],
        ],
    },
    "serve": {
        "model_path": "artifacts/best.pt",
        "internal_token": "CHANGE_ME",
        "model_version": "er-triage-0.1.0",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """读取 YAML 并与默认值合并。path 相对项目根目录时自动补全。"""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    user_cfg = {}
    if cfg_path.exists():
        user_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return _deep_merge(DEFAULTS, user_cfg)


def resolve_path(relative: str | Path) -> Path:
    """把配置里的相对路径解析为项目根目录下的绝对路径。"""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
