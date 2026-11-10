"""MIMIC-IV-ED -> 本项目的训练表。

两种数据来源（二选一）：

A. 无需授权的开放版：MIMIC-IV-ED **Demo**（PhysioNet 上 Access=Open，只需点击同意
   Data Use Agreement，不需要 CITI 培训、不走 credentialed 审批）。适合校验真实数据
   格式与管线，但样本量小，不足以训练模型。

B. 全量版：MIMIC-IV-ED（credentialed）。需要完成 CITI「Data or Specimens Only
   Research」培训并申请审批。

无论哪种，数据都不得再分发，仓库中只保留本脚本，不提交任何原始数据。

目录约定（解压后放在 data/raw/ 下，`.csv.gz` 与 `.csv` 两种命名都支持）：
    data/raw/triage.csv.gz  /  triage.csv
    data/raw/edstays.csv.gz /  edstays.csv
    data/raw/patients.csv.gz   # 可选，来自 MIMIC-IV hosp 模块，用于补年龄

字段映射依据 MIMIC-IV-ED 官方 schema：
    triage.acuity -> esi，triage.chiefcomplaint -> chief_complaint，
    temperature/heartrate/resprate/o2sat -> temp_c/hr/rr/spo2
急诊分诊表不含 GCS，特征层会自动跳过缺失列（FeatureSpec.used_cols）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import resolve_path

COLUMN_MAP = {
    "temperature": "temp_c",
    "heartrate": "hr",
    "resprate": "rr",
    "o2sat": "spo2",
    "sbp": "sbp",
    "dbp": "dbp",
    "pain": "pain",
    "acuity": "esi",
    "chiefcomplaint": "chief_complaint",
}


def _read_table(raw_dir: Path, stem: str) -> pd.DataFrame:
    """按 `stem.csv.gz` / `stem.csv` 的顺序查找并读取，兼容开放 Demo 与全量版。"""
    for candidate in (f"{stem}.csv.gz", f"{stem}.csv"):
        path = raw_dir / candidate
        if path.exists():
            return pd.read_csv(path, compression="infer")
    raise FileNotFoundError(
        f"缺少 {raw_dir / (stem + '.csv')}（或 .csv.gz）。请把 MIMIC-IV-ED Demo（开放版）"
        f"或全量版的 {stem} 表解压到 {raw_dir}。"
    )


def _read_patients(raw_dir: Path, filename: str) -> pd.DataFrame:
    stem = Path(filename).name.replace(".csv.gz", "").replace(".csv", "")
    return _read_table(raw_dir, stem)


def convert(raw_dir: str | Path = "data/raw",
            out_path: str | Path = "data/processed/mimic_ed.csv",
            patients_file: str | None = None) -> pd.DataFrame:
    raw = resolve_path(raw_dir)
    triage = _read_table(raw, "triage").rename(columns=COLUMN_MAP)
    stays = _read_table(raw, "edstays")

    need = ["subject_id", "stay_id", "esi", "chief_complaint"]
    missing = [c for c in need if c not in triage.columns]
    if missing:
        raise ValueError(f"triage 表缺少字段 {missing}，请核对 MIMIC-IV-ED 版本")

    df = triage[[c for c in ["subject_id", "stay_id", "esi", "chief_complaint",
                             "temp_c", "hr", "rr", "spo2", "sbp", "dbp", "pain"]
                 if c in triage.columns]].copy()

    # 性别、到达方式、就诊时间在 edstays
    join_cols = [c for c in ["subject_id", "stay_id", "gender", "arrival_transport", "intime"]
                 if c in stays.columns]
    df = df.merge(stays[join_cols], on=["subject_id", "stay_id"], how="left")
    df = df.rename(columns={"gender": "sex", "arrival_transport": "arrival_mode"})

    # 年龄：MIMIC-IV 用 anchor_age + 就诊年份偏移
    if patients_file:
        pts = _read_patients(raw, patients_file)
        if {"subject_id", "anchor_age", "anchor_year"} <= set(pts.columns):
            df = df.merge(pts[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left")
            visit_year = pd.to_datetime(df["intime"], errors="coerce").dt.year
            df["age"] = df["anchor_age"] + (visit_year - df["anchor_year"]).clip(lower=0)
            df = df.drop(columns=["anchor_age", "anchor_year"])

    df["esi"] = pd.to_numeric(df["esi"], errors="coerce")
    df = df[df["esi"].between(1, 5)].drop_duplicates(subset="stay_id")

    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="把 MIMIC-IV-ED 转换为训练表")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/mimic_ed.csv")
    parser.add_argument("--patients", default=None,
                        help="可选，MIMIC-IV hosp 的 patients.csv.gz，用于补年龄")
    args = parser.parse_args()
    df = convert(args.raw_dir, args.out, args.patients)
    print(f"已转换 {len(df)} 条记录 -> {resolve_path(args.out)}")
    print(df["esi"].value_counts(normalize=True).sort_index().round(3).to_string())


if __name__ == "__main__":
    main()
