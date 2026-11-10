"""ETL 兼容性测试：开放 Demo 常见是未压缩的 .csv，全量版是 .csv.gz。"""
import pandas as pd

from er_triage.data.mimic_ed import convert


def _make_raw(raw_dir, suffix=".csv"):
    triage = pd.DataFrame({
        "subject_id": [1, 2],
        "stay_id": [10, 11],
        "acuity": [2, 3],
        "chiefcomplaint": ["chest pain", "ankle sprain"],
        "temperature": [37.0, 36.8],
        "heartrate": [110, 85],
        "resprate": [22, 16],
        "o2sat": [94, 99],
        "sbp": [100, 130],
        "dbp": [60, 80],
        "pain": [7, 3],
    })
    stays = pd.DataFrame({
        "subject_id": [1, 2],
        "stay_id": [10, 11],
        "gender": ["M", "F"],
        "arrival_transport": ["ambulance", "walk_in"],
        "intime": ["2020-01-01 08:00:00", "2020-01-02 09:00:00"],
    })
    triage.to_csv(raw_dir / f"triage{suffix}", index=False, compression="infer")
    stays.to_csv(raw_dir / f"edstays{suffix}", index=False, compression="infer")


def _assert_mapped(df):
    assert len(df) == 2
    assert list(df["esi"]) == [2, 3]
    assert df["hr"].tolist() == [110, 85]
    assert df["spo2"].tolist() == [94, 99]
    assert df["sex"].tolist() == ["M", "F"]
    assert df["arrival_mode"].tolist() == ["ambulance", "walk_in"]
    assert "gcs" not in df.columns  # 急诊表没有 GCS，交由特征层跳过


def test_convert_accepts_uncompressed_demo_csv(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_raw(raw, ".csv")
    df = convert(raw, tmp_path / "out.csv")
    _assert_mapped(df)


def test_convert_accepts_gzip_full_version(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_raw(raw, ".csv.gz")
    df = convert(raw, tmp_path / "out.csv")
    _assert_mapped(df)
