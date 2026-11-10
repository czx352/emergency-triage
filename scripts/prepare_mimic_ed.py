"""把 MIMIC-IV-ED 转成训练表：py scripts/prepare_mimic_ed.py --patients patients.csv.gz"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from er_triage.data.mimic_ed import main  # noqa: E402

if __name__ == "__main__":
    main()
