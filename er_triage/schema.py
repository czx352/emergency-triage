"""领域常量：ESI 分级定义、字段名、临床合理区间、输出文案。

ESI（Emergency Severity Index）为 1-5 级的有序分级，
1 = 需立即抢救，5 = 非急诊。模型输出必须保留“有序”这一结构信息。
"""
from __future__ import annotations

N_CLASSES = 5
ESI_MIN, ESI_MAX = 1, 5
CLASS_IDS = list(range(ESI_MIN, ESI_MAX + 1))  # [1, 2, 3, 4, 5]
CLASS_INDEX_OFFSET = ESI_MIN                  # ESI 1-5 转 0-based 下标需减去 1

TEXT_COL = "chief_complaint"
TARGET_COL = "esi"
PATIENT_COL = "subject_id"
STAY_COL = "stay_id"

NUMERIC_COLS = ["age", "hr", "sbp", "dbp", "rr", "temp_c", "spo2", "pain", "gcs"]

# 生理参数的临床合理区间，超出视为录入错误并截断
CLIP_RANGES: dict[str, tuple[float, float]] = {
    "age": (0.0, 110.0),
    "hr": (20.0, 250.0),
    "sbp": (40.0, 260.0),
    "dbp": (20.0, 160.0),
    "rr": (4.0, 60.0),
    "temp_c": (32.0, 43.0),
    "spo2": (50.0, 100.0),
    "pain": (0.0, 10.0),
    "gcs": (3.0, 15.0),
}

ESI_LABELS = {
    1: "立即抢救（Resuscitation）",
    2: "危重（Emergent）",
    3: "紧急（Urgent）",
    4: "次紧急（Less Urgent）",
    5: "非急诊（Non-Urgent）",
}

# 按分级给出的处置建议，服务层直接返回
ESI_SUGGESTIONS = {
    1: "立即进入抢救室，启动绿色通道，通知急诊医师与护士长",
    2: "10 分钟内接诊，持续监护，优先安排检查与床位",
    3: "30 分钟内接诊，动态复评生命体征",
    4: "60 分钟内接诊，可安排至普通诊区候诊",
    5: "按序就诊，提供健康宣教与分流指引",
}

# 服务层把分级映射为风险标签
ESI_RISK = {1: "高", 2: "高", 3: "中", 4: "低", 5: "低"}

# 危重阈值：真实 ESI <= CRITICAL_ESI 视为危重，被预测为 >= UNDER_TRIAGE_PRED 即“降级漏诊”
CRITICAL_ESI = 2
UNDER_TRIAGE_PRED = 3
