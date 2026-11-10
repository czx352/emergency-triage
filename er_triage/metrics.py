"""评估指标：区分「分得准」和「分得安全」。

准确率在 ESI 任务上没有意义（39% 的病例都是 3 级），因此主指标用：
- Quadratic Weighted Kappa：有序一致性；
- 危重降级率（critical under-triage）：真实 ESI<=2 被判成 >=3，临床最关心的错误；
- 代价加权误差：用配置里的代价矩阵换算成临床风险。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from .schema import CRITICAL_ESI, N_CLASSES, UNDER_TRIAGE_PRED


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> float:
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)),
                          average="macro", zero_division=0))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def under_triage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """整体分诊不足比例：预测等级比真实更轻。"""
    return float(np.mean(y_pred > y_true))


def critical_under_triage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """危重降级率：真实 ESI<=2 却被判为 >=3，即可能延误抢救。"""
    mask = y_true <= (CRITICAL_ESI - 1)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[mask] >= (UNDER_TRIAGE_PRED - 1)))


def over_triage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """分诊过度比例：预测等级比真实更重，浪费抢救资源。"""
    return float(np.mean(y_pred < y_true))


def cost_weighted_error(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix) -> float:
    cost = np.asarray(cost_matrix, dtype=np.float64)
    return float(np.mean(cost[y_true, y_pred]))


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric, n_boot: int = 500,
                 seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values.append(metric(y_true[idx], y_pred[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix) -> dict:
    lo, hi = bootstrap_ci(y_true, y_pred, macro_f1)
    k_lo, k_hi = bootstrap_ci(y_true, y_pred, quadratic_weighted_kappa)
    return {
        "n": int(len(y_true)),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci95": [lo, hi],
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "qwk_ci95": [k_lo, k_hi],
        "accuracy": float(np.mean(y_true == y_pred)),
        "critical_under_triage": critical_under_triage_rate(y_true, y_pred),
        "under_triage": under_triage_rate(y_true, y_pred),
        "over_triage": over_triage_rate(y_true, y_pred),
        "cost_weighted_error": cost_weighted_error(y_true, y_pred, cost_matrix),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
