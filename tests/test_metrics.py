import numpy as np

from er_triage.metrics import (
    confusion,
    cost_weighted_error,
    critical_under_triage_rate,
    evaluate,
    macro_f1,
    quadratic_weighted_kappa,
    under_triage_rate,
)


def test_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4])
    assert macro_f1(y, y) == 1.0
    assert quadratic_weighted_kappa(y, y) == 1.0
    assert under_triage_rate(y, y) == 0.0
    assert critical_under_triage_rate(y, y) == 0.0


def test_critical_under_triage_only_counts_critical_cases():
    # 第 1 例真实 ESI1 被判成 ESI4（危重降级）；第 2 例真实 ESI4 被判成 ESI5（不致命）
    y_true = np.array([0, 3])
    y_pred = np.array([3, 4])
    assert critical_under_triage_rate(y_true, y_pred) == 1.0
    assert under_triage_rate(y_true, y_pred) == 1.0

    # 危重病例被正确识别时，该指标为 0
    y_pred_ok = np.array([1, 4])
    assert critical_under_triage_rate(y_true, y_pred_ok) == 0.0


def test_qwk_penalizes_far_errors_more_than_near_errors():
    y_true = np.array([0, 0, 4, 4])
    near = np.array([1, 1, 3, 3])
    far = np.array([4, 4, 0, 0])
    assert quadratic_weighted_kappa(y_true, near) > quadratic_weighted_kappa(y_true, far)


def test_cost_weighted_error_uses_matrix():
    cost = [[0, 1, 2, 4, 8],
            [1, 0, 1, 3, 6],
            [2, 1, 0, 1, 3],
            [4, 3, 1, 0, 1],
            [8, 6, 3, 1, 0]]
    y_true = np.array([0])
    assert cost_weighted_error(y_true, np.array([0]), cost) == 0.0
    assert cost_weighted_error(y_true, np.array([4]), cost) == 8.0


def test_evaluate_returns_expected_keys_and_confusion_shape():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 5, size=200)
    result = evaluate(y_true, y_pred, [[0] * 5] * 5)
    for key in ["macro_f1", "qwk", "accuracy", "critical_under_triage",
                "under_triage", "over_triage", "cost_weighted_error"]:
        assert key in result
    assert confusion(y_true, y_pred, 5).shape == (5, 5)
