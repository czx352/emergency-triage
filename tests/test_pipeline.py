import numpy as np
import torch

from er_triage.data.features import FeatureSpec, split_by_patient, targets_to_index
from er_triage.data.synthetic import generate
from er_triage.data.dataset import TriageDataset
from er_triage.metrics import evaluate
from er_triage.models.fusion import build_model, probs_from_logits
from er_triage.models.losses import build_loss
from er_triage.models.text import TextProcessor


def test_patient_split_has_no_overlap():
    df = generate(3000, seed=7)
    train, val, test = split_by_patient(df, 0.15, 0.15, seed=7)
    assert set(train["subject_id"]) & set(test["subject_id"]) == set()
    assert set(train["subject_id"]) & set(val["subject_id"]) == set()
    assert len(train) + len(val) + len(test) == len(df)


def test_feature_spec_drops_absent_column_and_imputes_nan():
    df = generate(200, seed=3).drop(columns=["gcs"])
    spec = FeatureSpec().fit(df, ["age", "hr", "sbp", "dbp", "rr", "temp_c", "spo2", "pain", "gcs"])
    assert "gcs" not in spec.used_cols
    x = spec.transform(df)
    assert x.shape == (len(df), spec.n_features)
    assert np.isfinite(x).all()


def test_one_training_step_runs():
    df = generate(160, seed=5)
    spec = FeatureSpec().fit(df)
    processor = TextProcessor("hash", max_len=16, vocab_size=512)
    x_num = torch.as_tensor(spec.transform(df))
    x_text = torch.as_tensor(processor.encode(df["chief_complaint"].tolist()))
    y = torch.as_tensor(targets_to_index(df["esi"]))

    ds = TriageDataset(x_num.numpy(), x_text.numpy(), y.numpy())
    cfg = {"model": {"text_dim": 16, "vitals_hidden": [16], "fusion": "gated",
                     "head": "coral", "dropout": 0.0}}
    encoder = processor.build_encoder(16, 0.0)
    model = build_model(cfg, spec.n_features, encoder)
    criterion = build_loss("coral")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    xb, tb, yb = [torch.stack([ds[i][k] for i in range(32)]) for k in range(3)]
    loss = criterion(model(xb, tb), yb)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)


def test_evaluate_end_to_end_pipeline():
    df = generate(400, seed=11)
    y_true = targets_to_index(df["esi"])
    rng = np.random.default_rng(0)
    y_pred = rng.integers(0, 5, size=len(df))
    result = evaluate(y_true, y_pred, [[0] * 5] * 5)
    assert result["n"] == len(df)

    probs = probs_from_logits(torch.randn(8, 4), "coral")
    assert probs.shape == (8, 5)
