import torch
import pytest
from fastapi.testclient import TestClient

from er_triage.config import load_config
from er_triage.data.features import FeatureSpec
from er_triage.data.synthetic import generate
from er_triage.models.fusion import build_model
from er_triage.models.text import TextProcessor
from er_triage.serve import app as app_module

BODY = {
    "age": 56,
    "sex": "M",
    "arrival_mode": "ambulance",
    "chief_complaint": "chest pain radiating to arm, for 2 days",
    "vitals": {"hr": 118, "sbp": 96, "dbp": 60, "rr": 24, "temp_c": 37.2,
               "spo2": 92, "pain": 7, "gcs": 15},
}


def _make_artifact(tmp_path, cfg):
    """构造一个未训练的小模型产物，只验证服务链路，不验证效果。"""
    df = generate(300, seed=1)
    spec = FeatureSpec().fit(df, cfg["data"]["numeric_cols"])
    processor = TextProcessor("hash", int(cfg["model"]["text_max_len"]),
                              int(cfg["model"]["text_vocab_size"]))
    encoder = processor.build_encoder(int(cfg["model"]["text_dim"]), float(cfg["model"]["dropout"]))
    model = build_model(cfg, spec.n_features, encoder)
    artifact = {
        "state_dict": model.state_dict(),
        "config": cfg,
        "feature_spec": spec.to_dict(),
        "text_state": processor.state(),
        "n_numeric": spec.n_features,
        "head": cfg["model"]["head"],
        "model_version": "test-0.0.1",
    }
    path = tmp_path / "tiny.pt"
    torch.save(artifact, path)
    return path


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = load_config("configs/default.yaml")
    cfg["serve"]["internal_token"] = "test-token"
    monkeypatch.setattr(app_module, "CFG", cfg)
    monkeypatch.setattr(app_module, "STATE",
                        {"model": None, "spec": None, "processor": None,
                         "version": None, "head": None, "device": "cpu"})
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(_make_artifact(tmp_path, cfg)))
    return TestClient(app_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_ready"] is False


def test_rejects_missing_or_wrong_token(client):
    assert client.post("/ai/emergency-triage", json=BODY).status_code == 401
    assert client.post("/ai/emergency-triage", json=BODY,
                       headers={"X-Internal-Token": "wrong"}).status_code == 401


def test_predicts_with_contract_fields(client):
    resp = client.post("/ai/emergency-triage", json=BODY,
                       headers={"X-Internal-Token": "test-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert 1 <= data["level"] <= 5
    assert len(data["probabilities"]) == 5
    assert abs(sum(data["probabilities"]) - 1.0) < 1e-3
    assert data["risk"] in {"高", "中", "低"}
    assert data["model_version"] == "test-0.0.1"


def test_returns_503_when_artifact_missing(client, monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_MODEL_PATH", str(tmp_path / "not_exist.pt"))
    monkeypatch.setattr(app_module, "STATE",
                        {"model": None, "spec": None, "processor": None,
                         "version": None, "head": None, "device": "cpu"})
    resp = client.post("/ai/emergency-triage", json=BODY,
                       headers={"X-Internal-Token": "test-token"})
    assert resp.status_code == 503


def test_validation_rejects_empty_complaint(client):
    bad = {**BODY, "chief_complaint": ""}
    resp = client.post("/ai/emergency-triage", json=bad,
                       headers={"X-Internal-Token": "test-token"})
    assert resp.status_code == 422
