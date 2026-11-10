"""FastAPI 推理服务。

启动：
    uvicorn er_triage.serve.app:app --port 8090

调用：
    curl -X POST http://127.0.0.1:8090/ai/emergency-triage \
      -H "Content-Type: application/json" \
      -H "X-Internal-Token: CHANGE_ME" \
      -d '{"age":56,"sex":"M","chief_complaint":"chest pain radiating to arm",
           "vitals":{"hr":118,"sbp":96,"rr":24,"spo2":92,"pain":7}}'

未加载到模型时返回 503，而不是用假数据兜底。
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Header, HTTPException

from ..config import load_config, resolve_path
from ..data.features import FeatureSpec
from ..models.fusion import build_model, probs_from_logits
from ..models.text import TextProcessor
from ..schema import ESI_RISK, ESI_SUGGESTIONS, NUMERIC_COLS
from .schemas import TriageRequest, TriageResponse

app = FastAPI(title="ER Triage Inference", version="0.1.0")

CFG = load_config(os.environ.get("TRIAGE_CONFIG", "configs/default.yaml"))
_LOCK = threading.Lock()
STATE: dict = {"model": None, "spec": None, "processor": None,
               "version": None, "head": None, "device": "cpu"}


def load_predictor() -> dict:
    """懒加载模型产物；同一个进程只加载一次。"""
    with _LOCK:
        if STATE["model"] is not None:
            return STATE
        path = resolve_path(os.environ.get("TRIAGE_MODEL_PATH", CFG["serve"]["model_path"]))
        if not path.exists():
            raise FileNotFoundError(f"模型产物不存在: {path}，请先运行训练脚本")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        cfg = artifact["config"]
        spec = FeatureSpec.from_dict(artifact["feature_spec"])
        processor = TextProcessor.from_state(artifact["text_state"])
        encoder = processor.build_encoder(int(cfg["model"]["text_dim"]), float(cfg["model"]["dropout"]))
        model = build_model(cfg, n_numeric=artifact["n_numeric"], text_encoder=encoder)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        STATE.update(model=model, spec=spec, processor=processor,
                     version=artifact["model_version"], head=artifact["head"])
        return STATE


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": STATE["model"] is not None,
        "model_version": STATE["version"],
    }


def predict_probs(state: dict, age: float | None, vitals: dict, chief_complaint: str) -> np.ndarray:
    """单条推理，API 与演示页共用，避免两处逻辑漂移。"""
    row = {"age": age}
    for col in NUMERIC_COLS:
        if col == "age":
            continue
        row[col] = vitals.get(col)
    frame = pd.DataFrame([row])

    x_num = torch.as_tensor(state["spec"].transform(frame), dtype=torch.float32)
    x_text = torch.as_tensor(state["processor"].encode([chief_complaint or ""]), dtype=torch.long)
    with torch.no_grad():
        logits = state["model"](x_num, x_text)
        return probs_from_logits(logits, state["head"])[0].numpy()


@app.post("/ai/emergency-triage", response_model=TriageResponse)
def emergency_triage(
    req: TriageRequest,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> TriageResponse:
    expected = str(CFG["serve"]["internal_token"])
    if not x_internal_token or x_internal_token != expected:
        raise HTTPException(status_code=401, detail="X-Internal-Token 无效")

    started = time.perf_counter()
    try:
        state = load_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    probs = predict_probs(state, req.age, req.vitals.model_dump(), req.chief_complaint)

    level = int(np.argmax(probs)) + 1
    return TriageResponse(
        level=level,
        probabilities=[round(float(p), 4) for p in probs],
        risk=ESI_RISK[level],
        suggestion=ESI_SUGGESTIONS[level],
        model_version=str(state["version"]),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
