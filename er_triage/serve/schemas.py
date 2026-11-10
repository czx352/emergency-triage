"""接口契约（与 BFF 侧 /ai/emergency-triage 对齐）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Vitals(BaseModel):
    hr: float | None = Field(default=None, ge=20, le=250, description="心率 bpm")
    sbp: float | None = Field(default=None, ge=40, le=260, description="收缩压 mmHg")
    dbp: float | None = Field(default=None, ge=20, le=160, description="舒张压 mmHg")
    rr: float | None = Field(default=None, ge=4, le=60, description="呼吸频率 次/分")
    temp_c: float | None = Field(default=None, ge=32, le=43, description="体温 ℃")
    spo2: float | None = Field(default=None, ge=50, le=100, description="血氧饱和度 %")
    pain: float | None = Field(default=None, ge=0, le=10, description="疼痛评分 NRS")
    gcs: float | None = Field(default=None, ge=3, le=15, description="格拉斯哥昏迷评分")


class TriageRequest(BaseModel):
    age: float | None = Field(default=None, ge=0, le=110)
    sex: str | None = Field(default=None, description="M/F")
    arrival_mode: str | None = Field(default=None, description="ambulance/walk_in/wheelchair/police")
    chief_complaint: str = Field(min_length=1, description="主诉")
    vitals: Vitals = Field(default_factory=Vitals)


class TriageResponse(BaseModel):
    level: int = Field(ge=1, le=5, description="ESI 分级，1 最危重")
    probabilities: list[float] = Field(description="5 个等级的概率，索引 0 对应 ESI 1")
    risk: str = Field(description="高/中/低")
    suggestion: str = Field(description="处置建议")
    model_version: str
    latency_ms: float
