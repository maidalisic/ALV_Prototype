from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

CFG = ConfigDict(protected_namespaces=())


class Anomaly(BaseModel):
    line_number: int = Field(..., ge=1)
    score: float = Field(..., ge=-1.0, le=1.0)
    message: str
    model_config = CFG


class Classification(BaseModel):
    line_number: int
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    message: str
    model_config = CFG


class AnalyseResponse(BaseModel):
    anomalies: List[Anomaly]
    classifications: Optional[List[Classification]] = None
    model_used: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_config = CFG


class TrainResponse(BaseModel):
    model_path: str
    trained_at: datetime = Field(default_factory=datetime.utcnow)
    model_config = CFG


class ModelInfo(BaseModel):
    name: str
    created_at: datetime
    path: str
    model_config = CFG
