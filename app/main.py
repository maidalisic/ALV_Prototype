from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Header
from fastapi.openapi.models import Contact, License

from .schemas import AnalyseResponse, TrainResponse, ModelInfo
from .service.analyser import Analyser
from .service.trainer import Trainer
from .service.chatgpt import ChatGPTAnalyser
from .service.classifier import Classifier

contact = Contact(name="Alisic Maid", email="maid@alisic.net")

app = FastAPI(
    title="ALV – AI-powered Logfile Validator",
    version="0.3.2",
    description="Detect anomalies and classify errors in CI/CD log files.",
    contact=contact,
)

# ------------------------------------------------------------------ Singletons
MODELS_DIR = Path(__file__).resolve().parent / "models"
analyser = Analyser(MODELS_DIR)
trainer = Trainer(MODELS_DIR)
classifier = Classifier(MODELS_DIR)


@app.post("/analyse", response_model=AnalyseResponse, summary="Analyse a logfile")
async def analyse_logs(
    file: UploadFile = File(
        ...,
        examples=[
            {
                "summary": "Segfault snippet",
                "description": "Minimal failing log section",
                "value": "ERROR 12:00:01 KERNEL : Segmentation fault ...",
            }
        ],
    ),
    mode: str = Query("local", enum=["local", "chatgpt"]),
    classify: bool = Query(False),
    openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
):
    data = (await file.read()).decode("utf-8", errors="ignore")
    if not data.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = (
        analyser.analyse(data)
        if mode == "local"
        else await ChatGPTAnalyser(api_key=openai_key).analyse(data)
    )
    if classify:
        result["classifications"] = classifier.classify(data)
    return AnalyseResponse(**result)


@app.post("/train", response_model=TrainResponse, summary="Train a new model")
async def train_model(
    files: List[UploadFile] = File(...),
    contamination: float = Query(0.05, ge=0.0, le=0.5),
    n_estimators: int = Query(100, ge=50, le=500),
):
    texts: List[str] = []
    for f in files:
        raw = await f.read()
        txt = raw.decode("utf-8", errors="ignore")
        if txt.strip():
            texts.append(txt)
    if not texts:
        raise HTTPException(status_code=400, detail="No non-empty files uploaded.")

    path = trainer.train_from_texts(
        texts, contamination=contamination, n_estimators=n_estimators
    )
    return TrainResponse(model_path=path)


@app.get("/models", response_model=List[ModelInfo], summary="List models")
async def list_models():
    return trainer.list_models()
