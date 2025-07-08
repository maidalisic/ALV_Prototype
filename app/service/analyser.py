from __future__ import annotations
from pathlib import Path
from typing import List
import re, joblib
from ..schemas import Anomaly
from .preprocess import clean_line

_ERR_PAT = re.compile(r"\b(ERROR|FAIL|FATAL)\b", re.I)


class Analyser:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir

    def analyse(self, text: str) -> dict:
        model_path = max(self.models_dir.glob("model_*.joblib"), default=None)
        if model_path is None:
            raise RuntimeError("No model trained")

        bundle = joblib.load(model_path)
        vec, forest, threshold = (
            bundle["vectorizer"],
            bundle["model"],
            bundle["threshold"],
        )

        raw_lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        lines = [clean_line(l) for l in raw_lines]
        if not lines:
            return {"anomalies": [], "model_used": model_path.name}

        scores = forest.decision_function(vec.transform(lines))

        anomalies: List[Anomaly] = [
            Anomaly(line_number=i + 1, score=float(s), message=raw_lines[i])
            for i, s in enumerate(scores)
            if s <= threshold
        ]

        if not anomalies:
            for i, (raw, s) in enumerate(zip(raw_lines, scores)):
                if s > threshold and _ERR_PAT.search(raw):
                    anomalies.append(
                        Anomaly(line_number=i + 1, score=threshold - 0.001, message=raw)
                    )
        return {"anomalies": anomalies, "model_used": model_path.name}
