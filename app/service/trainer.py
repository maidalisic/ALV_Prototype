from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Sequence, List
import joblib, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from ..schemas import ModelInfo
from .preprocess import clean_line


class Trainer:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------ Anomaly ------------
    def train_from_texts(
        self, texts: Sequence[str], *, contamination: float, n_estimators: int
    ) -> str:
        lines = [
            clean_line(ln) for txt in texts for ln in txt.splitlines() if ln.strip()
        ]
        if not lines:
            raise ValueError("Empty training corpus")

        vec = TfidfVectorizer(ngram_range=(1, 2))
        X = vec.fit_transform(lines)

        forest = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=42
        ).fit(X)

        scores = forest.decision_function(X)
        mu, sigma = scores.mean(), scores.std()
        threshold = mu - 2 * sigma

        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        path = self.models_dir / f"model_{ts}.joblib"
        joblib.dump(
            {"vectorizer": vec, "model": forest, "threshold": float(threshold)}, path
        )
        return str(path.relative_to(self.models_dir.parent))

    # ------------ Classifier (Random Forest) ------------
    def train_classifier(
        self,
        csv_path: Path,
        *,
        trees: int = 400,
        max_depth: int = 30,
    ) -> str:
        df = pd.read_csv(csv_path)
        
        # ---- Remove empty labels -----------------------------------
        before = len(df)
        df = df.dropna(subset=["label"])
        df = df[df["label"].astype(str).str.strip() != ""]
        skipped = before - len(df)
        if skipped:
            print(f"⚠️  {skipped} row(s) without label – ignored.")
        if df.empty:
            raise ValueError("No labeled rows found in CSV.")

        # ---- Recalculate line_norm if necessary --------------------
        if "line_norm" not in df.columns:
            print("ℹ️  Column 'line_norm' missing – will be computed from 'line'.")
            from .preprocess import clean_line
            df["line_norm"] = df["line"].apply(clean_line)

        vec = TfidfVectorizer(ngram_range=(1, 2))
        X = vec.fit_transform(df["line_norm"])
        y = df["label"]

        clf = RandomForestClassifier(
            n_estimators=trees,
            max_depth=max_depth,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ).fit(X, y)

        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        path = self.models_dir / f"classifier_{ts}.joblib"
        joblib.dump({"vectorizer": vec, "classifier": clf}, path)
        return str(path.relative_to(self.models_dir.parent))

    # ----------------------------------------------------
    def list_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name=p.stem,
                created_at=datetime.utcfromtimestamp(p.stat().st_mtime),
                path=str(p),
            )
            for p in sorted(self.models_dir.glob("*.joblib"))
        ]
