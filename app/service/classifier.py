from __future__ import annotations
import re, joblib
from pathlib import Path
from typing import List, Dict
from ..schemas import Classification
from .preprocess import clean_line

_PATTERNS = [
    ("TimeoutError", re.compile(r"timeout", re.I)),
    ("SegmentationFault", re.compile(r"segmentation fault", re.I)),
    ("NullPointer", re.compile(r"null pointer", re.I)),
    ("TestFailure", re.compile(r"test(failure| failed)", re.I)),
    ("MemoryLeak", re.compile(r"memory leak", re.I)),
]

CONF_THRESHOLD = 0.5


class Classifier:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._ml = self._load_latest()

    def classify(self, text: str) -> List[Classification]:
        raw_lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        lines = [clean_line(l) for l in raw_lines]
        ml_preds: Dict[int, Classification] = {}
        if self._ml:
            vec, clf = self._ml["vectorizer"], self._ml["classifier"]
            probs = clf.predict_proba(vec.transform(lines))
            for i, (p_vec, raw) in enumerate(zip(probs, raw_lines)):
                conf = p_vec.max()
                if conf >= CONF_THRESHOLD:
                    ml_preds[i] = Classification(
                        line_number=i + 1,
                        label=clf.classes_[p_vec.argmax()],
                        confidence=float(conf),
                        message=raw,
                    )

        results = list(ml_preds.values())
        for i, raw in enumerate(raw_lines):
            if i in ml_preds:
                continue
            for label, pat in _PATTERNS:
                if pat.search(raw):
                    results.append(
                        Classification(
                            line_number=i + 1, label=label, confidence=1.0, message=raw
                        )
                    )
                    break
        return results

    def _load_latest(self):
        files = sorted(self.models_dir.glob("classifier_*.joblib"))
        return joblib.load(files[-1]) if files else None
