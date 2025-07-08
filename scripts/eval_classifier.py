#!/usr/bin/env python
# ------------------------------------------------------------
#  RF Classifier – Precision/Recall/F1 (val & test splits)
# ------------------------------------------------------------
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.service.classifier import Classifier            # noqa: E402
from app.service.preprocess import clean_line            # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test-logs", type=Path, required=True,
                   help="Root folder with split/{mac,ssh}/{val,test}.log")
    p.add_argument("--csv", type=Path, default=Path("data/labels.csv"))
    return p.parse_args()


def gather_logs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.log"))


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["label"])
    truth = dict(zip(df["line_norm"], df["label"]))

    clf = Classifier(ROOT / "app" / "models")

    y_true, y_pred = [], []
    for lp in gather_logs(args.test_logs):
        if lp.stem not in {"val", "test"}:   # only evaluate val/test splits
            continue

        raw = lp.read_text(errors="ignore")
        preds = {clean_line(c.message): c.label for c in clf.classify(raw)}

        for ln in raw.splitlines():
            ln_norm = clean_line(ln)
            if ln_norm in truth:
                y_true.append(truth[ln_norm])
                y_pred.append(preds.get(ln_norm, "None"))

    if not y_true:
        sys.exit("❌  No labeled lines found in val/test – check paths.")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"\nPrecision: {prec:.2%}  Recall: {rec:.2%}  F1: {f1:.2%}\n")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
