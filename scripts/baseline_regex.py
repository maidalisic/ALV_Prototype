#!/usr/bin/env python3
"""
Simple regex baseline (intentionally uses *different* patterns than labeling).

 ▸ compares against labels.csv
 ▸ outputs micro and optionally macro scores
"""

from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from app.service.preprocess import clean_line as normalise

# ------------------------------------------------------------------ #
#  PATTERNS_BASE – intentionally broader/supplementary rules         #
# ------------------------------------------------------------------ #
PATTERNS_BASE: list[tuple[str, str]] = [
    ("SSHInvalidUser", r"invalid user"),
    ("SSHFailedPass",  r"failed password"),
    ("SSHTooManyAuth", r"too many authentication failures"),
    ("BluetoothError", r"bluetooth.*error"),          # some overlap is acceptable
    ("NoNetworkRoute", r"no network route"),
]

COMPILED = [(lbl, re.compile(rx, re.I)) for lbl, rx in PATTERNS_BASE]

def match(lbl_raw: str) -> str | None:
    for lbl, rx in COMPILED:
        if rx.search(lbl_raw):
            return lbl
    return None

def safe_auprc(y_true_bin, y_score) -> str:
    if sum(y_true_bin) == 0:
        return "n/a"
    try:
        return f"{average_precision_score(y_true_bin, y_score):.3f}"
    except ValueError:
        return "n/a"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-logs", required=True, type=Path)
    ap.add_argument("--csv",       default=Path("data/labels.csv"), type=Path)
    ap.add_argument("--macro",     action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    truth = dict(zip(df["line_norm"], df["label"]))

    y_true, y_pred, y_true_bin, y_score = [], [], [], []

    for log in args.test_logs.rglob("test.log"):
        for raw in log.read_text(errors="ignore").splitlines():
            norm = normalise(raw)
            if norm not in truth:
                continue
            y_true.append(truth[norm])
            pred = match(raw) or "None"
            y_pred.append(pred)
            y_true_bin.append(1)           # all lines are positive
            y_score.append(1.0 if pred != "None" else 0.0)

    if not y_true:
        sys.exit("⚠️  No overlapping labeled lines found")

    avg = "macro" if args.macro else "micro"
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg, zero_division=0
    )
    print(f"{avg.capitalize()} Precision : {p:.3f}")
    print(f"{avg.capitalize()} Recall    : {r:.3f}")
    print(f"{avg.capitalize()} F1        : {f1:.3f}")
    print(f"{avg.capitalize()} AUPRC     : {safe_auprc(y_true_bin, y_score)}")

if __name__ == "__main__":
    main()
