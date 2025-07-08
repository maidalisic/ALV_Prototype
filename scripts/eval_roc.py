from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))        # <── newly inserted at the top

import json, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--label-csv", type=Path, default=Path("logs/labels.csv"))
    return p.parse_args()


def main():
    args = parse()
    gt = {
        (row.split(",")[0]): row.split(",")[1].strip()
        for row in args.label_csv.read_text().splitlines()[1:]
    }
    y_true, y_score = [], []
    for jf in args.json_dir.glob("*.json"):
        data = json.loads(jf.read_text())
        for a in data["anomalies"]:
            y_true.append(1 if a["message"] in gt else 0)
            y_score.append(-a["score"])

    if len(set(y_true)) < 2:
        print("🚫  ROC requires at least one positive and one negative class.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.3f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC – Anomaly Detection")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
