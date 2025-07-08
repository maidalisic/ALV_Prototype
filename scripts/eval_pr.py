from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))        # <── neu ganz oben

import argparse, json
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--json-dir", type=Path, required=True)
args = p.parse_args()

y_true, y_score = [], []

for jf in args.json_dir.glob("*.json"):
    with open(jf) as fp:
        data = json.load(fp)
    log_lines = jf.with_suffix(".log").read_text().splitlines()

    err_lines = {a["message"] for a in data["anomalies"]}
    for ln in log_lines:
        score = next((a["score"] for a in data["anomalies"] if a["message"] == ln), 0.0)
        y_true.append(1 if ln in err_lines else 0)
        y_score.append(-score)

prec, rec, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)

print(f"Average Precision: {ap:.3f}")

plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"PR-Curve (AP={ap:.3f})")
plt.show()
