#!/usr/bin/env python
# ------------------------------------------------------------
#  Isolation Forest training + ad-hoc analysis of multiple log sources
# ------------------------------------------------------------
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from fastapi.encoders import jsonable_encoder            # noqa: E402
from app.service.trainer import Trainer                  # noqa: E402
from app.service.analyser import Analyser                # noqa: E402
from app.service.classifier import Classifier            # noqa: E402

MODELS_DIR = ROOT / "app" / "models"

# ---------- CLI ------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Isolation Forest + Analyse Logfiles"
    )
    # Allow multiple paths (files *or* folders)
    p.add_argument("log_paths", type=Path, nargs="+",
                   help="*.log file or folder containing .log files")
    p.add_argument("--cont", type=float, default=0.05,
                   help="Contamination rate for Isolation Forest")
    p.add_argument("--trees", type=int, default=150,
                   help="#Trees for Isolation Forest")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training – use most recent model instead")
    return p.parse_args()

# ---------- Helper functions -----------------------------------------
def gather_logs(paths: List[Path]) -> List[Path]:
    logs: List[Path] = []
    for p in paths:
        if p.is_dir():
            logs.extend(sorted(p.rglob("*.log")))
        elif p.is_file() and p.suffix == ".log":
            logs.append(p)
    if not logs:
        sys.exit("❌  No .log files found.")
    return logs

# ---------- MAIN -----------------------------------------------------
def main() -> None:
    args = parse_args()
    log_files = gather_logs(args.log_paths)
    texts = [p.read_text(errors="ignore") for p in log_files]

    # ––– Train Isolation Forest ––––––––––––––––––––––––––––––––––––––
    if not args.skip_train:
        rel = Trainer(MODELS_DIR).train_from_texts(
            texts, contamination=args.cont, n_estimators=args.trees
        )
        print(f"✅  IF model saved as {rel}")
    else:
        print("ℹ️  Using most recently trained IF model (--skip-train)")

    # ––– Inference & Classification ––––––––––––––––––––––––––––––––––
    analyser   = Analyser(MODELS_DIR)
    classifier = Classifier(MODELS_DIR)

    rows = []
    for path, txt in zip(log_files, texts):
        res = analyser.analyse(txt)
        res["classifications"] = classifier.classify(txt)

        # Save JSON (next to logfile)
        out = path.with_suffix(".json")
        out.write_text(json.dumps(jsonable_encoder(res), indent=2))

        scores = [a.score for a in res["anomalies"]]
        avg_str = f"{sum(scores)/len(scores):.4f}" if scores else "—"
        min_str = f"{min(scores):.4f}"              if scores else "—"

        rows.append((path.name, len(scores), avg_str,
                     min_str, len(res["classifications"]),
                     res["model_used"]))

    # ––– Compact result table ––––––––––––––––––––––––––––––––––––––––
    print("\n### Evaluation results")
    print("| Logfile | #Anom | AvgScore | MinScore | #Class | Model |")
    print("|---------|------:|---------:|---------:|-------:|-------|")
    for fname, n_anom, avg, mn, n_cls, model in rows:
        print(f"| {fname} | {n_anom:^6} | {avg:^9} | {mn:^9} | "
              f"{n_cls:^7} | {model} |")
    print("\nJSON responses saved next to each .log.")


if __name__ == "__main__":
    main()
