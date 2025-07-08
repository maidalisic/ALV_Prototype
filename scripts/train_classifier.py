#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.service.trainer import Trainer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--csv",   type=Path, default=Path("data/labels.csv"))
p.add_argument("--trees", type=int,  default=400)
p.add_argument("--depth", type=int,  default=30)
args = p.parse_args()

trainer = Trainer(ROOT / "app" / "models")
rel = trainer.train_classifier(csv_path=args.csv,
                               trees=args.trees,
                               max_depth=args.depth)
print(f"✅  RF Classifier saved as {rel}")
