from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from app.service.trainer import Trainer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log_dir", type=Path)
    p.add_argument("--cont", type=float, default=0.05)
    p.add_argument("--trees", type=int, default=100)
    args = p.parse_args()

    files = sorted(args.log_dir.glob("*.log"))
    if not files:
        sys.exit("No .log files")

    texts = [p.read_text(errors="ignore") for p in files]
    trainer = Trainer(ROOT / "app" / "models")
    rel = trainer.train_from_texts(
        texts, contamination=args.cont, n_estimators=args.trees
    )
    print("Model saved as:", rel)


if __name__ == "__main__":
    main()
