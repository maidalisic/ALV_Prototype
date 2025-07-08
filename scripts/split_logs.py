#!/usr/bin/env python3
"""
Zeit­stratifizierter 70-15-15-Split eines einzelnen .log-Files.
Beispiel:
    python scripts/split_logs.py \
        --input data/raw/Mac.log \
        --outdir data/split/mac \
        --train 0.70 --val 0.15 --test 0.15
"""

import argparse, random
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--train",  type=float, default=0.70)
    p.add_argument("--val",    type=float, default=0.15)
    p.add_argument("--test",   type=float, default=0.15)
    args = p.parse_args()

    lines = args.input.read_text(errors="ignore").splitlines()
    n = len(lines)
    idx = list(range(n))
    random.seed(42)
    random.shuffle(idx)

    def slice_pct(frac: float, offset: int = 0):
        k = int(frac * n)
        return [lines[i] for i in idx[offset:offset + k]]

    train = slice_pct(args.train, 0)
    val   = slice_pct(args.val,   len(train))
    test  = slice_pct(args.test,  len(train) + len(val))

    args.outdir.mkdir(parents=True, exist_ok=True)
    Path(args.outdir / "train.log").write_text("\n".join(train))
    Path(args.outdir / "val.log").write_text("\n".join(val))
    Path(args.outdir / "test.log").write_text("\n".join(test))

    print(f"✅  {args.input.name}: {len(train)}/{len(val)}/{len(test)}  (train/val/test)")

if __name__ == "__main__":
    main()
