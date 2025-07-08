#!/usr/bin/env python
"""
Creates/updates data/labels.csv:
  * reads all *.log files under data/raw/
  * normalizes each line via app.service.preprocess.clean_line
  * writes CSV with columns: id,line_norm,label
  * existing labels.csv is preserved (only appends new lines)
After execution, you need to fill in the 'label' column manually – once.
"""
from __future__ import annotations
from pathlib import Path
import csv, itertools, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.service.preprocess import clean_line  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = ROOT / "data" / "labels.csv"

def collect_lines():
    for log_path in RAW_DIR.glob("*.log"):
        with log_path.open(errors="ignore") as fp:
            for ln in fp:
                ln = ln.rstrip()
                if not ln.strip():
                    continue
                yield clean_line(ln)

def main() -> None:
    existing = {}
    if CSV_PATH.exists():
        with CSV_PATH.open() as fp:
            rdr = csv.DictReader(fp)
            existing = {row["line_norm"]: row["label"] for row in rdr}

    all_lines = set(collect_lines())
    new_lines = sorted(all_lines - set(existing))
    if not new_lines:
        print("✔ labels.csv is already complete")
        return

    # Continue IDs
    start_id = len(existing) + 1
    with CSV_PATH.open("a", newline="") as fp:
        wr = csv.writer(fp)
        if fp.tell() == 0:
            wr.writerow(["id", "line_norm", "label"])
        for idx, ln in enumerate(new_lines, start=start_id):
            wr.writerow([idx, ln, ""])     # empty label to be filled in later
    print(f"➕ {len(new_lines)} new lines appended to labels.csv")

if __name__ == "__main__":
    main()
