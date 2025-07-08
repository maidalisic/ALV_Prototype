#!/usr/bin/env python3
"""
Creates (or extends) data/labels.csv for ground-truth labels.

 ▸ PATTERNS_LABEL – used ONLY here!
 ▸ The baseline rules are defined in baseline_regex.py (PATTERNS_BASE)
   and are intentionally different so the baseline does not evaluate itself.
"""

from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from app.service.preprocess import clean_line          # noqa: E402

# ------------------------------------------------------------------ #
#  PATTERNS used only for ground-truth labeling                     #
#  (more narrowly defined variants, about 60% of cases)             #
# ------------------------------------------------------------------ #
PATTERNS_LABEL: list[tuple[str, re.Pattern]] = [
    # --- macOS (three common errors) ------------------------------
    ("BluetoothError",   re.compile(r"\[bluetooth.*error", re.I)),
    ("NoNetworkRoute",   re.compile(r"no network route",    re.I)),
    ("LayoutConstraint", re.compile(r"unable to simultaneously satisfy constraints", re.I)),

    # --- OpenSSH (two clearly recognizable classes) ----------------
    ("SSHAuthFail",      re.compile(r"authentication failure", re.I)),
    ("SSHPossibleBreak", re.compile(r"possible break-in attempt", re.I)),
]

# ------------------------------------------------------------------ #
def detect_label(norm_line: str) -> Optional[str]:
    for lbl, pat in PATTERNS_LABEL:
        if pat.search(norm_line):
            return lbl
    return None
# ------------------------------------------------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", action="append", required=True, type=Path,
                    help="Directory/directories containing .log files (not individual files)")
    ap.add_argument("--out", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--inplace", action="store_true",
                    help="Append to existing CSV instead of overwriting")
    args = ap.parse_args()

    known: set[str] = set()
    rows: list[tuple[str, str]] = []

    if args.inplace and args.out.exists():
        with args.out.open() as fp:
            for row in csv.DictReader(fp):
                rows.append((row["line_norm"], row["label"]))
                known.add(row["line_norm"])

    for d in args.logs_dir:
        for logf in d.rglob("*.log"):
            for raw in logf.read_text(errors="ignore").splitlines():
                norm = clean_line(raw)
                if norm in known:
                    continue
                lbl = detect_label(norm)
                if lbl:
                    rows.append((norm, lbl))
                    known.add(norm)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["line_norm", "label"])
        w.writerows(rows)

    print(f"✅  {len(rows):,} labeled rows written → {args.out}")

if __name__ == "__main__":
    main()
