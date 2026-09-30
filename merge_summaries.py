"""
merge_summaries.py — fold parallel-run summary__<tag>.csv files into summary.csv.

Each `run_experiment.py --tag X` job writes its own summary__X.csv so concurrent
jobs never append to the same file. Run this once they finish to produce the
single summary.csv the analysers read.

  python merge_summaries.py            # merge all summary__*.csv (+ existing summary.csv)
  python merge_summaries.py --keep     # keep the per-tag files (default: delete)

De-dupes on (video, kind, model, yolo_requested, repeat); the last file wins for
a given key, so re-running a tag and re-merging is safe.
"""

import argparse
import csv
from pathlib import Path

EXP = Path(__file__).parent / "eval_results_experiment"
KEY = ("video", "kind", "model", "yolo_requested", "repeat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="don't delete summary__*.csv after merge")
    args = ap.parse_args()

    parts = sorted(EXP.glob("summary__*.csv"))
    main_csv = EXP / "summary.csv"
    if not parts:
        raise SystemExit("no summary__*.csv files to merge")

    rows, header = {}, None
    sources = ([main_csv] if main_csv.exists() else []) + parts
    for p in sources:
        with open(p, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            header = rd.fieldnames
            n = 0
            for r in rd:
                rows[tuple(r.get(k, "") for k in KEY)] = r
                n += 1
        print(f"  {p.name}: {n} rows")

    with open(main_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    print(f"\nmerged {len(rows)} unique rows -> {main_csv.name}")

    if not args.keep:
        for p in parts:
            p.unlink()
        print(f"removed {len(parts)} per-tag file(s)")


if __name__ == "__main__":
    main()
