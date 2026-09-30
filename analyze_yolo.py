"""
analyze_yolo.py - Summarise the stock-YOLO baseline and place it next to the LLMs.

Reads eval_results_experiment/yolo_summary.csv (from yolo_baseline.py) and, when
present, eval_results_experiment/summary.csv (from run_experiment.py) so the YOLO
recall sits in the same table as gpt4 / llama_vision / moondream.

Ground truth is ANOMALY for every clip in the dataset (no negative controls), so
the metric is recall (detection rate), reported with a Wilson 95% interval.
Stdlib only.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent / "eval_results_experiment"
YOLO_CSV = BASE / "yolo_summary.csv"
LLM_CSV = BASE / "summary.csv"
Z95 = 1.959963985
HIT = {"ANOMALIA", "ANOMALY"}  # accept both the PT-era and current tokens


def wilson_ci(successes, n, z=Z95):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (center - adj) / denom), min(1.0, (center + adj) / denom))


def pct(x):
    return f"{x * 100:.1f}%"


def load(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    yolo = load(YOLO_CSV)
    if not yolo:
        raise SystemExit(f"No rows in {YOLO_CSV}. Run: python yolo_baseline.py")
    llm = [r for r in load(LLM_CSV)
           if r.get("status") != "ERROR" and r.get("yolo_requested") == "off"]

    tracks = ["external", "internal"]
    yolo_models = sorted(set(r["model"] for r in yolo))

    print("=" * 74)
    print("STOCK-YOLO BASELINE - recall (ground truth = ANOMALY for all clips)")
    print("=" * 74)
    print(f"{'Model':<12}{'Track':<11}{'n':<5}{'Hits':<6}{'Recall':<9}{'95% CI':<20}{'Latency':<10}")
    for m in yolo_models:
        for tr in tracks:
            sub = [r for r in yolo if r["model"] == m and r["kind"] == tr]
            if not sub:
                continue
            n = len(sub)
            hits = sum(1 for r in sub if r["status"] in HIT)
            p, lo, hi = wilson_ci(hits, n)
            lat = sum(float(r["elapsed_s"]) for r in sub) / n
            print(f"{m:<12}{tr:<11}{n:<5}{hits:<6}{pct(p):<9}"
                  f"[{pct(lo)}, {pct(hi)}]{'':<3}{lat:.2f}s")

    # Per-video hit/miss grid
    print("\n" + "=" * 74)
    print("PER-VIDEO VERDICTS")
    print("=" * 74)
    videos = sorted(set(r["video"] for r in yolo),
                    key=lambda s: (len(s), s))
    hdr = f"{'video':<8}{'track':<11}" + "".join(f"{m:<12}" for m in yolo_models)
    print(hdr)
    for tr in tracks:
        for v in videos:
            cells = []
            any_row = False
            for m in yolo_models:
                match = [r for r in yolo if r["video"] == v and r["kind"] == tr and r["model"] == m]
                if not match:
                    cells.append(f"{'-':<12}")
                    continue
                any_row = True
                mark = "ANOM" if match[0]["status"] in HIT else "norm"
                cells.append(f"{mark:<12}")
            if any_row:
                print(f"{v:<8}{tr:<11}" + "".join(cells))

    # Side-by-side with the LLM experiment, if it exists
    if llm:
        print("\n" + "=" * 74)
        print("SIDE-BY-SIDE - overall recall, YOLO vs LLM pipeline")
        print("(LLM numbers pooled over all repeats; YOLO-OFF frames only)")
        print("=" * 74)
        rows = []
        for m in yolo_models:
            sub = [r for r in yolo if r["model"] == m]
            hits = sum(1 for r in sub if r["status"] in HIT)
            rows.append((m, hits, len(sub)))
        for m in sorted(set(r["model"] for r in llm)):
            sub = [r for r in llm if r["model"] == m]
            hits = sum(1 for r in sub if r["status"] in HIT)
            rows.append((m, hits, len(sub)))
        print(f"{'Model':<14}{'n':<7}{'Hits':<7}{'Recall':<9}{'95% CI':<20}")
        for m, hits, n in rows:
            p, lo, hi = wilson_ci(hits, n)
            print(f"{m:<14}{n:<7}{hits:<7}{pct(p):<9}[{pct(lo)}, {pct(hi)}]")

    print("\nNote: no NORMAL clips in the dataset, so this is recall only - it "
          "cannot\nshow YOLO's false-positive rate (it flags `person` whenever one is visible).")


if __name__ == "__main__":
    main()
