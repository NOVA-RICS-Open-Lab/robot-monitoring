"""
analyze_results.py — Analysis for the LLM anomaly-detection experiment.

The YOLO-crop-ON condition was dropped from the study: the LLM pipeline is now
evaluated only on the YOLO-OFF frames (9 uniform frames, blue-colour ROI
heuristic, no modelo/best.pt). This script therefore keeps ONLY the
yolo_requested == "off" rows of summary.csv and ignores the rest.

Remaining design: 15 videos x 3 models x 10 repeats = 450 runs.

Ground truth: every clip lives under ANOMALIAS / ANOMALIAS_OPERACIONAIS by
construction (internal logs independently confirm a real "last_error":
"gripper" event in all 5 internal clips), so the dataset has no negative
control — we report ANOMALY-detection rate (recall), not full accuracy, and
say so explicitly. Stdlib only: Wilson score interval for proportions, exact
McNemar test for paired model comparisons.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).parent / "eval_results_experiment" / "summary.csv"
Z95 = 1.959963985
# extract_verdict() now returns "ANOMALY"; older CSV rows say "ANOMALIA". Count both.
HIT = {"ANOMALIA", "ANOMALY"}


def wilson_ci(successes, n, z=Z95):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (center - adj) / denom), min(1.0, (center + adj) / denom))


def mcnemar_exact_p(b, c):
    """Exact two-sided McNemar test on the two off-diagonal discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    return min(p, 1.0)


def load_rows():
    with open(CSV_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_pct(x):
    return f"{x * 100:.1f}%"


def main():
    rows = load_rows()
    print(f"Total rows in CSV: {len(rows)}")

    errors = [r for r in rows if r["status"] == "ERROR"]
    if errors:
        print(f"WARNING: {len(errors)} ERROR rows excluded")
    rows = [r for r in rows if r["status"] != "ERROR"]

    kept = [r for r in rows if r.get("yolo_requested") == "off"]
    dropped = len(rows) - len(kept)
    if dropped:
        print(f"Dropped {dropped} YOLO-crop-ON rows (condition removed from the study); "
              f"{len(kept)} YOLO-OFF rows kept.")
    rows = kept

    models = sorted(set(r["model"] for r in rows))

    # ── 1. External track: detection rate per model ──────────────────────────
    print("\n" + "=" * 78)
    print("1) EXTERNAL TRACK — detection rate per model")
    print("   (ground truth = ANOMALY for all clips; 10 videos x 10 repeats = 100 runs)")
    print("=" * 78)
    print(f"{'Model':<14}{'n':<6}{'Hits':<6}{'Rate':<9}{'95% CI':<18}")
    for model in models:
        sub = [r for r in rows if r["kind"] == "external" and r["model"] == model]
        n = len(sub)
        hits = sum(1 for r in sub if r["status"] in HIT)
        p, lo, hi = wilson_ci(hits, n)
        print(f"{model:<14}{n:<6}{hits:<6}{fmt_pct(p):<9}[{fmt_pct(lo)}, {fmt_pct(hi)}]")

    # ── 2. Internal track: detection rate per model ─────────────────────────
    print("\n" + "=" * 78)
    print("2) INTERNAL TRACK — detection rate per model")
    print("   (5 videos x 10 repeats = 50 runs; model also gets the robot state log)")
    print("=" * 78)
    print(f"{'Model':<14}{'n':<6}{'Hits':<6}{'Rate':<9}{'95% CI':<18}")
    for model in models:
        sub = [r for r in rows if r["kind"] == "internal" and r["model"] == model]
        n = len(sub)
        hits = sum(1 for r in sub if r["status"] in HIT)
        p, lo, hi = wilson_ci(hits, n)
        print(f"{model:<14}{n:<6}{hits:<6}{fmt_pct(p):<9}[{fmt_pct(lo)}, {fmt_pct(hi)}]")

    # ── 3. Overall per model (external + internal) ──────────────────────────
    print("\n" + "=" * 78)
    print("3) OVERALL detection rate per model (all 150 runs: external + internal)")
    print("=" * 78)
    for model in models:
        sub = [r for r in rows if r["model"] == model]
        n = len(sub)
        hits = sum(1 for r in sub if r["status"] in HIT)
        p, lo, hi = wilson_ci(hits, n)
        print(f"{model:<14}n={n:<6}hits={hits:<6}rate={fmt_pct(p):<9}CI=[{fmt_pct(lo)}, {fmt_pct(hi)}]")

    # ── 4. Inter-run agreement (consistency across the 10 repeats) ──────────
    print("\n" + "=" * 78)
    print("4) INTER-RUN AGREEMENT — mean % of repeats matching the majority verdict")
    print("   (per video x model x kind group of 10 repeats)")
    print("=" * 78)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["video"], r["model"])].append(r["status"])
    agreement_by_model = defaultdict(list)
    for (kind, video, model), statuses in groups.items():
        counts = defaultdict(int)
        for s in statuses:
            counts[s] += 1
        agreement_by_model[model].append(max(counts.values()) / len(statuses))
    print(f"{'Model':<14}{'Mean agreement':<18}{'Min':<8}{'Max':<8}")
    for model in models:
        vals = agreement_by_model[model]
        print(f"{model:<14}{fmt_pct(sum(vals)/len(vals)):<18}{fmt_pct(min(vals)):<8}{fmt_pct(max(vals)):<8}")

    # ── 5. McNemar: pairwise model comparison (paired by kind+video+repeat) ─
    print("\n" + "=" * 78)
    print("5) McNEMAR TEST — pairwise model comparison (150 paired runs per pair)")
    print("=" * 78)
    by_key = {}
    for r in rows:
        by_key.setdefault((r["kind"], r["video"], r["repeat"]), {})[r["model"]] = r["status"]
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            b = c = 0
            for d in by_key.values():
                if m1 not in d or m2 not in d:
                    continue
                h1, h2 = d[m1] in HIT, d[m2] in HIT
                if h1 and not h2:
                    b += 1
                elif h2 and not h1:
                    c += 1
            p_val = mcnemar_exact_p(b, c)
            print(f"{m1} vs {m2:<14}{m1}-only-hit={b:<5}{m2}-only-hit={c:<5}p={p_val:.4f}"
                  f"{'  (significant, p<0.05)' if p_val < 0.05 else ''}")

    # ── 6. Latency ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("6) LATENCY per model (seconds per run, stage 1 + stage 2)")
    print("=" * 78)
    for model in models:
        vals = sorted(float(r["elapsed_s"]) for r in rows if r["model"] == model)
        mean = sum(vals) / len(vals)
        med = vals[len(vals) // 2]
        print(f"{model:<14}mean={mean:.1f}s  median={med:.1f}s  min={vals[0]:.1f}s  max={vals[-1]:.1f}s")

    # ── 7. Two-stage effect (only if the CSV carries the columns) ──────────
    ts = [r for r in rows if r.get("two_stage") == "1" and r.get("s1_status")]
    if ts:
        print("\n" + "=" * 78)
        print("7) TWO-STAGE EFFECT — stage-1 verdict vs final (after dense re-check)")
        print("=" * 78)
        print(f"{'Model':<14}{'runs':<7}{'s2 ran':<8}{'confirmed':<11}{'overturned':<12}{'s2 error':<9}")
        for model in models:
            sub = [r for r in ts if r["model"] == model]
            ran = [r for r in sub if r["s2_status"]]
            over = sum(1 for r in ran if (r["s1_status"] in HIT) != (r["status"] in HIT))
            err = sum(1 for r in ran if r["s2_status"] == "ERROR")
            conf = len(ran) - over - err
            print(f"{model:<14}{len(sub):<7}{len(ran):<8}{conf:<11}{over:<12}{err:<9}")


if __name__ == "__main__":
    main()
