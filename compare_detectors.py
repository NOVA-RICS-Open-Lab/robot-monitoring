"""
compare_detectors.py - Put the stock-YOLO baseline next to the LLM pipeline.

Reads:
  eval_results_experiment/summary.csv       (run_experiment.py: gpt4/llama/moondream,
                                             15 videos x 10 repeats, YOLO-OFF rows only)
  eval_results_experiment/yolo_summary.csv  (yolo_baseline.py: yolo_coco/yolo_pose,
                                             1 deterministic verdict per video)

The YOLO-crop-ON condition was dropped from the study, so only yolo_requested
== "off" rows of summary.csv are used here.

Every clip is an anomaly, so the metric is recall. LLM cells show the hit rate
over the 10 runs per (video, model); a video counts as an LLM "hit" when the
majority of those runs said anomaly. YOLO has one verdict per video.

Caveats printed at the end - read them before quoting the numbers.
Stdlib only.
"""

import csv
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent / "eval_results_experiment"
HIT = {"ANOMALIA", "ANOMALY"}
LLM_MODELS = ["gpt4", "qwen3_vl", "llama_vision", "moondream"]
YOLO_MODELS = ["yolo_coco", "yolo_pose"]


def load_llm():
    runs = defaultdict(list)   # (kind, video, model) -> [status, ...]
    lat = defaultdict(list)    # model -> [seconds]
    path = BASE / "summary.csv"
    if not path.exists():
        return runs, lat
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ERROR", "status"):
                continue
            if r.get("yolo_requested") != "off":   # YOLO-crop-ON dropped from the study
                continue
            runs[(r["kind"], r["video"], r["model"])].append(r["status"])
            lat[r["model"]].append(float(r["elapsed_s"]))
    return runs, lat


def load_yolo():
    verdict = {}               # (kind, video, model) -> status
    lat = defaultdict(list)
    path = BASE / "yolo_summary.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} - run: python yolo_baseline.py")
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            verdict[(r["kind"], r["video"], r["model"])] = r["status"]
            lat[r["model"]].append(float(r["elapsed_s"]))
    return verdict, lat


def videos(runs, kind):
    return sorted({v for (k, v, _m) in runs if k == kind}, key=lambda s: (len(s), s))


def main():
    llm, llm_lat = load_llm()
    yolo, yolo_lat = load_yolo()
    have_llm = bool(llm)

    for kind in ("external", "internal"):
        print("\n" + "=" * 92)
        print(f"{kind.upper()} TRACK - per-video verdicts")
        print("=" * 92)
        cols = (LLM_MODELS if have_llm else []) + YOLO_MODELS
        print(f"{'video':<7}" + "".join(f"{c:<15}" for c in cols))
        vs = videos(llm or yolo, kind)
        for v in vs:
            row = f"{v:<7}"
            for m in LLM_MODELS if have_llm else []:
                s = llm.get((kind, v, m), [])
                if not s:
                    row += f"{'-':<15}"
                    continue
                h = sum(1 for x in s if x in HIT)
                mark = "ANOM" if h * 2 > len(s) else ("tie " if h * 2 == len(s) else "norm")
                row += f"{mark} {h:>2}/{len(s):<2}     "
            for m in YOLO_MODELS:
                st = yolo.get((kind, v, m), "?")
                row += f"{'ANOM' if st in HIT else 'norm':<15}"
            print(row)

    # recall summary
    print("\n" + "=" * 92)
    print("RECALL  (ground truth = anomaly for every clip)")
    print("=" * 92)
    for kind in ("external", "internal", "ALL"):
        print(f"\n{kind}:")
        if have_llm:
            for m in LLM_MODELS:
                tot = hit = 0
                for (k, _v, mm), s in llm.items():
                    if mm != m or (kind != "ALL" and k != kind):
                        continue
                    tot += len(s)
                    hit += sum(1 for x in s if x in HIT)
                if tot:
                    print(f"  {m:<14}{hit / tot * 100:5.1f}%   ({hit}/{tot} runs)")
        for m in YOLO_MODELS:
            tot = hit = 0
            for (k, _v, mm), st in yolo.items():
                if mm != m or (kind != "ALL" and k != kind):
                    continue
                tot += 1
                hit += 1 if st in HIT else 0
            if tot:
                print(f"  {m:<14}{hit / tot * 100:5.1f}%   ({hit}/{tot} videos)")

    # latency
    print("\n" + "=" * 92)
    print("LATENCY  (seconds per video analysis)")
    print("=" * 92)
    for m in (LLM_MODELS if have_llm else []) + YOLO_MODELS:
        src = llm_lat.get(m) or yolo_lat.get(m) or []
        if not src:
            continue
        s = sorted(src)
        print(f"  {m:<14}mean {sum(s) / len(s):6.2f}   median {s[len(s) // 2]:6.2f}   "
              f"min {s[0]:.2f}   max {s[-1]:.2f}")

    print("\n" + "-" * 92)
    print("CAVEATS")
    print("-" * 92)
    print("""\
1. No NORMAL clips -> recall only. YOLO keys on `person`, so on clean footage it
   would fire constantly; this dataset cannot measure that false-positive rate.
2. INTERNAL track is not like-for-like: the LLMs read the robot state log (which
   contains the real gripper error), YOLO sees only pixels. yolo_pose's 100% is
   spurious -- it detects the operator's hand that happens to be in frame, not
   the process fault.
3. Input framing differs: the LLM off-runs use the blue-colour ROI crop, the
   YOLO baseline runs on the full uncropped frame (both on the same 9 timestamps).
4. n = 10 external / 5 internal videos for YOLO -> very wide confidence intervals
   (see analyze_yolo.py for the Wilson intervals).
5. LLM 'tie'/'50%' rows mean the model flipped its verdict across the 10 repeats;
   YOLO is deterministic (same answer every run).""")


if __name__ == "__main__":
    main()
