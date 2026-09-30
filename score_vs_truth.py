"""
score_vs_truth.py — grade every detector against ground_truth.py.

Three criteria per (track, video, model):
  detected      : final verdict == ANOMALY
  reason_ok     : the stated basis matches the TRUE anomaly
                    external + human GT  -> mentions hand/arm/person
                    external + object GT -> names the object / "foreign object"
                    internal (non-mixed) -> cites the control-log event
                                            (YOLO can't -> always fails here)
                    mixed                -> either side counts
  localisation_ok : the reported anomaly window (two-stage) overlaps the GT
                    window (as a fraction of clip length); only scored when a
                    window is present.

YOLO: one deterministic row per clip (eval_results_experiment/yolo_summary.csv +
      eval_results/<track>/<stem>__<variant>.txt).
LLM : aggregated over the repeats in eval_results_experiment/summary.csv +
      eval_results_experiment/<track>/<stem>__<model>__yolooff__runNN.txt;
      detected = majority, reason/localisation = mean over repeats.

Reason matching is keyword-based and intentionally lenient — it separates
"right for the right reason" from "right label, wrong/absent reason", not
fine-grained correctness.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import frame_sampling as fs
from ground_truth import GROUND_TRUTH

BASE = Path(__file__).parent
EXP = BASE / "eval_results_experiment"
HIT = {"ANOMALIA", "ANOMALY"}

HUMAN_RE = re.compile(r"m[aã]os?\b|m[aã]o\b|bra[cç]o|dedos?\b|pessoa|hum[ao]|corpo|operador|"
                      r"\bhand\b|\bwrist\b|\barm\b|person|\bfinger", re.I)
OBJ_GENERIC_RE = re.compile(r"objeto (estranho|externo|amarelo|branco|que n[aã]o pertence)|"
                            r"foreign object|n[aã]o faz parte|does not belong", re.I)
LOG_RE = re.compile(r"gripper|garra|last_error|\bidle\b|press[aã]o|no log\b|do log\b|"
                    r"estado mudou|last_result", re.I)
# visual cues that the pick/place cycle failed, usable when there is NO log
GRASP_RE = re.compile(r"n[aã]o (est[aá]|parece|conseguiu|consegue) (a )?(segurar|agarrar|pegar|prender)|"
                      r"sem (segurar|agarrar|nada nas? garras?)|garra vazia|n[aã]o h[aá] pe[cç]a|"
                      r"pe[cç]a (n[aã]o|inexistente|ausente)|sem pe[cç]a|falha (no|ao) (agarr|pegar|grip)|"
                      r"not (holding|grasping|gripping)|empty gripper|no (piece|part|object) (held|present)|"
                      r"fails? to (grasp|pick|grip)|phantom", re.I)


def _clip_seconds(track, video):
    folder = "ANOMALIAS" if track == "external" else "ANOMALIAS_OPERACIONAIS"
    total, fps = fs.video_meta(BASE / "videos" / folder / f"{video}.mp4")
    return total / fps if fps else 0.0


def _obj_terms(gt):
    terms = []
    for o, c in zip(gt["objects"], gt["coco_class"]):
        terms += [w for w in re.split(r"\s+", o) if len(w) > 2]
        if c:
            terms += [w for w in re.split(r"\s+", c) if len(w) > 2]
    # PT synonyms for the recurring ones
    extra = {"ball": "bola", "bola": "ball", "bottle": "garrafa", "garrafa": "bottle",
             "plant": "planta", "planta": "plant", "wallet": "carteira", "carteira": "wallet"}
    terms += [extra[t.lower()] for t in terms if t.lower() in extra]
    return [t.lower() for t in terms if t]


def reason_ok(track, video, text):
    gt = GROUND_TRUTH[(track, video)]
    t = (text or "").lower()
    human_hit = bool(HUMAN_RE.search(t))
    obj_hit = bool(OBJ_GENERIC_RE.search(t)) or any(term in t for term in _obj_terms(gt))
    log_hit = bool(LOG_RE.search(t))

    if track == "internal":
        grasp_hit = bool(GRASP_RE.search(t))
        if gt["mixed"]:
            return log_hit or human_hit or grasp_hit
        if gt.get("has_log", True):
            return log_hit
        return grasp_hit          # no-log clip: a visual "not holding a piece" observation
    # external
    if gt["mixed"]:
        return human_hit
    if gt["human"] and gt["objects"]:
        return human_hit or obj_hit
    if gt["human"]:
        return human_hit
    return obj_hit                       # object-only clip (an5, an6, an9)


def localisation_ok(track, video, start_s, end_s, tol=0.08):
    if start_s == "" or end_s == "":
        return None
    dur = _clip_seconds(track, video)
    if dur <= 0:
        return None
    lo, hi = float(start_s) / dur, float(end_s) / dur
    glo, ghi = GROUND_TRUTH[(track, video)]["window"]
    return not (hi < glo - tol or lo > ghi + tol)   # intervals overlap (with slack)


def _run_text(track, video, model, rep=None):
    if model in ("yolo_coco", "yolo_pose"):
        p = BASE / "eval_results" / track / f"{video}__{model}.txt"
    else:
        p = EXP / track / f"{video}__{model}__yolooff__run{rep:02d}.txt"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def load_yolo():
    rows = {}
    p = EXP / "yolo_summary.csv"
    if not p.exists():
        return rows
    for r in csv.DictReader(open(p, encoding="utf-8")):
        rows[(r["kind"], r["video"], r["model"])] = r
    return rows


def load_llm():
    runs = defaultdict(list)
    p = EXP / "summary.csv"
    if not p.exists():
        return runs
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("yolo_requested") not in (None, "off"):
            continue
        if r.get("status") == "ERROR":
            continue
        runs[(r["kind"], r["video"], r["model"])].append(r)
    return runs


def grade():
    yolo = load_yolo()
    llm = load_llm()
    out = []   # (track, video, model, detected, reason, loc)

    for (track, video, model), r in sorted(yolo.items()):
        det = r["status"] in HIT
        txt = _run_text(track, video, model) or r.get("basis", "")
        rea = det and reason_ok(track, video, txt)
        loc = localisation_ok(track, video, r.get("anomaly_start_s", ""), r.get("anomaly_end_s", "")) if det else None
        out.append((track, video, model, det, rea, loc))

    for (track, video, model), rs in sorted(llm.items()):
        n = len(rs)
        det_frac = sum(1 for r in rs if r["status"] in HIT) / n
        det = det_frac > 0.5
        reas, locs = [], []
        for r in rs:
            det_run = r["status"] in HIT
            txt = _run_text(track, video, model, int(r["repeat"])) or r.get("verdict_line", "")
            reas.append(det_run and reason_ok(track, video, txt))
            if not det_run:
                continue
            lo = localisation_ok(track, video, r.get("anomaly_start_s", ""), r.get("anomaly_end_s", ""))
            if lo is not None:
                locs.append(lo)
        out.append((track, video, model, det,
                    sum(reas) / n, (sum(locs) / len(locs)) if locs else None))
    return out


def main():
    rows = grade()
    if not rows:
        raise SystemExit("no results found — run yolo_baseline.py / run_experiment.py first")
    models = sorted({r[2] for r in rows})

    print("=" * 90)
    print("SCORE vs GROUND TRUTH")
    print("=" * 90)
    print(f"{'model':<14}{'track':<10}{'detect':<18}{'right reason':<18}{'localised':<14}")
    for model in models:
        for track in ("external", "internal", "ALL"):
            sub = [r for r in rows if r[2] == model and (track == "ALL" or r[0] == track)]
            if not sub:
                continue
            n = len(sub)
            det = sum(1 for r in sub if r[3])
            rea = sum((r[4] if isinstance(r[4], float) else (1.0 if r[4] else 0.0)) for r in sub)
            locv = [r[5] for r in sub if isinstance(r[5], (int, float))]
            loc = f"{sum(locv)/len(locv)*100:.0f}% (n={len(locv)})" if locv else "-"
            print(f"{model:<14}{track:<10}{f'{det}/{n} ({det/n*100:.0f}%)':<18}"
                  f"{f'{rea:.1f}/{n} ({rea/n*100:.0f}%)':<18}{loc:<14}")
    print()

    # per-clip grid
    print("=" * 90)
    print("PER-CLIP  (D=detected  R=right reason  L=localised;  '.' = no / n/a)")
    print("=" * 90)
    clips = sorted({(r[0], r[1]) for r in rows}, key=lambda x: (x[0], len(x[1]), x[1]))
    print(f"{'clip':<16}" + "".join(f"{m:<14}" for m in models))
    for (track, video) in clips:
        cells = []
        for m in models:
            hit = [r for r in rows if r[0] == track and r[1] == video and r[2] == m]
            if not hit:
                cells.append(f"{'-':<14}")
                continue
            _, _, _, d, rea, loc = hit[0]
            rflag = "R" if (rea is True or (isinstance(rea, float) and rea >= 0.5)) else "."
            lflag = "L" if loc is True or (isinstance(loc, float) and loc >= 0.5) else ("." if loc is None else "x")
            cells.append(f"{('D' if d else '.')+rflag+lflag:<14}")
        gt = GROUND_TRUTH[(track, video)]
        print(f"{track[:3]+' '+video:<16}" + "".join(cells) + f"  <- {gt['desc']}")


if __name__ == "__main__":
    main()
