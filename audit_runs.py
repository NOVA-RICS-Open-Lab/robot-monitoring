"""
audit_runs.py — Cold read of every YOLO-OFF run: does the verdict hold up?

All 450 YOLO-OFF runs returned ANOMALY (100% recall). This pass parses each
run's saved text and asks whether that ANOMALY is *supported by the run's own
reasoning*, so "right label / wrong reason" and "degenerate output" get
separated from genuine detections.

Per run it extracts:
  - verdict line + explanation
  - whether the explanation is degenerate (too short / just numbers / a bbox)
  - EXTERNAL: does the explanation name a human body part? a foreign object?
              which frame number(s) does it blame?
  - INTERNAL: does the explanation cite the log event (gripper / idle / last_error /
              pressure)?  or does it lean only on a visible hand?
  - per-frame description / caption lines that mention a hand

Ground truth: every clip is an anomaly; internal clips all have
last_error=gripper + an unexpected idle mid-PickAndPlace (see the .txt logs in
videos/ANOMALIAS_OPERACIONAIS/). External clips all have a human intrusion.

Output: per (model, track) aggregates + a list of the runs that need eyeballing.
Stdlib only.
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "eval_results_experiment"
HUMAN_RE = re.compile(r"m[aã]os?\b|m[aã]o\b|bra[cç]o|dedos?\b|pessoa|hum[ao]|corpo|operador", re.I)
OBJ_RE = re.compile(r"extintor|garrafa|telem[oó]vel|caneta|chave inglesa|ferramenta|"
                    r"objeto estranho|objeto externo|objeto que n[aã]o pertence|copo|"
                    r"\bbola\b|bal[aã]o|bolinha|esfera|\bball\b", re.I)
LOG_RE = re.compile(r"gripper|garra|last_error|idle|press[aã]o|no log\b|do log\b|estado mudou|"
                    r"\berro\b.*gripper|last_result", re.I)
FRAME_RE = re.compile(r"frames?\s*(\d+)", re.I)
VERDICT_RE = re.compile(r"^(ANOMALIA|ANOMALY|NORMAL)\s*:?\s*(.*)$", re.I)
DEGENERATE_RE = re.compile(r"^[\s\[\]\(\)0-9.,;:%-]*$")


def parse(path: Path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    stem = path.stem  # e.g. an1__gpt4__yolooff__run01
    parts = stem.split("__")
    video, model = parts[0], parts[1]
    kind = "internal" if "(interna)" in txt or "(internal)" in txt or "internal" in str(path.parent) else "external"

    # Two-stage format: use the STAGE 2 block if present (that is the final
    # verdict), else STAGE 1, else the legacy single "Resultado:" block.
    if "STAGE 2" in txt and "STAGE 2  : not run" not in txt:
        body = txt.split("STAGE 2", 1)[1]
    elif "STAGE 1" in txt:
        body = txt.split("STAGE 1", 1)[1].split("STAGE 2", 1)[0]
    elif "Resultado:" in txt:
        body = txt.split("Resultado:", 1)[1]
    else:
        body = txt

    # numbered per-frame description lines (gpt4 style)
    desc = {int(m.group(1)): m.group(2).strip()
            for m in re.finditer(r"^\s*(\d{1,2})\.\s+(.*)$", body, re.M)}
    # intermediate captions (local models)
    caps = {int(m.group(1)): m.group(2).strip()
            for m in re.finditer(r"^\s*Frame\s+(\d{1,2}):\s+(.*)$", txt, re.M)}

    # last verdict line in the body
    verdict, expl = None, ""
    for line in reversed([l.strip() for l in body.splitlines() if l.strip()]):
        m = VERDICT_RE.match(line)
        if m:
            verdict = "ANOMALY" if m.group(1).upper().startswith("ANOMAL") else "NORMAL"
            expl = m.group(2).strip()
            break

    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", expl)
    degenerate = len(words) < 5 or bool(DEGENERATE_RE.match(expl))

    return {
        "path": path, "video": video, "model": model, "kind": kind,
        "verdict": verdict, "expl": expl, "degenerate": degenerate,
        "n_words": len(words),
        "human": bool(HUMAN_RE.search(expl)),
        "obj": bool(OBJ_RE.search(expl)),
        "log": bool(LOG_RE.search(expl)),
        "blamed_frames": sorted(set(int(x) for x in FRAME_RE.findall(expl) if int(x) <= 20)),
        "desc_hand_frames": sorted(f for f, t in desc.items() if HUMAN_RE.search(t)),
        "cap_hand_frames": sorted(f for f, t in caps.items()
                                  if re.search(r"\bhand\b|finger|m[aã]o", t, re.I)),
        "has_desc": bool(desc), "has_caps": bool(caps),
    }


def main():
    files = sorted(RUNS_DIR.glob("*/*__yolooff__*.txt"))
    if not files:
        sys.exit("no YOLO-OFF run files under eval_results_experiment/")
    runs = [parse(p) for p in files]
    print(f"Parsed {len(runs)} YOLO-OFF run texts\n")

    by = defaultdict(list)
    for r in runs:
        by[(r["model"], r["kind"])].append(r)

    for (model, kind) in sorted(by):
        rs = by[(model, kind)]
        n = len(rs)
        v = Counter(r["verdict"] for r in rs)
        deg = sum(r["degenerate"] for r in rs)
        print("=" * 78)
        print(f"{model}  /  {kind}   (n={n})")
        print("=" * 78)
        print(f"  verdicts            : {dict(v)}")
        print(f"  degenerate explanation : {deg}/{n}  ({deg/n*100:.0f}%)   "
              f"[<5 real words, or just numbers/bbox]")
        print(f"  median explanation words: {sorted(r['n_words'] for r in rs)[n//2]}")
        if kind == "external":
            hum = sum(r["human"] for r in rs)
            obj = sum(r["obj"] and not r["human"] for r in rs)
            neither = sum(not r["human"] and not r["obj"] for r in rs)
            print(f"  explanation names a human : {hum}/{n}  ({hum/n*100:.0f}%)")
            print(f"  names only a foreign obj  : {obj}/{n}")
            print(f"  names NEITHER (unsupported): {neither}/{n}")
            bf = Counter(f for r in rs for f in r["blamed_frames"])
            print(f"  frame(s) blamed (all runs): {dict(sorted(bf.items()))}")
        else:
            lg = sum(r["log"] for r in rs)
            honly = sum(r["human"] and not r["log"] for r in rs)
            neither = sum(not r["human"] and not r["log"] and not r["degenerate"] for r in rs)
            print(f"  explanation cites the log event : {lg}/{n}  ({lg/n*100:.0f}%)")
            print(f"  leans ONLY on a visible hand    : {honly}/{n}")
            print(f"  neither log nor hand (vague)    : {neither}/{n}")
        print()

    # runs that need a human look
    print("#" * 78)
    print("RUNS TO EYEBALL")
    print("#" * 78)
    susp = []
    for r in runs:
        why = []
        if r["degenerate"]:
            why.append("degenerate-explanation")
        if r["kind"] == "external" and not r["human"] and not r["obj"]:
            why.append("external-verdict-not-grounded")
        if r["kind"] == "internal" and not r["log"] and not r["degenerate"]:
            why.append("internal-verdict-ignores-log")
        if r["kind"] == "external" and r["blamed_frames"] and r["desc_hand_frames"] \
           and not set(r["blamed_frames"]) & set(r["desc_hand_frames"]):
            why.append(f"blames f{r['blamed_frames']} but desc has hand in f{r['desc_hand_frames']}")
        if why:
            susp.append((r, why))
    print(f"{len(susp)}/{len(runs)} runs flagged\n")
    flagcount = Counter(w.split(" ")[0] for _, ws in susp for w in ws)
    for k, c in flagcount.most_common():
        print(f"  {c:>4}  {k}")
    print()
    # show a compact sample grouped by model/kind
    shown = defaultdict(int)
    for r, why in susp:
        key = (r["model"], r["kind"], why[0])
        if shown[key] >= 3:
            continue
        shown[key] += 1
        print(f"  [{r['model']}/{r['kind']}] {r['path'].name}")
        print(f"      -> {'; '.join(why)}")
        print(f"      expl: {r['expl'][:160]}")


if __name__ == "__main__":
    main()
