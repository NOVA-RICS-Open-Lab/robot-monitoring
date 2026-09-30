"""
stage3_match.py — semantic adjudication layer ("stage 3").

For every run that ended ANOMALY, an LLM judge compares the model's own final
explanation against the TRUE anomaly text (ground_truth.py) and grades:

  MATCH     the model identified the same anomaly as the reference
  PARTIAL   related / overlapping, but incomplete or mixed with a wrong claim
  MISMATCH  a different or hallucinated anomaly (also: any claim on a NORMAL clip)

Runs that ended NORMAL/UNKNOWN are recorded as NO_DETECTION (not sent to the judge).

Reads the saved run texts only — the VLMs are NOT re-run. Identical explanations
are judged once and cached.

  python stage3_match.py                       # all VLM runs + false-alarm runs
  python stage3_match.py --models gpt4
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from ground_truth import GROUND_TRUTH

load_dotenv(Path(__file__).parent / ".env")
EXP = Path(__file__).parent / "eval_results_experiment"
HIT = {"ANOMALIA", "ANOMALY"}
OUT = EXP / "stage3.csv"
JUDGE_MODEL = "gpt-4o"

_client = OpenAI()
_cache = {}

JUDGE_SYS = (
    "You grade whether an anomaly report describes the SAME anomaly as a reference. "
    "Reply with exactly one word: MATCH, PARTIAL, or MISMATCH.\n\n"
    "MATCH  = the report names the same anomaly TYPE and AGENT as the reference "
    "(e.g. a human hand/arm; a specific foreign object like a ball/bottle/plant; or the "
    "same process fault like the gripper not holding a piece). Different wording, missing "
    "minor detail, or not mentioning exact frames is still MATCH.\n"
    "PARTIAL = the core is right but clearly incomplete or vague, OR the report gets the "
    "core right but ALSO makes a clearly wrong extra claim.\n"
    "MISMATCH = a different anomaly, or a hallucinated one not in the reference."
)


def judge(ref, report):
    key = (ref, report.strip()[:1500])
    if key in _cache:
        return _cache[key]
    try:
        r = _client.chat.completions.create(
            model=JUDGE_MODEL, temperature=0, max_tokens=4,
            messages=[{"role": "system", "content": JUDGE_SYS},
                      {"role": "user", "content":
                       f"REFERENCE (true anomaly): {ref}\n\nMODEL REPORT: {report.strip()[:1500]}\n\n"
                       "One word:"}],
        )
        out = r.choices[0].message.content.strip().upper()
        g = next((x for x in ("MATCH", "PARTIAL", "MISMATCH") if x in out), "MISMATCH")
    except Exception as e:
        g = f"JUDGE_ERR({e})"
    _cache[key] = g
    return g


def final_explanation(txt_path: Path):
    """The verdict-line explanation of the FINAL stage (stage 2 if it ran, else stage 1)."""
    if not txt_path.exists():
        return ""
    t = txt_path.read_text(encoding="utf-8", errors="replace")
    if "STAGE 2" in t and "STAGE 2  : not run" not in t and "STAGE 2 : not run" not in t:
        block = t.split("STAGE 2", 1)[1].split("Result:", 1)[0]
    elif "STAGE 1" in t:
        block = t.split("STAGE 1", 1)[1].split("STAGE 2", 1)[0]
    else:
        block = t.split("Resultado:", 1)[-1]
    for ln in reversed([l.strip() for l in block.splitlines() if l.strip()]):
        m = re.match(r"(ANOMALIA|ANOMALY|NORMAL)\s*:\s*(.*)$", ln, re.I)
        if m:
            return m.group(2).strip()
    return block.strip()[:400]


def ref_text(kind, video):
    g = GROUND_TRUTH.get((kind, video))
    if not g:
        return None
    r = g["desc"]
    if g.get("internal_cause"):
        r += f" — cause: {g['internal_cause']}"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="gpt4,qwen3_vl,llama_vision")
    ap.add_argument("--no-false-alarm", action="store_true")
    args = ap.parse_args()
    models = set(m.strip() for m in args.models.split(",") if m.strip())

    jobs = []  # (model, kind, video, repeat, final_status, txt_path, ref)
    sm = EXP / "summary.csv"
    if sm.exists():
        for r in csv.DictReader(open(sm, encoding="utf-8")):
            if r.get("yolo_requested") != "off" or r["model"] not in models:
                continue
            tp = EXP / r["kind"] / f"{r['video']}__{r['model']}__yolooff__run{int(r['repeat']):02d}.txt"
            jobs.append((r["model"], r["kind"], r["video"], r["repeat"], r["status"],
                         tp, ref_text(r["kind"], r["video"])))

    fa = EXP / "false_alarm.csv"
    if fa.exists() and not args.no_false_alarm:
        for r in csv.DictReader(open(fa, encoding="utf-8")):
            if r["model"] not in models:
                continue
            tp = EXP / "false_alarm" / f"{r['model']}_run{int(r['run']):02d}.txt"
            jobs.append((r["model"], "false_alarm", "teste_falso_alarm", r["run"], r["final"],
                         tp, "NORMAL OPERATION — no anomaly is present"))

    rows = []
    for i, (model, kind, video, rep, status, tp, ref) in enumerate(jobs, 1):
        if ref is None:
            grade = "NO_GT"
        elif status not in HIT:
            grade = "NO_DETECTION"
        elif kind == "false_alarm":
            grade = "MISMATCH"            # normal video: any fire is a false positive
        else:
            grade = judge(ref, final_explanation(tp))
        rows.append((model, kind, video, rep, status, grade))
        if i % 50 == 0:
            print(f"  judged {i}/{len(jobs)}  (cache {len(_cache)})")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind", "video", "repeat", "final_status", "stage3"])
        w.writerows(rows)

    # ---- report ----
    def block(kinds, title):
        print("\n" + "=" * 88)
        print(title)
        print("=" * 88)
        print(f"{'model':<14}{'runs':<6}{'fired':<7}{'MATCH':<8}{'PARTIAL':<9}"
              f"{'MISMATCH':<10}{'M+P/fired':<11}{'MATCH/runs':<11}")
        for model in sorted(models):
            for k in kinds:
                sub = [r for r in rows if r[0] == model and r[1] == k]
                if not sub:
                    continue
                n = len(sub)
                fired = [r for r in sub if r[5] in ("MATCH", "PARTIAL", "MISMATCH")]
                mm = sum(1 for r in fired if r[5] == "MATCH")
                pp = sum(1 for r in fired if r[5] == "PARTIAL")
                xx = sum(1 for r in fired if r[5] == "MISMATCH")
                mp = f"{(mm+pp)/len(fired)*100:.0f}%" if fired else "-"
                print(f"{model:<14}{n:<6}{len(fired):<7}{mm:<8}{pp:<9}{xx:<10}{mp:<11}"
                      f"{f'{mm/n*100:.0f}%':<11}")

    block(["external"], "STAGE 3 — ANOMALIAS COMUM (external): does the report match the true anomaly?")
    block(["internal"], "STAGE 3 — ANOMALIAS OPERACIONAIS (internal)")
    block(["false_alarm"], "STAGE 3 — FALSE ALARM (reference = NORMAL, so any fire = MISMATCH)")

    # per-clip MATCH rate, external + internal
    print("\n" + "=" * 88)
    print("per-clip  MATCH / fired  (blank = model never fired here)")
    print("=" * 88)
    clips = sorted({(r[1], r[2]) for r in rows if r[1] in ("external", "internal")},
                   key=lambda x: (x[0], len(x[1]), x[1]))
    print(f"{'clip':<16}" + "".join(f"{m:<16}" for m in sorted(models)))
    for (k, v) in clips:
        line = f"{k[:3]} {v:<12}"
        for m in sorted(models):
            sub = [r for r in rows if r[0] == m and r[1] == k and r[2] == v]
            fired = [r for r in sub if r[5] in ("MATCH", "PARTIAL", "MISMATCH")]
            mm = sum(1 for r in fired if r[5] == "MATCH")
            line += f"{(f'{mm}/{len(fired)}' if fired else '.'):<16}"
        print(line)
    print(f"\nCSV: {OUT}")


if __name__ == "__main__":
    main()
