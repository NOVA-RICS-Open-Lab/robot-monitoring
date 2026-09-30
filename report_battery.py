"""report_battery.py — assemble the three test tables from the battery outputs."""
import csv
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).parent / "eval_results_experiment"
HIT = {"ANOMALIA", "ANOMALY"}
VLM = ["gpt4", "qwen3_vl", "llama_vision"]


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def vids(rows, kind):
    return sorted({r["video"] for r in rows if r["kind"] == kind}, key=lambda s: (len(s), s))


def pct(a, b):
    return f"{a/b*100:.0f}%" if b else "-"


def cell(rows, kind, video, model, field):
    sub = [r for r in rows if r["kind"] == kind and r["video"] == video and r["model"] == model]
    if not sub:
        return "  -  "
    n = len(sub)
    h = sum(1 for r in sub if r.get(field, "") in HIT)
    return f"{h:2d}/{n:<2d}"


SHORT = {"gpt4": "GPT", "qwen3_vl": "QWEN", "llama_vision": "LLAMA"}


def track_table(rows, yolo, kind, title, clip_note=None):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
    hdr = f"{'video':<6}"
    for m in VLM:
        s = SHORT[m]
        hdr += f"{s+'.s1':>8}{s+'.fin':>8}"
    if yolo:
        hdr += f"{'coco':>7}{'pose':>7}"
    hdr += "   description"
    print(hdr)
    for v in vids(rows, kind):
        line = f"{v:<6}"
        for m in VLM:
            line += f"{cell(rows, kind, v, m, 's1_status'):>8}{cell(rows, kind, v, m, 'status'):>8}"
        if yolo:
            for ym in ("yolo_coco", "yolo_pose"):
                yr = [r for r in yolo if r["kind"] == kind and r["video"] == v and r["model"] == ym]
                line += f"{(yr[0]['status'][:4] if yr else '-'):>7}"
        note = "   " + clip_note.get((kind, v), "") if clip_note else ""
        print(line + note)

    print("-" * 100)
    print(f"{'RATE':<7}" + "".join(
        f"{pct(sum(1 for r in rows if r['kind']==kind and r['model']==m and r['s1_status'] in HIT), sum(1 for r in rows if r['kind']==kind and r['model']==m)):<9}"
        f"{pct(sum(1 for r in rows if r['kind']==kind and r['model']==m and r['status'] in HIT), sum(1 for r in rows if r['kind']==kind and r['model']==m)):<9}"
        for m in VLM))

    # two-stage effect
    print("\ntwo-stage effect (of the stage-1 ANOMALY runs, how many stage 2 changed):")
    for m in VLM:
        sub = [r for r in rows if r["kind"] == kind and r["model"] == m and r.get("s2_status")]
        s1a = [r for r in sub if r["s1_status"] in HIT]
        over = sum(1 for r in s1a if (r["s1_status"] in HIT) != (r["status"] in HIT))
        unk = sum(1 for r in s1a if r["s2_status"] not in HIT and r["s2_status"] != "NORMAL")
        print(f"  {m:<13} s1=ANOM & s2 ran: {len(s1a):3d}   overturned->NORMAL: {over:3d}   s2 unusable(kept s1): {unk:3d}")


def main():
    rows = [r for r in load(EXP / "summary.csv") if r.get("yolo_requested") == "off"]
    yolo = load(EXP / "yolo_summary.csv")
    fa = load(EXP / "false_alarm.csv")

    clip_desc = {}
    try:
        from ground_truth import GROUND_TRUTH
        clip_desc = {(t, v): g["desc"] for (t, v), g in GROUND_TRUTH.items()}
    except Exception:
        pass

    track_table(rows, yolo, "external", "TEST 1 - ANOMALIAS COMUM (external)  |  S1 = stage-1 hits/10 , FIN = final hits/10",
                clip_desc)
    track_table(rows, None, "internal",
                "TEST 2 - ANOMALIAS OPERACIONAIS (internal)  |  an1-an5 have a log, an6-an10 do NOT",
                clip_desc)

    print("\n" + "=" * 100)
    print("TEST 3 - TESTE_FALSO_ALARM.mp4 (normal video: every ANOMALY = false alarm)")
    print("=" * 100)
    print(f"{'model':<14}{'run':<5}{'frames':<8}{'stage1':<10}{'stage2':<10}{'final':<10}{'idx (first..last)':<22}")
    for r in fa:
        idx = r["s1_frame_idx"].split()
        print(f"{r['model']:<14}{r['run']:<5}{r['s1_frames']:<8}{r['s1_status']:<10}"
              f"{r['s2_status'] or '-':<10}{r['final']:<10}{idx[0]+'..'+idx[-1]:<22}")
    print("-" * 100)
    print(f"{'model':<14}{'runs':<6}{'frames min/mean/max':<22}{'S1 false alarms':<18}{'FINAL false alarms':<18}")
    for m in VLM:
        sub = [r for r in fa if r["model"] == m]
        if not sub:
            continue
        nf = [int(r["s1_frames"]) for r in sub]
        s1a = sum(1 for r in sub if r["s1_status"] in HIT)
        fina = sum(1 for r in sub if r["final"] in HIT)
        unk = sum(1 for r in sub if r["s1_status"] not in HIT and r["s1_status"] != "NORMAL")
        print(f"{m:<14}{len(sub):<6}{f'{min(nf)}/{sum(nf)/len(nf):.1f}/{max(nf)}':<22}"
              f"{f'{s1a}/{len(sub)}':<18}{f'{fina}/{len(sub)}':<18}" + (f"  (+{unk} UNKNOWN)" if unk else ""))


if __name__ == "__main__":
    main()
