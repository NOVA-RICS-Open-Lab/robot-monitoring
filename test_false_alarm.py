"""
test_false_alarm.py — continuous-monitoring false-alarm test on a NORMAL video.

videos/teste_falso_alarm.mp4 is ~10 min of normal robot operation, no anomaly.
This simulates the live monitor: a window slides across the whole recording, one
Stage-1 analysis per window (9 frames sampled inside it), Stage-2 dense re-check
on any fire. Every ANOMALY verdict is a false positive. We report the
false-alarm rate per analysis and extrapolated per hour of monitoring at this
cadence.

An anti-hallucination clause is prepended to the prompt (the earlier version of
this test showed the models inventing a human hand whenever the robot moved the
yellow blocks). Disable it with --no-anti-hallucination for an A/B.

  python test_false_alarm.py
  python test_false_alarm.py --window-seconds 15 --step-seconds 15 --models gpt4
  python test_false_alarm.py --no-anti-hallucination
"""

import argparse
import csv
import time
from pathlib import Path

import benchmark_models as bm
import vision_backends as vb
import frame_sampling as fs
from run_experiment import frames_b64_at, anchor_indices

BASE = Path(__file__).parent
VIDEO = BASE / "videos" / "teste_falso_alarm.mp4"
OUT_DIR = BASE / "eval_results_experiment"
TXT_DIR = OUT_DIR / "false_alarm_continuous"
CSV_PATH = OUT_DIR / "false_alarm_continuous.csv"
HIT = {"ANOMALIA", "ANOMALY"}
CSV_HEADER = ["model", "window", "start_s", "end_s", "n_frames",
              "s1_status", "s2_frames", "s2_status", "final", "elapsed_s"]

ANTI_HALLUC = (
    "ATENCAO: O proprio robo pega e larga os blocos amarelos como parte da sua "
    "operacao normal. Blocos a mudarem de posicao entre frames e operacao NORMAL "
    "do robo e NAO e prova de uma mao humana. So classifiques como ANOMALIA se "
    "conseguires mesmo VER, numa das imagens, uma mao, um dedo, um braco ou uma "
    "pessoa, ou um objeto que claramente nao pertence ao processo (garrafa, bola, "
    "planta, carteira, etc.). Na duvida, responde NORMAL."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="gpt4,qwen3_vl,llama_vision")
    ap.add_argument("--window-seconds", type=float, default=15.0)
    ap.add_argument("--step-seconds", type=float, default=15.0)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--dense-stride", type=int, default=4)
    ap.add_argument("--dense-halfcount", type=int, default=10)
    ap.add_argument("--dense-cap", type=int, default=12)
    ap.add_argument("--no-two-stage", dest="two_stage", action="store_false", default=True)
    ap.add_argument("--no-anti-hallucination", dest="anti", action="store_false", default=True)
    args = ap.parse_args()

    if not VIDEO.exists():
        raise SystemExit(f"missing {VIDEO}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in bm.MODELS:
            raise SystemExit(f"unknown model {m}")

    total, fps = fs.video_meta(VIDEO)
    dur = total / fps
    aas = bm.load_aas_context()
    extra = ANTI_HALLUC if args.anti else ""

    # one window per step across the whole recording
    starts = []
    s = 0.0
    while s < dur - 1.0:
        starts.append(s)
        s += args.step_seconds
    windows = []
    for wi, st in enumerate(starts, 1):
        en = min(st + args.window_seconds, dur)
        lo, hi = int(st * fps), max(int(en * fps) - 1, int(st * fps) + 1)
        idx = fs.uniform_indices(hi - lo + 1, args.frames)
        idx = [lo + i for i in idx]
        windows.append((wi, st, en, idx))

    TXT_DIR.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    cf = open(CSV_PATH, "a", newline="", encoding="utf-8")
    w = csv.writer(cf)
    if new:
        w.writerow(CSV_HEADER)

    print(f"video: {VIDEO.name}  {dur/60:.1f} min  {fps:.0f} fps")
    print(f"window {args.window_seconds:.0f} s, step {args.step_seconds:.0f} s  ->  "
          f"{len(windows)} analyses per model  |  {args.frames} frames each  |  "
          f"anti-hallucination clause: {args.anti}\n")

    rows = []
    t_start = time.time()
    for model in models:
        for (wi, st, en, s1_idx) in windows:
            t0 = time.time()
            s1_b64 = frames_b64_at(VIDEO, s1_idx, False)
            r1, _e, _ = bm.run_model(model, "external", aas, s1_b64, None, extra=extra)
            s1_status = "ERROR" if r1.startswith("[ERRO") else vb.extract_verdict(r1)[0]

            s2_status, s2_n, r2, final = "", 0, "", s1_status
            if args.two_stage and s1_status == "ANOMALY":
                _, expl = vb.extract_verdict(r1)
                anchors = anchor_indices(expl, r1, s1_idx)
                d_idx = fs.dense_window_indices(anchors, total, args.dense_stride,
                                                args.dense_halfcount, cap=args.dense_cap)
                d_b64 = frames_b64_at(VIDEO, d_idx, False)
                r2, _e2, _ = bm.run_model(model, "external", aas, d_b64, None, extra=extra)
                s2_n = len(d_b64)
                s2_status = "ERROR" if r2.startswith("[ERRO") else vb.extract_verdict(r2)[0]
                if s2_status in ("ANOMALY", "NORMAL"):
                    final = s2_status

            elapsed = time.time() - t0
            (TXT_DIR / f"{model}_w{wi:03d}.txt").write_text(
                f"model: {model}   window {wi}   {st:.0f}-{en:.0f} s   idx={s1_idx}\n\n"
                f"STAGE 1 ({len(s1_idx)} frames) -> {s1_status}\n{r1.strip()}\n\n"
                + (f"STAGE 2 ({s2_n} frames) -> {s2_status}\n{r2.strip()}\n\n" if r2 else "STAGE 2: not run\n\n")
                + f"FINAL: {final}\n", encoding="utf-8")
            w.writerow([model, wi, f"{st:.0f}", f"{en:.0f}", len(s1_idx),
                        s1_status, s2_n, s2_status, final, f"{elapsed:.1f}"])
            cf.flush()
            rows.append((model, wi, len(s1_idx), s1_status, s2_status, final))
            flag = "  <-- FALSE ALARM" if final in HIT else ""
            print(f"  {model:<13} w{wi:>2}/{len(windows)} [{st:>4.0f}-{en:<4.0f}s] "
                  f"s1={s1_status}" + (f" s2={s2_status}" if s2_status else "")
                  + f" -> {final} ({elapsed:.0f}s){flag}")
    cf.close()

    print("\n" + "=" * 84)
    print("PER MODEL  (video is normal, so every ANOMALY is a false positive)")
    print("=" * 84)
    print(f"{'model':<14}{'analyses':<10}{'frames total':<14}{'S1 false':<10}"
          f"{'final false':<13}{'UNKNOWN':<9}{'FP rate':<10}{'~per hour':<10}")
    per_hour_factor = 3600.0 / args.step_seconds
    for m in models:
        sub = [r for r in rows if r[0] == m]
        n = len(sub)
        s1fa = sum(1 for r in sub if r[3] in HIT)
        finfa = sum(1 for r in sub if r[5] in HIT)
        unk = sum(1 for r in sub if r[3] not in HIT and r[3] != "NORMAL")
        fr_total = sum(r[2] for r in sub)  # Stage-1 frames actually shown to the model
        rate = finfa / n if n else 0
        print(f"{m:<14}{n:<10}{fr_total:<14}{f'{s1fa}/{n}':<10}"
              f"{f'{finfa}/{n}':<13}{unk:<9}{f'{rate*100:.1f}%':<10}"
              f"{f'{rate*per_hour_factor:.1f}':<10}")
    print(f"\ntotal wall time: {(time.time()-t_start)/60:.1f} min")
    print(f"CSV: {CSV_PATH}\nper-window text: {TXT_DIR}/")


if __name__ == "__main__":
    main()
