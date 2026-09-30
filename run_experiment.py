"""
run_experiment.py — LLM anomaly-detection experiment, two-stage.

Stage 1 : 9 uniform frames (colour-heuristic ROI crop, YOLO-OFF path) -> model
          verdict + explanation.
Stage 2 : ONLY when stage 1 says ANOMALY and a frame can be identified from the
          model's own output. Re-sample densely around that frame
          (--dense-stride apart, +/- --dense-halfcount samples), re-query the
          same model. The stage-2 verdict is FINAL and may overturn stage 1.
          Stage 1 NORMAL is never revisited.

15 videos (10 external + 5 internal) x 3 models x N repeats.

The YOLO-crop-ON condition (best.pt ROI) was dropped from the study; the path is
still behind --yolo on/both but the default is off.

Cost note: with two-stage on, every ANOMALY run adds one more model call over
~20 frames. For the local models (llama_vision, moondream) that is ~1 min/run
extra — use --two-stage-models gpt4 to restrict it, or --no-two-stage.

Usage:
  python run_experiment.py --smoke
  python run_experiment.py
  python run_experiment.py --two-stage-models gpt4
  python run_experiment.py --no-two-stage
"""

import argparse, csv, re, time, base64
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import benchmark_models as bm
import vision_backends as vb
import frame_sampling as fs

BASE_DIR     = Path(__file__).parent
OUT_DIR      = BASE_DIR / "eval_results_experiment"
MODEL_PATH   = BASE_DIR / "modelo" / "best.pt"
CLASSES_FILE = BASE_DIR / "data" / "classes.txt"

MAX_DIM  = 560
AZUL_MIN = np.array([90, 40, 40])
AZUL_MAX = np.array([135, 255, 255])

CSV_HEADER = ["video", "kind", "model", "yolo_requested", "yolo_effective", "repeat",
              "two_stage", "s1_status", "s2_status", "status", "elapsed_s",
              "s1_frames", "s2_frames", "anomaly_start_s", "anomaly_end_s", "verdict_line"]

CLASSES = [l.strip() for l in open(CLASSES_FILE, encoding="utf-8") if l.strip()]
_yolo = YOLO(str(MODEL_PATH))

_HAND_RE = re.compile(r"m[aã]os?\b|bra[cç]o|dedos?\b|pessoa|hum[ao]|corpo\s+humano|"
                      r"\bbola\b|bal[aã]o|garrafa|\bplanta\b|carteira|extintor|"
                      r"objeto branco|objeto estranho|objeto externo", re.I)
_FRAME_RE = re.compile(r"frames?\s*(\d+)", re.I)


def _ctr(pts):
    return np.array(pts, dtype=np.float32).mean(0)


def _roi(tabelas, shape):
    if not tabelas:
        return None
    if len(tabelas) < 2:
        pts = tabelas[0]["pontos"]
        return (max(0, int(pts[:, 0].min()) - 20), max(0, int(pts[:, 1].min()) - 20),
                min(shape[1], int(pts[:, 0].max()) + 20), min(shape[0], int(pts[:, 1].max()) + 20))
    tb = sorted(tabelas, key=lambda t: _ctr(t["pontos"])[1])
    p0, p1 = tb[0]["pontos"], tb[1]["pontos"]
    y1 = int(p0[:, 1].max()); y2 = int(p1[:, 1].min())
    x1 = max(0, int(max(p0[:, 0].min(), p1[:, 0].min())) - 10)
    x2 = min(shape[1], int(min(p0[:, 0].max(), p1[:, 0].max())) + 10)
    if y2 <= y1:
        ap = np.vstack([t["pontos"] for t in tabelas])
        return (max(0, int(ap[:, 0].min()) - 20), max(0, int(ap[:, 1].min()) - 20),
                min(shape[1], int(ap[:, 0].max()) + 20), min(shape[0], int(ap[:, 1].max()) + 20))
    return (x1, y1, x2, y2)


def _azul(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, AZUL_MIN, AZUL_MAX)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = frame.shape[:2]
        return (0, 0, w, h)
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (max(0, x - 15), max(0, y - 15), min(frame.shape[1], x + w + 15), min(frame.shape[0], y + h + 15))


def _yolo_roi(frame):
    res = _yolo.predict(frame, verbose=False)[0]
    tabelas = []
    if res.obb is not None:
        for pts, cls_id, conf in zip(res.obb.xyxyxyxy.cpu().numpy(),
                                      res.obb.cls.cpu().numpy().astype(int),
                                      res.obb.conf.cpu().numpy()):
            if CLASSES[cls_id] == "tabela_completa":
                tabelas.append({"pontos": pts, "conf": float(conf)})
    tabelas.sort(key=lambda t: _ctr(t["pontos"])[1])
    roi = _roi(tabelas, frame.shape)
    if roi is None:
        roi = _azul(frame)
    return roi


def frames_b64_at(video_path: Path, indices, yolo_mode: bool):
    """Crop (ROI) + resize + JPEG-b64 the given source-frame indices."""
    out = []
    for _src, frame in fs.read_frames(video_path, indices):
        x1, y1, x2, y2 = _yolo_roi(frame) if yolo_mode else _azul(frame)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        h, w = crop.shape[:2]
        if max(h, w) > MAX_DIM:
            s = MAX_DIM / max(h, w)
            crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append(base64.b64encode(buf).decode())
    return out


def anchor_indices(explanation, full_result, s1_idx, max_anchors=2):
    """Map the frame number(s) the model blamed (1-based, in the stage-1 sample)
    back to source-frame indices. Prefers explicit 'frame N' mentions in the
    verdict explanation; falls back to description lines that mention a
    hand/object; last resort the middle stage-1 frame. Capped at max_anchors."""
    ks = [int(m.group(1)) for m in _FRAME_RE.finditer(explanation)
          if 1 <= int(m.group(1)) <= len(s1_idx)]
    if not ks:
        ks = [int(m.group(1)) for m in re.finditer(r"^\s*(\d{1,2})\.\s+(.*)$", full_result, re.M)
              if 1 <= int(m.group(1)) <= len(s1_idx) and _HAND_RE.search(m.group(2))]
    if not ks:
        ks = [len(s1_idx) // 2 + 1]
    ks = sorted(set(ks))[:max_anchors]
    return [s1_idx[k - 1] for k in ks]


def _write_run_txt(out_path, video_name, kind, model_key, spec, log_lines,
                   s1, s2, final_status, win):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    L = [
        f"Video: {video_name} ({kind})",
        f"Model: {model_key} ({spec['model']})",
        f"Log lines used: {log_lines}" if log_lines is not None else "",
        "",
        f"STAGE 1  ({s1['n']} frames, {s1['elapsed']:.2f}s)  -> {s1['status']}",
        s1["result"].strip(),
        "",
    ]
    if s2 is not None:
        L += [
            f"STAGE 2  ({s2['n']} dense frames around src {s2['anchors']}, {s2['elapsed']:.2f}s)"
            f"  -> {s2['status']}",
            s2["result"].strip(),
            "",
        ]
    else:
        L += ["STAGE 2  : not run", ""]
    L += ["Result:",
          final_status + (f"   anomaly window ~ {win[0]:.2f}s .. {win[1]:.2f}s" if win else ""), ""]
    out_path.write_text("\n".join(x for x in L if x is not None), encoding="utf-8")


def _ensure_csv(csv_path):
    if csv_path.exists():
        head = csv_path.read_text(encoding="utf-8").splitlines()[:1]
        if head and head[0].split(",") != CSV_HEADER:
            bak = csv_path.with_suffix(".pre-twostage.csv")
            csv_path.rename(bak)
            print(f"archived old-schema summary -> {bak.name}")
    new = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(CSV_HEADER)
    return f, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--models", type=str, default=",".join(bm.MODELS.keys()))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--yolo", type=str, default="off", choices=["both", "on", "off"])
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--kind", type=str, default="", choices=["", "external", "internal"])
    ap.add_argument("--two-stage", dest="two_stage", action="store_true", default=True)
    ap.add_argument("--no-two-stage", dest="two_stage", action="store_false")
    ap.add_argument("--two-stage-models", type=str, default="",
                    help="restrict stage 2 to these models (comma list); default = all")
    ap.add_argument("--dense-stride", type=int, default=4)
    ap.add_argument("--dense-halfcount", type=int, default=10)
    ap.add_argument("--dense-confirm-only", action="store_true",
                    help="stage 2 only localises; it never overturns the stage-1 verdict "
                         "(recommended on an all-anomaly dataset)")
    ap.add_argument("--dense-cap", type=int, default=12,
                    help="max frames sent to the model in stage 2 (GPT-4o multi-image "
                         "+ TPM limits make >~14 unreliable)")
    ap.add_argument("--tag", type=str, default="",
                    help="write to eval_results_experiment/summary__<tag>.csv instead of "
                         "summary.csv, so parallel jobs don't fight over one file. "
                         "Merge afterwards with merge_summaries.py.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.repeats = 1
        args.only = args.only or "an1"

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_keys:
        if m not in bm.MODELS:
            raise SystemExit(f"Unknown model: {m}. Options: {list(bm.MODELS)}")
    ts_models = set(m.strip() for m in args.two_stage_models.split(",") if m.strip()) or set(model_keys)
    yolo_modes = {"both": [True, False], "on": [True], "off": [False]}[args.yolo]

    aas_section = bm.load_aas_context()
    jobs = bm.collect_jobs(args.only, args.kind)
    if not jobs:
        raise SystemExit("No videos matched the given filters.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = f"summary__{args.tag}.csv" if args.tag else "summary.csv"
    csv_f, writer = _ensure_csv(OUT_DIR / csv_name)

    total_runs = len(jobs) * len(yolo_modes) * len(model_keys) * args.repeats
    run_i = 0
    t_start = time.time()
    print(f"two-stage: {'on' if args.two_stage else 'off'}"
          + (f" (models: {sorted(ts_models)})" if args.two_stage else "")
          + f"  |  dense: stride {args.dense_stride}, +/-{args.dense_halfcount}")

    for kind, video_path, txt_path in jobs:
        log_txt = txt_path.read_text(encoding="utf-8", errors="replace") \
            if (kind == "internal" and txt_path) else None
        log_lines = len([l for l in (log_txt or "").splitlines() if l.strip()]) if log_txt else None
        vtotal, fps = fs.video_meta(video_path)
        s1_idx = fs.uniform_indices(vtotal, args.frames)

        for yolo_mode in yolo_modes:
            eff = yolo_mode if kind == "external" else False
            s1_b64 = frames_b64_at(video_path, s1_idx, eff)
            yr = "on" if yolo_mode else "off"

            for model_key in model_keys:
                spec = bm.MODELS[model_key]
                for rep in range(1, args.repeats + 1):
                    run_i += 1
                    out_path = OUT_DIR / kind / (f"{video_path.stem}__{model_key}"
                                                 f"__yolo{yr}__run{rep:02d}.txt")
                    if out_path.exists() and "STAGE 1" in out_path.read_text(encoding="utf-8", errors="ignore"):
                        print(f"  [{run_i}/{total_runs}] {video_path.stem} {model_key} r{rep}: skip")
                        continue

                    r1, e1, cap1 = bm.run_model(model_key, kind, aas_section, s1_b64, log_txt)
                    if r1.startswith("[ERRO"):
                        s1 = dict(status="ERROR", result=r1, explanation=r1, elapsed=e1, n=len(s1_b64))
                    else:
                        st, ex = vb.extract_verdict(r1)
                        s1 = dict(status=st, result=r1, explanation=ex, elapsed=e1, n=len(s1_b64))

                    s2 = None
                    final = s1["status"]
                    win = None
                    if (args.two_stage and s1["status"] == "ANOMALY"
                            and model_key in ts_models):
                        anchors = anchor_indices(s1["explanation"], s1["result"], s1_idx)
                        d_idx = fs.dense_window_indices(anchors, vtotal, args.dense_stride,
                                                       args.dense_halfcount, cap=args.dense_cap)
                        d_b64 = frames_b64_at(video_path, d_idx, eff)

                        r2, e2 = "", 0.0
                        for attempt in range(2):          # 1 retry: 429s and refusals are non-deterministic
                            r2, e, _ = bm.run_model(model_key, kind, aas_section, d_b64, log_txt)
                            e2 += e
                            if "rate_limit" in r2 or "429" in r2:
                                time.sleep(25); continue
                            st2, _ex2 = vb.extract_verdict(r2)
                            if st2 != "UNKNOWN" or r2.startswith("[ERRO"):
                                break

                        if r2.startswith("[ERRO"):
                            st2, ex2 = "ERROR", r2
                        else:
                            st2, ex2 = vb.extract_verdict(r2)
                        s2 = dict(status=st2, result=r2, explanation=ex2, elapsed=e2,
                                  n=len(d_b64), anchors=anchors)

                        # localisation window is always recorded; the verdict only
                        # moves when stage 2 is a clean ANOMALY/NORMAL and
                        # --dense-confirm-only was not set.
                        win = (fs.to_seconds(min(d_idx), fps), fs.to_seconds(max(d_idx), fps))
                        if not args.dense_confirm_only and st2 in ("ANOMALY", "NORMAL"):
                            final = st2

                    _write_run_txt(out_path, video_path.name, kind, model_key, spec, log_lines,
                                   s1, s2, final, win)
                    writer.writerow([
                        video_path.stem, kind, model_key, yr, "on" if eff else "off", rep,
                        int(args.two_stage and model_key in ts_models),
                        s1["status"], s2["status"] if s2 else "", final,
                        f"{s1['elapsed'] + (s2['elapsed'] if s2 else 0):.2f}",
                        s1["n"], s2["n"] if s2 else 0,
                        f"{win[0]:.2f}" if win else "", f"{win[1]:.2f}" if win else "",
                        (s2 or s1)["explanation"][:80].replace("\n", " "),
                    ])
                    csv_f.flush()
                    flip = ""
                    if s2:
                        flip = ("  (OVERTURNED)" if final != s1["status"]
                                else "  (s2 error, kept s1)" if s2["status"] == "ERROR"
                                else "  (confirmed)")
                    print(f"  [{run_i}/{total_runs}] {video_path.stem} {model_key} r{rep}: "
                          f"s1={s1['status']}" + (f" s2={s2['status']}" if s2 else "")
                          + f" -> {final}{flip}")

    csv_f.close()
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Summary: {OUT_DIR / csv_name}")


if __name__ == "__main__":
    main()
