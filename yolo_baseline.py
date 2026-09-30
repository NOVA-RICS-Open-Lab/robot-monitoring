"""
yolo_baseline.py - Off-the-shelf YOLO baseline, now two-stage.

Stage 1  : sample the whole clip uniformly (--frames, default 9), run the
           detector, apply the decision rule -> verdict + the source-frame
           indices that triggered it (anchors).
Stage 2  : ONLY if stage 1 said ANOMALY and produced anchors. Re-sample densely
           around each anchor (--dense-stride frames apart, +/- --dense-halfcount
           samples) and re-run the detector. The stage-2 verdict is FINAL and may
           overturn stage 1 (ANOMALY -> NORMAL when the dense window does not
           confirm). Stage 1 NORMAL is never revisited.

Two stock Ultralytics models, used as-is:
  yolo_coco  (yolo11n.pt)      - ANOMALY if `person` (conf >= --conf) in any frame.
                                 Foreign COCO objects recorded but not verdict-
                                 changing unless --foreign-objects.
  yolo_pose  (yolo11n-pose.pt) - ANOMALY if a wrist keypoint (idx 9/10) is visible
                                 with conf >= --conf in any frame.

Writes:
  eval_results/<track>/<stem>__<variant>.txt
  eval_results_experiment/yolo_summary.csv   (final `status` + both stage verdicts
                                              + localised anomaly window)

Usage:
  python yolo_baseline.py --smoke
  python yolo_baseline.py
  python yolo_baseline.py --no-two-stage --only an5
  python yolo_baseline.py --dense-stride 4 --dense-halfcount 10
"""

import argparse
import csv
import time
from pathlib import Path

from ultralytics import YOLO

import benchmark_models as bm
import frame_sampling as fs

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "eval_results"
CSV_PATH = BASE_DIR / "eval_results_experiment" / "yolo_summary.csv"

CSV_HEADER = ["video", "kind", "model", "repeat", "conf", "two_stage",
              "s1_status", "s2_status", "status", "elapsed_s",
              "s1_frames", "s2_frames", "frames",
              "anomaly_start_s", "anomaly_end_s", "basis"]

VARIANTS = {
    "yolo_coco": {"weights": "yolo11n.pt",      "task": "detect"},
    "yolo_pose": {"weights": "yolo11n-pose.pt", "task": "pose"},
}

FOREIGN_COCO = {
    "cell phone", "bottle", "cup", "book", "remote", "scissors", "knife",
    "fork", "spoon", "mouse", "keyboard", "backpack", "handbag", "wine glass",
    "sports ball", "potted plant", "bowl", "vase", "teddy bear",
}
WRIST_KPTS = {9: "left_wrist", 10: "right_wrist"}

_model_cache = {}


def _get_model(weights: str) -> YOLO:
    if weights not in _model_cache:
        _model_cache[weights] = YOLO(weights)  # auto-downloads on first use
    return _model_cache[weights]


def _run_coco(model, framepairs, fps, conf, foreign_counts):
    """framepairs: list[(src_idx, bgr)]. Returns dict(status, basis, lines, hits)."""
    lines, person_hits, foreign_hits = [], [], []
    for src, frame in framepairs:
        res = model.predict(frame, conf=conf, verbose=False)[0]
        names = res.names
        dets = []
        for cls_id, c in zip(res.boxes.cls.cpu().numpy().astype(int),
                             res.boxes.conf.cpu().numpy()):
            label = names[cls_id]
            dets.append(f"{label}({c:.2f})")
            if label == "person":
                person_hits.append(src)
            elif label in FOREIGN_COCO:
                foreign_hits.append((src, label))
        lines.append(f"  src#{src:<5} t={fs.to_seconds(src, fps):>6.2f}s : "
                     f"{', '.join(dets) if dets else '(nothing)'}")

    person_hits = sorted(set(person_hits))
    ftxt = ", ".join(sorted({f"{l}" for _, l in foreign_hits}))
    if person_hits:
        return dict(status="ANOMALY", hits=person_hits,
                    basis=f"person in {len(person_hits)} frame(s)"
                          + (f"; also: {ftxt}" if ftxt else ""), lines=lines)
    if foreign_hits and foreign_counts:
        return dict(status="ANOMALY", hits=sorted({s for s, _ in foreign_hits}),
                    basis=f"foreign object(s) only: {ftxt}", lines=lines)
    return dict(status="NORMAL", hits=[],
                basis="no person detected" + (f" (foreign seen, not counted: {ftxt})" if ftxt else ""),
                lines=lines)


def _run_pose(model, framepairs, fps, conf):
    lines, wrist_hits = [], []
    for src, frame in framepairs:
        res = model.predict(frame, conf=conf, verbose=False)[0]
        n_person = 0 if res.boxes is None else len(res.boxes)
        seen = []
        kp = res.keypoints
        if kp is not None and kp.data is not None and len(kp.data):
            for person in kp.data.cpu().numpy():          # (n, 17, 3)
                for idx, wname in WRIST_KPTS.items():
                    wc = float(person[idx, 2])
                    if wc >= conf:
                        wrist_hits.append(src)
                        seen.append(f"{wname}({wc:.2f})")
        lines.append(f"  src#{src:<5} t={fs.to_seconds(src, fps):>6.2f}s : "
                     f"{n_person} person(s)" + (f"; wrists {', '.join(seen)}" if seen else ""))
    wrist_hits = sorted(set(wrist_hits))
    if wrist_hits:
        return dict(status="ANOMALY", hits=wrist_hits,
                    basis=f"wrist keypoint in {len(wrist_hits)} frame(s)", lines=lines)
    return dict(status="NORMAL", hits=[], basis="no wrist keypoint above threshold", lines=lines)


def _run(spec, model, framepairs, fps, conf, foreign):
    if spec["task"] == "detect":
        return _run_coco(model, framepairs, fps, conf, foreign)
    return _run_pose(model, framepairs, fps, conf)


def _write_txt(out_path, video_name, kind, vkey, weights, conf, fps,
               s1, s2, final_status, win):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"Video: {video_name} ({kind})",
        f"Model: {vkey} ({weights})   conf>={conf}",
        "",
        f"STAGE 1  ({len(s1['lines'])} uniform frames)  -> {s1['status']}",
        *s1["lines"],
        f"  basis: {s1['basis']}",
        "",
    ]
    if s2 is not None:
        body += [
            f"STAGE 2  ({len(s2['lines'])} dense frames around src {s1['hits']})  -> {s2['status']}",
            *s2["lines"],
            f"  basis: {s2['basis']}",
            "",
        ]
    else:
        body += ["STAGE 2  : not run (stage 1 not ANOMALY, or no anchor frame)", ""]
    body += ["Result:",
             f"{final_status}"
             + (f"   anomaly window ~ {win[0]:.2f}s .. {win[1]:.2f}s" if win else ""),
             ""]
    out_path.write_text("\n".join(body), encoding="utf-8")


def _ensure_csv():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        first = CSV_PATH.read_text(encoding="utf-8").splitlines()[:1]
        if first and first[0].split(",") != CSV_HEADER:
            bak = CSV_PATH.with_suffix(f".bak-{int(time.time())}.csv")
            CSV_PATH.rename(bak)
            print(f"archived old-schema summary -> {bak.name}")
    new = not CSV_PATH.exists()
    f = open(CSV_PATH, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(CSV_HEADER)
    return f, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=9, help="stage-1 uniform frame count")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--variants", type=str, default=",".join(VARIANTS))
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--kind", type=str, default="", choices=["", "external", "internal"])
    ap.add_argument("--foreign-objects", action="store_true")
    ap.add_argument("--two-stage", dest="two_stage", action="store_true", default=True)
    ap.add_argument("--no-two-stage", dest="two_stage", action="store_false")
    ap.add_argument("--dense-stride", type=int, default=4)
    ap.add_argument("--dense-halfcount", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1,
                    help="write N rows per clip (YOLO is deterministic, so they are "
                         "identical) — only for a uniform n against the 10x VLM runs")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.only = args.only or "an1"

    alias = {"coco": "yolo_coco", "pose": "yolo_pose"}
    variant_keys = [alias.get(v.strip(), v.strip()) for v in args.variants.split(",") if v.strip()]
    for v in variant_keys:
        if v not in VARIANTS:
            raise SystemExit(f"Unknown variant: {v}. Options: {list(VARIANTS)}")

    jobs = bm.collect_jobs(args.only, args.kind)
    if not jobs:
        raise SystemExit("No videos matched the given filters.")

    csv_f, writer = _ensure_csv()
    total = len(jobs) * len(variant_keys)
    done = 0
    t_start = time.time()

    for kind, video_path, _txt in jobs:
        vtotal, fps = fs.video_meta(video_path)
        s1_idx = fs.uniform_indices(vtotal, args.frames)
        s1_pairs = fs.read_frames(video_path, s1_idx)
        print(f"\n[{video_path.stem}] ({kind}) total={vtotal}f fps={fps:.0f}  "
              f"stage-1 idx={s1_idx}")

        for vkey in variant_keys:
            done += 1
            spec = VARIANTS[vkey]
            model = _get_model(spec["weights"])
            t0 = time.time()

            # computed once; YOLO is deterministic so --repeats just duplicates the row
            s1 = _run(spec, model, s1_pairs, fps, args.conf, args.foreign_objects)
            s2 = None
            final = s1["status"]
            win = None

            if args.two_stage and s1["status"] == "ANOMALY" and s1["hits"]:
                dense_idx = fs.dense_window_indices(s1["hits"], vtotal,
                                                    args.dense_stride, args.dense_halfcount)
                dense_pairs = fs.read_frames(video_path, dense_idx)
                s2 = _run(spec, model, dense_pairs, fps, args.conf, args.foreign_objects)
                final = s2["status"]
                if s2["hits"]:
                    win = (fs.to_seconds(min(s2["hits"]), fps),
                           fs.to_seconds(max(s2["hits"]), fps))
            elif s1["hits"]:
                win = (fs.to_seconds(min(s1["hits"]), fps),
                       fs.to_seconds(max(s1["hits"]), fps))

            elapsed = time.time() - t0
            out_path = OUT_DIR / kind / f"{video_path.stem}__{vkey}.txt"
            _write_txt(out_path, video_path.name, kind, vkey, spec["weights"],
                       args.conf, fps, s1, s2, final, win)
            s1n, s2n = len(s1["lines"]), (len(s2["lines"]) if s2 else 0)
            for rep in range(1, args.repeats + 1):
                writer.writerow([
                    video_path.stem, kind, vkey, rep, args.conf, int(args.two_stage),
                    s1["status"], s2["status"] if s2 else "", final, f"{elapsed:.2f}",
                    s1n, s2n, s1n + s2n,
                    f"{win[0]:.2f}" if win else "", f"{win[1]:.2f}" if win else "",
                    (s2 or s1)["basis"],
                ])
            csv_f.flush()
            flip = "" if not s2 else ("  (STAGE-2 OVERTURNED)" if s2["status"] != s1["status"] else "  (confirmed)")
            print(f"  [{done}/{total}] {vkey}: s1={s1['status']}"
                  + (f" s2={s2['status']}" if s2 else "")
                  + f" -> {final} ({elapsed:.2f}s){flip}")

    csv_f.close()
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Summary: {CSV_PATH}")


if __name__ == "__main__":
    main()
