"""
frame_sampling.py — shared frame-index helpers for the two-stage anomaly pipeline.

Stage 1 samples the whole clip uniformly (uniform_indices). If the detector /
model flags an anomaly, stage 2 re-samples densely around the flagged frame(s)
(dense_window_indices) and re-evaluates; the stage-2 verdict is final and may
overturn stage 1.

All indices are 0-based source-video frame numbers.
"""

from __future__ import annotations

import cv2


def uniform_indices(total: int, n: int) -> list[int]:
    """`n` frame indices spread uniformly over [0, total-1] (the historical rule
    used by benchmark_models.py / run_experiment.py)."""
    if total <= 0:
        return list(range(n))
    return sorted(set(int(i * (total - 1) / max(1, n - 1)) for i in range(n)))


def dense_window_indices(anchors, total: int, stride: int = 4, half_count: int = 10,
                         cap: int = 60) -> list[int]:
    """Union of `anchor + k*stride` for k in [-half_count, half_count] around each
    anchor, clamped to [0, total-1], de-duplicated and sorted.

    `cap` bounds the total number of frames returned (protects the LLM stage from
    a runaway multi-anchor window); if exceeded the set is uniformly thinned.
    """
    if total <= 0:
        total = max(anchors) + 1 if anchors else 1
    idx = set()
    for a in anchors:
        for k in range(-half_count, half_count + 1):
            f = int(a) + k * stride
            if 0 <= f <= total - 1:
                idx.add(f)
    out = sorted(idx)
    if len(out) > cap:
        step = len(out) / cap
        out = [out[int(i * step)] for i in range(cap)]
    return out


def read_frames(video_path: str, indices) -> list:
    """Read the given source-frame indices from a video as BGR arrays (skips any
    that fail to decode). Order follows `indices` sorted ascending."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    out = []
    for i in sorted(set(int(x) for x in indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            out.append((i, frame))
    cap.release()
    return out


def video_meta(video_path: str):
    """(total_frames, fps) — fps falls back to 30.0 when the container omits it."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return total, fps


def to_seconds(frame_idx: int, fps: float) -> float:
    return round(frame_idx / fps, 2) if fps else 0.0
