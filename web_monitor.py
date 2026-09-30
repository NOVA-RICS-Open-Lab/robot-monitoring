"""
web_monitor.py — Robot workspace anomaly detection
  - EXTERNAL: camera-based detection (VLM + LLM + YOLO)
  - INTERNAL: robot process monitoring via HTTP API
  - AAS context | Configurable models | AI chat
"""

import argparse, base64, json, threading, time, xml.etree.ElementTree as ET, zipfile
from collections import deque
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory
from ultralytics import YOLO

import vision_backends

load_dotenv(Path(__file__).parent / ".env")

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

BASE_DIR = Path(__file__).parent
CFG_FILE = BASE_DIR / "config.json"

PROMPT_VISAO_DEFAULT = (
    "List the most prominent objects in this image. "
    "Only list what you clearly see.\n"
    "Reply with a single comma-separated line. One short label per object. No repetitions.\n"
    "Example: robot arm, white tray, yellow block"
)

PROMPT_TEXTO_DEFAULT = (
    "An industrial robot workspace shows: {objetos}\n\n"
    "{aas_section}\n"
    "Using ONLY the context above (if provided), classify each detected object:\n"
    "- Parts of the robot, its tools, the materials it handles, or its environment → NORMAL\n"
    "- Any human body part (hand, finger, arm, person, human) → ALWAYS ANOMALY\n"
    "- Objects that clearly do not belong to this robot's operation → ANOMALY\n"
    "- If you are uncertain → NORMAL\n\n"
    "Reply with one line only:\n"
    "  ANOMALY: [specific object name]\n"
    "  NORMAL"
)

DEFAULT_CFG = {
    "vision_model":       "gpt-4o",
    "yolo_enabled":       True,
    "camera":             0,
    "cooldown":           5.0,
    "aas_filename":       "",
    "robot_operation":    "",
    "aas_context":        "",
    "prompt_visao":       "",
    "prompt_texto":       "",
    "external_enabled":   True,
    "internal_enabled":   True,
    "robot_api_url":      "http://192.168.1.100:8000/api/status",
    "robot_api_interval": 3.0,
    "source_type":        "camera",
    "video_path":         "",
    "video_loop":         True,
    "internal_log_path":  "",
}

def _load_cfg():
    if CFG_FILE.exists():
        try:
            saved = json.loads(CFG_FILE.read_text(encoding="utf-8"))
            if "analysis_enabled" in saved and "external_enabled" not in saved:
                saved["external_enabled"] = saved.pop("analysis_enabled")
            return {**DEFAULT_CFG, **saved}
        except Exception:
            pass
    return dict(DEFAULT_CFG)

def _save_cfg(c):
    CFG_FILE.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")

_cfg      = _load_cfg()
_cfg_lock = threading.Lock()

def _find(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"Not found: {candidates}")

MODEL_PATH   = _find([BASE_DIR/"modelo"/"best.pt",
                       BASE_DIR.parent/"runs"/"obb"/"train-2"/"weights"/"best.pt"])
CLASSES_FILE = _find([BASE_DIR/"data"/"classes.txt",
                       BASE_DIR.parent/"data"/"classes.txt"])
CLASSES  = [l.strip() for l in open(CLASSES_FILE, encoding="utf-8") if l.strip()]
MAX_DIM  = 560
AZUL_MIN = np.array([90,  40,  40])
AZUL_MAX = np.array([135, 255, 255])

_yolo = YOLO(str(MODEL_PATH))
print(f"YOLO: {MODEL_PATH}")

# ── Prompts ───────────────────────────────────────────────────────────────────
def _get_prompt_visao():
    v = _cfg.get("prompt_visao", "")
    return v.strip() if v and v.strip() else PROMPT_VISAO_DEFAULT

def _build_aas_section():
    op  = _cfg.get("robot_operation", "")
    ctx = _cfg.get("aas_context", "")
    if not op and not ctx: return ""
    s = "Robot context (from AAS):\n"
    if op:  s += f"  Operation: {op}\n"
    if ctx: s += f"  Context: {ctx}\n"
    return s

def _build_texto_prompt(objetos):
    with _cfg_lock:
        tmpl = _cfg.get("prompt_texto", "")
        aas_section = _build_aas_section()
    if not tmpl or not tmpl.strip():
        tmpl = PROMPT_TEXTO_DEFAULT
    try:
        return tmpl.format(objetos=objetos, aas_section=aas_section)
    except (KeyError, ValueError):
        return tmpl.replace("{objetos}", objetos).replace("{aas_section}", aas_section)

# ── AAS Parser ────────────────────────────────────────────────────────────────
_KEEP_TAGS = {"idShort","value","description","langString","text","note","semanticId"}

def _xml_texts(root):
    seen, out = set(), []
    for el in root.iter():
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        t   = (el.text or "").strip()
        if tag in _KEEP_TAGS and t and len(t) > 2 and t not in seen:
            seen.add(t); out.append(t)
    return out

def _json_texts(obj, depth=0):
    if depth > 8: return []
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and len(v) > 2: out.append(f"{k}: {v}")
            else: out.extend(_json_texts(v, depth+1))
    elif isinstance(obj, list):
        for i in obj: out.extend(_json_texts(i, depth+1))
    return out

def parse_aas(file_bytes: bytes, filename: str) -> str:
    fname = filename.lower()
    texts = []
    try:
        if fname.endswith(".aasx"):
            with zipfile.ZipFile(BytesIO(file_bytes)) as zf:
                for name in zf.namelist():
                    if name.endswith((".xml", ".aas")):
                        try: texts.extend(_xml_texts(ET.fromstring(zf.read(name))))
                        except Exception: pass
        elif fname.endswith(".json"):
            texts = _json_texts(json.loads(file_bytes.decode("utf-8", "replace")))
        else:
            texts = _xml_texts(ET.fromstring(file_bytes))
    except Exception as e:
        return f"[Parse error: {e}]"
    seen, out = set(), []
    for t in texts:
        if t not in seen: seen.add(t); out.append(t)
    return "\n".join(out[:300])

def aas_summarize(raw: str):
    if not raw or raw.startswith("[Parse"): return ("", "")
    from openai import OpenAI
    try:
        r = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"Extract from this Asset Administration Shell description:\n{raw[:2500]}\n\n"
                "Answer:\nOPERATION: <one sentence — what task the robot performs>\n"
                "CONTEXT: <2 sentences — objects/materials involved and any anomaly-relevant info>"}],
            max_tokens=130, temperature=0,
        )
        txt = r.choices[0].message.content
        op = ctx = ""
        for line in txt.splitlines():
            if line.startswith("OPERATION:"): op = line[10:].strip()
            elif line.startswith("CONTEXT:"): ctx = line[8:].strip()
        return (op, ctx)
    except Exception as e:
        return ("", f"[LLM error: {e}]")

# ── Model calls ───────────────────────────────────────────────────────────────
def _enc(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if max(h, w) > MAX_DIM:
        s = MAX_DIM / max(h, w)
        frame_bgr = cv2.resize(frame_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()

def call_vision(frame_bgr) -> str:
    with _cfg_lock:
        model  = _cfg["vision_model"]
        prompt = _get_prompt_visao()
    try:
        if model.startswith("ollama/"):
            import requests as req
            r = req.post("http://localhost:11434/api/generate", json={
                "model": model[7:], "prompt": prompt,
                "images": [_enc(frame_bgr)], "stream": False,
                "options": {"num_predict": 30, "temperature": 0},
            }, timeout=60)
            return r.json()["response"].strip()
        else:
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{_enc(frame_bgr)}", "detail": "low"}},
                ]}],
                max_tokens=30, temperature=0,
            )
            return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR vision: {e}]"

def call_text(objetos) -> str:
    with _cfg_lock: model = _cfg["vision_model"]
    prompt = _build_texto_prompt(objetos)
    try:
        if model.startswith("ollama/"):
            import requests as req
            r = req.post("http://localhost:11434/api/generate", json={
                "model": model[7:], "prompt": prompt, "stream": False,
                "options": {"num_predict": 20, "temperature": 0},
            }, timeout=60)
            return r.json()["response"].strip()
        else:
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20, temperature=0,
            )
            return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR text: {e}]"

# ── Internal anomaly agent (multi-frame, 4fps) ────────────────────────────────
MAX_INTERNAL_FRAMES = 60          # cap for whole-video analysis (~15s @ 4fps)
INTERNAL_FPS         = 4
INTERNAL_WINDOW_SEC  = 10          # live-trigger window: last N seconds @ 4fps

def _is_strategic_moment(curr: dict, prev: dict) -> bool:
    if not prev: return False
    return (
        curr.get("state")         != prev.get("state") or
        curr.get("last_error")    != prev.get("last_error") or
        curr.get("current_phase") != prev.get("current_phase") or
        curr.get("pending_move")  != prev.get("pending_move")
    )

INTERNAL_PROMPT_IMG = """\
You are an industrial robot process monitor (INTERNAL anomaly detection — failures in \
the robot's own execution, not external interference).

AAS Operation: {aas_op}
AAS Context: {aas_ctx}

Execution log:
{log_context}

You are given {n} sequential frames (chronological order) covering the relevant window.

Log note: the "gripper" field is a NUMERIC value indicating the pressure/force the \
gripper is exerting (not a boolean or an error flag). 0 means no pressure / nothing \
gripped; a high value means the gripper is firmly holding something.

Anomaly criteria:
- last_error is not null → ALWAYS ANOMALY
- state is 'error', 'fault', 'stopped' unexpectedly → ANOMALY
- present_board differs from target_board unexpectedly → ANOMALY
- A sudden drop in gripper pressure (from a high value to ~0) during a phase where it \
should be holding an object, or pressure staying at 0 when the cycle requires gripping \
something → ANOMALY (object was dropped or never gripped)
- Normal transitions (idle↔running, expected phase changes, gripper pressure consistent \
with the task phase) → NORMAL

Reply in this exact structure:
OBSERVATIONS (summarize into at most 6 time segments, each covering several consecutive \
frames — do NOT write one line per individual frame, that format is rejected):
1. <factual observation for the 1st segment>
2. <...>

Then a final line, exactly one of:
ANOMALY: <factual and elaborate explanation — several sentences, justify using what you saw>
NORMAL: <factual and elaborate explanation — several sentences, justify using what you saw>
"""

INTERNAL_PROMPT_SYNTH = """\
You are an industrial robot process monitor (INTERNAL anomaly detection — failures in \
the robot's own execution, not external interference).

AAS Operation: {aas_op}
AAS Context: {aas_ctx}

Execution log:
{log_context}

A vision model captioned {n} sequential frames (chronological order) covering the \
relevant window:

{captions}

Log note: the "gripper" field is a NUMERIC value indicating the pressure/force the \
gripper is exerting (not a boolean or an error flag). 0 means no pressure / nothing \
gripped; a high value means the gripper is firmly holding something.

Anomaly criteria:
- last_error is not null → ALWAYS ANOMALY
- state is 'error', 'fault', 'stopped' unexpectedly → ANOMALY
- present_board differs from target_board unexpectedly → ANOMALY
- A sudden drop in gripper pressure (from a high value to ~0) during a phase where it \
should be holding an object, or pressure staying at 0 when the cycle requires gripping \
something → ANOMALY (object was dropped or never gripped)
- Normal transitions (idle↔running, expected phase changes, gripper pressure consistent \
with the task phase) → NORMAL

Reply with ONLY the final verdict line, exactly one of:
ANOMALY: <factual and elaborate explanation — several sentences, justify using the captions and log>
NORMAL: <factual and elaborate explanation — several sentences, justify using the captions and log>
"""

def _build_internal_prompts(aas_op, aas_ctx, log_context, n):
    img_prompt = INTERNAL_PROMPT_IMG.format(
        aas_op=aas_op or "unspecified", aas_ctx=aas_ctx or "none",
        log_context=log_context or "(empty)", n=n)

    def synth_prompt_fn(captions):
        captions_txt = "\n".join(f"Frame {i+1}: {c}" for i, c in enumerate(captions))
        return INTERNAL_PROMPT_SYNTH.format(
            aas_op=aas_op or "unspecified", aas_ctx=aas_ctx or "none",
            log_context=log_context or "(empty)", n=len(captions), captions=captions_txt)

    return img_prompt, synth_prompt_fn

def call_internal_agent(robot_state: dict, prev_state: dict, frames_b64=None):
    """Multi-frame internal analysis. frames_b64: chronological list of base64 JPEGs
    (last INTERNAL_WINDOW_SEC seconds @ INTERNAL_FPS from the live buffer).
    Returns (status, explanation, frame_notes)."""
    with _cfg_lock:
        aas_op  = _cfg.get("robot_operation", "")
        aas_ctx = _cfg.get("aas_context", "")
        model   = _cfg.get("vision_model", "gpt-4o")

    log_context = (
        f"Previous state:\n{json.dumps(prev_state, indent=2)}\n\n"
        f"Current state:\n{json.dumps(robot_state, indent=2)}"
    )
    frames_b64 = frames_b64 or []
    if not frames_b64:
        return "WAITING", "[No frames available for analysis]", None

    img_prompt, synth_prompt_fn = _build_internal_prompts(aas_op, aas_ctx, log_context, len(frames_b64))
    try:
        result_text, frame_notes = vision_backends.analyze_frames(model, frames_b64, img_prompt, synth_prompt_fn)
        status, explanation = vision_backends.extract_verdict(result_text)
        return status, explanation, frame_notes
    except Exception as e:
        return "WAITING", f"[Error: {e}]", None

# ── Geometry ──────────────────────────────────────────────────────────────────
def _ord(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)

def _ctr(pts): return np.array(pts, dtype=np.float32).mean(0)

def _in_poly(pt, poly):
    return cv2.pointPolygonTest(np.array(poly, np.float32).reshape(-1,1,2),
                                (float(pt[0]), float(pt[1])), False) >= 0

def _pos(pb, pt):
    dst = np.array([[0,0],[400,0],[400,200],[0,200]], np.float32)
    M   = cv2.getPerspectiveTransform(_ord(pt), dst)
    p   = cv2.perspectiveTransform(np.array([[pb]], np.float32), M)[0][0]
    x, y = p
    if x<0 or x>400 or y<0 or y>200: return None
    return min(int(y/100),1)*4 + min(int(x/100),3) + 1

def _roi(tabelas, shape):
    if not tabelas: return None
    if len(tabelas) < 2:
        pts = tabelas[0]["pontos"]
        return (max(0,int(pts[:,0].min())-20), max(0,int(pts[:,1].min())-20),
                min(shape[1],int(pts[:,0].max())+20), min(shape[0],int(pts[:,1].max())+20))
    tb = sorted(tabelas, key=lambda t: _ctr(t["pontos"])[1])
    p0, p1 = tb[0]["pontos"], tb[1]["pontos"]
    y1 = int(p0[:,1].max()); y2 = int(p1[:,1].min())
    x1 = max(0, int(max(p0[:,0].min(), p1[:,0].min()))-10)
    x2 = min(shape[1], int(min(p0[:,0].max(), p1[:,0].max()))+10)
    if y2 <= y1:
        ap = np.vstack([t["pontos"] for t in tabelas])
        return (max(0,int(ap[:,0].min())-20), max(0,int(ap[:,1].min())-20),
                min(shape[1],int(ap[:,0].max())+20), min(shape[0],int(ap[:,1].max())+20))
    return (x1, y1, x2, y2)

def _azul(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, AZUL_MIN, AZUL_MAX)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(40,40)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(10,10)))
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = frame.shape[:2]; return (0, 0, w, h)
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (max(0,x-15), max(0,y-15), min(frame.shape[1],x+w+15), min(frame.shape[0],y+h+15))

# ── Frame buffer ──────────────────────────────────────────────────────────────
_frame_buffer = deque(maxlen=25)
_buffer_lock  = threading.Lock()

# Dense 4fps ring buffer for live internal-anomaly analysis (last INTERNAL_WINDOW_SEC seconds)
_frame_buffer_4fps = deque(maxlen=INTERNAL_FPS * INTERNAL_WINDOW_SEC)
_buffer4_lock       = threading.Lock()

PROMPT_RELATORIO = """\
You are a safety analyst for an industrial robot workspace. An anomaly was just detected.

Robot context (from AAS):
  Operation: {operation}
  Context:   {context}

Detected anomaly: {anomaly}
Anomaly type: {anomaly_type}

You are given {n} sequential images captured before and at the moment of anomaly detection \
(chronological order, earliest first). Analyze the full sequence carefully.

Provide a structured report:

1. FRAME-BY-FRAME OBSERVATIONS
2. ANOMALY TIMELINE
3. PROBABLE CAUSE
4. RISK ASSESSMENT (LOW / MEDIUM / HIGH)
5. RECOMMENDATION

Be concise and factual. Output in English.\
"""

# ── Global state ──────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "frame_num": 0, "fps": 0.0, "ultimo_update": 0.0,
    "blocos": {}, "roi": None, "metodo": "",
    # external
    "ext_status":         "WAITING",
    "ext_detalhe":        "",
    "ext_visao_raw":      "",
    "ext_decisao_raw":    "",
    "ext_t_visao":        None,
    "ext_t_texto":        None,
    "ext_t_total":        None,
    "ext_ultima_analise": None,
    # internal
    "int_status":         "WAITING",
    "int_detalhe":        "",
    "int_robot_state":    {},
    "int_ultima_analise": None,
    "int_frame_notes":    None,
    "int_video_analysis_status": "idle",
    # reports
    "report_status": "",
    "last_report":   "",
    # chat (kept separate, not sent with /api/status)
    "chat_history":  [],
}
_jpeg_frame = None
_jpeg_crop  = None

def generate_report(anomaly_text, anomaly_type, buf_snap, current_jpeg, aas_op, aas_ctx):
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts_str   = time.strftime("%Y%m%d_%H%M%S")
    out_path = reports_dir / f"anomaly_{ts_str}.txt"

    frames = list(buf_snap)
    if len(frames) > 5:
        idx = [int(i * (len(frames)-1) / 4) for i in range(5)]
        frames = [frames[i] for i in idx]

    content = [{"type": "text", "text": PROMPT_RELATORIO.format(
        operation=aas_op or "Not specified",
        context=aas_ctx   or "No AAS context available",
        anomaly=anomaly_text,
        anomaly_type=anomaly_type,
        n=len(frames) + (1 if current_jpeg else 0),
    )}]
    for _, jpeg_bytes in frames:
        b64 = base64.b64encode(jpeg_bytes).decode()
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
    if current_jpeg:
        b64 = base64.b64encode(current_jpeg).decode()
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

    analysis = ""
    try:
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": content}],
            max_tokens=700, temperature=0.2,
        )
        analysis = resp.choices[0].message.content.strip()
    except Exception as e:
        analysis = f"[Analysis error: {e}]"

    sep = "=" * 64
    lines = [
        sep, "  ANOMALY REPORT — ROBOT MONITOR", sep,
        f"  Date/Time  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Type       : {anomaly_type}",
        f"  File       : {out_path.name}", sep, "",
        "ROBOT CONTEXT (AAS)", "-" * 64,
        f"  Operation : {aas_op  or 'Not specified'}",
        f"  Context   : {aas_ctx or 'Not specified'}", "",
        "ANOMALY DETECTED", "-" * 64,
        f"  {anomaly_text}", "",
        f"FRAMES ANALYZED: {len(frames) + (1 if current_jpeg else 0)}",
        "", "DETAILED ANALYSIS (GPT-4o)", "-" * 64,
        analysis, "", sep,
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Saved: {out_path}")
    return out_path.name

def _resolve_internal_log_path(video_path_str):
    with _cfg_lock:
        manual = _cfg.get("internal_log_path", "")
    if manual and Path(manual).exists():
        return Path(manual)
    if video_path_str:
        sibling = Path(video_path_str).with_suffix(".txt")
        if sibling.exists():
            return sibling
    return None

def _extract_video_frames_4fps(video_path_str, fps=INTERNAL_FPS, max_frames=MAX_INTERNAL_FRAMES):
    """Returns list of (timestamp_sec, b64_jpeg) sampled at `fps` across the whole video,
    downsampled uniformly to `max_frames` if that would exceed the cap."""
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path_str}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / src_fps if total > 0 else 0.0

    n_target = max(1, int(duration * fps)) if duration > 0 else max_frames
    n_target = min(n_target, max_frames)
    timestamps = [i * duration / max(1, n_target - 1) for i in range(n_target)] if n_target > 1 else [0.0]

    out = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * src_fps))
        ok, frame = cap.read()
        if ok:
            out.append((round(t, 2), _enc(frame)))
    cap.release()
    return out

def analyze_internal_video():
    """One-shot whole-video internal analysis (simulation mode), run in a background thread."""
    with _cfg_lock:
        video_path = _cfg.get("video_path", "")
        aas_op     = _cfg.get("robot_operation", "")
        aas_ctx    = _cfg.get("aas_context", "")
        model      = _cfg.get("vision_model", "gpt-4o")

    with _lock: _state["int_video_analysis_status"] = "a_correr"
    try:
        if not video_path:
            raise RuntimeError("No video loaded")
        log_path = _resolve_internal_log_path(video_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path else ""
        log_context = (
            f"Log file: {log_path.name if log_path else '(none found)'}\n"
            f"{log_text or '(empty)'}"
        )

        # Stage 1: coarse pass on ~9 uniform frames. Stage 2 (the dense whole-video
        # pass) runs only if stage 1 flags an anomaly; its verdict is final and may
        # overturn stage 1. A stage-1 NORMAL is not revisited.
        coarse = _extract_video_frames_4fps(video_path, max_frames=9)
        if not coarse:
            raise RuntimeError("Could not extract frames from the video")
        c_b64 = [b64 for _, b64 in coarse]
        c_img, c_synth = _build_internal_prompts(aas_op, aas_ctx, log_context, len(c_b64))
        c_text, c_notes = vision_backends.analyze_frames(model, c_b64, c_img, c_synth)
        s1_status, s1_expl = vision_backends.extract_verdict(c_text)

        if s1_status != "ANOMALY":
            status, explanation, frame_notes = s1_status, s1_expl, c_notes
        else:
            sampled = _extract_video_frames_4fps(video_path)
            if not sampled:
                raise RuntimeError("Could not extract frames from the video")
            frames_b64 = [b64 for _, b64 in sampled]
            img_prompt, synth_prompt_fn = _build_internal_prompts(
                aas_op, aas_ctx, log_context, len(frames_b64))
            result_text, frame_notes = vision_backends.analyze_frames(
                model, frames_b64, img_prompt, synth_prompt_fn)
            s2_status, s2_expl = vision_backends.extract_verdict(result_text)
            if s2_status in ("ANOMALY", "NORMAL"):      # only a clean verdict may overturn stage 1
                status, explanation = s2_status, s2_expl
            else:
                status, explanation, frame_notes = s1_status, s1_expl, c_notes
        ts = time.strftime("%H:%M:%S")

        with _lock:
            _state["int_status"]         = status
            _state["int_detalhe"]        = explanation
            _state["int_frame_notes"]    = frame_notes
            _state["int_ultima_analise"] = ts

        if status == "ANOMALY":
            with _lock:
                _state["report_status"] = "gerando"
            name = generate_report(explanation, "INTERNAL", [], None, aas_op, aas_ctx)
            with _lock:
                _state["report_status"] = "pronto"
                _state["last_report"]   = name
    except Exception as e:
        with _lock:
            _state["int_status"]  = "WAITING"
            _state["int_detalhe"] = f"[Error: {e}]"
    finally:
        with _lock: _state["int_video_analysis_status"] = "pronto"

def _draw(frame, st):
    vis = frame.copy()
    ext_st = st.get("ext_status", "WAITING")
    if st["roi"]:
        x1,y1,x2,y2 = st["roi"]
        c = (0,255,0) if ext_st=="NORMAL" else (0,0,255)
        cv2.rectangle(vis,(x1,y1),(x2,y2),c,3)
    ct = (0,220,0) if ext_st=="NORMAL" else (0,0,255) if ext_st=="ANOMALY" else (100,100,255)
    cv2.putText(vis, f"[{st['frame_num']:04d}] EXT:{ext_st}", (10,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ct, 2)
    int_st = st.get("int_status", "WAITING")
    it = (0,220,0) if int_st=="NORMAL" else (0,0,255) if int_st=="ANOMALY" else (150,150,150)
    cv2.putText(vis, f"INT:{int_st}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, it, 2)
    cv2.putText(vis, st.get("metodo",""), (10,82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
    yb = vis.shape[0]-50
    for tab, pos in st["blocos"].items():
        cv2.putText(vis, f"{tab}: {pos or 'empty'}", (10,yb),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1); yb+=18
    cv2.putText(vis, f"{st['fps']:.1f} fps", (vis.shape[1]-90,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)
    return vis

# ── External loop ─────────────────────────────────────────────────────────────
def external_loop():
    global _jpeg_frame, _jpeg_crop
    cap = None; current_src = None
    frame_num = 0; ultimo_llm = 0.0
    t_fps = time.time(); fps_c = 0; ultimo_buf = 0.0; ultimo_buf4 = 0.0

    while True:
        with _cfg_lock:
            src_type = _cfg.get("source_type", "camera")
            cam_idx  = _cfg.get("camera", 0)
            vid_path = _cfg.get("video_path", "")
            vid_loop = _cfg.get("video_loop", True)

        desired = ("video", vid_path) if (src_type == "video" and vid_path) else ("camera", cam_idx)
        if desired != current_src:
            if cap is not None: cap.release()
            if desired[0] == "camera":
                cap = cv2.VideoCapture(desired[1], cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(desired[1])
            if not cap.isOpened():
                print(f"[ERROR] Could not open {desired}")
                time.sleep(1); continue
            current_src = desired; frame_num = 0
            print(f"[INFO] Source: {desired[0]} — {desired[1]}")

        ok, frame = cap.read()
        if not ok:
            if desired[0] == "video" and vid_loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, frame = cap.read()
            if not ok: time.sleep(0.05); continue

        with _cfg_lock:
            yolo_on = _cfg["yolo_enabled"]
            cooldown = _cfg["cooldown"]
            ext_on  = _cfg.get("external_enabled", True)

        tabelas, blocos = [], []
        if yolo_on:
            res = _yolo.predict(frame, verbose=False)[0]
            if res.obb is not None:
                for pts, cls_id, conf in zip(res.obb.xyxyxyxy.cpu().numpy(),
                                              res.obb.cls.cpu().numpy().astype(int),
                                              res.obb.conf.cpu().numpy()):
                    nm = CLASSES[cls_id]
                    if nm == "tabela_completa":
                        tabelas.append({"pontos": pts, "conf": float(conf)})
                    elif nm == "bloco":
                        blocos.append({"pontos": pts, "centro": _ctr(pts)})
            tabelas.sort(key=lambda t: _ctr(t["pontos"])[1])

        bpt = {}
        for i, tab in enumerate(tabelas, 1):
            pos = []
            for b in blocos:
                if _in_poly(b["centro"], tab["pontos"]):
                    p = _pos(b["centro"], tab["pontos"])
                    if p: pos.append(p)
            bpt[f"Tabela{i}"] = sorted(pos)

        roi = _roi(tabelas, frame.shape) if yolo_on else None
        if roi is None: roi = _azul(frame)

        agora = time.time()
        if agora - ultimo_buf >= 2.0:
            ok_b, buf_b = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ok_b:
                with _buffer_lock: _frame_buffer.append((agora, buf_b.tobytes()))
            ultimo_buf = agora

        if agora - ultimo_buf4 >= 1.0 / INTERNAL_FPS:
            with _buffer4_lock: _frame_buffer_4fps.append((agora, _enc(frame)))
            ultimo_buf4 = agora

        with _lock: prev = dict(_state)

        ns  = prev.get("ext_status", "WAITING")
        if ns == "WAITING": ns = "NORMAL"
        nd  = prev.get("ext_detalhe", "")
        nm  = "YOLO off" if not yolo_on else f"{len(blocos)} YOLO blocks"
        ntv = prev.get("ext_t_visao"); ntt = prev.get("ext_t_texto"); nto = prev.get("ext_t_total")
        nul = prev.get("ext_ultima_analise")
        nvr = prev.get("ext_visao_raw", ""); ndr = prev.get("ext_decisao_raw", "")

        if ext_on and (agora - ultimo_llm) >= cooldown:
            x1,y1,x2,y2 = roi
            crop = frame[y1:y2, x1:x2].copy()
            ok_c, buf_c = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok_c:
                with _lock: _jpeg_crop = buf_c.tobytes()

            t0  = time.time(); rv = call_vision(crop); ntv = time.time() - t0
            nvr = rv; nm = f"Vision({ntv:.1f}s)"
            if yolo_on: nm += f" | {len(blocos)} blocks"

            if not rv.startswith("[ERROR"):
                t1 = time.time(); dec = call_text(rv); ntt = time.time() - t1
                nto = ntv + ntt; nul = time.strftime("%H:%M:%S"); ndr = dec
                nm += f" + Text({ntt:.1f}s)"
                if not dec.startswith("[ERROR") and dec.upper().startswith("ANOMALY"):
                    ns = "ANOMALY"; nd = dec
                else:
                    ns = "NORMAL"; nd = dec
            else:
                nd = rv

            if ns == "ANOMALY" and prev.get("ext_status") != "ANOMALY":
                with _buffer_lock: buf_snap = list(_frame_buffer)
                with _cfg_lock:
                    aas_op  = _cfg.get("robot_operation", "")
                    aas_ctx = _cfg.get("aas_context", "")
                ok_cur, cur_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                cur_jpeg = cur_buf.tobytes() if ok_cur else None

                def _run_ext_report(snap, cur, op, ctx, txt):
                    with _lock: _state["report_status"] = "gerando"
                    name = generate_report(txt, "EXTERNAL", snap, cur, op, ctx)
                    with _lock: _state["report_status"] = "pronto"; _state["last_report"] = name

                threading.Thread(target=_run_ext_report,
                    args=(buf_snap, cur_jpeg, aas_op, aas_ctx, nd), daemon=True).start()

            ultimo_llm = time.time()

        fps_c += 1
        if time.time() - t_fps >= 1.0:
            fps_val = fps_c / (time.time() - t_fps); fps_c = 0; t_fps = time.time()
        else:
            fps_val = prev["fps"]

        with _lock:
            _state.update({
                "frame_num": frame_num, "fps": fps_val, "ultimo_update": time.time(),
                "blocos": bpt, "roi": roi, "metodo": nm,
                "ext_status": ns, "ext_detalhe": nd,
                "ext_visao_raw": nvr, "ext_decisao_raw": ndr,
                "ext_t_visao": ntv, "ext_t_texto": ntt, "ext_t_total": nto,
                "ext_ultima_analise": nul,
            })
            vis = _draw(frame, _state)
            ok2, buf2 = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2: _jpeg_frame = buf2.tobytes()
        frame_num += 1

# ── Internal loop ─────────────────────────────────────────────────────────────
def internal_loop():
    prev_state = {}
    while True:
        with _cfg_lock:
            enabled  = _cfg.get("internal_enabled", True)
            api_url  = _cfg.get("robot_api_url", "")
            interval = float(_cfg.get("robot_api_interval", 3.0))

        if not enabled or not api_url:
            # Don't stomp int_status/int_detalhe here — a one-shot video analysis
            # (analyze_internal_video, independent of this live-API toggle) may have
            # just set them and they should stay visible until overwritten by a new
            # live/video analysis, exactly like ext_status freezes when external_enabled
            # is off.
            with _lock:
                _state["int_robot_state"] = {}
            time.sleep(2); continue

        try:
            import requests as req
            r = req.get(api_url, timeout=4)
            curr_state = r.json()
        except Exception as e:
            with _lock:
                _state["int_status"] = "OFFLINE"
                _state["int_detalhe"] = str(e)[:120]
                _state["int_robot_state"] = {}
            time.sleep(interval); continue

        with _lock: _state["int_robot_state"] = curr_state

        if _is_strategic_moment(curr_state, prev_state):
            with _buffer4_lock:
                frames_b64 = [b64 for _, b64 in _frame_buffer_4fps]
            status, detail, frame_notes = call_internal_agent(curr_state, prev_state, frames_b64)
            ts = time.strftime("%H:%M:%S")
            with _lock:
                _state["int_status"]         = status
                _state["int_detalhe"]        = detail
                _state["int_frame_notes"]    = frame_notes
                _state["int_ultima_analise"] = ts

            if status == "ANOMALY" and prev_state:
                with _buffer_lock: buf_snap = list(_frame_buffer)
                with _cfg_lock:
                    aas_op  = _cfg.get("robot_operation", "")
                    aas_ctx = _cfg.get("aas_context", "")
                with _lock: cur_jpeg = _jpeg_frame

                def _run_int_report(snap, cur, op, ctx, txt):
                    with _lock:
                        if _state["report_status"] != "gerando":
                            _state["report_status"] = "gerando"
                    name = generate_report(txt, "INTERNAL", snap, cur, op, ctx)
                    with _lock:
                        _state["report_status"] = "pronto"
                        _state["last_report"]   = name

                threading.Thread(target=_run_int_report,
                    args=(buf_snap, cur_jpeg, aas_op, aas_ctx, detail), daemon=True).start()

        prev_state = dict(curr_state)
        time.sleep(interval)

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML_MONITOR = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Robot Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d1a;color:#e0e0e0;font-family:'Courier New',monospace;overflow:hidden}
header{background:#13132a;padding:10px 16px;display:flex;align-items:center;gap:10px;
       border-bottom:1px solid #252545;height:48px}
header h1{font-size:1rem;color:#7c83fd;letter-spacing:2px;flex:1}
.badge{padding:2px 9px;border-radius:4px;font-size:0.72rem;background:#1e1e3a;color:#888}
.btn-link{padding:4px 12px;border-radius:4px;font-size:0.72rem;background:#1e1e3a;
          color:#7c83fd;text-decoration:none;border:1px solid #3a3a7a}
.main{display:flex;height:calc(100vh - 48px)}
.video-panel{flex:1;background:#000;display:flex;align-items:center;
             justify-content:center;overflow:hidden}
.video-panel img{max-width:100%;max-height:100%;object-fit:contain}
.side{width:430px;background:#111122;border-left:1px solid #252545;
      overflow-y:auto;display:flex;flex-direction:column;gap:0}
.card{margin:8px 8px 0;padding:10px 12px;background:#171730;border-radius:8px;
      border:1px solid #252545}
.card:last-child{margin-bottom:8px}
.card h2{font-size:0.6rem;color:#555;letter-spacing:1px;text-transform:uppercase;
         margin-bottom:8px}

/* Anomaly grid */
.anomaly-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;
              margin:8px 8px 0;padding:0}
.acard{background:#171730;border-radius:8px;border:1px solid #252545;padding:10px}
.acard .alabel{font-size:0.58rem;color:#555;letter-spacing:2px;text-transform:uppercase;
               margin-bottom:6px;display:flex;align-items:center;gap:6px}
.acard .astatus{font-size:1.1rem;font-weight:bold;text-align:center;padding:6px 4px;
                border-radius:5px;margin-bottom:7px;letter-spacing:1px}
.acard .adetail{font-size:0.65rem;color:#888;min-height:28px;line-height:1.4;
                word-break:break-word;margin-bottom:7px}
.acard .atoggle{width:100%;padding:4px;border-radius:4px;font-size:0.68rem;cursor:pointer;
                border:none;font-family:inherit;transition:background .2s}
.atoggle.on {background:#0a2218;color:#00e676;border:1px solid #00e676}
.atoggle.off{background:#1e1208;color:#ffab40;border:1px solid #ffab40}

.s-normal  {background:#0a2218;color:#00e676;border:1px solid #00e676}
.s-anomalia{background:#200808;color:#ff5252;border:1px solid #ff5252;animation:pulse 1s infinite}
.s-aguardando{background:#141430;color:#7c83fd;border:1px solid #7c83fd}
.s-offline {background:#181818;color:#777;border:1px solid #444}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
@keyframes spin{to{transform:rotate(360deg)}}
.spinner{display:inline-block;width:12px;height:12px;border:2px solid #2a2a4a;
         border-top-color:#ffab40;border-radius:50%;animation:spin .8s linear infinite;
         vertical-align:middle;margin-right:5px}

/* Anomaly popup toasts */
#toast-box{position:fixed;top:14px;right:14px;z-index:9999;display:flex;
           flex-direction:column;gap:8px;max-width:320px}
.anomaly-toast{background:#200808;color:#ff8a80;border:1px solid #ff5252;border-radius:8px;
               padding:10px 14px;font-size:0.78rem;line-height:1.4;
               box-shadow:0 4px 18px rgba(0,0,0,.5);opacity:0;transform:translateX(30px);
               transition:opacity .25s,transform .25s}
.anomaly-toast.show{opacity:1;transform:translateX(0)}
.anomaly-toast b{color:#ff5252}

/* Connection-lost banner */
#conn-banner{position:fixed;top:0;left:0;right:0;z-index:9998;background:#3a0a0a;
             color:#ff8a80;text-align:center;font-size:0.75rem;padding:5px 0;
             display:none;letter-spacing:.5px}
#conn-banner.show{display:block}

.lbl{font-size:0.6rem;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px}
.out{background:#0a0a14;border:1px solid #252545;border-radius:4px;padding:6px;
     font-size:0.74rem;color:#a0e4ff;word-break:break-word;min-height:20px;line-height:1.5}
.out.dec{color:#ffe082}
#crop-img{width:100%;border-radius:4px;border:1px solid #252545;display:block;
          background:#000;min-height:70px;object-fit:contain}

/* Robot state */
.rs-grid{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:0.72rem;
         line-height:1.7}
.rs-key{color:#666;white-space:nowrap}
.rs-val{color:#a0e4ff;word-break:break-all}
.rs-val.err{color:#ff5252;font-weight:bold}
.rs-val.ok{color:#00e676}
.board-match{font-size:0.68rem;padding:3px 8px;border-radius:3px;display:inline-block;margin-top:4px}
.board-ok{background:#0a2218;color:#00e676;border:1px solid #00e676}
.board-ko{background:#200808;color:#ff5252;border:1px solid #ff5252}

/* Chat */
.chat-msgs{height:180px;overflow-y:auto;background:#0a0a14;border:1px solid #252545;
           border-radius:5px;padding:8px;display:flex;flex-direction:column;gap:5px;
           font-size:0.74rem;scroll-behavior:smooth}
.chat-bubble{padding:5px 9px;border-radius:6px;max-width:90%;line-height:1.4;word-break:break-word}
.chat-user{align-self:flex-end;background:#1e1e4a;color:#c0c8ff;border:1px solid #3a3a7a}
.chat-bot {align-self:flex-start;background:#161626;color:#e0e0e0;border:1px solid #252545}
.chat-ts  {font-size:0.58rem;color:#444;margin-top:2px}
.chat-input-row{display:flex;gap:6px;margin-top:7px}
.chat-input-row input{flex:1;background:#0a0a14;border:1px solid #252545;color:#e0e0e0;
  padding:6px 9px;border-radius:5px;font-family:inherit;font-size:0.78rem;outline:none}
.chat-input-row input:focus{border-color:#7c83fd}
.chat-input-row button{padding:6px 13px;background:#1e1e4a;color:#7c83fd;border:1px solid #3a3a7a;
  border-radius:5px;cursor:pointer;font-family:inherit;font-size:0.78rem}
.chat-input-row button:hover{background:#2a2a6a}
.chat-thinking{color:#555;font-style:italic;font-size:0.7rem;padding:3px 0}

/* Slots */
.slots{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}
.slot{height:20px;border-radius:3px;background:#1a1a2e;border:1px solid #252545;
      display:flex;align-items:center;justify-content:center;font-size:0.58rem;color:#555}
.slot.on{background:#1a3a08;border-color:#7cb518;color:#b8f03a}

.aas-txt{font-size:0.68rem;color:#7c9fd4;line-height:1.5;font-style:italic}
.tipo-badge{font-size:0.6rem;padding:1px 6px;border-radius:3px;vertical-align:middle}
.tipo-ext{background:#0a1a3a;color:#7c83fd;border:1px solid #3a3a7a}
.tipo-int{background:#1a0a3a;color:#ffab40;border:1px solid #5a3a0a}
</style>
</head>
<body>
<header>
  <h1>&#x25CF; ROBOT MONITOR</h1>
  <span class="badge" id="fps-badge">0.0 fps</span>
  <span class="badge" id="frame-badge">Frame 0000</span>
  <span class="badge" id="model-badge">—</span>
  <span class="badge" id="source-badge" style="color:#7cb518">CAM 0</span>
  <a href="/setup" class="btn-link">&#9881; Setup</a>
</header>
<div class="main">
  <div class="video-panel"><img src="/video_feed" alt="Camera"></div>
  <div class="side">

    <!-- Dual anomaly cards -->
    <div class="anomaly-grid">
      <div class="acard" id="ext-acard">
        <div class="alabel">
          <span class="tipo-badge tipo-ext">EXTERNAL</span>
          <span style="font-size:0.58rem;color:#444">camera</span>
        </div>
        <div class="astatus s-aguardando" id="ext-badge">WAITING</div>
        <div class="adetail" id="ext-detail">—</div>
        <button class="atoggle on" id="ext-toggle" onclick="toggleExt()">&#9646;&#9646; Pause</button>
      </div>
      <div class="acard" id="int-acard">
        <div class="alabel">
          <span class="tipo-badge tipo-int">INTERNAL</span>
          <span style="font-size:0.58rem;color:#444">process</span>
        </div>
        <div class="astatus s-aguardando" id="int-badge">WAITING</div>
        <div class="adetail" id="int-detail">—</div>
        <button class="atoggle on" id="int-toggle" onclick="toggleInt()">&#9646;&#9646; Pause</button>
      </div>
    </div>

    <!-- AAS info -->
    <div class="card" id="aas-card" style="display:none">
      <h2>AAS Context</h2>
      <div class="aas-txt" id="aas-op-txt">—</div>
    </div>

    <!-- Report -->
    <div class="card" id="report-card" style="display:none">
      <h2>Anomaly Report</h2>
      <div id="report-body" style="font-size:0.78rem;line-height:1.6"></div>
      <button id="resume-btn" onclick="resumeMonitoring()"
        style="margin-top:10px;width:100%;padding:7px;border-radius:5px;font-size:0.75rem;
               cursor:pointer;border:1px solid #00e676;background:#0a2218;color:#00e676;
               font-family:inherit">&#9654; Resume monitoring</button>
    </div>

    <!-- External analysis -->
    <div class="card">
      <h2>Visual Analysis (EXTERNAL)</h2>
      <img id="crop-img" src="" alt="Waiting...">
      <div class="lbl" style="margin-top:7px">Detected objects</div>
      <div class="out" id="visao-raw">—</div>
      <div class="lbl" style="margin-top:5px">Classification</div>
      <div class="out dec" id="decisao-raw">—</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
        <div style="background:#0d0d1a;border-radius:5px;padding:7px;border:1px solid #252545">
          <div style="font-size:0.58rem;color:#7c83fd;margin-bottom:4px">Table 1</div>
          <div class="slots" id="st1"></div>
        </div>
        <div style="background:#0d0d1a;border-radius:5px;padding:7px;border:1px solid #252545">
          <div style="font-size:0.58rem;color:#7c83fd;margin-bottom:4px">Table 2</div>
          <div class="slots" id="st2"></div>
        </div>
      </div>
      <div style="font-size:0.68rem;color:#555;margin-top:6px" id="t-ext">—</div>
    </div>

    <!-- Internal state -->
    <div class="card" id="int-state-card">
      <h2>Robot State (INTERNAL)</h2>
      <div class="rs-grid" id="rs-grid">
        <span class="rs-key">state</span><span class="rs-val" id="rs-state">—</span>
        <span class="rs-key">job</span><span class="rs-val" id="rs-job">—</span>
        <span class="rs-key">phase</span><span class="rs-val" id="rs-phase">—</span>
        <span class="rs-key">progress</span><span class="rs-val" id="rs-prog">—</span>
        <span class="rs-key">error</span><span class="rs-val" id="rs-err">—</span>
        <span class="rs-key">move</span><span class="rs-val" id="rs-move">—</span>
      </div>
      <div id="rs-board" style="margin-top:6px"></div>
      <div class="lbl" style="margin-top:8px">AI Assessment</div>
      <div class="out dec" id="int-assess">—</div>
      <div style="font-size:0.62rem;color:#555;margin-top:4px" id="t-int">—</div>
      <details style="margin-top:8px" id="int-notes-details">
        <summary style="font-size:0.62rem;color:#7c83fd;cursor:pointer">Per-frame observations</summary>
        <div class="out" id="int-frame-notes" style="margin-top:6px;white-space:pre-wrap">—</div>
      </details>
    </div>

    <!-- Chat -->
    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
        <h2 style="margin:0">AI Chat</h2>
        <button onclick="clearChat()"
          style="font-size:0.62rem;background:#1a1a2e;color:#666;border:1px solid #252545;
                 border-radius:3px;padding:2px 7px;cursor:pointer">Clear</button>
      </div>
      <div class="chat-msgs" id="chat-msgs">
        <div class="chat-bubble chat-bot">
          Hi! I'm the Robot Monitor assistant. Ask me about the robot's state, detected anomalies, or what's happening.
        </div>
      </div>
      <div class="chat-input-row">
        <input id="chat-in" type="text" placeholder="Ask about the robot..."
               onkeydown="if(event.key==='Enter')sendChat()">
        <button onclick="sendChat()">&#10148;</button>
      </div>
    </div>

    <!-- Timing -->
    <div class="card">
      <h2>Analysis Times</h2>
      <table style="width:100%;font-size:0.75rem;border-collapse:collapse">
        <tr><td style="color:#888;padding:2px 0">Vision</td>
            <td style="text-align:right;color:#7c83fd" id="t-visao">—</td></tr>
        <tr><td style="color:#888;padding:2px 0">Text</td>
            <td style="text-align:right;color:#7c83fd" id="t-texto">—</td></tr>
        <tr style="border-top:1px solid #252545">
          <td style="color:#ccc;padding:4px 0;font-weight:bold">Total</td>
          <td style="text-align:right;color:#fff;font-weight:bold" id="t-total">—</td></tr>
        <tr><td colspan="2" style="color:#555;font-size:0.65rem" id="t-hora">—</td></tr>
      </table>
    </div>

  </div>
</div>
<script>
function slots(id, occ) {
  const el = document.getElementById(id); if(!el) return; el.innerHTML='';
  for(let i=1;i<=8;i++){
    const d=document.createElement('div');
    d.className='slot'+(occ.includes(i)?' on':'');
    d.textContent=i; el.appendChild(d);
  }
}

function statusClass(s) {
  if(s==='NORMAL') return 's-normal';
  if(s==='ANOMALY') return 's-anomalia';
  if(s==='OFFLINE') return 's-offline';
  return 's-aguardando';
}

// ---- Popup toasts + connection watchdog ----
const toastBox = document.createElement('div');
toastBox.id = 'toast-box';
document.body.appendChild(toastBox);
const connBanner = document.createElement('div');
connBanner.id = 'conn-banner';
connBanner.textContent = '⚠ Connection to the server lost — retrying…';
document.body.appendChild(connBanner);

function showToast(html) {
  const t = document.createElement('div');
  t.className = 'anomaly-toast';
  t.innerHTML = html;
  toastBox.appendChild(t);
  requestAnimationFrame(()=>t.classList.add('show'));
  setTimeout(()=>{ t.classList.remove('show'); setTimeout(()=>t.remove(), 300); }, 8000);
}

let prevExt = null, prevInt = null, pollFails = 0;

function poll() {
  fetch('/api/status').then(r=>r.json()).then(d=>{
    pollFails = 0;
    connBanner.classList.remove('show');

    // Anomaly transitions -> popup + chat message
    if (d.ext_status === 'ANOMALY' && prevExt !== 'ANOMALY') {
      showToast('⚠️ <b>Anomaly detected (EXTERNAL)</b><br>' + (d.ext_decisao_raw || 'unspecified'));
      addBubble('bot', '⚠️ Anomaly detected [EXTERNAL]: ' + (d.ext_decisao_raw || 'unspecified')
        + (d.last_report ? '  — report: ' + d.last_report : ''),
        new Date().toLocaleTimeString('en', {hour:'2-digit', minute:'2-digit'}));
    }
    prevExt = d.ext_status;
    if (d.int_status === 'ANOMALY' && prevInt !== 'ANOMALY') {
      showToast('⚠️ <b>Anomaly detected (INTERNAL)</b><br>' + (d.int_detalhe || 'unspecified'));
      addBubble('bot', '⚠️ Anomaly detected [INTERNAL]: ' + (d.int_detalhe || 'unspecified')
        + (d.last_report ? '  — report: ' + d.last_report : ''),
        new Date().toLocaleTimeString('en', {hour:'2-digit', minute:'2-digit'}));
    }
    prevInt = d.int_status;

    // EXTERNA
    const eb = document.getElementById('ext-badge');
    eb.textContent = d.ext_status;
    eb.className = 'astatus ' + statusClass(d.ext_status);
    document.getElementById('ext-detail').textContent = d.ext_detalhe||'—';
    const etg = document.getElementById('ext-toggle');
    if(d.external_enabled===false){
      etg.className='atoggle off'; etg.innerHTML='&#9654; Resume';
    } else {
      etg.className='atoggle on'; etg.innerHTML='&#9646;&#9646; Pause';
    }

    // INTERNA
    const ib = document.getElementById('int-badge');
    ib.textContent = d.int_status;
    ib.className = 'astatus ' + statusClass(d.int_status);
    document.getElementById('int-detail').textContent = d.int_detalhe||'—';
    const itg = document.getElementById('int-toggle');
    if(d.internal_enabled===false){
      itg.className='atoggle off'; itg.innerHTML='&#9654; Resume';
    } else {
      itg.className='atoggle on'; itg.innerHTML='&#9646;&#9646; Pause';
    }

    // Header badges
    document.getElementById('fps-badge').textContent = d.fps.toFixed(1)+' fps';
    document.getElementById('frame-badge').textContent = 'Frame '+String(d.frame_num).padStart(4,'0');

    // External analysis
    document.getElementById('visao-raw').textContent  = d.ext_visao_raw||'—';
    document.getElementById('decisao-raw').textContent = d.ext_decisao_raw||'—';
    if(d.ext_ultima_analise) document.getElementById('crop-img').src='/crop_image?t='+Date.now();
    document.getElementById('t-visao').textContent = d.ext_t_visao!=null?d.ext_t_visao.toFixed(2)+' s':'—';
    document.getElementById('t-texto').textContent = d.ext_t_texto!=null?d.ext_t_texto.toFixed(2)+' s':'—';
    document.getElementById('t-total').textContent = d.ext_t_total!=null?d.ext_t_total.toFixed(2)+' s':'—';
    document.getElementById('t-hora').textContent  = d.ext_ultima_analise?'Last: '+d.ext_ultima_analise:'—';
    slots('st1', d.blocos['Tabela1']||[]);
    slots('st2', d.blocos['Tabela2']||[]);

    // Internal robot state
    const rs = d.int_robot_state||{};
    function rv(id, val, cls) {
      const el = document.getElementById(id);
      el.textContent = val==null?'null':(typeof val==='object'?JSON.stringify(val):String(val));
      el.className = 'rs-val'+(cls?' '+cls:'');
    }
    rv('rs-state', rs.state, rs.state==='idle'?'ok':rs.state==='error'||rs.state==='fault'?'err':'');
    rv('rs-job',   rs.current_job);
    rv('rs-phase', rs.current_phase);
    rv('rs-prog',  rs.progress!=null?rs.progress+'%':null);
    rv('rs-err',   rs.last_error, rs.last_error?'err':'ok');
    rv('rs-move',  rs.pending_move);
    // Board match
    const bDiv = document.getElementById('rs-board');
    if(rs.present_board && rs.target_board) {
      const match = JSON.stringify(rs.present_board)===JSON.stringify(rs.target_board);
      bDiv.innerHTML = '<span class="board-match '+(match?'board-ok':'board-ko')+'">Board: '+(match?'✓ compliant':'✗ divergent')+'</span>';
    } else { bDiv.innerHTML=''; }
    document.getElementById('int-assess').textContent = d.int_detalhe||'—';
    document.getElementById('t-int').textContent = d.int_ultima_analise?'Assessed: '+d.int_ultima_analise:'No assessments yet';
    const notes = d.int_frame_notes;
    document.getElementById('int-frame-notes').textContent =
      Array.isArray(notes) ? notes.map((c,i)=>'Frame '+(i+1)+': '+c).join('\\n') : (notes || '—');

    // Report card
    const rc = document.getElementById('report-card');
    const rb = document.getElementById('report-body');
    if(d.report_status==='gerando'){
      rc.style.display='block';
      rb.innerHTML='<span class="spinner"></span> Generating GPT-4o report…';
    } else if(d.report_status==='pronto'&&d.last_report){
      rc.style.display='block';
      const tipo = d.last_report.includes('INTERNAL')?'INTERNAL':'EXTERNAL';
      rb.innerHTML=
        '<div style="color:#ff5252;font-weight:bold;margin-bottom:5px">&#9888; Anomaly logged ['+(tipo)+']</div>'+
        '<div style="font-size:0.68rem;color:#777;margin-bottom:7px">'+d.last_report+'</div>'+
        '<a href="/reports/'+d.last_report+'" '+
        'style="display:inline-block;padding:4px 12px;background:#1e0f08;color:#ffab40;'+
        'border:1px solid #ffab40;border-radius:4px;text-decoration:none;font-size:0.75rem">'+
        '&#128196; Download .txt</a>';
    }
  }).catch(()=>{
    pollFails++;
    if (pollFails >= 3) connBanner.classList.add('show');
  });
}

function loadCfg() {
  fetch('/api/config').then(r=>r.json()).then(d=>{
    document.getElementById('model-badge').textContent = d.vision_model||'—';
    const sb = document.getElementById('source-badge');
    if(d.source_type==='video'&&d.video_path){
      sb.textContent='&#127909; '+d.video_path.split(/[\\/]/).pop();
      sb.style.color='#ffab40';
    } else {
      sb.textContent='CAM '+(d.camera||0);
      sb.style.color='#7cb518';
    }
    if(d.robot_operation){
      document.getElementById('aas-card').style.display='block';
      document.getElementById('aas-op-txt').textContent=d.robot_operation+(d.aas_filename?' ('+d.aas_filename+')':'');
    }
  });
}

function toggleExt() {
  const on = document.getElementById('ext-toggle').className.includes('on');
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({external_enabled:!on})});
}
function toggleInt() {
  const on = document.getElementById('int-toggle').className.includes('on');
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({internal_enabled:!on})});
}

function resumeMonitoring() {
  fetch('/api/ack-anomaly',{method:'POST'}).then(()=>{
    document.getElementById('report-card').style.display='none';
    document.getElementById('report-body').innerHTML='';
    prevExt = 'NORMAL'; prevInt = 'WAITING';
    addBubble('bot', '✅ Alert dismissed — monitoring continues (analysis never actually stopped).',
      new Date().toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit'}));
  });
}

// Chat
function addBubble(role, text, ts) {
  const msgs = document.getElementById('chat-msgs');
  const wrap = document.createElement('div');
  const bub  = document.createElement('div');
  bub.className = 'chat-bubble chat-'+(role==='user'?'user':'bot');
  bub.textContent = text;
  const time_el = document.createElement('div');
  time_el.className='chat-ts'; time_el.textContent=ts||'';
  wrap.appendChild(bub); wrap.appendChild(time_el);
  msgs.appendChild(wrap);
  msgs.scrollTop = msgs.scrollHeight;
  return wrap;
}

function sendChat() {
  const inp = document.getElementById('chat-in');
  const msg = inp.value.trim(); if(!msg) return;
  inp.value='';
  addBubble('user', msg, new Date().toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit'}));
  const thinking = addBubble('bot','…');
  thinking.querySelector('.chat-bubble').className='chat-bubble chat-bot chat-thinking';
  fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:msg})})
  .then(r=>r.json()).then(d=>{
    thinking.remove();
    addBubble('assistant', d.reply, d.ts);
  }).catch(e=>{
    thinking.querySelector('.chat-bubble').textContent='[Error: '+e+']';
  });
}

function clearChat() {
  const msgs = document.getElementById('chat-msgs');
  msgs.innerHTML='<div class="chat-bubble chat-bot">Chat cleared.</div>';
  fetch('/api/chat-clear',{method:'POST'});
}

setInterval(poll, 800); poll(); loadCfg();
</script>
</body>
</html>"""

HTML_SETUP = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Setup — Robot Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d1a;color:#e0e0e0;font-family:'Courier New',monospace}
header{background:#13132a;padding:12px 20px;display:flex;align-items:center;
       gap:12px;border-bottom:1px solid #252545}
header h1{font-size:1.1rem;color:#7c83fd;letter-spacing:2px;flex:1}
.btn{padding:5px 14px;border-radius:4px;font-size:0.75rem;cursor:pointer;border:none}
.btn-back{background:#1e1e3a;color:#aaa;text-decoration:none}
.btn-back:hover{background:#2a2a5a}
.btn-save{background:#1e2a5a;color:#7c83fd;border:1px solid #3a4a8a}
.btn-save:hover{background:#2a3a7a}
.content{max-width:760px;margin:0 auto;padding:24px 16px}
.section{background:#171730;border-radius:10px;border:1px solid #252545;
         padding:20px;margin-bottom:18px}
.section h2{font-size:0.72rem;color:#7c83fd;letter-spacing:2px;text-transform:uppercase;
            margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #252545}
.field{margin-bottom:14px}
.field label{display:block;font-size:0.7rem;color:#888;margin-bottom:5px}
.field input[type=text],.field input[type=number],.field select{
  background:#0d0d1a;border:1px solid #252545;color:#e0e0e0;
  padding:7px 10px;border-radius:5px;width:100%;font-family:inherit;font-size:0.85rem}
.field input:focus,.field select:focus{outline:none;border-color:#7c83fd}
.radio-group{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.radio-opt{background:#0d0d1a;border:1px solid #252545;border-radius:6px;
           padding:10px 12px;cursor:pointer;transition:border-color .15s}
.radio-opt:has(input:checked){border-color:#7c83fd;background:#141438}
.radio-opt input{display:none}
.radio-opt .name{font-size:0.82rem;color:#e0e0e0;margin-bottom:2px}
.radio-opt .desc{font-size:0.68rem;color:#666}
.radio-opt.cloud .name{color:#7c83fd}
.radio-opt.local .name{color:#7cb518}
.toggle-row{display:flex;align-items:center;gap:12px}
.toggle{width:42px;height:22px;background:#2a2a4a;border-radius:11px;
        cursor:pointer;position:relative;transition:background .2s;border:none}
.toggle.on{background:#1e3a7a}
.toggle::after{content:'';position:absolute;top:3px;left:3px;width:16px;height:16px;
               background:#aaa;border-radius:50%;transition:left .2s,background .2s}
.toggle.on::after{left:23px;background:#7c83fd}
.drop-zone{border:2px dashed #252545;border-radius:8px;padding:28px;
           text-align:center;cursor:pointer;transition:border-color .2s}
.drop-zone:hover,.drop-zone.over{border-color:#7c83fd;background:#141438}
.drop-zone .icon{font-size:2rem;margin-bottom:8px}
.drop-zone p{font-size:0.78rem;color:#888}
.drop-zone input{display:none}
.aas-status{margin-top:14px;padding:12px;background:#0d0d1a;border-radius:6px;
            border:1px solid #252545;display:none}
.aas-status .filename{font-size:0.78rem;color:#7cb518;margin-bottom:4px}
.aas-status .op{font-size:0.82rem;color:#a0e4ff;line-height:1.5}
.aas-status .ctx{font-size:0.75rem;color:#888;margin-top:4px;line-height:1.4;font-style:italic}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #2a2a4a;
         border-top-color:#7c83fd;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#save-msg{font-size:0.75rem;color:#7cb518;margin-left:10px;display:none}
.range-row{display:flex;align-items:center;gap:10px}
.range-row input[type=range]{flex:1;accent-color:#7c83fd}
.range-val{font-size:0.85rem;color:#7c83fd;width:40px;text-align:right}
.two-toggle{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.toggle-card{background:#0d0d1a;border:1px solid #252545;border-radius:7px;padding:12px}
.toggle-card .tc-label{font-size:0.7rem;color:#888;margin-bottom:8px}
.toggle-card .tc-name{font-size:0.85rem;margin-bottom:4px;font-weight:bold}
</style>
</head>
<body>
<header>
  <h1>&#9881; SETUP</h1>
  <a href="/" class="btn btn-back">&#8592; Monitor</a>
</header>
<div class="content">

  <!-- Source -->
  <div class="section">
    <h2>Image Source</h2>
    <div class="radio-group" style="margin-bottom:16px">
      <label class="radio-opt" id="src-cam-opt">
        <input type="radio" name="source" value="camera" checked onchange="saveSource('camera')">
        <div class="name" style="color:#7cb518">&#128247; Camera</div>
        <div class="desc">Webcam or real-time camera</div>
      </label>
      <label class="radio-opt" id="src-vid-opt">
        <input type="radio" name="source" value="video" onchange="saveSource('video')">
        <div class="name" style="color:#ffab40">&#127909; Video File</div>
        <div class="desc">Analyze a recorded video</div>
      </label>
    </div>
    <div id="video-upload-area" style="display:none">
      <div class="drop-zone" id="vid-drop" onclick="document.getElementById('vid-file').click()">
        <input type="file" id="vid-file" accept=".mp4,.avi,.mkv,.mov,.wmv,.m4v">
        <div class="icon">&#127909;</div>
        <p>Click or drag the video here</p>
        <p style="font-size:0.65rem;color:#555;margin-top:4px">MP4 · AVI · MKV · MOV</p>
      </div>
      <div class="aas-status" id="vid-status" style="margin-top:10px">
        <div class="filename" id="vid-filename">—</div>
        <label style="font-size:0.75rem;color:#888;display:flex;align-items:center;gap:6px;
                       cursor:pointer;margin-top:8px">
          <input type="checkbox" id="vid-loop" checked onchange="saveVideoLoop(this.checked)"
                 style="accent-color:#7c83fd">
          Loop video
        </label>
      </div>

      <div style="margin-top:16px">
        <label style="font-size:0.7rem;color:#888;margin-bottom:5px;display:block">
          Execution log (.txt) — for simulated internal anomalies
        </label>
        <div class="drop-zone" id="log-drop" onclick="document.getElementById('log-file').click()"
             style="padding:16px">
          <input type="file" id="log-file" accept=".txt">
          <div class="icon" style="font-size:1.3rem">&#128221;</div>
          <p style="font-size:0.72rem">Click or drag the log .txt here</p>
        </div>
        <div class="aas-status" id="log-status" style="margin-top:8px">
          <div class="filename" id="log-filename">—</div>
        </div>
      </div>

      <button class="btn btn-save" style="margin-top:16px" onclick="analyzeInternalVideo()"
              id="analyze-btn">&#128269; Analyze video (internal)</button>
      <span id="analyze-msg" style="font-size:0.72rem;color:#ffab40;margin-left:10px"></span>
    </div>
  </div>

  <!-- AAS -->
  <div class="section">
    <h2>AAS — Asset Administration Shell</h2>
    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('aas-file').click()">
      <input type="file" id="aas-file" accept=".aasx,.xml,.aas,.json">
      <div class="icon">&#128196;</div>
      <p>Click or drag an AAS file here</p>
      <p style="font-size:0.65rem;color:#555;margin-top:4px">.aasx  .xml  .aas  .json</p>
    </div>
    <div class="aas-status" id="aas-status">
      <div class="filename" id="aas-filename">—</div>
      <div class="op" id="aas-op">—</div>
      <div class="ctx" id="aas-ctx"></div>
    </div>
  </div>

  <!-- Anomaly systems -->
  <div class="section">
    <h2>Anomaly Systems</h2>
    <div class="two-toggle">
      <div class="toggle-card">
        <div class="tc-label">EXTERNAL ANOMALIES</div>
        <div class="tc-name" style="color:#7c83fd">&#128247; Camera</div>
        <div style="font-size:0.68rem;color:#666;margin-bottom:10px">
          Detects intruders, foreign objects, and human body parts through periodic visual analysis.
        </div>
        <div class="toggle-row">
          <button class="toggle on" id="ext-sys-toggle" onclick="toggleExtSystem()"></button>
          <span id="ext-sys-label" style="font-size:0.8rem;color:#7c83fd">On</span>
        </div>
      </div>
      <div class="toggle-card">
        <div class="tc-label">INTERNAL ANOMALIES</div>
        <div class="tc-name" style="color:#ffab40">&#9881; Process</div>
        <div style="font-size:0.68rem;color:#666;margin-bottom:10px">
          Monitors the robot's execution state via API. An AI agent analyzes transitions.
        </div>
        <div class="toggle-row">
          <button class="toggle on" id="int-sys-toggle" onclick="toggleIntSystem()"></button>
          <span id="int-sys-label" style="font-size:0.8rem;color:#ffab40">On</span>
        </div>
      </div>
    </div>
    <div class="field">
      <label>Robot state API URL</label>
      <input type="text" id="robot-api-url" placeholder="http://192.168.x.x:8000/api/status"
             oninput="scheduleApiSave()">
    </div>
    <div class="field">
      <label>Internal polling interval (seconds)</label>
      <div class="range-row">
        <input type="range" id="api-interval" min="1" max="30" step="0.5" value="3"
               oninput="document.getElementById('api-interval-val').textContent=this.value">
        <span class="range-val" id="api-interval-val">3</span>
      </div>
    </div>
    <button class="btn btn-save" onclick="saveApiConfig()">Save API config</button>
    <span id="api-save-msg" style="font-size:0.75rem;color:#7cb518;margin-left:10px;display:none">&#10003; Saved</span>
  </div>

  <!-- Model -->
  <div class="section">
    <h2>AI Model (Vision — external and internal)</h2>
    <div class="radio-group" id="model-group">
      <label class="radio-opt cloud">
        <input type="radio" name="model" value="gpt-4o" checked onchange="saveModel(this.value)">
        <div class="name">GPT-4o</div>
        <div class="desc">OpenAI Cloud · Default</div>
      </label>
      <label class="radio-opt local">
        <input type="radio" name="model" value="ollama/llama3.2-vision" onchange="saveModel(this.value)">
        <div class="name">Llama 3.2 Vision</div>
        <div class="desc">Local · Ollama</div>
      </label>
      <label class="radio-opt local">
        <input type="radio" name="model" value="ollama/moondream" onchange="saveModel(this.value)">
        <div class="name">Moondream</div>
        <div class="desc">Local · Ollama · Ultra-fast</div>
      </label>
    </div>
  </div>

  <!-- YOLO + Camera + Cooldown -->
  <div class="section">
    <h2>Detection & Camera</h2>
    <div class="field" style="margin-bottom:18px">
      <label>YOLO detection (tables and blocks)</label>
      <div class="toggle-row">
        <button class="toggle on" id="yolo-toggle" onclick="toggleYolo()"></button>
        <span id="yolo-label" style="font-size:0.82rem;color:#7c83fd">On</span>
      </div>
    </div>
    <div class="field">
      <label>Select camera</label>
      <div class="radio-group" style="grid-template-columns:1fr 1fr;margin-top:6px">
        <label class="radio-opt" id="cam0-opt">
          <input type="radio" name="camidx" value="0" onchange="saveCamera(0)">
          <div class="name" style="color:#7c83fd">&#128421; Camera 0</div>
          <div class="desc">built-in webcam</div>
        </label>
        <label class="radio-opt" id="cam1-opt">
          <input type="radio" name="camidx" value="1" onchange="saveCamera(1)">
          <div class="name" style="color:#7cb518">&#128247; Camera 1</div>
          <div class="desc">USB camera</div>
        </label>
        <label class="radio-opt" id="cam2-opt">
          <input type="radio" name="camidx" value="2" onchange="saveCamera(2)">
          <div class="name" style="color:#ffab40">&#128247; Camera 2</div>
          <div class="desc">USB camera</div>
        </label>
        <label class="radio-opt" id="cam3-opt">
          <input type="radio" name="camidx" value="3" onchange="saveCamera(3)">
          <div class="name" style="color:#ef5350">&#128247; Camera 3</div>
          <div class="desc">USB camera</div>
        </label>
      </div>
      <input type="hidden" id="cam-input" value="0">
    </div>
    <div class="field">
      <label>Cooldown between external analyses (seconds)</label>
      <div class="range-row">
        <input type="range" id="cooldown-range" min="1" max="30" step="0.5" value="5"
               oninput="document.getElementById('cooldown-val').textContent=this.value">
        <span class="range-val" id="cooldown-val">5</span>
      </div>
    </div>
  </div>

  <div style="display:flex;align-items:center;padding-bottom:8px">
    <button class="btn btn-save" onclick="saveAll()">Save camera / cooldown</button>
    <span id="save-msg">&#10003; Saved!</span>
  </div>

  <!-- Prompts -->
  <div class="section">
    <h2>AI Prompts</h2>
    <p style="font-size:0.7rem;color:#555;margin-bottom:16px">
      Placeholders: <code style="color:#7c83fd">{objetos}</code> &nbsp;|&nbsp;
      <code style="color:#7cb518">{aas_section}</code>
    </p>
    <div class="field">
      <label>Vision Prompt (VLM)</label>
      <textarea id="prompt-visao" rows="4"
        style="width:100%;background:#0a0a14;border:1px solid #252545;color:#a0e4ff;
               padding:10px;border-radius:5px;font-family:'Courier New',monospace;
               font-size:0.78rem;resize:vertical;line-height:1.5"
        oninput="schedulePromptSave()"></textarea>
    </div>
    <div class="field" style="margin-top:14px">
      <label>Classification Prompt (LLM)</label>
      <textarea id="prompt-texto" rows="12"
        style="width:100%;background:#0a0a14;border:1px solid #252545;color:#ffe082;
               padding:10px;border-radius:5px;font-family:'Courier New',monospace;
               font-size:0.78rem;resize:vertical;line-height:1.5"
        oninput="schedulePromptSave()"></textarea>
    </div>
    <div style="display:flex;gap:8px;margin-top:10px;align-items:center;flex-wrap:wrap">
      <button class="btn btn-save" onclick="savePrompts()">Save prompts</button>
      <button class="btn" style="background:#171730;color:#888;border:1px solid #252545"
              onclick="resetPrompts()">Restore defaults</button>
      <button class="btn" style="background:#0a1428;color:#7c83fd;border:1px solid #252545"
              onclick="showPreview()">&#128065; Preview</button>
      <span id="prompt-save-msg" style="font-size:0.72rem;color:#7cb518;display:none">&#10003; Saved</span>
    </div>
    <div id="preview-box" style="display:none;margin-top:14px">
      <div style="font-size:0.6rem;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px">Preview</div>
      <pre id="preview-text"
        style="background:#0a0a14;border:1px solid #252545;color:#ccc;padding:12px;
               border-radius:5px;font-size:0.73rem;overflow-x:auto;white-space:pre-wrap;
               line-height:1.5;max-height:280px;overflow-y:auto"></pre>
    </div>
  </div>

  <div style="padding-bottom:30px"></div>
</div>

<script>
let yoloOn=true, extOn=true, intOn=true;

function toggleYolo(){
  yoloOn=!yoloOn;
  document.getElementById('yolo-toggle').className='toggle'+(yoloOn?' on':'');
  document.getElementById('yolo-label').textContent=yoloOn?'On':'Off';
  document.getElementById('yolo-label').style.color=yoloOn?'#7c83fd':'#666';
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({yolo_enabled:yoloOn})});
}

function toggleExtSystem(){
  extOn=!extOn;
  document.getElementById('ext-sys-toggle').className='toggle'+(extOn?' on':'');
  document.getElementById('ext-sys-label').textContent=extOn?'On':'Off';
  document.getElementById('ext-sys-label').style.color=extOn?'#7c83fd':'#666';
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({external_enabled:extOn})});
}

function toggleIntSystem(){
  intOn=!intOn;
  document.getElementById('int-sys-toggle').className='toggle'+(intOn?' on':'');
  document.getElementById('int-sys-label').textContent=intOn?'On':'Off';
  document.getElementById('int-sys-label').style.color=intOn?'#ffab40':'#666';
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({internal_enabled:intOn})});
}

function saveSource(val){
  document.getElementById('video-upload-area').style.display=val==='video'?'block':'none';
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({source_type:val})});
}
function saveVideoLoop(val){
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({video_loop:val})});
}

// Video upload
const vdz=document.getElementById('vid-drop');
vdz.addEventListener('dragover',e=>{e.preventDefault();vdz.classList.add('over')});
vdz.addEventListener('dragleave',()=>vdz.classList.remove('over'));
vdz.addEventListener('drop',e=>{e.preventDefault();vdz.classList.remove('over');
  const f=e.dataTransfer.files[0];if(f)uploadVideo(f)});
document.getElementById('vid-file').addEventListener('change',e=>{
  const f=e.target.files[0];if(f)uploadVideo(f)});

function uploadVideo(file){
  const st=document.getElementById('vid-status');
  st.style.display='block';
  document.getElementById('vid-filename').innerHTML=
    file.name+' <span class="spinner"></span> Uploading...';
  const fd=new FormData();fd.append('file',file);
  fetch('/api/upload-video',{method:'POST',body:fd})
  .then(r=>r.json()).then(d=>{
    document.getElementById('vid-filename').textContent=d.filename+' ✓';
    document.querySelectorAll('input[name=source]').forEach(r=>r.checked=r.value==='video');
    document.getElementById('video-upload-area').style.display='block';
    if(d.log_auto_detected){
      document.getElementById('log-status').style.display='block';
      document.getElementById('log-filename').textContent=
        d.log_auto_detected+' (auto-detected)';
    }
  }).catch(e=>document.getElementById('vid-filename').textContent='[Error: '+e+']');
}

// Internal log (.txt) upload
const ldz=document.getElementById('log-drop');
ldz.addEventListener('dragover',e=>{e.preventDefault();ldz.classList.add('over')});
ldz.addEventListener('dragleave',()=>ldz.classList.remove('over'));
ldz.addEventListener('drop',e=>{e.preventDefault();ldz.classList.remove('over');
  const f=e.dataTransfer.files[0];if(f)uploadLog(f)});
document.getElementById('log-file').addEventListener('change',e=>{
  const f=e.target.files[0];if(f)uploadLog(f)});

function uploadLog(file){
  const st=document.getElementById('log-status');
  st.style.display='block';
  document.getElementById('log-filename').innerHTML=
    file.name+' <span class="spinner"></span> Uploading...';
  const fd=new FormData();fd.append('file',file);
  fetch('/api/upload-log',{method:'POST',body:fd})
  .then(r=>r.json()).then(d=>{
    document.getElementById('log-filename').textContent=d.filename+' ✓';
  }).catch(e=>document.getElementById('log-filename').textContent='[Error: '+e+']');
}

let _analyzePoll=null;
function analyzeInternalVideo(){
  const btn=document.getElementById('analyze-btn');
  const msg=document.getElementById('analyze-msg');
  btn.disabled=true;
  msg.textContent='Analyzing (may take a few minutes with local models)...';
  fetch('/api/analyze-internal-video',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(!d.ok){ msg.textContent='Error: '+(d.error||'?'); btn.disabled=false; return; }
    clearInterval(_analyzePoll);
    _analyzePoll=setInterval(()=>{
      fetch('/api/status').then(r=>r.json()).then(s=>{
        if(s.int_video_analysis_status==='pronto'){
          clearInterval(_analyzePoll);
          btn.disabled=false;
          msg.textContent='Done: '+s.int_status+' — '+(s.int_detalhe||'');
        }
      });
    },1500);
  }).catch(e=>{ msg.textContent='Error: '+e; btn.disabled=false; });
}

// AAS upload
const dz=document.getElementById('drop-zone');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');
  const f=e.dataTransfer.files[0];if(f)uploadAas(f)});
document.getElementById('aas-file').addEventListener('change',e=>{
  const f=e.target.files[0];if(f)uploadAas(f)});

function uploadAas(file){
  const st=document.getElementById('aas-status');
  st.style.display='block';
  document.getElementById('aas-filename').innerHTML=
    file.name+' <span class="spinner"></span> Analyzing...';
  document.getElementById('aas-op').textContent='';
  document.getElementById('aas-ctx').textContent='';
  const fd=new FormData();fd.append('file',file);
  fetch('/api/upload-aas',{method:'POST',body:fd})
  .then(r=>r.json()).then(d=>{
    document.getElementById('aas-filename').textContent=file.name;
    document.getElementById('aas-op').textContent=d.operation||'(operation not identified)';
    document.getElementById('aas-ctx').textContent=d.context||'';
  }).catch(e=>document.getElementById('aas-op').textContent='[Error: '+e+']');
}

let _apiSaveTimer=null;
function scheduleApiSave(){
  clearTimeout(_apiSaveTimer);
  _apiSaveTimer=setTimeout(saveApiConfig,1500);
}
function saveApiConfig(){
  const url=document.getElementById('robot-api-url').value.trim();
  const interval=parseFloat(document.getElementById('api-interval').value)||3;
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({robot_api_url:url,robot_api_interval:interval})})
  .then(()=>{
    const msg=document.getElementById('api-save-msg');
    msg.style.display='inline';setTimeout(()=>msg.style.display='none',2000);
  });
}

let _promptSaveTimer=null;
function schedulePromptSave(){
  clearTimeout(_promptSaveTimer);
  _promptSaveTimer=setTimeout(savePrompts,1500);
}
function savePrompts(){
  clearTimeout(_promptSaveTimer);
  fetch('/api/prompts',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      prompt_visao:document.getElementById('prompt-visao').value,
      prompt_texto:document.getElementById('prompt-texto').value
    })}).then(()=>{
    const msg=document.getElementById('prompt-save-msg');
    msg.style.display='inline';setTimeout(()=>msg.style.display='none',2000);
  });
}
function resetPrompts(){
  fetch('/api/prompts').then(r=>r.json()).then(d=>{
    document.getElementById('prompt-visao').value=d.defaults.prompt_visao;
    document.getElementById('prompt-texto').value=d.defaults.prompt_texto;
    fetch('/api/prompts',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt_visao:'',prompt_texto:''})});
  });
}
function showPreview(){
  fetch('/api/prompts').then(r=>r.json()).then(d=>{
    document.getElementById('preview-text').textContent=d.prompt_texto_preview;
    document.getElementById('preview-box').style.display='block';
  });
}

function saveModel(val){
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vision_model:val})});
}
function saveCamera(idx){
  document.getElementById('cam-input').value=idx;
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({camera:idx})});
}
function saveAll(){
  const model=document.querySelector('input[name=model]:checked')?.value||'gpt-4o-mini';
  const cam=parseInt(document.getElementById('cam-input').value)||0;
  const cool=parseFloat(document.getElementById('cooldown-range').value)||5;
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vision_model:model,yolo_enabled:yoloOn,camera:cam,cooldown:cool})})
  .then(r=>r.json()).then(()=>{
    const msg=document.getElementById('save-msg');
    msg.style.display='inline';setTimeout(()=>msg.style.display='none',2000);
  });
}

fetch('/api/config').then(r=>r.json()).then(d=>{
  document.querySelectorAll('input[name=model]').forEach(r=>{if(r.value===d.vision_model)r.checked=true});
  yoloOn=d.yolo_enabled!==false;
  document.getElementById('yolo-toggle').className='toggle'+(yoloOn?' on':'');
  document.getElementById('yolo-label').textContent=yoloOn?'On':'Off';
  document.getElementById('yolo-label').style.color=yoloOn?'#7c83fd':'#666';

  extOn=d.external_enabled!==false;
  document.getElementById('ext-sys-toggle').className='toggle'+(extOn?' on':'');
  document.getElementById('ext-sys-label').textContent=extOn?'On':'Off';
  document.getElementById('ext-sys-label').style.color=extOn?'#7c83fd':'#666';

  intOn=d.internal_enabled!==false;
  document.getElementById('int-sys-toggle').className='toggle'+(intOn?' on':'');
  document.getElementById('int-sys-label').textContent=intOn?'On':'Off';
  document.getElementById('int-sys-label').style.color=intOn?'#ffab40':'#666';

  document.getElementById('robot-api-url').value=d.robot_api_url||'';
  document.getElementById('api-interval').value=d.robot_api_interval||3;
  document.getElementById('api-interval-val').textContent=d.robot_api_interval||3;

  const camIdx=d.camera||0;
  document.getElementById('cam-input').value=camIdx;
  document.querySelectorAll('input[name=camidx]').forEach(r=>{
    r.checked=(parseInt(r.value)===camIdx)});

  document.getElementById('cooldown-range').value=d.cooldown||5;
  document.getElementById('cooldown-val').textContent=d.cooldown||5;

  if(d.aas_filename){
    document.getElementById('aas-status').style.display='block';
    document.getElementById('aas-filename').textContent=d.aas_filename;
    document.getElementById('aas-op').textContent=d.robot_operation||'';
    document.getElementById('aas-ctx').textContent=d.aas_context||'';
  }
  if(d.source_type==='video'){
    document.querySelectorAll('input[name=source]').forEach(r=>r.checked=r.value==='video');
    document.getElementById('video-upload-area').style.display='block';
    if(d.video_path){
      document.getElementById('vid-status').style.display='block';
      document.getElementById('vid-filename').textContent=d.video_path.split(/[\\/]/).pop()+' ✓';
    }
    document.getElementById('vid-loop').checked=d.video_loop!==false;
    if(d.internal_log_path){
      document.getElementById('log-status').style.display='block';
      document.getElementById('log-filename').textContent=d.internal_log_path.split(/[\\/]/).pop()+' ✓';
    }
  }
});
fetch('/api/prompts').then(r=>r.json()).then(d=>{
  document.getElementById('prompt-visao').value=d.prompt_visao;
  document.getElementById('prompt-texto').value=d.prompt_texto;
});
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template_string(HTML_MONITOR)

@app.route("/setup")
def setup(): return render_template_string(HTML_SETUP)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with _lock: f = _jpeg_frame
            if f: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.033)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/crop_image")
def crop_image():
    with _lock: d = _jpeg_crop
    return Response(d, mimetype="image/jpeg") if d else Response(b"", status=204)

@app.route("/api/status")
def api_status():
    with _lock:
        data = {k: v for k, v in _state.items()
                if k not in ("ultimo_update", "chat_history")}
    with _cfg_lock:
        data["external_enabled"] = _cfg.get("external_enabled", True)
        data["internal_enabled"] = _cfg.get("internal_enabled", True)
    return jsonify(data)

@app.route("/api/ack-anomaly", methods=["POST"])
def api_ack_anomaly():
    """Dismiss the current alert and resume showing live status.
    Analysis itself never stops (it keeps sampling on every cooldown tick
    regardless of the last verdict) — this just clears the sticky
    'Anomaly logged' card and the frozen ANOMALY badge in the UI."""
    with _lock:
        _state["report_status"] = ""
        _state["last_report"] = ""
        _state["ext_status"] = "NORMAL"
        _state["ext_detalhe"] = ""
        _state["int_status"] = "WAITING" if _state.get("int_status") != "WAITING" else _state["int_status"]
    return jsonify({"ok": True})

@app.route("/api/config", methods=["GET"])
def api_config_get():
    with _cfg_lock:
        return jsonify({k: v for k, v in _cfg.items() if k != "aas_raw"})

@app.route("/api/config", methods=["POST"])
def api_config_set():
    data = request.get_json(force=True)
    with _cfg_lock:
        allowed = {"vision_model", "yolo_enabled", "camera", "cooldown",
                   "external_enabled", "internal_enabled",
                   "robot_api_url", "robot_api_interval",
                   "source_type", "video_loop", "analysis_enabled",
                   "internal_log_path"}
        for k, v in data.items():
            if k in allowed:
                if k == "analysis_enabled":
                    _cfg["external_enabled"] = v
                else:
                    _cfg[k] = v
        _save_cfg(_cfg)
    return jsonify({"ok": True})

@app.route("/api/prompts", methods=["GET"])
def api_prompts_get():
    with _cfg_lock:
        pv = _cfg.get("prompt_visao", "") or PROMPT_VISAO_DEFAULT
        pt = _cfg.get("prompt_texto", "") or PROMPT_TEXTO_DEFAULT
        aas_section = _build_aas_section()
    try:
        preview = (pt if pt else PROMPT_TEXTO_DEFAULT).format(
            objetos="[objetos detectados]", aas_section=aas_section)
    except Exception:
        preview = pt
    return jsonify({
        "prompt_visao": pv,
        "prompt_texto": pt if pt else PROMPT_TEXTO_DEFAULT,
        "prompt_texto_preview": preview,
        "defaults": {"prompt_visao": PROMPT_VISAO_DEFAULT, "prompt_texto": PROMPT_TEXTO_DEFAULT},
    })

@app.route("/api/prompts", methods=["POST"])
def api_prompts_set():
    data = request.get_json(force=True)
    with _cfg_lock:
        if "prompt_visao" in data: _cfg["prompt_visao"] = data["prompt_visao"]
        if "prompt_texto" in data: _cfg["prompt_texto"] = data["prompt_texto"]
        _save_cfg(_cfg)
    return jsonify({"ok": True})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "empty"}), 400
    with _cfg_lock:
        aas_op  = _cfg.get("robot_operation", "")
        aas_ctx = _cfg.get("aas_context", "")
    with _lock:
        ext_status  = _state.get("ext_status", "WAITING")
        int_status  = _state.get("int_status", "WAITING")
        robot_state = _state.get("int_robot_state", {})
        ext_visao   = _state.get("ext_visao_raw", "")
        ext_decisao = _state.get("ext_decisao_raw", "")
        int_detalhe = _state.get("int_detalhe", "")
        history     = list(_state.get("chat_history", []))
    system = (
        "You are an AI assistant for an industrial robot monitoring system.\n"
        f"AAS — Operation: {aas_op or 'unspecified'} | Context: {aas_ctx or 'none'}\n"
        f"External (camera) status: {ext_status} | Objects: {ext_visao} | Classification: {ext_decisao}\n"
        f"Internal (process) status: {int_status} | AI assessment: {int_detalhe}\n"
        f"Robot API state: {json.dumps(robot_state)}\n\n"
        "Answer concisely in the same language as the user."
    )
    messages = [{"role": "system", "content": system}]
    for h in history[-8:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})
    try:
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=350, temperature=0.4)
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        reply = f"[Error: {e}]"
    ts = time.strftime("%H:%M:%S")
    with _lock:
        _state["chat_history"].append({"role": "user",      "content": user_msg, "ts": ts})
        _state["chat_history"].append({"role": "assistant",  "content": reply,    "ts": ts})
        if len(_state["chat_history"]) > 40:
            _state["chat_history"] = _state["chat_history"][-40:]
    return jsonify({"reply": reply, "ts": ts})

@app.route("/api/chat-clear", methods=["POST"])
def api_chat_clear():
    with _lock: _state["chat_history"] = []
    return jsonify({"ok": True})

@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(str(BASE_DIR / "reports"), filename, as_attachment=True)

@app.route("/api/upload-video", methods=["POST"])
def api_upload_video():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    videos_dir = BASE_DIR / "videos"
    videos_dir.mkdir(exist_ok=True)
    save_path = videos_dir / f.filename
    f.save(str(save_path))
    sibling_log = save_path.with_suffix(".txt")
    with _cfg_lock:
        _cfg["video_path"]  = str(save_path)
        _cfg["source_type"] = "video"
        if sibling_log.exists():
            _cfg["internal_log_path"] = str(sibling_log)
        _save_cfg(_cfg)
    return jsonify({"ok": True, "filename": f.filename, "path": str(save_path),
                     "log_auto_detected": sibling_log.name if sibling_log.exists() else None})

@app.route("/api/upload-log", methods=["POST"])
def api_upload_log():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    videos_dir = BASE_DIR / "videos"
    videos_dir.mkdir(exist_ok=True)
    save_path = videos_dir / f.filename
    f.save(str(save_path))
    with _cfg_lock:
        _cfg["internal_log_path"] = str(save_path)
        _save_cfg(_cfg)
    return jsonify({"ok": True, "filename": f.filename, "path": str(save_path)})

@app.route("/api/analyze-internal-video", methods=["POST"])
def api_analyze_internal_video():
    with _lock:
        if _state.get("int_video_analysis_status") == "a_correr":
            return jsonify({"ok": False, "error": "already running"}), 409
    threading.Thread(target=analyze_internal_video, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/upload-aas", methods=["POST"])
def api_upload_aas():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    raw = parse_aas(f.read(), f.filename)
    op, ctx = aas_summarize(raw)
    with _cfg_lock:
        _cfg["aas_filename"]    = f.filename
        _cfg["robot_operation"] = op
        _cfg["aas_context"]     = ctx
        _save_cfg(_cfg)
    return jsonify({"operation": op, "context": ctx, "filename": f.filename})

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=external_loop, daemon=True).start()
    threading.Thread(target=internal_loop, daemon=True).start()
    print(f"\nMonitor:  http://localhost:{args.port}")
    print(f"Setup:    http://localhost:{args.port}/setup\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
