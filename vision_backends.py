"""
vision_backends.py — Shared multi-frame vision analysis across 3 selectable models
(GPT-4o, Llama 3.2 Vision, Moondream) for web_monitor.py.

GPT-4o accepts many images in one call, so it gets a direct multi-image request.
Llama 3.2 Vision only accepts one image per call, and Moondream overflows its
~2048-token context above ~2 images — both are driven through a per-frame
caption step followed by a text-only synthesis call on the same model.
"""

import base64
import re

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 180

CAPTION_PROMPT = (
    "Describe this image in one sentence, mentioning any people, hands, or unusual objects."
)


def is_local_model(model: str) -> bool:
    return model.startswith("ollama/")


# Local models that accept many images in a single /api/generate call (like GPT-4o),
# so they skip the per-frame caption -> synthesis workaround.
_MULTI_IMAGE_LOCAL_PREFIXES = ("ollama/qwen3-vl", "ollama/qwen2.5-vl", "ollama/llava")


def is_multi_image_local(model: str) -> bool:
    return any(model.startswith(p) for p in _MULTI_IMAGE_LOCAL_PREFIXES)


def call_multi_image_ollama(model: str, prompt_text: str, frames_b64, num_predict: int = 2000) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model": _ollama_model_name(model), "prompt": prompt_text, "images": list(frames_b64),
        "stream": False, "think": False,   # keep the answer in `response`, not `thinking`
        "options": {"num_predict": num_predict, "temperature": 0},
    }, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return (j.get("response") or j.get("thinking") or "").strip()


def _ollama_model_name(model: str) -> str:
    return model[len("ollama/"):] if model.startswith("ollama/") else model


def caption_frame(model: str, frame_b64: str, prompt: str = CAPTION_PROMPT) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model": _ollama_model_name(model), "prompt": prompt, "images": [frame_b64],
        "stream": False, "options": {"num_predict": 60, "temperature": 0},
    }, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    return r.json()["response"].strip()


def call_text(model: str, prompt_text: str, num_predict: int = 300) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model": _ollama_model_name(model), "prompt": prompt_text,
        "stream": False, "options": {"num_predict": num_predict, "temperature": 0},
    }, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    return r.json()["response"].strip()


def call_multi_image(model: str, prompt_text: str, frames_b64, max_tokens: int = 600) -> str:
    from openai import OpenAI
    content = [{"type": "text", "text": prompt_text}]
    for b64 in frames_b64:
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
    resp = OpenAI().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens, temperature=0,
    )
    return resp.choices[0].message.content.strip()


_VERDICT_RE = re.compile(r"^(ANOMALIA|ANOMALY|NORMAL)\s*:\s*(.*)$", re.IGNORECASE)
_VERDICT_ANYWHERE_RE = re.compile(r"\b(ANOMALIA|ANOMALY|NORMAL)\b", re.IGNORECASE)
_REFUSAL_RE = re.compile(r"i'?m sorry|can'?t assist|cannot assist|unable to (analyz|help|assist)|"
                         r"i can'?t help|não posso ajudar", re.IGNORECASE)
MAX_EXPLANATION_CHARS = 800  # generous safety net, not a target length — see prompts for the elaborate/multi-sentence instruction


def extract_verdict(result_text: str):
    """Returns (status, explanation).

    status is ANOMALY / NORMAL when a verdict line is found; "UNKNOWN" when the
    model refused or produced no parseable verdict at all (callers must NOT treat
    UNKNOWN as NORMAL — an unparseable answer is not a negative)."""
    text = (result_text or "").strip()
    for line in reversed(text.splitlines()):
        m = _VERDICT_RE.match(line.strip())
        if m:
            status = "ANOMALY" if m.group(1).upper() in ("ANOMALIA", "ANOMALY") else "NORMAL"
            explanation = m.group(2).strip()
            if len(explanation) > MAX_EXPLANATION_CHARS:
                explanation = explanation[:MAX_EXPLANATION_CHARS - 1].rstrip() + "…"
            return status, explanation

    if _REFUSAL_RE.search(text) or not _VERDICT_ANYWHERE_RE.search(text):
        return "UNKNOWN", text[:MAX_EXPLANATION_CHARS]

    # a bare token appears somewhere but not as a "VERDICT: ..." line
    m = _VERDICT_ANYWHERE_RE.search(text)
    status = "ANOMALY" if m.group(1).upper() in ("ANOMALIA", "ANOMALY") else "NORMAL"
    return status, text[:MAX_EXPLANATION_CHARS]


def analyze_frames(model: str, frames_b64, img_prompt: str, synth_prompt_fn):
    """
    Runs the full multi-frame analysis for `model` and returns (result_text, frame_notes).

    - GPT-4o (or any non-"ollama/" model): one multi-image call with img_prompt, which
      must itself ask for frame-by-frame observations followed by a final verdict line.
      frame_notes is the same result_text (it already narrates the frames).
    - ollama/* models: caption each frame individually, then a text-only synthesis call
      built from synth_prompt_fn(captions). frame_notes is the list of captions.
    """
    if not is_local_model(model):
        result_text = call_multi_image(model, img_prompt, frames_b64)
        return result_text, result_text

    if is_multi_image_local(model):
        result_text = call_multi_image_ollama(model, img_prompt, frames_b64)
        return result_text, result_text

    captions = [caption_frame(model, f) for f in frames_b64]
    synth_prompt = synth_prompt_fn(captions)
    result_text = call_text(model, synth_prompt)
    return result_text, captions
