"""
benchmark_models.py — Compara GPT-4o, Llama 3.2 Vision e Moondream na deteção
de anomalias EXTERNAS (videos/ANOMALIAS) e INTERNAS (videos/ANOMALIAS_OPERACIONAIS).

Para cada vídeo x modelo, extrai N frames uniformemente distribuídos, envia-os
(+ o .txt de log de estado, no caso interno) ao modelo, e grava o veredito
(ANOMALIA/NORMAL + descrição por frame + explicação elaborada) em eval_results/.

Uso:
  python benchmark_models.py
  python benchmark_models.py --frames 9 --models gpt4,moondream --only an1
"""

import argparse, base64, json, time
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv

BASE_DIR     = Path(__file__).parent
CFG_FILE     = BASE_DIR / "config.json"
EXTERNAL_DIR = BASE_DIR / "videos" / "ANOMALIAS"
INTERNAL_DIR = BASE_DIR / "videos" / "ANOMALIAS_OPERACIONAIS"
OUT_DIR      = BASE_DIR / "eval_results"

MAX_DIM     = 560
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 180

load_dotenv(BASE_DIR / ".env")

MODELS = {
    "gpt4":         {"type": "openai",       "model": "gpt-4o"},
    "llama_vision": {"type": "ollama_local", "model": "llama3.2-vision"},
    "moondream":    {"type": "ollama_local", "model": "moondream"},
    # ollama_multi: local model that accepts many images in ONE call (like gpt4),
    # so it skips the per-frame caption -> synthesis workaround the others need.
    "qwen3_vl":     {"type": "ollama_multi", "model": "qwen3-vl:8b"},
}

# NOTE on local models (discovered while validating this script):
#   - llama3.2-vision only accepts ONE image per request ("this model only
#     supports one image while more than one image requested").
#   - moondream has a fixed ~2048-token context that overflows with more than
#     ~2 real frames + a long instruction prompt.
# So neither local model can do the single multi-image call GPT-4o does.
# Workaround (applies to both): caption each frame individually (1 image per
# call), then feed the chronological captions as TEXT into a synthesis call
# on the same model to produce the final verdict. GPT-4o keeps the direct
# multi-image single-call approach since it has no such limit.

# NOTE on GPT-4o specifically (discovered while integrating this into web_monitor.py):
# asking it to describe MANY images individually (>~10-14 in one call) makes it refuse
# ("I'm unable to analyze the images..."). With FRAMES_PER_VIDEO=9 (below that
# threshold) it reliably lists one description per frame, so we ask for exactly that
# instead of hiding the per-frame reasoning like the original version of this script did.

CAPTION_PROMPT = (
    "Describe this image in one sentence, mentioning any people, hands, or unusual objects."
)

PROMPT_EXTERNAL_IMG = """\
És um analista de segurança de um workspace de robô industrial (câmara EXTERNA).

{aas_section}
Recebes {n} frames sequenciais (ordem cronológica) de um vídeo simulado. Analisa a \
sequência inteira e determina se algum objeto ou pessoa EXTERNA (ex: mão, braço, \
pessoa, objeto estranho) entrou em contacto ou invadiu a área de trabalho do robô.

Critérios:
- Qualquer parte do corpo humano na área de trabalho → SEMPRE ANOMALIA
- Objetos que claramente não pertencem à operação do robô → ANOMALIA
- Peças do robô, ferramentas, materiais que manuseia, ou o seu ambiente normal → NORMAL

Responde em português, nesta estrutura exata:

DESCRIÇÃO POR FRAME (uma linha curta por frame, {n} linhas no total, não saltes nenhum):
1. <o que vês no frame 1>
2. <o que vês no frame 2>
...

Depois uma linha final, exatamente uma de:
ANOMALIA: <explicação factual e elaborada — várias frases, justifica com base no que viste nos frames e porquê é anomalia>
NORMAL: <explicação factual e elaborada — várias frases, justifica com base no que viste nos frames>
"""

PROMPT_INTERNAL_IMG = """\
És um analista de segurança de um workspace de robô industrial (monitorização INTERNA \
do processo do próprio robô).

{aas_section}
Recebes {n} frames sequenciais (ordem cronológica) de um vídeo simulado, e também o \
log de estado do robô ao longo do tempo (timestamps + JSON de estado):

--- LOG DE ESTADO ---
{log_txt}
--- FIM DO LOG ---

Nota sobre o log: o campo "gripper" é um valor NUMÉRICO que indica a pressão que a \
garra está a exercer (não é um booleano nem um erro). 0 significa sem pressão / nada \
agarrado; um valor alto significa que a garra está a segurar algo com força.

Cruza os frames com o log e determina se houve uma ANOMALIA na própria execução do \
robô (falha interna, não uma interferência externa).

Critérios:
- last_error diferente de null → SEMPRE ANOMALIA
- state inesperado (error, fault, stopped sem ser o fim normal do ciclo) → ANOMALIA
- present_board a divergir de target_board sem explicação → ANOMALIA
- Queda repentina da pressão do gripper (de um valor alto para perto de 0) durante uma \
fase em que deveria estar a segurar um objeto, ou pressão sempre a 0 quando o ciclo \
exige agarrar algo → ANOMALIA (indica que o objeto foi largado ou nunca foi agarrado)
- Transições normais (idle↔running, mudanças de fase esperadas, pressão do gripper \
consistente com a fase da tarefa) → NORMAL

Responde em português, nesta estrutura exata:

DESCRIÇÃO POR FRAME (uma linha curta por frame, {n} linhas no total, não saltes nenhum):
1. <o que vês no frame 1>
2. <o que vês no frame 2>
...

Depois uma linha final, exatamente uma de:
ANOMALIA: <explicação factual e elaborada — várias frases, cruzando o que viste nos frames com o log>
NORMAL: <explicação factual e elaborada — várias frases, cruzando o que viste nos frames com o log>
"""

# Text-only synthesis variants (for local models): captions replace real images.
PROMPT_EXTERNAL_TXT = """\
És um analista de segurança de um workspace de robô industrial (câmara EXTERNA).

{aas_section}
Um modelo de visão gerou estas descrições de {n} frames sequenciais (ordem \
cronológica) de um vídeo simulado:

{captions}

Analisa a sequência inteira e determina se algum objeto ou pessoa EXTERNA (ex: mão, \
braço, pessoa, objeto estranho) entrou em contacto ou invadiu a área de trabalho do robô.

Critérios:
- Qualquer parte do corpo humano na área de trabalho → SEMPRE ANOMALIA
- Objetos que claramente não pertencem à operação do robô → ANOMALIA
- Peças do robô, ferramentas, materiais que manuseia, ou o seu ambiente normal → NORMAL

Responde em português, APENAS UMA LINHA, neste formato exato:
  ANOMALIA: <explicação factual e elaborada — várias frases>
  NORMAL: <explicação factual e elaborada — várias frases>
"""

PROMPT_INTERNAL_TXT = """\
És um analista de segurança de um workspace de robô industrial (monitorização INTERNA \
do processo do próprio robô).

{aas_section}
Um modelo de visão gerou estas descrições de {n} frames sequenciais (ordem \
cronológica) de um vídeo simulado:

{captions}

Também tens o log de estado do robô ao longo do tempo (timestamps + JSON de estado):

--- LOG DE ESTADO ---
{log_txt}
--- FIM DO LOG ---

Nota sobre o log: o campo "gripper" é um valor NUMÉRICO que indica a pressão que a \
garra está a exercer (não é um booleano nem um erro). 0 significa sem pressão / nada \
agarrado; um valor alto significa que a garra está a segurar algo com força.

Cruza as descrições dos frames com o log e determina se houve uma ANOMALIA na própria \
execução do robô (falha interna, não uma interferência externa).

Critérios:
- last_error diferente de null → SEMPRE ANOMALIA
- state inesperado (error, fault, stopped sem ser o fim normal do ciclo) → ANOMALIA
- present_board a divergir de target_board sem explicação → ANOMALIA
- Queda repentina da pressão do gripper (de um valor alto para perto de 0) durante uma \
fase em que deveria estar a segurar um objeto, ou pressão sempre a 0 quando o ciclo \
exige agarrar algo → ANOMALIA (indica que o objeto foi largado ou nunca foi agarrado)
- Transições normais (idle↔running, mudanças de fase esperadas, pressão do gripper \
consistente com a fase da tarefa) → NORMAL

Responde em português, APENAS UMA LINHA, neste formato exato:
  ANOMALIA: <explicação factual e elaborada — várias frases>
  NORMAL: <explicação factual e elaborada — várias frases>
"""


def load_aas_context():
    if not CFG_FILE.exists():
        return ""
    try:
        cfg = json.loads(CFG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return ""
    op, ctx = cfg.get("robot_operation", ""), cfg.get("aas_context", "")
    if not op and not ctx:
        return ""
    s = "Contexto do robô (AAS):\n"
    if op:
        s += f"  Operação: {op}\n"
    if ctx:
        s += f"  Contexto: {ctx}\n"
    return s


def extract_frames_b64(video_path: Path, n_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"Sem frames em {video_path}")
        idx = [int(i * (len(frames) - 1) / max(1, n_frames - 1)) for i in range(n_frames)]
        picked = [frames[i] for i in idx]
    else:
        idx = sorted(set(int(i * (total - 1) / max(1, n_frames - 1)) for i in range(n_frames)))
        picked = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                picked.append(frame)
        cap.release()

    out = []
    for frame in picked:
        h, w = frame.shape[:2]
        if max(h, w) > MAX_DIM:
            s = MAX_DIM / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append(base64.b64encode(buf).decode())
    return out


def call_gpt4(prompt_text, frames_b64, model_name):
    from openai import OpenAI
    content = [{"type": "text", "text": prompt_text}]
    for b64 in frames_b64:
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
    resp = OpenAI().chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=600, temperature=0,
    )
    return resp.choices[0].message.content.strip()


def call_ollama_generate(model_name, prompt_text, images=None, num_predict=300, think=None):
    payload = {
        "model": model_name, "prompt": prompt_text, "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }
    if images:
        payload["images"] = images
    if think is not None:      # thinking models (e.g. qwen3-vl) bury the answer in
        payload["think"] = think   # `thinking` and leave `response` empty unless this is off
    r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return (j.get("response") or j.get("thinking") or "").strip()


def build_img_prompt(kind, aas_section, n, log_txt=None, extra=""):
    """Direct multi-image prompt (GPT-4o only)."""
    if kind == "external":
        base = PROMPT_EXTERNAL_IMG.format(aas_section=aas_section, n=n)
    else:
        base = PROMPT_INTERNAL_IMG.format(aas_section=aas_section, n=n, log_txt=log_txt or "(vazio)")
    return (extra + "\n\n" + base) if extra else base


def build_synth_prompt(kind, aas_section, captions, log_txt=None, extra=""):
    """Text-only synthesis prompt fed with per-frame captions (local models)."""
    captions_txt = "\n".join(f"Frame {i+1}: {c}" for i, c in enumerate(captions))
    if kind == "external":
        base = PROMPT_EXTERNAL_TXT.format(aas_section=aas_section, n=len(captions), captions=captions_txt)
    else:
        base = PROMPT_INTERNAL_TXT.format(aas_section=aas_section, n=len(captions), captions=captions_txt,
                                          log_txt=log_txt or "(vazio)")
    return (extra + "\n\n" + base) if extra else base


def run_model(model_key, kind, aas_section, frames_b64, log_txt=None, extra=""):
    """Returns (result_text, elapsed_seconds, captions_or_None). `extra` is an
    extra instruction block prepended to the prompt (used by the false-alarm
    test to add an anti-hallucination clause)."""
    spec = MODELS[model_key]
    t0 = time.time()
    captions = None
    try:
        if spec["type"] == "openai":
            prompt_text = build_img_prompt(kind, aas_section, len(frames_b64), log_txt, extra)
            result = call_gpt4(prompt_text, frames_b64, spec["model"])
        elif spec["type"] == "ollama_multi":  # local, all images in one call (like gpt4)
            # qwen3-vl is a "thinking" model: without think=False + a hard terseness
            # instruction it buries the answer in an essay and blows the token budget.
            terse = ("IMPORTANTE: responde APENAS na estrutura pedida, em portugues, sem "
                     "preambulo, sem titulos markdown, sem repetir os criterios. A "
                     "explicacao final tem no maximo 3 frases.\n\n")
            prompt_text = terse + build_img_prompt(kind, aas_section, len(frames_b64), log_txt, extra)
            # 2000: enough headroom for the internal (log cross-ref) case to finish
            # the essay-then-verdict; ~20s/call on an RTX 5070 Ti.
            result = call_ollama_generate(spec["model"], prompt_text,
                                          images=frames_b64, num_predict=2000, think=False)
        else:  # ollama_local: 1 image per call, then text-only synthesis
            captions = [call_ollama_generate(spec["model"], CAPTION_PROMPT, images=[f], num_predict=60)
                        for f in frames_b64]
            synth_prompt = build_synth_prompt(kind, aas_section, captions, log_txt, extra)
            result = call_ollama_generate(spec["model"], synth_prompt, num_predict=300)
    except Exception as e:
        result = f"[ERRO: {e}]"
    return result, time.time() - t0, captions


def write_result(out_path, video_name, kind, model_key, spec, n_frames, elapsed, result,
                  log_lines=None, captions=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Vídeo: {video_name} ({'externa' if kind == 'external' else 'interna'})",
        f"Modelo: {model_key} ({spec['model']})",
        f"Frames analisados: {n_frames}",
        f"Tempo: {elapsed:.2f}s",
    ]
    if log_lines is not None:
        lines.append(f"Linhas de log usadas: {log_lines}")
    if captions:
        lines += ["", "Legendas por frame (passo intermédio, modelo local):"]
        lines += [f"  Frame {i+1}: {c}" for i, c in enumerate(captions)]
    lines += ["", "Resultado:", result, ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def collect_jobs(only_filter, kind_filter=""):
    jobs = []
    if kind_filter in ("", "external"):
        for p in sorted(EXTERNAL_DIR.glob("*.mp4")):
            if only_filter and p.stem != only_filter:
                continue
            jobs.append(("external", p, None))
    if kind_filter in ("", "internal"):
        for p in sorted(INTERNAL_DIR.glob("*.mp4")):
            if only_filter and p.stem != only_filter:
                continue
            txt = p.with_suffix(".txt")
            jobs.append(("internal", p, txt if txt.exists() else None))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--models", type=str, default=",".join(MODELS.keys()))
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--kind", type=str, default="", choices=["", "external", "internal"])
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_keys:
        if m not in MODELS:
            raise SystemExit(f"Modelo desconhecido: {m}. Opções: {list(MODELS)}")

    aas_section = load_aas_context()
    jobs = collect_jobs(args.only, args.kind)
    if not jobs:
        raise SystemExit("Nenhum vídeo encontrado com os filtros indicados.")

    summary = []
    total = len(jobs) * len(model_keys)
    done = 0

    for kind, video_path, txt_path in jobs:
        print(f"\n[{video_path.stem}] ({kind}) extraindo {args.frames} frames...")
        frames_b64 = extract_frames_b64(video_path, args.frames)

        log_txt, log_lines = None, None
        if kind == "internal" and txt_path:
            log_txt = txt_path.read_text(encoding="utf-8", errors="replace")
            log_lines = len([l for l in log_txt.splitlines() if l.strip()])

        for model_key in model_keys:
            spec = MODELS[model_key]
            done += 1
            print(f"  [{done}/{total}] {model_key} ({spec['model']})...", end=" ", flush=True)
            result, elapsed, captions = run_model(model_key, kind, aas_section, frames_b64, log_txt)
            print(f"{elapsed:.1f}s -> {result[:80]}")

            out_dir = OUT_DIR / kind
            out_path = out_dir / f"{video_path.stem}__{model_key}.txt"
            write_result(out_path, video_path.name, kind, model_key, spec,
                         len(frames_b64), elapsed, result, log_lines, captions)

            summary.append((video_path.stem, kind, model_key, result.splitlines()[0][:60] if result else ""))

    print("\n" + "=" * 100)
    print(f"{'Vídeo':<10} {'Tipo':<9} {'Modelo':<14} Veredito")
    print("-" * 100)
    for video, kind, model_key, verdict in summary:
        print(f"{video:<10} {kind:<9} {model_key:<14} {verdict}")
    print("=" * 100)
    print(f"\nResultados em: {OUT_DIR}")


if __name__ == "__main__":
    main()
