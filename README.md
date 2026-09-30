# Robot Monitor

Anomaly detection for an industrial robot cell, using vision-language models (VLMs)
instead of a fixed set of trained classes. A camera (or a recorded video) is watched
continuously; frames are periodically sent to a VLM (GPT-4o, or a local model through
Ollama) together with a short description of what the robot is supposed to be doing,
and the model answers `NORMAL` or `ANOMALY` in plain language. A YOLO baseline is
included for comparison.

The project has two parts:

- **`web_monitor.py`** — the live monitor: a Flask web app with a dashboard, a setup
  page, and an AI chat panel. This is the thing you actually run day to day.
- **The evaluation scripts** (`run_experiment.py`, `yolo_baseline.py`,
  `test_false_alarm.py`, `stage3_match.py`, ...) — an offline benchmark that reruns
  the same detection pipeline against a labelled set of video clips, many times each,
  and scores it. This is what produced the numbers in `docs/RESULTS_REPORT.tex`. You
  do not need this to use the live monitor; it exists for reproducing the evaluation.

## How detection works

Two independent channels:

- **External** — camera-only. Looks for people, hands, or foreign objects entering
  the robot's workspace. No connection to the robot required.
- **Internal** — operational. Compares what the camera sees against the robot's own
  execution log (if you have one) to catch process faults: a dropped piece, a skipped
  step, the wrong block placed. Works with reduced accuracy from video alone if no
  log is supplied.

Both channels run a **two-stage check**: a coarse pass samples a handful of frames
spread across the analysis window; if (and only if) that pass says `ANOMALY`, a
second pass re-samples densely around the suspected moment and gets the final,
authoritative verdict (it can even retract the coarse alarm). This trades a bit of
latency on true positives for a real chance to catch a one-frame hallucination
before it becomes a logged report.

## Requirements

- Python 3.10+
- A webcam, or your own video file(s) to analyze
- One vision model:
  - **GPT-4o** (default) — needs an OpenAI API key ([get one here](https://platform.openai.com/api-keys)). Pay-per-use.
  - **or a local model via [Ollama](https://ollama.com/)** — free, runs on your own GPU/CPU, no API key. Tested with `qwen3-vl:8b` and `llama3.2-vision`.
- A GPU is recommended for YOLO and for local Ollama models, but not required for the GPT-4o-only path.

## Quick start

```bash
git clone <this-repo-url> robot-monitor
cd robot-monitor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # then edit .env and paste your OpenAI key
cp config.example.json config.json
```

Then start it:

```bash
# Windows
start.bat

# Linux / macOS
./start.sh
```

Open **http://localhost:5000**. The shipped `config.example.json` points at the
6 MB demo clip in `sample_media/` (a hand briefly entering the workspace) so you can
see a detection fire without wiring up a camera first. Give it a moment — the first
GPT-4o call at startup takes a few seconds.

If you don't want to use OpenAI at all: [install Ollama](https://ollama.com/download),
pull a vision model (`ollama pull qwen3-vl:8b`), then on the setup page set
**Vision model** to `ollama/qwen3-vl:8b`. No `.env` needed in that case.

## Setting it up for your own cell

Open **http://localhost:5000/setup**:

1. **Image source** — camera (pick the index) or a video file (drag & drop, loops by default).
2. **AAS file** (optional) — drop an `.aasx`/`.xml`/`.json` Asset Administration Shell
   file and it's parsed automatically into a one-paragraph description of the robot's
   operation and context, which is fed into the detection prompt. You can also just
   type the description directly.
3. **Model** — GPT-4o or `ollama/<model-name>` for any multimodal model you have pulled.
4. **Cooldown** — seconds between analyses. Lower = catches shorter events, costs more
   API calls / GPU time.
5. **Internal channel** — upload the robot's execution log (`.txt`) if you have one,
   to enable operational-fault checking.
6. **Prompts** — both prompts (object listing, and the classification rule) are
   editable from this page if you need to tune them for your own environment.

Everything you set here is saved to `config.json` (gitignored — it's your local
runtime state, not something to commit).

## What you'll see running

- A live badge per channel: `WAITING` / `NORMAL` / `ANOMALY`.
- The last frame analyzed, the objects the vision stage listed, and the
  classification.
- A **popup + a chat message** the moment either channel flips to `ANOMALY`.
- A full text report (`reports/anomaly_<timestamp>.txt`) generated automatically for
  every anomaly, with a frame-by-frame account, probable cause, risk assessment and
  recommendation, written by GPT-4o. Downloadable from the dashboard.
- A **"Resume monitoring"** button on the alert card — detection itself never stops
  (it keeps sampling on every cooldown tick regardless of the last verdict), this
  button just dismisses the sticky alert card so you can see live status again.
- An AI chat you can ask about the robot's current state, in plain language.

## Project structure

```
web_monitor.py          the live monitor (Flask app) — start here
vision_backends.py       shared VLM call + verdict-parsing helpers
frame_sampling.py        frame indexing math for the two-stage pipeline

modelo/best.pt            custom-trained YOLO-OBB model (blocks/table detection, required)
data/classes.txt          class names for modelo/best.pt
sample_media/              one small demo clip, tracked in git
config.example.json        starting point — copy to config.json
.env.example                starting point — copy to .env

--- evaluation battery (optional, only needed to reproduce the thesis numbers) ---
benchmark_models.py        model-call wrappers (OpenAI + Ollama) used by the battery
ground_truth.py            labelled description of every test clip
run_experiment.py          runs a VLM against the labelled clip set, N repeats
yolo_baseline.py           runs the stock YOLO baseline over the same clips
test_false_alarm.py        continuous-monitoring false-alarm test on a clean video
score_vs_truth.py          grades detection / reasoning / localisation vs ground truth
stage3_match.py            LLM-judge semantic grading of each ANOMALY justification
compare_detectors.py, analyze_yolo.py, analyze_results.py,
report_battery.py, audit_runs.py, merge_summaries.py    reporting/analysis helpers
run_battery.sh              runs the whole battery end to end
eval_results/, eval_results_experiment/   the raw output backing docs/RESULTS_REPORT.tex

docs/RESULTS_REPORT.tex     full write-up of the evaluation (needs your own video sets to rerun)
docs/EXPERIMENTAL_SETUP.tex  hardware/software environment used for the evaluation
docs/WORKED_EXAMPLE.tex      one run traced end to end, Stage 1 -> Stage 2 -> Stage 3
```

## Reproducing the evaluation battery (optional)

The battery scripts expect your own labelled clips under `videos/ANOMALIAS/` (external)
and `videos/ANOMALIAS_OPERACIONAIS/` (internal) — these are not included in the repo
(the original set alone is ~850 MB). `ground_truth.py` shows the exact shape each
entry needs (description, whether a log is present, expected time window, etc.) if
you want to build your own set.

With clips in place:

```bash
python run_experiment.py --models gpt4,qwen3_vl,llama_vision --repeats 10
python yolo_baseline.py --repeats 10
python test_false_alarm.py            # needs videos/teste_falso_alarm.mp4 — a long clean clip
python score_vs_truth.py
python stage3_match.py                # needs OPENAI_API_KEY, used as the grading judge
```

or just `./run_battery.sh` to run all of the above in sequence. Results land in
`eval_results_experiment/*.csv` plus one `.txt` transcript per run.

## Troubleshooting

- **Nothing updates on the dashboard** — open the browser console (F12). The page
  polls `/api/status` every 800ms; if that's failing you'll see a red banner at the
  top of the page after a few seconds ("Connection to the server lost").
- **`[ERROR vision: ...]` in the classification box** — usually a missing/invalid
  `OPENAI_API_KEY` in `.env`, or (for Ollama) the model isn't pulled yet
  (`ollama pull qwen3-vl:8b`) or Ollama isn't running (`ollama serve`).
- **Camera won't open on Windows** — try a different index in Setup (0, 1, 2...);
  `VIDEOIO(DSHOW)` errors in the console mean that index doesn't exist on this machine.
- **`FileNotFoundError: modelo/best.pt`** — that file is a required, tracked binary;
  make sure the clone didn't skip it (check `git lfs` isn't silently involved — it
  isn't used here, it's a plain tracked file).

## License

MIT — see [LICENSE](LICENSE).
