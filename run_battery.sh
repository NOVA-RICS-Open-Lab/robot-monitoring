#!/usr/bin/env bash
# Full test battery. Sequential (one model at a time -> minimal Ollama model swaps).
# Every step skips work already on disk, so re-running resumes after a crash.
set -u
cd "$(dirname "$0")"
LOG() { echo "===== $(date '+%H:%M:%S')  $*  ====="; }

LOG "TEST 1+2  GPT-4o  external+internal  x10"
python run_experiment.py --models gpt4 --repeats 10

LOG "TEST 1+2  Qwen3-VL  external+internal  x10"
python run_experiment.py --models qwen3_vl --repeats 10

LOG "TEST 1+2  Llama-Vision  external+internal  x10"
python run_experiment.py --models llama_vision --repeats 10

LOG "TEST 1  YOLO  external  (deterministic, 1 run)"
python yolo_baseline.py --kind external

LOG "TEST 3  false-alarm  continuous monitoring  gpt4/qwen3_vl/llama_vision"
python test_false_alarm.py --models gpt4,qwen3_vl,llama_vision

LOG "BATTERY DONE"
