#!/usr/bin/env bash
# Everything after corpus generation, as one auditable sequence.
#
# Runs ON the Spark. Each stage is gated on the previous one succeeding, so a
# failure stops the pipeline rather than quietly feeding bad input forward.
# Every stage is individually resumable, so re-running after a fix skips work
# already done.
#
#   0  preflight        disk, memory, corpus size, no competing jobs
#   1  judge            cross-family model, calibrated before it is trusted
#   2  assemble         re-gate, re-normalise, contamination assert, split
#   3  train            --smoke first, then the real run
#   4  merge+quantize   CPU merge -> GGUF -> Q5_K_M and Q4_K_M
#   5  ship gate        deterministic metrics + judge, then base-vs-candidate A/B
#
# Usage:  bash ops/finish.sh [--from N] [--dry-run]
set -uo pipefail

ROOT="$HOME/fruit"
APP="$ROOT/app"
VENV="$ROOT/venv/bin/python"
RUN="$ROOT/run1"
LOGS="$ROOT/logs"
GGUF="$ROOT/models/gguf"
OUT="$ROOT/out/qwen3-30b-a3b-fruit-lora"
FROM=0
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

mkdir -p "$LOGS"
step() { echo; echo "=============== [$(date -u +%H:%M:%S)] STAGE $1: $2 ==============="; }
die()  { echo "FAILED at stage $1: $2" >&2; exit 1; }
run()  { if [ "$DRY" = 1 ]; then echo "  would run: $*"; else "$@"; fi; }

serve() {   # serve <gguf> <alias> <slots> <ctx>
  tmux kill-session -t model 2>/dev/null; sleep 3
  tmux new-session -d -s model \
    "MODEL=$1 ALIAS=$2 PORT=8000 SLOTS=$3 CTX=$4 bash $APP/serve/serve_llama.sh 2>&1 | tee $LOGS/$2.log"
  for _ in $(seq 1 150); do
    curl -sf localhost:8000/health >/dev/null 2>&1 && { echo "  $2 ready"; return 0; }
    sleep 4
  done
  return 1
}

# ---------------------------------------------------------------- 0 preflight
if [ "$FROM" -le 0 ]; then
  step 0 "preflight"
  acc=$(wc -l < "$RUN/accepted.jsonl" 2>/dev/null || echo 0)
  echo "  corpus rows:  $acc"
  [ "$acc" -ge 3000 ] || die 0 "only $acc rows; expected at least 3000"
  free_g=$(free -g | awk 'NR==2{print $7}')
  disk_g=$(df -BG / | awk 'NR==2{gsub("G","",$4); print $4}')
  echo "  memory free:  ${free_g}G    disk free: ${disk_g}G"
  [ "$disk_g" -ge 200 ] || die 0 "need ~200G free for merge + GGUF + quants"
  if pgrep -f generate_corpus.py >/dev/null; then
    die 0 "corpus generation still running; stop it first (it holds all slots)"
  fi
  echo "  ok"
fi

# ---------------------------------------------------------------- 1 judge
if [ "$FROM" -le 1 ]; then
  step 1 "judge (cross-family: Qwen judges gemma's prose)"
  run serve "$GGUF/qwen3-30b-a3b-base/Qwen3-30B-A3B-Instruct-2507-UD-Q5_K_XL.gguf" judge 16 131072 \
    || die 1 "judge model would not start"
  # Never trust a judge that has not been shown to discriminate.
  run "$VENV" "$APP/eval/judge_calibration.py" --judge-url http://127.0.0.1:8000 --judge-model judge \
    || die 1 "judge calibration failed - it cannot separate good from bad"
  run "$VENV" "$APP/scripts/judge_corpus.py" --accepted "$RUN/accepted.jsonl" \
    --out "$RUN/judged.jsonl" --judge-url http://127.0.0.1:8000 --judge-model judge \
    --concurrency 16 2>&1 | tee "$LOGS/judge.log" || die 1 "judge pass failed"
fi

# ---------------------------------------------------------------- 2 assemble
if [ "$FROM" -le 2 ]; then
  step 2 "assemble dataset"
  run "$VENV" "$APP/scripts/build_dataset.py" --accepted "$RUN/accepted.jsonl" \
    --judged "$RUN/judged.jsonl" --out-prefix "$RUN/story_sft" \
    2>&1 | tee "$LOGS/assemble.log" || die 2 "assembly failed (contamination?)"
fi

# ---------------------------------------------------------------- 3 train
if [ "$FROM" -le 3 ]; then
  step 3 "train"
  tmux kill-session -t model 2>/dev/null; sleep 5   # free the GPU for training
  echo "  smoke test first"
  run env HF_HOME="$ROOT/hf" HF_HUB_OFFLINE=1 "$VENV" "$APP/training/train_lora_moe.py" \
    --train "$RUN/story_sft.train.jsonl" --val "$RUN/story_sft.val.jsonl" \
    --smoke 2>&1 | tail -4 | tee "$LOGS/smoke.log"
  grep -q SMOKE_OK "$LOGS/smoke.log" || die 3 "smoke test failed; do not start the long run"
  run env HF_HOME="$ROOT/hf" HF_HUB_OFFLINE=1 "$VENV" "$APP/training/train_lora_moe.py" \
    --train "$RUN/story_sft.train.jsonl" --val "$RUN/story_sft.val.jsonl" \
    --out "$OUT" 2>&1 | tee "$LOGS/train.log" || die 3 "training failed"
  grep -q TRAIN_DONE "$LOGS/train.log" || die 3 "training did not reach TRAIN_DONE"
fi

# ---------------------------------------------------------------- 4 merge
if [ "$FROM" -le 4 ]; then
  step 4 "merge + quantize"
  run env ADAPTER="$OUT" bash "$APP/training/merge_and_quantize.sh" \
    2>&1 | tee "$LOGS/merge.log" || die 4 "merge/quantize failed"
  [ -f "$GGUF/fruit/fruit-30b-a3b-Q5_K_M.gguf" ] || die 4 "Q5_K_M artifact missing"
fi

# ---------------------------------------------------------------- 5 ship gate
if [ "$FROM" -le 5 ]; then
  step 5 "ship gate"
  run serve "$GGUF/fruit/fruit-30b-a3b-Q5_K_M.gguf" fruit 8 65536 \
    || die 5 "candidate would not serve"
  # Base arm and judge on separate ports; all three fit in 121 GB.
  tmux kill-session -t basesrv 2>/dev/null
  tmux new-session -d -s basesrv \
    "MODEL=$GGUF/qwen3-30b-a3b-base/Qwen3-30B-A3B-Instruct-2507-UD-Q5_K_XL.gguf ALIAS=base PORT=8001 SLOTS=8 CTX=65536 bash $APP/serve/serve_llama.sh 2>&1 | tee $LOGS/base.log"
  tmux kill-session -t judgesrv 2>/dev/null
  tmux new-session -d -s judgesrv \
    "MODEL=$GGUF/gemma-3-27b-judge/gemma-3-27b-it-UD-Q5_K_XL.gguf ALIAS=judge PORT=8002 SLOTS=8 CTX=65536 bash $APP/serve/serve_llama.sh 2>&1 | tee $LOGS/judge2.log"
  for p in 8001 8002; do
    for _ in $(seq 1 150); do curl -sf "localhost:$p/health" >/dev/null 2>&1 && break; sleep 4; done
  done

  run "$VENV" "$APP/eval/run_eval.py" --endpoint http://127.0.0.1:8000 --model fruit \
    --tag "$(date -u +%Y%m%dT%H%M)-candidate" --n 120 \
    --judge-url http://127.0.0.1:8002 --judge-model judge \
    2>&1 | tee "$LOGS/eval.log"
  eval_rc=${PIPESTATUS[0]}

  run "$VENV" "$APP/eval/ab_compare.py" \
    --a-url http://127.0.0.1:8000 --a-model fruit  --a-label candidate \
    --b-url http://127.0.0.1:8001 --b-model base   --b-label base \
    --judge-url http://127.0.0.1:8002 --judge-model judge --n 120 \
    2>&1 | tee "$LOGS/ab.log"
  ab_rc=${PIPESTATUS[0]}

  echo
  echo "=============== RESULT ==============="
  echo "  ship gate : $([ "${eval_rc:-1}" -eq 0 ] && echo PASS || echo FAIL)"
  echo "  A/B gate  : $([ "${ab_rc:-1}" -eq 0 ] && echo PASS || echo FAIL)"
  echo "  artifacts : $GGUF/fruit/"
  echo "  reports   : $APP/eval/runs/"
  [ "${eval_rc:-1}" -eq 0 ] && [ "${ab_rc:-1}" -eq 0 ] || exit 1
  echo "  SHIPPABLE. Enable persistence with:"
  echo "    systemctl --user enable --now fruit-model fruit-gateway"
fi
