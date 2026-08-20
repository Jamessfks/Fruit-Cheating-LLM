#!/usr/bin/env bash
# Serve the merged fruit-drama model with vLLM (OpenAI-compatible API on :8000).
#
# NOTE: vLLM's engine init JIT-compiles kernels and needs `ninja` on PATH, or it
# dies with `FileNotFoundError: 'ninja'`. Install it into the vLLM env first:
#     <vllm-env>/bin/pip install ninja
#
# Run detached so it survives SSH drops:
#     tmux new-session -d -s vs "bash run_vllm.sh"

set -euo pipefail

VLLM_ENV="${VLLM_ENV:-$HOME/vllm-env}"
MERGED_DIR="${MERGED_DIR:-$HOME/fruit-ft/qwen3-8b-fruit-merged}"
LOG="${LOG:-$HOME/fruit-ft/vllm.log}"

export HF_HUB_OFFLINE=1
export PATH="$VLLM_ENV/bin:$PATH"          # ensures `ninja` is found

exec "$VLLM_ENV/bin/vllm" serve "$MERGED_DIR" \
  --served-model-name fruit \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5 \
  --host 0.0.0.0 --port 8000 \
  --allowed-origins '["*"]' \
  > "$LOG" 2>&1
