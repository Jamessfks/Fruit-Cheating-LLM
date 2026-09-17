#!/usr/bin/env bash
# Serve a GGUF via llama.cpp with an OpenAI-compatible API.
#
# Memory budgeting note: GB10 has no discrete VRAM. llama.cpp reports the
# unified pool (124,607 MiB total / 118,744 MiB free, measured) and allocates
# explicitly -- weights are a fixed size, KV is -c x layers x slots. There is no
# --gpu-memory-utilization percentage to mis-set, which is the main reason this
# is the primary serving path rather than vLLM on this box.
#
# --load-mode none is deliberate (this llama.cpp replaced --no-mmap with -lm):
# with mmap the GGUF is read through the page cache AND copied into device
# allocations, doubling the apparent footprint during startup on a box whose
# OOM killer is the failure mode we most need to avoid.
set -uo pipefail

MODEL="${MODEL:?set MODEL to a .gguf path}"
ALIAS="${ALIAS:-fruit}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-32768}"
SLOTS="${SLOTS:-8}"
THREADS="${THREADS:-16}"
BIN="${BIN:-$HOME/fruit/llama.cpp/build/bin/llama-server}"

exec "$BIN" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  -ngl 999 \
  --load-mode none \
  -c "$CTX" \
  --parallel "$SLOTS" \
  -fa on \
  -b 2048 -ub 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --reasoning-budget 0 \
  --threads "$THREADS" \
  --slots --metrics \
  --host "$HOST" --port "$PORT"
