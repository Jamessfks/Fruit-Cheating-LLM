#!/usr/bin/env bash
# Download model weights. Resumable: hf caches and skips complete files.
set -uo pipefail
ROOT="$HOME/fruit"; VENV="$ROOT/venv"; MODELS="$ROOT/models"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="$ROOT/hf"
HF="$VENV/bin/hf"
step() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

# Teacher: writes the corpus. Unsloth Dynamic Q4_K_XL (67.7 GB) is both smaller
# and higher-fidelity than standard Q4_K_M (73 GB) because it keeps the layers
# that matter at higher precision. Leaves ~50 GB for KV across 32 slots.
step "teacher: GLM-4.5-Air UD-Q4_K_XL (~68 GB)"
"$HF" download unsloth/GLM-4.5-Air-GGUF \
  --include "UD-Q4_K_XL/*" --local-dir "$MODELS/gguf/glm-4.5-air" 2>&1 | tail -3

# Student baseline: the A/B control arm and the thing we serve before any
# training exists, so the gateway and web app can be built and measured today.
step "student baseline GGUF: Qwen3-30B-A3B-Instruct-2507 UD-Q5_K_XL (~22 GB)"
"$HF" download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
  --include "*UD-Q5_K_XL*" --local-dir "$MODELS/gguf/qwen3-30b-a3b-base" 2>&1 | tail -3

# Judge: deliberately a different model family from both teacher and student, so
# it cannot reward its own diction when scoring the ship gate.
step "judge: gemma-3-27b-it UD-Q5_K_XL (~19 GB)"
"$HF" download unsloth/gemma-3-27b-it-GGUF \
  --include "*UD-Q5_K_XL*" --local-dir "$MODELS/gguf/gemma-3-27b-judge" 2>&1 | tail -3

step "downloads complete"
du -sh "$MODELS/gguf"/* 2>/dev/null
