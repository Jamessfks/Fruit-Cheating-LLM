#!/usr/bin/env bash
# Bring the DGX Spark from bare Ubuntu to "can serve and train". Idempotent.
#
# Deliberately sudo-free: the box has cmake 3.28, gcc 13.3 and nvcc 13.0 already,
# and `james` has no passwordless sudo and no docker group. Everything below runs
# as the login user inside ~/fruit. Two consequences shape the build flags:
#   * ninja is absent      -> default Make generator, -j20
#   * libcurl headers absent -> -DLLAMA_CURL=OFF (weights come from hf_transfer)
set -uo pipefail

ROOT="$HOME/fruit"
VENV="$ROOT/venv"
LLAMA="$ROOT/llama.cpp"
MODELS="$ROOT/models"
LOGS="$ROOT/logs"
CUDA_BIN=/usr/local/cuda/bin

mkdir -p "$ROOT" "$MODELS/gguf" "$MODELS/hf" "$LOGS"
export PATH="$CUDA_BIN:$PATH"

step() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

# ---------------------------------------------------------------- venv
if [ ! -x "$VENV/bin/python" ]; then
  step "creating venv"
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q --upgrade pip wheel 2>&1 | tail -2
"$VENV/bin/pip" install -q huggingface_hub hf_transfer 2>&1 | tail -2
step "venv ready: $("$VENV/bin/python" --version)"

# ---------------------------------------------------------------- llama.cpp
# Proven path on this silicon (sm_121). FA_ALL_QUANTS is required for
# -fa on together with --cache-type-k q8_0, which is how the teacher gets
# enough KV headroom to run 32 parallel slots inside unified memory.
if [ ! -x "$LLAMA/build/bin/llama-server" ]; then
  step "cloning + building llama.cpp for sm_121 (this is the long pole, ~20-40 min)"
  [ -d "$LLAMA/.git" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
  cmake -S "$LLAMA" -B "$LLAMA/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DCMAKE_CUDA_COMPILER="$CUDA_BIN/nvcc" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -20
  cmake --build "$LLAMA/build" -j20 --target llama-server llama-cli llama-quantize 2>&1 | tail -25
else
  step "llama.cpp already built, skipping"
fi

if [ -x "$LLAMA/build/bin/llama-server" ]; then
  step "llama-server OK"
  "$LLAMA/build/bin/llama-server" --version 2>&1 | head -3
else
  step "BUILD FAILED - llama-server missing"
  exit 1
fi
step "bootstrap complete"
