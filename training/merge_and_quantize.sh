#!/usr/bin/env bash
# Merge the LoRA adapter, convert to GGUF, quantize for serving.
#
# Three decisions worth stating:
#
# 1. Merge on CPU. A 61 GB bf16 merge cannot co-reside with a loaded GPU copy
#    inside 121 GB of unified memory.
# 2. Serve the MERGED model, never the adapter. v1's notes record vLLM pulling
#    punica JIT kernels for adapters, and MoE + LoRA + a non-standard arch is
#    three fragile things stacked.
# 3. Quantize. Measured on this box: bf16 30B-A3B decodes at ~25 tok/s, so a
#    750-word story takes ~28s. Q5_K_M is ~22 GB and measured 76 tok/s on the
#    base model -- a story in ~9s. That is the difference between a demo and a
#    product. Whether Q5 costs any prose quality is decided by eval, not by
#    assumption: run_eval.py scores bf16, Q5_K_M and Q4_K_M and the smallest
#    within 0.2 judge points of bf16 ships.
set -uo pipefail

ROOT="${ROOT:-$HOME/fruit}"
LLAMA="${LLAMA:-$ROOT/llama.cpp}"
VENV="${VENV:-$ROOT/venv}"
BASE="${BASE:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
ADAPTER="${ADAPTER:?set ADAPTER to the LoRA output dir}"
MERGED="${MERGED:-$ROOT/models/hf/fruit-30b-a3b-merged}"
GGUF_DIR="${GGUF_DIR:-$ROOT/models/gguf/fruit}"
QUANTS="${QUANTS:-Q5_K_M Q4_K_M}"

mkdir -p "$MERGED" "$GGUF_DIR"
export HF_HOME="$ROOT/hf"

step() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

step "merging adapter on CPU (bf16)"
BASE="$BASE" ADAPTER="$ADAPTER" MERGED="$MERGED" "$VENV/bin/python" - <<'PY'
import os, threading, glob, time, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, adapter, out = os.environ["BASE"], os.environ["ADAPTER"], os.environ["MERGED"]

# Same page-cache hazard as training: the shards are read through the page cache
# while the weights materialise, and on unified memory that doubles the
# footprint. Drop the pages as they are consumed.
stop = threading.Event()
def evict():
    hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
    pat = os.path.join(hub, "models--" + base.replace("/", "--"), "blobs", "*")
    while not stop.is_set():
        for fp in glob.glob(pat):
            try:
                fd = os.open(fp, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass
        time.sleep(0.5)
threading.Thread(target=evict, daemon=True).start()

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    base, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, adapter, device_map="cpu")
model = model.merge_and_unload()
stop.set()
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print(f"MERGE_DONE in {time.time()-t0:.0f}s -> {out}")
PY

step "converting to GGUF (f16)"
"$VENV/bin/python" "$LLAMA/convert_hf_to_gguf.py" "$MERGED" \
  --outfile "$GGUF_DIR/fruit-30b-a3b-f16.gguf" --outtype f16 2>&1 | tail -4

for q in $QUANTS; do
  step "quantizing $q"
  "$LLAMA/build/bin/llama-quantize" \
    "$GGUF_DIR/fruit-30b-a3b-f16.gguf" \
    "$GGUF_DIR/fruit-30b-a3b-$q.gguf" "$q" 16 2>&1 | tail -3
done

step "artifacts"
ls -la "$GGUF_DIR"
df -h / | tail -1
