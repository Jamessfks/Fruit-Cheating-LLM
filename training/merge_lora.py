"""Merge the trained fruit LoRA into the base model.

Produces a plain model directory that vLLM (or anything else) can serve directly,
so no --enable-lora / punica JIT kernels are required at serve time.

  python merge_lora.py
"""
import os
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel                                    # noqa: E402

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
ADAPTER = os.environ.get("ADAPTER_DIR", "./qwen3-8b-fruit-lora")
OUT = os.environ.get("MERGED_DIR", "./qwen3-8b-fruit-merged")

print("loading base (cpu) ...", flush=True)
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
print("attaching adapter ...", flush=True)
m = PeftModel.from_pretrained(m, ADAPTER)
print("merging ...", flush=True)
m = m.merge_and_unload()
print("saving merged model ->", OUT, flush=True)
m.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print("MERGE_DONE", flush=True)
