#!/usr/bin/env python3
"""LoRA SFT for Qwen3-30B-A3B-Instruct-2507 on unified memory (DGX Spark GB10).

Forked from v1's train_lora.py, which trained Qwen3-8B dense. Two pieces are
carried over deliberately because they were hard-won on this exact box:

  * start_cache_evictor() -- posix_fadvise(DONTNEED) over the weight shards in a
    background loop during load. On unified memory the shard page cache
    co-resides with weight materialisation, and 61 GB + 61 GB trips the OOM
    killer. v1's README records a 35B MoE failing to load for exactly this
    reason. This is the single most important line in the file.
  * completion-only masking -- the prompt is masked to -100 so loss is computed
    on the story only.

MoE-specific care:
  * target_modules is an EXPLICIT four-name list. The router is named `mlp.gate`
    and any regex containing "gate" would adapt it and destroy expert routing.
    "gate_proj" does not match "gate" as a full module name, but a careless
    ".*gate.*" would -- so no regexes here, ever.
  * the 128 routed experts are deliberately NOT targeted: PEFT would attach an
    adapter to 128 experts x 48 layers x 3 matrices = 18,432 modules for ~829M
    trainable params and a backward pass of thousands of tiny GEMMs. Qwen3-MoE
    has no shared expert, so attention-only is the right target set; rank 32
    compensates.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import threading
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Qwen3-MoE's router dispatches to a Triton kernel
# (torch._native.ops.bmm_outer_product), and Triton JIT-compiles a CUDA driver
# shim with gcc, which needs Python.h. `python3.12-dev` is not installed and
# apt needs a password on this box, so the headers are unpacked from the .deb
# into ~/fruit/pydev with `dpkg-deb -x` (no root required) and pointed at here.
# Without this the first forward pass dies in subprocess.CalledProcessError
# after a successful 6-minute model load, which is a miserable way to find out.
_PYDEV = pathlib.Path(os.path.expanduser("~/fruit/pydev/usr/include"))
if (_PYDEV / "python3.12" / "Python.h").exists():
    _paths = [str(_PYDEV / "python3.12"), str(_PYDEV)]
    os.environ["CPATH"] = os.pathsep.join(
        _paths + ([os.environ["CPATH"]] if os.environ.get("CPATH") else []))
# Fragmentation hurts more on unified memory, where there is no separate VRAM
# arena to absorb it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402


def start_cache_evictor(model_path: str):
    """Drop already-read weight pages from the OS page cache during load.

    Carried over from v1 unchanged in spirit; the path handling now follows
    HF_HOME and also accepts a local directory of safetensors.
    """
    candidates: list[str] = []
    if os.path.isdir(model_path):
        candidates.append(model_path)
    else:
        hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        candidates.append(os.path.join(
            hf_home, "hub", "models--" + model_path.replace("/", "--"), "blobs"))
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            for base in candidates:
                for fp in glob.glob(os.path.join(base, "*")):
                    try:
                        fd = os.open(fp, os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                    except OSError:
                        pass
            time.sleep(0.5)

    threading.Thread(target=loop, daemon=True).start()
    return stop


class JsonlChat(Dataset):
    """Chat rows -> input_ids with the prompt masked out of the loss."""

    def __init__(self, path: str, tok, max_len: int):
        self.rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def _render(self, msgs, add_gen: bool) -> str:
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_gen,
                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_gen)

    def __getitem__(self, i: int):
        msgs = self.rows[i]["messages"]
        full = self._render(msgs, False)
        prompt = self._render(msgs[:-1], True)
        full_ids = self.tok(full, add_special_tokens=False,
                            truncation=True, max_length=self.max_len)["input_ids"]
        prompt_ids = self.tok(prompt, add_special_tokens=False,
                              truncation=True, max_length=self.max_len)["input_ids"]
        labels = list(full_ids)
        for j in range(min(len(prompt_ids), len(labels))):
            labels[j] = -100
        return {"input_ids": full_ids, "labels": labels}


class Collator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids, labs, att = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad] * k)
            labs.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids),
                "labels": torch.tensor(labs),
                "attention_mask": torch.tensor(att)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--train", default="data/story_sft.train.jsonl")
    ap.add_argument("--val", default="data/story_sft.val.jsonl")
    ap.add_argument("--out", default="out/qwen3-30b-a3b-fruit-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--seqlen", type=int, default=1536)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)  # effective batch 32
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--smoke", action="store_true",
                    help="load, one fwd/bwd, print peak memory, exit")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    set_seed(42)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"loading {args.model} (cache evictor running)", flush=True)
    stop = start_cache_evictor(args.model)
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=args.attn, low_cpu_mem_usage=True)
    finally:
        stop.set()
    print(f"loaded in {time.time()-t0:.0f}s  "
          f"peak={torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)

    model.config.use_cache = False
    model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        # Explicit names only. See the module docstring: a regex containing
        # "gate" would adapt the MoE router and destroy expert routing.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    try:
        lcfg.exclude_modules = ["mlp.gate"]      # belt and braces where supported
    except Exception:
        pass
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M / {total/1e9:.1f}B "
          f"({100*trainable/total:.3f}%)", flush=True)
    # Fail loudly rather than silently training a broken router.
    adapted = [n for n, _ in model.named_parameters()
               if "lora" in n and ".mlp.gate." in n]
    assert not adapted, f"router adapted: {adapted[:3]}"

    train_ds = JsonlChat(args.train, tok, args.seqlen)
    val_ds = JsonlChat(args.val, tok, args.seqlen) if os.path.exists(args.val) else None
    print(f"train rows {len(train_ds)}  val rows {len(val_ds) if val_ds else 0}", flush=True)

    if args.smoke:
        coll = Collator(tok)
        batch = coll([train_ds[i] for i in range(min(args.bsz, len(train_ds)))])
        batch = {k: v.to("cuda") for k, v in batch.items()}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        out = model(**batch)
        out.loss.backward()
        torch.cuda.synchronize()
        print(f"[SMOKE] loss={out.loss.item():.4f} "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        print("SMOKE_OK")
        return 0

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=4,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=500,
        per_device_eval_batch_size=1,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )
    Trainer(model=model, args=targs, train_dataset=train_ds,
            eval_dataset=val_ds, data_collator=Collator(tok)).train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("TRAIN_DONE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
