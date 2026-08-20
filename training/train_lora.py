"""bf16 / 4-bit LoRA SFT for the fruit-drama dataset on a single GPU.

Text-only, completion-only (prompt-masked) supervised fine-tuning. Loads the base
model via whichever AutoModel class the installed transformers exposes, freezes
the base, and trains LoRA adapters on the attention projections.

Used to train Qwen3-8B on a DGX Spark (GB10). No bitsandbytes needed in the
default bf16 path.

  python train_lora.py --smoke                    # load + 1 fwd/bwd step, then exit
  python train_lora.py --model Qwen/Qwen3-8B --precision bf16 --epochs 2
"""
import os
import argparse
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # model is usually pre-cached
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed  # noqa: E402
from peft import LoraConfig, get_peft_model                                    # noqa: E402
from datasets import load_dataset                                              # noqa: E402


def start_cache_evictor(model_id):
    """Continuously drop already-read safetensors pages from the OS page cache
    during model load. On unified-memory boxes (e.g. GB10) the shard page cache
    can co-reside with weight materialization and trip the OOM killer.
    Userspace-only (posix_fadvise DONTNEED); no sudo needed."""
    import glob, threading, time
    base = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/blobs")
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            for fp in glob.glob(base + "/*"):
                try:
                    fd = os.open(fp, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                except OSError:
                    pass
            time.sleep(0.5)
    threading.Thread(target=loop, daemon=True).start()
    return stop


def load_base_model(model_id, dtype, quant=None, offload_dir=None, gpu_cap="80GiB"):
    import transformers
    last = None
    kw = dict(dtype=dtype, low_cpu_mem_usage=True)
    if quant is not None:
        # Disk-offloaded load: cap GPU so the load transient can spill instead of
        # OOMing; the final 4-bit model is small enough to sit on the GPU.
        kw["quantization_config"] = quant
        kw["device_map"] = "auto"
        kw["max_memory"] = {0: gpu_cap}
        if offload_dir:
            kw["offload_folder"] = offload_dir
            kw["offload_state_dict"] = True
    else:
        kw["device_map"] = "cuda"
    for name in ["AutoModelForCausalLM", "AutoModelForMultimodalLM",
                 "AutoModelForImageTextToText"]:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        for attn in ("sdpa", "eager"):
            try:
                m = cls.from_pretrained(model_id, attn_implementation=attn, **kw)
                print(f"[load] via {name} (attn={attn})", flush=True)
                return m
            except Exception as e:
                last = e
                print(f"[load] {name}/{attn} failed: {str(e)[:160]}", flush=True)
    raise RuntimeError(f"could not load model: {last}")


def get_tokenizer(model_id):
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        from transformers import AutoProcessor
        tok = AutoProcessor.from_pretrained(model_id).tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def render_text(tok, msgs, add_gen):
    """Render chat messages to TEXT (not token objects — those aren't Arrow-serializable).
    enable_thinking=False so the model learns to emit the episode directly."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=add_gen,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)


def make_tokenize_fn(tok, max_len):
    def fn(ex):
        msgs = ex["messages"]
        full = tok(render_text(tok, msgs, False), add_special_tokens=False)["input_ids"]
        prompt = tok(render_text(tok, msgs[:-1], True), add_special_tokens=False)["input_ids"]
        labels = list(full)
        for i in range(min(len(prompt), len(full))):   # mask the prompt (completion-only)
            labels[i] = -100
        full, labels = full[:max_len], labels[:max_len]
        return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full)}
    return fn


class Collator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id

    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            n = L - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad] * n)
            lab.append(f["labels"] + [-100] * n)
            att.append(f["attention_mask"] + [0] * n)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train", default="data/fruit_drama_sft.train.jsonl")
    ap.add_argument("--val", default="data/fruit_drama_sft.val.jsonl")
    ap.add_argument("--out", default="qwen3-8b-fruit-lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--precision", choices=["4bit", "bf16"], default="bf16")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    set_seed(42)

    tok = get_tokenizer(args.model)
    print("[data] loading + tokenizing", flush=True)
    ds = load_dataset("json", data_files={"train": args.train, "validation": args.val})
    tfn = make_tokenize_fn(tok, args.seqlen)
    ds = ds.map(tfn, remove_columns=ds["train"].column_names, num_proc=8, desc="tokenize")
    ds = ds.filter(lambda e: any(x != -100 for x in e["labels"]))   # drop prompt-only rows
    print(f"[data] train={len(ds['train'])} val={len(ds['validation'])}", flush=True)

    quant = None
    if args.precision == "4bit":
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True)

    offload_dir = os.path.join(os.path.dirname(args.out.rstrip("/")) or ".", "offload")
    os.makedirs(offload_dir, exist_ok=True)
    _evict = start_cache_evictor(args.model)
    try:
        model = load_base_model(args.model, torch.bfloat16, quant, offload_dir)
    finally:
        _evict.set()

    model.config.use_cache = False
    if args.precision == "4bit":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    print(f"[mem] after load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    collator = Collator(tok)

    if args.smoke:
        batch = collator([ds["train"][i] for i in range(2)])
        batch = {k: v.to("cuda") for k, v in batch.items()}
        model.train()
        out = model(**batch)
        out.loss.backward()
        torch.cuda.synchronize()
        print(f"[SMOKE] loss={float(out.loss):.4f} | "
              f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        print("SMOKE_OK")
        return

    steps_per_epoch = max(1, len(ds["train"]) // (args.bsz * args.accum))
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz, gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=1, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.03, max_grad_norm=1.0, bf16=True, optim="adamw_torch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, logging_first_step=True, save_strategy="steps", save_steps=200,
        save_total_limit=3, eval_strategy="steps", eval_steps=200,
        dataloader_num_workers=4, remove_unused_columns=False, report_to="none",
        seed=42)
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=ds["validation"], data_collator=collator)
    print(f"[train] ~{steps_per_epoch} optimizer steps/epoch x {args.epochs} epochs",
          flush=True)
    trainer.train()
    trainer.save_model(args.out)      # saves the LoRA adapter only
    tok.save_pretrained(args.out)
    print("[done] adapter saved to", args.out)


if __name__ == "__main__":
    main()
