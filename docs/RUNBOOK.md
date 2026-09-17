# Runbook

Every command runs from the Mac in the repo root. `./ops/fruit` resolves the
Spark host itself — never hardcode an address.

## Cold start on a wiped box

```bash
scp ops/spark_bootstrap.sh ops/fetch_models.sh spark-2e6c:~/
ssh spark-2e6c 'bash ~/spark_bootstrap.sh'     # venv, llama.cpp for sm_121, py headers
ssh spark-2e6c 'bash ~/fetch_models.sh'        # teacher + student + judge GGUFs
./ops/fruit sync
./ops/fruit status
```

`spark_bootstrap.sh` is idempotent and needs no sudo. It builds llama.cpp with
`-DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=OFF` (no libcurl headers on the
box, no ninja) and unpacks the Python dev headers Triton needs from the `.deb`
with `dpkg-deb -x`.

Verify CUDA actually compiled in, rather than assuming:

```bash
ssh spark-2e6c '~/fruit/llama.cpp/build/bin/llama-server --list-devices'
# expect: CUDA0: NVIDIA GB10 (124607 MiB, ... free)
```

## 1. Corpus generation (~3 days for 10k rows)

```bash
./ops/fruit serve '~/fruit/models/gguf/gemma-3-27b-judge/gemma-3-27b-it-UD-Q5_K_XL.gguf' gemma 32 196608
ssh spark-2e6c 'cd ~/fruit/app && tmux new-session -d -s corpus \
  "~/fruit/venv/bin/python -u scripts/generate_corpus.py --target 10000 \
   --out ~/fruit/run1 --teacher-model gemma --concurrency 32 --fewshot 1 \
   --fewshot-block 400 2>&1 | tee -a ~/fruit/logs/corpus.log"'
./ops/fruit corpus     # progress, accept rate, per-gate rejects
```

**Do not restart this casually.** A restart discards the in-flight wave (up to
128 generations that have not been written yet). It *is* fully resumable — keyed
on premise id across three append-only JSONLs — but each restart costs a wave.

## 2. Judge pass (~10 h for 10k)

Swap to a model from a **different family than the writer**, then:

```bash
./ops/fruit serve '~/fruit/models/gguf/qwen3-30b-a3b-base/Qwen3-30B-A3B-Instruct-2507-UD-Q5_K_XL.gguf' judge 16 131072
./ops/fruit judge
```

## 3. Dataset assembly

```bash
./ops/fruit dataset
```

Fails loudly if any training row shares a prompt with the frozen eval set.
Unjudged rows are kept, so a partial judge run cannot silently shrink the corpus.

## 4. Training (~14–18 h)

```bash
./ops/fruit train      # runs --smoke first and asks before the long run
./ops/fruit logs train
```

Expected smoke output: `peak≈57.6 GiB`, `trainable 26.7M / 30.6B (0.087%)`,
`SMOKE_OK`. If peak exceeds ~100 GiB, stop and take the fallback ladder:
`--bsz 1 --accum 16`, then `--seqlen 1024`, then `Qwen3-14B` dense with
`all-linear` targets.

## 5. Merge, quantize, serve

```bash
ssh spark-2e6c 'cd ~/fruit/app && ADAPTER=~/fruit/out/qwen3-30b-a3b-fruit-lora \
  bash training/merge_and_quantize.sh'
./ops/fruit serve '~/fruit/models/gguf/fruit/fruit-30b-a3b-Q5_K_M.gguf' fruit 8 65536
./ops/fruit up
```

## 6. Ship gate

```bash
TAG=ckpt-final JUDGE_URL=http://127.0.0.1:8001 ./ops/fruit eval
./ops/fruit ab         # candidate vs base, blind and position-swapped
```

Both exit non-zero on failure. Every serving artifact is small enough that
candidate + base + judge co-reside in 121 GB, so the A/B needs no model swaps.

Ship the smallest quant within **0.2 judge points** of bf16 on campiness, emoji
integration and coherence.

## Persistence

```bash
ssh spark-2e6c 'bash ~/fruit/app/serve/systemd/install.sh'
ssh spark-2e6c 'systemctl --user enable --now fruit-model fruit-gateway'
```

User units plus `loginctl enable-linger` give reboot survival without sudo. If
polkit refuses linger, fall back to `@reboot` in crontab.

## Traps

- **`pkill -f <pattern>` over SSH kills your own session** when the pattern
  matches your command line. Use
  `ps -eo pid=,args= | awk '/pat/ && !/awk/ {print $1; exit}'`.
- **`free`'s "used" column is misleading during a model load** — it counts the
  unified-memory mapping. Trust `torch.cuda.max_memory_allocated()`.
- **`llama-server` has no `--no-mmap`**; it is `-lm/--load-mode none`.
- **`hf_transfer` must be installed explicitly.** The
  `huggingface_hub[hf_transfer]` extra silently does not pull it, and downloads
  run ~7× slower without it.
- **The Spark is on WiFi.** `enP7s7` is `NO-CARRIER`; plugging in ethernet would
  materially speed up the ~170 GB of model transfers.
