# Fruit-Cheating-LLM — dataset (data-preprocessing phase)

Training data for a **specialist LLM that writes viral "AI fruit drama" scripts** —
short, telenovela-style episodes where anthropomorphic fruits and vegetables cheat,
scheme, and betray each other, built around jaw-dropping plot twists.

> **Status: data preprocessing only. No model has been fine-tuned yet.**
> This repo produces one condensed, LoRA-ready SFT dataset. Training is a later phase.

## What this produces

A single condensed dataset in **chat-messages JSONL** (the standard SFT/LoRA format
for TRL / Unsloth / Axolotl):

```json
{"messages":[
  {"role":"system","content":"<system prompt>"},
  {"role":"user","content":"<instruction>"},
  {"role":"assistant","content":"<dramatic script / episode>"}],
 "meta":{"source":"...","slice":"...","license":"..."}}
```

Output files (git-ignored — large files belong on the HF Hub / Git LFS, not GitHub):
- `data/fruit_drama_sft.train.jsonl`
- `data/fruit_drama_sft.val.jsonl`
- `data/dataset_card.json` — build stats (rows per slice, per license, token estimate)
- `data/samples/sample.jsonl` — a small committed peek

## Design: two-tier

The target output format (a 6-scene fruit episode) is defined by the
[Flashloop soap-opera formula](https://www.flashloop.app/blog/how-to-make-ai-fruit-drama-videos),
baked into `prompts/system_fruit.txt`.

- **Fruit tier** (`fruit_synth`) — synthetic episodes generated directly in the exact
  6-scene schema (`NARRATION` + `VISUAL` per scene). Carries the **fruit** system prompt.
  This is what teaches the precise output shape.
- **Voice tier** (everything else) — real, permissively-licensed dramatic writing that
  teaches soap-opera voice, escalation, and shocking twists. Carries a generic
  **voice** system prompt (`prompts/system_voice.txt`).

At inference you use the fruit system prompt; the model applies the dramatic skill it
learned from the voice tier to the fruit format it learned from the fruit tier.

## The mix (built: 44,585 rows; ~32.6M tokens)

All permissive sources are enabled. Exact live counts are always in
`data/dataset_card.json`.

Chosen policy: **permissive license only + platform-safe (SFW)**. This deliberately
excludes copyrighted TV/movie scripts (Cornell, IMSDB, TV transcripts, MELD, etc.),
which are not safe to train on for a publishable/monetizable product.

| Slice | Source | License | System | Rows |
|---|---|---|---|---|
| `writingprompts_twist` | [euclaise/writingprompts](https://huggingface.co/datasets/euclaise/writingprompts) (twist/drama-filtered, SFW) | MIT | voice | 21,811 |
| `multichar` | [agentlans/multi-character-dialogue](https://huggingface.co/datasets/agentlans/multi-character-dialogue) | CC-BY-4.0 | voice | 12,994 |
| `fruit_synth` | synthesized from the Flashloop formula | ours | fruit | 7,500 |
| `gutenberg_melodrama` | [manu/project_gutenberg](https://huggingface.co/datasets/manu/project_gutenberg) (en, dramatic chunks) | public domain | voice | 1,181 |
| `dramabench` | [FutureMa/DramaBench](https://huggingface.co/datasets/FutureMa/DramaBench) | MIT | voice | 925 |
| `screenplay_voice` + `screenplay_structure` | [Atum09/screenplay-dataset](https://huggingface.co/datasets/Atum09/screenplay-dataset) (`<think>` stripped) | MIT | voice | 174 |

**Every slice is 100% unique** (verified) — global dedup guarantees no duplicate assistant
text enters the set, so each source contributes only its distinct rows.

Two notes on the smaller sources:

- `screenplay_*` (Atum09/screenplay-dataset): looks ideal (MIT, ~1M rows, chat-format,
  genre-tagged) but is **heavily duplicated** — 45,244 drama rows collapse to only **174
  unique** assistant outputs, so despite its size it contributes 174 clean rows (with the
  `<think>` reasoning block stripped). Dedup makes including it safe; it simply adds little.
- `gutenberg_melodrama` (manu/project_gutenberg): dialogue-rich dramatic chunks pulled from
  ~one 200 MB shard's worth of public-domain books (keyword-filtered for drama). Bump
  `max_scan` (books) in `config/mix.json` to pull more.

The voice tier is carried mainly by `writingprompts` (diverse, MIT) and `multichar`
(~13k unique, CC-BY); permissive + de-duplicated soap dialogue is otherwise scarce.

## Filtering (every row)

- **Detokenization** — fixes whitespace-tokenized Reddit text (`filters.detokenize`).
- **SFW** — drops explicit sexual content and slurs (`filters.is_sfw`).
- **Quality** — length bounds, English/ASCII ratio, anti-repetition (`filters.quality_ok`).
- **Drama bias** — generic story sources are filtered toward betrayal/affair/revenge/
  twist keywords (`filters.is_dramatic`).
- **Dedup** — global normalized-hash dedup across all slices (`filters.Dedup`).

## Ingestion note

HF row-by-row streaming was far too slow here (a few rows/sec, rate-limited). The
pipeline instead **bulk-downloads parquet shards and parses them locally with pyarrow**,
one shard at a time then deletes it, keeping peak disk to a single shard (~450 MB).
Small/oddly-packaged sources are pulled by their native file (`train.jsonl.gz`) or the
datasets-server API.

## Reproduce

```bash
pip install -r requirements.txt
python src/build_dataset.py                 # full build (~50k)
python src/build_dataset.py --skip-external # fruit slice only (offline, no downloads)
python src/build_dataset.py --scale 0.05    # small end-to-end test
```

Layout:
```
config/mix.json        caps, scan limits, genre/category filters, licenses
prompts/system_*.txt   fruit + voice system prompts
src/fruit_generator.py synthetic 6-scene episode generator
src/adapters.py        per-source -> unified chat rows
src/filters.py         cleaning / SFW / quality / dedup
src/build_dataset.py   orchestrator (dedup, shuffle, split, write, data card)
src/augment_dataset.py add new sources to an existing build without re-downloading
src/measure.py         measures real *unique* yield per source (used to set caps)
```

To add a source to an existing dataset without rebuilding from scratch, enable it in
`config/mix.json` and run `python src/augment_dataset.py` (reuses the current
`data/*.jsonl`, dedupes new rows against them, and rewrites a complete dataset).

## Next phase (not done here)

LoRA SFT on a 7–8B instruct base (e.g. Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct)
with TRL/Unsloth, applying the base model's chat template to the `messages` field.
