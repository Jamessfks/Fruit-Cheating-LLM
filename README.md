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

## The mix (built: 43,230 rows; ~31.8M tokens)

Target was ~50k; the honest permissive+SFW+de-duplicated yield came to **43,230**
(quality over padding — see the two disabled sources below). Exact live counts are in
`data/dataset_card.json`.

Chosen policy: **permissive license only + platform-safe (SFW)**. This deliberately
excludes copyrighted TV/movie scripts (Cornell, IMSDB, TV transcripts, MELD, etc.),
which are not safe to train on for a publishable/monetizable product.

| Slice | Source | License | System | ~Target |
|---|---|---|---|---|
| `fruit_synth` | synthesized from the Flashloop formula | ours | fruit | 7,500 |
| `writingprompts_twist` | [euclaise/writingprompts](https://huggingface.co/datasets/euclaise/writingprompts) (twist/drama-filtered, SFW) | MIT | voice | up to 30,000 |
| `multichar` | [agentlans/multi-character-dialogue](https://huggingface.co/datasets/agentlans/multi-character-dialogue) | CC-BY-4.0 | voice | 13,000 |
| `dramabench` | [FutureMa/DramaBench](https://huggingface.co/datasets/FutureMa/DramaBench) | MIT | voice | ~925 |
| `screenplay_*` | [Atum09/screenplay-dataset](https://huggingface.co/datasets/Atum09/screenplay-dataset) | MIT | voice | *disabled* |
| `gutenberg_melodrama` | [manu/project_gutenberg](https://huggingface.co/datasets/manu/project_gutenberg) | public domain | voice | *disabled* |

Two sources were measured and **disabled** during preprocessing:

- `screenplay_*` (Atum09/screenplay-dataset): looked ideal (MIT, ~1M rows, chat-format,
  genre-tagged) but is **heavily duplicated** — a scan of 45,244 drama rows yielded only
  **174 unique** assistant outputs. Not worth including. Adapters/config are kept so you
  can re-enable it if a de-duplicated version appears.
- `gutenberg_melodrama` (manu/project_gutenberg): the full 61k-book English corpus is too
  heavy to pull quickly; re-enable with a curated public-domain subset for long-form prose.

Because permissive + de-duplicated soap dialogue is scarce, the voice tier leans on
`writingprompts` (diverse, MIT) and `multichar` (13k unique, CC-BY). Exact per-slice
counts for the last build are always in `data/dataset_card.json`.

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
```

## Next phase (not done here)

LoRA SFT on a 7–8B instruct base (e.g. Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct)
with TRL/Unsloth, applying the base model's chat template to the `messages` field.
