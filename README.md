<div align="center">

# 🍓 Fruit-Cheating-LLM

**Type any drama. Get a 2–3 minute telenovela short story about fruit — emoji woven through the prose, three to five escalating twists, and a cliffhanger you didn't see coming.**

Distilled corpus → LoRA fine-tune → quantized local API → streaming web app. Runs entirely on one NVIDIA DGX Spark.

</div>

---

## What this is

A consumer story engine. You type a premise — `strawberry catches her husband with the eggplant at the farmers market`, or `my roommate ate my leftovers and lied about it`, or just `🍆💔` — and it writes one complete prose short story: ~620 words, campy and vivid, emoji inside the sentences, a twist ladder that reframes what you already read, and no tidy ending.

Premises with no fruit in them get fruits cast into the roles automatically. That is a trained behaviour, not luck.

## How it is built

| Stage | What happens |
|---|---|
| **Premises** | A quota-enforced combinatorial engine over ~75M facet combinations: 64 characters × 4 life stages × 24 betrayal engines × 34 venues × 20 discovery mechanisms × 8 twist architectures × 10 input registers. |
| **Generation** | A dense teacher (`gemma-3-27b-it`, Q5) writes each story from a rich internal brief. The brief is discarded; only the *degraded* user prompt is kept. |
| **Filtering** | 13 deterministic gates, then an LLM judge that must quote each twist verbatim before it counts. |
| **Training** | LoRA on `Qwen3-30B-A3B-Instruct-2507`, attention-only, rank 32, completion-only loss. |
| **Serving** | Merged, quantized to GGUF Q5_K_M, served by llama.cpp behind a hardened gateway. |
| **Eval** | A frozen held-out premise set and a ship gate that a checkpoint must pass to be deployed. |

### The asymmetry that makes it work

The teacher receives a full brief — cast psychology from the fruit bible, the betrayal engine, the venue, the twist architecture. The stored training row keeps **only** the messy user prompt:

```
"eggplant affair pickling plant wife finds out"   →   620-word story
```

That is the inference-time job. Training on the rich brief would produce a model that collapses the moment a real user types `🍓💔🍆`.

### The format contract

Every threshold lives once, in [`src/fruitdrama/contract.py`](src/fruitdrama/contract.py), and is imported by the generation prompt, the gates, the judge rubric, the eval metrics and the serve-time system prompt — so the target cannot drift between what we ask for, what we filter on, and what we score.

| Property | Target |
|---|---|
| Length | 500–750 words (2–3 min at 250 wpm) |
| Emoji | 1.2–5.0 per 100 words, inside sentences, spread across the whole story, never in runs |
| Twists | 3–5, escalating, each reframing earlier text |
| Ending | A reveal or an unanswered question. Never resolved |
| Content | PG-13. Implied, never depicted |
| Leakage | Zero scene headings, labels, markdown or stage directions |

## Layout

```
src/fruitdrama/   contract, textstats (emoji graphemes), gates, premises, prompts, teacher, judge, filters
scripts/          generate_corpus.py, build_dataset.py
training/         train_lora_moe.py, merge_and_quantize.sh
serve/            serve_llama.sh, gateway/app.py, gateway/static/index.html, systemd/
eval/             metrics.py, run_eval.py
ops/              spark_bootstrap.sh, fetch_models.sh, fruit (Mac-side CLI)
data/             fruit_bible.json, gold/seeds.jsonl, eval_holdout.json
```

## Running it

```bash
./ops/fruit status     # box, models, jobs
./ops/fruit corpus     # corpus generation progress
./ops/fruit up         # start model + gateway, open the app
./ops/fruit eval       # ship-gate eval
```

The app is served same-origin by the gateway, so there is no endpoint to configure. Reachable on the tailnet only.

## Engineering notes

Things that cost time and are worth knowing.

- **Dense beats sparse for bulk generation on unified memory.** GLM-4.5-Air (106B/12B active) is 2× faster single-stream than dense gemma-3-27b and then plateaus at 56 tok/s — concurrency 8 and 16 are identical, because concurrent requests route to different experts and each decode step ends up reading the whole model. The dense 27B scales 10.2× to 135 tok/s. Measure aggregate, not single-stream.
- **`filters.is_sfw()` had no word boundaries** despite a comment claiming otherwise. `cum` matched inside "documents", "circumstance", "accumulated" — and "Cucumber", one of the 64 characters. Every Cucumber story was being silently discarded.
- **`quality_ok(min_ascii=0.85)` rejects every correct story**, because emoji are non-ASCII by construction. Strip emoji and wanted typography before measuring ASCII purity.
- **Emoji position needs checking on both sides.** A forward-only test scores `She left. 🍓 He lied.` as woven, which is the most common bolted-on pattern. And a *title-leading* emoji must not count as floating, or it consumes the whole budget — that single fix doubled the corpus accept rate.
- **Quotas must scale with pool cardinality.** Three separate absolute caps silently limited the corpus far below target: `tone` at 6% across 6 values (a 36% ceiling), a character cap below its own uniform share, and an engine×venue pair cap that imposed a hard 9,792-row limit.
- **Asking a model to repair a story makes it worse.** 22 surgical-fix attempts produced 1 success and *more* defects than before (ending failures 7→12). Repair is worth it only for length.
- **Few-shot examples work and do not cause copying.** One exemplar eliminated every ending failure, with zero stories echoing its specifics.
- **`pkill -f` over SSH kills your own session** when the pattern matches your command line.
- **`llama-server` renamed `--no-mmap`** to `-lm/--load-mode none`.

## Licence

MIT (code). `data/fruit_bible.json` is ours. The corpus is teacher-generated; see the model's own licence for downstream terms.
