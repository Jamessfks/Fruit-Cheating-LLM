"""Augment the existing SFT dataset with the two re-enabled sources
(screenplay + gutenberg) WITHOUT re-downloading the slices already built.

Reuses data/fruit_drama_sft.{train,val}.jsonl as the base, adds only the new
sources (global-deduped against everything present), then rewrites a complete,
shuffled, re-split dataset and refreshes the data card. Atomic writes.
"""
import os
import sys
import json
import time
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PQ_TMP", os.path.join(ROOT, "pq_tmp"))
os.makedirs(os.environ["PQ_TMP"], exist_ok=True)

import filters as F      # noqa: E402
import adapters as A     # noqa: E402

OUT = os.path.join(ROOT, "data", "fruit_drama_sft")
cfg = json.load(open(os.path.join(ROOT, "config", "mix.json")))
sl = cfg["slices"]
systems = {
    "fruit": open(os.path.join(ROOT, "prompts", "system_fruit.txt")).read().strip(),
    "voice": open(os.path.join(ROOT, "prompts", "system_voice.txt")).read().strip(),
}


def wrap(rec):
    sysname = sl[rec["meta"]["slice"]]["system"]
    return {"messages": [{"role": "system", "content": systems[sysname]},
                         {"role": "user", "content": rec["user"]},
                         {"role": "assistant", "content": rec["assistant"]}],
            "meta": rec["meta"]}


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def dump(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    t0 = time.time()
    base = load_jsonl(f"{OUT}.train.jsonl") + load_jsonl(f"{OUT}.val.jsonl")
    print(f"base rows loaded: {len(base)}", flush=True)

    dedup = F.Dedup()
    for r in base:
        dedup.is_new(r["messages"][2]["content"])  # seed with existing assistants

    added = []
    stats = Counter()

    def absorb(rec):
        s = rec["meta"]["slice"]
        if stats[s] >= int(sl[s]["cap"]):
            return
        if not dedup.is_new(rec["assistant"]):
            return
        added.append(wrap(rec))
        stats[s] += 1

    def run(name, make_gen):
        try:
            n0 = sum(stats.values())
            for rec in make_gen():
                absorb(rec)
            print(f"[{name}] added {sum(stats.values())-n0}", flush=True)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)

    if sl["screenplay_voice"].get("enabled") or sl["screenplay_structure"].get("enabled"):
        print("adding screenplay (dedup + <think> stripped) ...", flush=True)
        run("screenplay", lambda: A.adapt_screenplay(
            int(sl["screenplay_voice"]["cap"]) if sl["screenplay_voice"].get("enabled") else 0,
            int(sl["screenplay_structure"]["cap"]) if sl["screenplay_structure"].get("enabled") else 0,
            cfg["screenplay_genres_allow"], cfg["screenplay_voice_categories"],
            cfg["screenplay_structure_categories"], int(sl["screenplay_voice"]["max_scan"])))

    if sl["gutenberg_melodrama"].get("enabled"):
        print("adding gutenberg melodrama ...", flush=True)
        run("gutenberg", lambda: A.adapt_gutenberg(
            int(sl["gutenberg_melodrama"]["cap"]),
            int(sl["gutenberg_melodrama"]["max_scan"])))

    combined = base + added
    rng = random.Random(cfg["seed"])
    rng.shuffle(combined)
    n = len(combined)
    n_val = max(1, int(n * (1 - cfg["train_val_split"])))
    val, train = combined[:n_val], combined[n_val:]

    dump(f"{OUT}.train.jsonl", train)
    dump(f"{OUT}.val.jsonl", val)
    # refreshed sample
    dump(os.path.join(ROOT, "data", "samples", "sample.jsonl"),
         [combined[i] for i in range(0, n, max(1, n // 30))][:30])

    by_slice = Counter(r["meta"]["slice"] for r in combined)
    by_lic = Counter(r["meta"].get("license", "?") for r in combined)
    chars = sum(len(m["content"]) for r in combined for m in r["messages"])
    card = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": n, "train_rows": len(train), "val_rows": len(val),
        "approx_tokens": int(chars / 4),
        "by_slice": dict(by_slice), "by_license": dict(by_lic),
        "format": "chat messages (system/user/assistant) JSONL",
        "policy": "permissive-license + platform-safe (SFW) only",
        "note": "augmented with screenplay (deduped, <think> stripped) + gutenberg melodrama",
    }
    json.dump(card, open(os.path.join(ROOT, "data", "dataset_card.json"), "w"), indent=2)

    print("\n== RESULT ==", flush=True)
    for s, c in by_slice.most_common():
        print(f"  {s:24} {c}")
    print(f"  {'TOTAL':24} {n}  (train {len(train)} / val {len(val)})")
    print(f"  added this run: {dict(stats)}")
    print(f"  licenses: {dict(by_lic)}   ~tokens: {card['approx_tokens']:,}")
    print(f"  elapsed: {time.time()-t0:.0f}s", flush=True)
    print("AUGMENT DONE", flush=True)


if __name__ == "__main__":
    main()
