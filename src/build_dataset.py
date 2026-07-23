"""Build the condensed Fruit-Cheating-LLM SFT dataset (chat-messages JSONL).

Two-tier: 'fruit_synth' rows carry the fruit system prompt; every other slice
carries the generic 'voice' system prompt. All sources are permissive-license
and SFW-filtered. Streaming keeps disk usage low.

Usage:
  python src/build_dataset.py                 # full build (~50k)
  python src/build_dataset.py --scale 0.01    # tiny end-to-end validation run
  python src/build_dataset.py --skip-external # fruit slice only (fully offline)
"""
import os
import sys
import json
import time
import random
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("PQ_TMP", os.path.join(ROOT, "pq_tmp"))
os.makedirs(os.environ["PQ_TMP"], exist_ok=True)

import filters as F          # noqa: E402
import fruit_generator as fg  # noqa: E402
import adapters as A          # noqa: E402


def load_prompt(name):
    with open(os.path.join(ROOT, "prompts", f"system_{name}.txt")) as f:
        return f.read().strip()


def wrap(rec, system_text):
    return {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": rec["user"]},
            {"role": "assistant", "content": rec["assistant"]},
        ],
        "meta": rec["meta"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply all caps + scan limits (use <1 for quick tests)")
    ap.add_argument("--skip-external", action="store_true",
                    help="only generate the synthetic fruit slice (offline)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "fruit_drama_sft"))
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(ROOT, "config", "mix.json")))
    sl = cfg["slices"]
    scale = args.scale
    seed = cfg["seed"]
    systems = {"fruit": load_prompt("fruit"), "voice": load_prompt("voice")}

    def cap(name):
        return max(1, int(sl[name]["cap"] * scale))

    def scan(name):
        return max(50, int(sl[name].get("max_scan", 10000) * scale))

    def enabled(name):
        return sl[name].get("enabled", True)

    dedup = F.Dedup()
    collected = []
    stats = Counter()
    lic = Counter()
    t0 = time.time()

    def absorb(rec):
        s = rec["meta"]["slice"]
        if stats[s] >= cap(s):
            return False
        if not dedup.is_new(rec["assistant"]):
            return False
        collected.append(wrap(rec, systems[sl[s]["system"]]))
        stats[s] += 1
        lic[rec["meta"].get("license", "?")] += 1
        return True

    print(f"== BUILD (scale={scale}) ==")

    # 1) synthetic fruit slice (offline, on-target)
    print("[1/7] fruit_synth ...")
    for rec in fg.generate(cap("fruit_synth") * 2, seed=seed):
        if stats["fruit_synth"] >= cap("fruit_synth"):
            break
        absorb(rec)

    if not args.skip_external:
        # 2) + 3) screenplay voice + structure (single pass); disabled by default
        if enabled("screenplay_voice") or enabled("screenplay_structure"):
            print("[2/7] screenplay_voice + [3/7] screenplay_structure ...")
            for rec in A.adapt_screenplay(
                    cap("screenplay_voice") if enabled("screenplay_voice") else 0,
                    cap("screenplay_structure") if enabled("screenplay_structure") else 0,
                    cfg["screenplay_genres_allow"], cfg["screenplay_voice_categories"],
                    cfg["screenplay_structure_categories"], scan("screenplay_voice")):
                absorb(rec)
        else:
            print("[2/7]+[3/7] screenplay ... SKIPPED (disabled in config)")

        # 4) multi-character dialogue
        print("[4/7] multichar ...")
        for rec in A.adapt_multichar(cap("multichar"), scan("multichar")):
            absorb(rec)

        # 5) drama continuation
        print("[5/7] dramabench ...")
        for rec in A.adapt_dramabench(cap("dramabench"), scan("dramabench")):
            absorb(rec)

        # 6) public-domain melodrama prose (disabled by default; see config note)
        if enabled("gutenberg_melodrama"):
            print("[6/7] gutenberg_melodrama ...")
            for rec in A.adapt_gutenberg(cap("gutenberg_melodrama"), scan("gutenberg_melodrama")):
                absorb(rec)
        else:
            print("[6/7] gutenberg_melodrama ... SKIPPED (disabled in config)")

        # 7) plot twists
        print("[7/7] writingprompts_twist ...")
        for rec in A.adapt_writingprompts(cap("writingprompts_twist"), scan("writingprompts_twist")):
            absorb(rec)

    # shuffle + split
    rng = random.Random(seed)
    rng.shuffle(collected)
    n = len(collected)
    n_val = max(1, int(n * (1 - cfg["train_val_split"]))) if n > 20 else 0
    val, train = collected[:n_val], collected[n_val:]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "samples"), exist_ok=True)

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(f"{args.out}.train.jsonl", train)
    dump(f"{args.out}.val.jsonl", val)
    dump(os.path.join(ROOT, "data", "samples", "sample.jsonl"),
         [collected[i] for i in range(0, n, max(1, n // 30))][:30])

    chars = sum(len(m["content"]) for r in collected for m in r["messages"])
    card = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scale": scale,
        "total_rows": n,
        "train_rows": len(train),
        "val_rows": len(val),
        "approx_tokens": int(chars / 4),
        "by_slice": dict(stats),
        "by_license": dict(lic),
        "format": "chat messages (system/user/assistant) JSONL",
        "policy": "permissive-license + platform-safe (SFW) only",
        "sources": {s: sl[s].get("license") for s in sl},
    }
    json.dump(card, open(os.path.join(ROOT, "data", "dataset_card.json"), "w"), indent=2)

    print("\n== SUMMARY ==")
    for s, c in stats.most_common():
        print(f"  {s:24} {c}")
    print(f"  {'TOTAL':24} {n}  (train {len(train)} / val {len(val)})")
    print(f"  licenses: {dict(lic)}")
    print(f"  ~tokens: {card['approx_tokens']:,}   elapsed: {time.time()-t0:.0f}s")
    print(f"  wrote: {args.out}.train.jsonl / .val.jsonl")


if __name__ == "__main__":
    main()
