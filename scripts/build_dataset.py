#!/usr/bin/env python3
"""Assemble accepted stories into train/val SFT files plus a dataset card.

Replaces v1's build_dataset.py / augment_dataset.py / config/mix.json. The
mixing logic is gone because there is nothing to mix: v1 blended 83% off-format
prose with 17% storyboards, and this corpus is on-format by construction.

What this step is actually for:
  * a reproducible 98/2 split on a fixed seed,
  * a final contamination assertion against the frozen eval premises,
  * a dataset card recording accept accounting and facet distribution, so a
    later quality regression can be traced to the data rather than guessed at.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fruitdrama import contract as C          # noqa: E402
from fruitdrama.filters import normalize_story  # noqa: E402
from fruitdrama import prompts as P           # noqa: E402
from fruitdrama import textstats as T         # noqa: E402
from fruitdrama.gates import check_story      # noqa: E402
from fruitdrama.premises import PremiseEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", default="data/run1/accepted.jsonl")
    ap.add_argument("--out-prefix", default="data/story_sft")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--judged", default="",
                    help="judged.jsonl from judge_corpus.py; rows that failed "
                         "the judge are excluded")
    ap.add_argument("--no-recheck", action="store_true",
                    help="skip re-gating (not advised: rows accepted under "
                         "older gate versions would reach training)")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    src = pathlib.Path(args.accepted)
    if not src.exists():
        print(f"FATAL: {src} not found", file=sys.stderr)
        return 2

    rows = []
    for line in src.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("story") and r.get("messages"):
                rows.append(r)
    # Re-normalise: rows written before the generate_corpus fix stored raw text.
    fixed = 0
    for r in rows:
        clean = normalize_story(r["story"])
        if clean != r["story"]:
            r["story"] = clean
            r["messages"] = P.training_messages(r["premise"], clean)
            fixed += 1
    print(f"loaded {len(rows)} gate-accepted rows"
          + (f"; re-normalised {fixed} that still carried formatting artifacts" if fixed else ""))

    if args.judged:
        jp = pathlib.Path(args.judged)
        if not jp.exists():
            print(f"FATAL: {jp} not found", file=sys.stderr)
            return 2
        verdict = {}
        for line in jp.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                verdict[j["premise_id"]] = j
        before = len(rows)
        # Unjudged rows are kept (the judge pass may be partial); only explicit
        # failures are dropped, so a half-finished judge run cannot silently
        # shrink the corpus.
        rows = [r for r in rows
                if verdict.get(r["premise_id"], {}).get("judge_ok") is not False]
        for r in rows:
            j = verdict.get(r["premise_id"])
            if j:
                r["judge"] = j.get("scores")
                r["n_twists"] = j.get("n_twists")
        print(f"judge pass: {before} -> {len(rows)} rows "
              f"({before - len(rows)} rejected, "
              f"{sum(1 for r in rows if r['premise_id'] not in verdict)} unjudged kept)")

    # Contamination assertion. The holdout makes overlap structurally
    # impossible, so any hit here means the holdout was bypassed -- fail loudly
    # rather than shipping a corpus that quietly invalidates the eval.
    eval_prompts = {p.user_prompt for p in PremiseEngine(seed=99, mode="eval").generate(200)}
    leaked = [r for r in rows if r["premise"] in eval_prompts]
    if leaked:
        print(f"FATAL: {len(leaked)} training rows share a prompt with the frozen "
              f"eval set; e.g. {leaked[0]['premise']!r}", file=sys.stderr)
        return 3
    print("contamination check: 0 overlaps with the frozen eval premises")

    # Re-gate by default. A long corpus run spans gate fixes, so early rows can
    # have been admitted by a buggier version. Measured on the 6,185-row corpus:
    # 83 rows (1.3%) failed the current gates, every one of them a G2 leakage
    # in the first 1,000 rows, with rows 1,000+ at 100%. Cheap to check, and the
    # alternative is training on stories the gates would now reject.
    if not args.no_recheck:
        import collections
        bad, why = [], collections.Counter()
        for r in rows:
            res = check_story(r["story"], premise=r["premise"])
            if not res.ok:
                bad.append(id(r))
                for g in dict.fromkeys(res.failures):
                    why[g] += 1
        if bad:
            badset = set(bad)
            rows = [r for r in rows if id(r) not in badset]
            print(f"re-gate: dropped {len(bad)} rows admitted under earlier gate "
                  f"versions {dict(why.most_common(5))}; {len(rows)} remain")
        else:
            print(f"re-gate: all {len(rows)} rows pass the current gates")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    val, train = rows[:n_val], rows[n_val:]

    def dump(path: pathlib.Path, recs: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({"messages": r["messages"],
                                     "meta": {k: r.get(k) for k in
                                              ("premise_id", "register",
                                               "is_human_premise",
                                               "twist_architecture",
                                               "prompt_version")}},
                                    ensure_ascii=False) + "\n")

    tp = pathlib.Path(f"{args.out_prefix}.train.jsonl")
    vp = pathlib.Path(f"{args.out_prefix}.val.jsonl")
    tp.parent.mkdir(parents=True, exist_ok=True)
    dump(tp, train)
    dump(vp, val)

    words = [T.word_count(r["story"]) for r in rows]
    emoji = [T.count_emoji(r["story"]) for r in rows]
    pct = lambda v, q: sorted(v)[int(len(v) * q)] if v else 0
    card = {
        "built_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "prompt_version": P.prompt_version(),
        "contract": {
            "word_range": [C.WORD_MIN, C.WORD_MAX],
            "emoji_per_100w": [C.EMOJI_PER_100W_MIN, C.EMOJI_PER_100W_MAX],
            "twists_min": C.TWIST_MIN,
        },
        "rows": {"total": len(rows), "train": len(train), "val": len(val)},
        "words": {"p10": pct(words, .10), "p50": pct(words, .50), "p90": pct(words, .90),
                  "in_band": round(sum(C.WORD_MIN <= w <= C.WORD_MAX for w in words) / max(1, len(words)), 3)},
        "emoji": {"p10": pct(emoji, .10), "p50": pct(emoji, .50), "p90": pct(emoji, .90)},
        "by_register": dict(collections.Counter(r.get("register") for r in rows)),
        "by_twist_architecture": dict(collections.Counter(r.get("twist_architecture") for r in rows)),
        "human_premise_share": round(sum(bool(r.get("is_human_premise")) for r in rows) / max(1, len(rows)), 3),
        "distinct_3_diversity": round(T.distinct_n([r["story"] for r in rows[:500]], 3), 4),
        "unique_openings": len({T.opening_ngram(r["story"], 6) for r in rows}),
        "train_sha256": hashlib.sha256(tp.read_bytes()).hexdigest()[:16],
    }
    cp = pathlib.Path("data/dataset_card.json")
    cp.write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))
    print(f"\nwrote {tp} ({len(train)}) and {vp} ({len(val)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
