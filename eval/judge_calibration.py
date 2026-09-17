#!/usr/bin/env python3
"""Does the judge discriminate? Run this before trusting any judge model.

A judge that scores everything highly is worse than no judge: it costs GPU time
and launders bad rows as good. This scores the hand-authored gold seeds against
three deliberately broken stories and requires clean separation.

Measured behaviour that motivated it (Qwen3-30B-A3B judging gemma's prose):
the seven rubric DIMENSIONS saturate at 5 for anything fluent -- a pasted v1
storyboard scored 5 on every axis -- while the quote-anchored twist enumeration
correctly returned 0 for the storyboard and 1 for twistless prose. So the twist
count is the load-bearing signal and the dimensions are only a backstop.

Exits non-zero unless every gold seed is accepted and every negative rejected.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fruitdrama import prompts as P     # noqa: E402
from fruitdrama import teacher as T     # noqa: E402
from fruitdrama.judge import score_story  # noqa: E402

STORYBOARD = """EPISODE 4: The Affair at Midnight

Scene - Recap
NARRATION: Last time, an elderly broccoli woman believed the marriage was safe.
VISUAL: Dramatic 9:16 vertical scene in a marble bathroom, soft golden light.

Scene - Rising Tension
NARRATION: But a strawberry woman kept slipping away at midnight.
VISUAL: Dramatic 9:16 vertical scene in a boardroom at night, cold light.

Scene - Cliffhanger
NARRATION: And everything is about to shatter.
VISUAL: Dramatic 9:16 vertical scene, a slow knowing half-smile in shadow."""

TWISTLESS = "A Quiet Tuesday\n\n" + (
    "Priya Peach went to the market and bought three plums. She walked home "
    "along the canal and the weather was mild. At home she made tea and read a "
    "little of her book. Peter Banana came in later and mentioned the weather "
    "too. They agreed it had been mild. He washed the cups and she dried them. "
    "Nothing in particular happened. The evening passed pleasantly enough. "
) * 6

SPAM = "Drama \U0001F353\U0001F494\n\n" + (
    "She wept. \U0001F62D\U0001F494 He lied. \U0001F621\U0001F631 The door "
    "slammed. \U0001F4A5\U0001F494 Nobody spoke. \U0001F62D\U0001F631 It was "
    "over. \U0001F494\U0001F62D "
) * 22

DIMS = [k for k, _ in P.JUDGE_DIMENSIONS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-url", default="http://127.0.0.1:8001")
    ap.add_argument("--judge-model", default="judge")
    args = ap.parse_args()

    cfg = T.GenConfig(base_url=args.judge_url, model=args.judge_model,
                      max_tokens=900, concurrency=3, timeout=600)
    if not T.server_ready(cfg):
        print(f"FATAL: no judge at {args.judge_url}", file=sys.stderr)
        return 2

    gold = [json.loads(l) for l in
            (_ROOT / "data" / "gold" / "seeds.jsonl").read_text().splitlines()
            if l.strip()]
    cases = [(f"GOOD {g['id']}", g["premise"], g["story"], True) for g in gold]
    cases += [("BAD storyboard", "strawberry affair at midnight", STORYBOARD, False),
              ("BAD twistless", "a peach buys plums", TWISTLESS, False),
              ("BAD emoji-spam", "a messy little drama", SPAM, False)]

    print(f"{'case':18s} {'accept':7s} {'twists':>6s} "
          + " ".join(d[:9].rjust(9) for d in DIMS))
    failures = []
    saturated = 0
    for label, prem, story, want_accept in cases:
        j = score_story(cfg, prem, story, order_seed=2)
        if j is None:
            failures.append(f"{label}: judge output unparseable")
            print(f"{label:18s} UNPARSED")
            continue
        row = " ".join(str(j.scores.get(d, "-")).rjust(9) for d in DIMS)
        print(f"{label:18s} {str(j.ok):7s} {j.n_real_twists:6d} {row}")
        if j.ok != want_accept:
            failures.append(
                f"{label}: expected accept={want_accept}, got {j.ok} "
                f"(twists={j.n_real_twists})")
        if not want_accept and all(j.scores.get(d) == 5 for d in DIMS):
            saturated += 1

    print()
    if saturated:
        print(f"NOTE: {saturated} negative case(s) scored 5 on every dimension. "
              f"Absolute dimension scores are not a usable quality signal for "
              f"this judge; rely on the twist enumeration and on "
              f"eval/ab_compare.py's pairwise comparison.")
    if failures:
        print("CALIBRATION FAILED")
        for f in failures:
            print("  " + f)
        return 1
    print("CALIBRATION PASSED: every gold seed accepted, every negative rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
