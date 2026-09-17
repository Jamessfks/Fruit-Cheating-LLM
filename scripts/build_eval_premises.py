#!/usr/bin/env python3
"""Freeze the held-out eval premise set to disk with a manifest hash.

Why freeze rather than regenerate from a seed: the premise engine is code, and
code changes. Regenerating means an eval run before an engine edit and one after
it are silently measuring different premise sets, so their scores are not
comparable -- which is exactly the thing an eval is supposed to make safe.

The set is written once and then treated as data. run_eval.py and ab_compare.py
load it and refuse to run if the file's hash does not match the manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fruitdrama.premises import PremiseEngine  # noqa: E402

OUT = pathlib.Path("data/eval/heldout_premises.jsonl")
MANIFEST = pathlib.Path("data/eval/MANIFEST.sha256")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen set (invalidates every "
                         "previous eval score)")
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        print(f"{OUT} already exists. Overwriting would invalidate every eval "
              f"score recorded against it; pass --force if that is intended.",
              file=sys.stderr)
        return 1

    premises = list(PremiseEngine(seed=args.seed, mode="eval").generate(args.n))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in premises:
            f.write(json.dumps({
                "id": p.id, "premise": p.user_prompt, "register": p.register,
                "is_human_premise": p.is_human_premise,
                "is_adversarial": p.is_adversarial,
                "twist_architecture": p.facets.get("twist"),
                "cast": p.facets.get("cast"),
            }, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    MANIFEST.write_text(f"{digest}  {OUT.name}\n")

    reg = collections.Counter(p.register for p in premises)
    print(f"froze {len(premises)} premises -> {OUT}")
    print(f"sha256 {digest}")
    print(f"registers: {dict(reg)}")
    print(f"human premises: {sum(p.is_human_premise for p in premises)}")
    print(f"adversarial:    {sum(p.is_adversarial for p in premises)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
