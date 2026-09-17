#!/usr/bin/env python3
"""Blind, position-swapped A/B between two endpoints on the frozen premise set.

Design decisions that make the number defensible:

* **Both arms get the byte-identical serve-time system prompt.** The honest
  comparison is "did fine-tuning help beyond what the prompt already does" --
  not "tuned model with a good prompt versus base model with no prompt".
* **Every pair is judged twice with positions swapped**, and a win counts only
  if both orders agree. Otherwise it is a tie. LLM judges have a measurable
  preference for one position, and without this the win rate is partly an
  artifact of ordering.
* **Blinded.** The judge sees "Story A" and "Story B", never which is which,
  never the words "base" or "fine-tuned".
* Reports a two-sided sign test over non-tied pairs, so a win rate comes with a
  p-value rather than standing alone.

Usage:
  ab_compare.py --a-url ... --a-model fruit --b-url ... --b-model base \
      --judge-url ... --judge-model judge --n 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import re
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from fruitdrama import prompts as P            # noqa: E402
from fruitdrama import teacher as T            # noqa: E402
from fruitdrama.premises import PremiseEngine  # noqa: E402


def load_frozen_premises(n: int | None = None) -> list:
    """Load the frozen held-out set, refusing to run if it has drifted.

    Regenerating from a seed would mean an eval before a premise-engine edit and
    one after it silently measure different sets. The hash check makes that
    impossible to do by accident.
    """
    path = _ROOT / "data" / "eval" / "heldout_premises.jsonl"
    manifest = _ROOT / "data" / "eval" / "MANIFEST.sha256"
    if not path.exists():
        raise SystemExit(f"FATAL: {path} missing. Run scripts/build_eval_premises.py")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.exists():
        expected = manifest.read_text().split()[0]
        if digest != expected:
            raise SystemExit(
                f"FATAL: the frozen premise set has changed.\n"
                f"  expected {expected}\n  found    {digest}\n"
                f"Scores are only comparable within one premise-set version. "
                f"Either restore the file or re-freeze with --force and treat "
                f"previous scores as a different benchmark.")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if n:
        rows = rows[:n]

    class P:  # minimal stand-in for a Premise
        __slots__ = ("id", "user_prompt", "register", "is_human_premise",
                     "is_adversarial", "facets")

        def __init__(self, d):
            self.id = d["id"]
            self.user_prompt = d["premise"]
            self.register = d.get("register")
            self.is_human_premise = d.get("is_human_premise", False)
            self.is_adversarial = d.get("is_adversarial", False)
            self.facets = {"twist": d.get("twist_architecture"),
                           "cast": d.get("cast")}

    return [P(d) for d in rows], digest

_VERDICT = re.compile(r"\b([AB])\b")

PAIRWISE = """You are comparing two short stories written from the same reader prompt.

Pick the one that is more fun to read as a campy fruit-telenovela: better
twists that genuinely reframe earlier text, stronger escalation, a more
striking ending, more vivid concrete writing, and emoji that feel woven into
the prose rather than bolted on.

Answer with exactly one character: A or B. No explanation."""


def generate(cfg: T.GenConfig, premises, tag: str, out_dir: pathlib.Path) -> dict:
    tasks = [(p.id, [{"role": "system", "content": P.STORY_SYSTEM},
                     {"role": "user", "content": p.user_prompt}], {})
             for p in premises]
    w = T.ResumableWriter(str(out_dir / f"{tag}.jsonl"))
    T.run_batch(cfg, tasks, w, None)
    w.close()
    return {json.loads(l)["premise_id"]: json.loads(l).get("text", "")
            for l in (out_dir / f"{tag}.jsonl").read_text().splitlines() if l.strip()}


def ask(jcfg: T.GenConfig, prompt: str, first: str, second: str) -> str | None:
    msgs = [{"role": "system", "content": PAIRWISE},
            {"role": "user", "content":
             f"READER PROMPT:\n{prompt}\n\nStory A:\n{first}\n\nStory B:\n{second}\n\nA or B?"}]
    try:
        raw = T.chat(jcfg, msgs, temperature=0.0, top_p=1.0, max_tokens=4)
    except Exception:
        return None
    m = _VERDICT.search((raw or "").strip().upper()[:12])
    return m.group(1) if m else None


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test over decided pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-url", required=True)
    ap.add_argument("--a-model", default="fruit")
    ap.add_argument("--a-label", default="candidate")
    ap.add_argument("--b-url", required=True)
    ap.add_argument("--b-model", default="base")
    ap.add_argument("--b-label", default="base")
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--premise-seed", type=int, default=99)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = pathlib.Path(args.out or
                       f"eval/runs/ab-{time.strftime('%Y-%m-%dT%H-%M-%SZ', time.gmtime())}")
    out.mkdir(parents=True, exist_ok=True)

    # Benchmark integrity first, environment second.
    premises, pset_hash = load_frozen_premises(args.n)
    acfg = T.GenConfig(base_url=args.a_url, model=args.a_model,
                       concurrency=args.concurrency, max_tokens=1600)
    bcfg = T.GenConfig(base_url=args.b_url, model=args.b_model,
                       concurrency=args.concurrency, max_tokens=1600)
    jcfg = T.GenConfig(base_url=args.judge_url, model=args.judge_model,
                       concurrency=args.concurrency, max_tokens=8)
    for name, cfg in (("A", acfg), ("B", bcfg), ("judge", jcfg)):
        if not T.server_ready(cfg):
            print(f"FATAL: {name} endpoint not responding at {cfg.base_url}",
                  file=sys.stderr)
            return 2

    print(f"generating arm A ({args.a_label})", flush=True)
    a = generate(acfg, premises, "a", out)
    print(f"generating arm B ({args.b_label})", flush=True)
    b = generate(bcfg, premises, "b", out)

    rng = random.Random(5)
    wins = losses = ties = undecided = 0
    rows = []
    for p in premises:
        sa, sb = a.get(p.id, ""), b.get(p.id, "")
        if not sa or not sb:
            undecided += 1
            continue
        # Randomise which arm is presented first, then ask again with the
        # order reversed. A win requires agreement across both orders.
        a_first = rng.random() < 0.5
        v1 = ask(jcfg, p.user_prompt, sa if a_first else sb, sb if a_first else sa)
        v2 = ask(jcfg, p.user_prompt, sb if a_first else sa, sa if a_first else sb)
        if v1 is None or v2 is None:
            undecided += 1
            continue
        # Translate each verdict into "did arm A win this presentation".
        a_won_1 = (v1 == "A") == a_first
        a_won_2 = (v2 == "B") == a_first
        if a_won_1 and a_won_2:
            wins += 1
            verdict = "A"
        elif not a_won_1 and not a_won_2:
            losses += 1
            verdict = "B"
        else:
            ties += 1
            verdict = "tie"
        rows.append({"premise_id": p.id, "premise": p.user_prompt,
                     "verdict": verdict, "order_a_first": a_first})

    (out / "pairs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    decided = wins + losses
    summary = {
        "a_label": args.a_label, "a_model": args.a_model,
        "b_label": args.b_label, "b_model": args.b_model,
        "judge_model": args.judge_model, "n_premises": args.n,
        "wins_a": wins, "wins_b": losses, "ties": ties, "undecided": undecided,
        "win_rate_excl_ties": round(wins / decided, 3) if decided else None,
        "sign_test_p": round(sign_test_p(wins, losses), 5),
        "prompt_version": P.prompt_version(),
        "premise_set_sha256": pset_hash,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Ship gate: >=65% of decided pairs, significant at p<0.01.
    gate = (decided >= 20 and wins / decided >= 0.65 and summary["sign_test_p"] < 0.01)
    print("\nA/B GATE:", "PASS" if gate else "FAIL")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
