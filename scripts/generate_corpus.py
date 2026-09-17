#!/usr/bin/env python3
"""Corpus generation: premises -> teacher -> gates -> repair -> judge -> SFT rows.

Built for a multi-day job on one box. Design constraints that shaped it:

* **Resumable at every stage.** Three append-only JSONL files keyed on premise
  id. A restart re-reads them and skips finished work, so a kill costs seconds.
* **Every rejection is recorded with its reason.** The per-gate accept rate is
  the earliest signal that the teacher has drifted; noticing at row 500 instead
  of row 20,000 is the difference between a bad hour and a bad day.
* **Repair before discard.** Mechanical misses (length, emoji count/placement,
  ending) cost one extra generation to fix and two to replace.
* **Live throughput and ETA**, because a job whose finish time is unknown
  cannot be planned around.

Usage:
  generate_corpus.py --target 15000 --out data/run1 \
      --teacher-url http://127.0.0.1:8000 --teacher-model teacher \
      [--judge-url http://127.0.0.1:8081 --judge-model judge] \
      [--concurrency 16] [--stats-every 50]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fruitdrama import prompts as P            # noqa: E402
from fruitdrama import teacher as T            # noqa: E402
from fruitdrama.gates import CorpusDedup, check_story  # noqa: E402
from fruitdrama.judge import score_story       # noqa: E402
from fruitdrama.premises import PremiseEngine  # noqa: E402


def fewshot_pool(paths: list[pathlib.Path]) -> list[dict]:
    pool = []
    for p in paths:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("story") and rec.get("premise"):
                        pool.append(rec)
    return pool


def pick_fewshots(pool: list[dict], want_arch: str, rng: random.Random, k: int = 2):
    """Prefer exemplars from a DIFFERENT twist architecture than the target.

    Matching the architecture teaches the plot; contrasting it teaches the form
    while leaving the plot to the brief. The measured run showed no plot copying
    with this scheme.
    """
    if not pool:
        return []
    other = [e for e in pool if e.get("twist_architecture") != want_arch] or pool
    return rng.sample(other, min(k, len(other)))


def block_fewshots(pool: list[dict], block: int, k: int, seed: int = 0):
    """Few-shots held FIXED within a block of requests, rotated between blocks.

    Prefill is expensive here: measured prompt processing is 311-577 tok/s, and
    two exemplars plus the system prompt is ~2,000 tokens, i.e. 4-6s per request
    before a single token is generated. Holding the exemplars fixed makes
    system+exemplars a shared prefix that llama.cpp's slot prefix cache serves
    almost for free, so only the ~500-token brief needs processing.

    Rotating between blocks preserves style variety across the corpus; keeping
    them fixed *within* a block is what buys the cache hit.
    """
    if not pool:
        return []
    r = random.Random(seed * 1000003 + block)
    return r.sample(pool, min(k, len(pool)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=15000)
    ap.add_argument("--out", default="data/run1")
    ap.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    ap.add_argument("--teacher-model", default="teacher")
    ap.add_argument("--judge-url", default="")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--fewshot", type=int, default=2)
    ap.add_argument("--stats-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-attempts-multiple", type=float, default=3.0,
                    help="premises to draw per accepted row")
    ap.add_argument("--fewshot-block", type=int, default=250,
                    help="requests per fixed few-shot block (prefix-cache win)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(__file__).resolve().parents[1]

    tcfg = T.GenConfig(base_url=args.teacher_url, model=args.teacher_model,
                       concurrency=args.concurrency, max_tokens=1600)
    if not T.server_ready(tcfg):
        print(f"FATAL: teacher not responding at {args.teacher_url}", file=sys.stderr)
        return 2
    jcfg = None
    if args.judge_url:
        jcfg = T.GenConfig(base_url=args.judge_url, model=args.judge_model,
                           concurrency=max(4, args.concurrency // 2), max_tokens=900)
        if not T.server_ready(jcfg):
            print(f"WARN: judge not responding at {args.judge_url}; "
                  "storing gate verdicts only and deferring judging",
                  file=sys.stderr)
            jcfg = None

    raw_w = T.ResumableWriter(str(out / "raw.jsonl"))
    acc_w = T.ResumableWriter(str(out / "accepted.jsonl"))
    rej_w = T.ResumableWriter(str(out / "rejected.jsonl"))
    done = raw_w.done | rej_w.done

    pool = fewshot_pool([root / "data" / "gold" / "seeds.jsonl",
                         out / "accepted.jsonl"])
    print(f"few-shot pool: {len(pool)}")

    n_premises = int(args.target * args.max_attempts_multiple)
    eng = PremiseEngine(seed=args.seed)
    premises = [p for p in eng.generate(n_premises) if p.id not in done]
    print(f"premises: {n_premises} drawn, {len(premises)} outstanding, "
          f"target {args.target} accepted")

    dedup = CorpusDedup()
    for line in (out / "accepted.jsonl").read_text().splitlines() if (out / "accepted.jsonl").exists() else []:
        if line.strip():
            dedup.check(json.loads(line).get("story", ""))

    rng = random.Random(args.seed)
    stats = collections.Counter()
    gate_rejects = collections.Counter()
    judge_rejects = collections.Counter()
    t0 = time.time()
    prompt_v = P.prompt_version()

    def report():
        el = time.time() - t0
        acc = stats["accepted"]
        rate = acc / el * 3600 if el else 0
        eta = (args.target - acc) / rate if rate else 0
        print(
            f"[{el/60:6.1f}m] seen={stats['seen']:6d} accepted={acc:6d} "
            f"({100*acc/max(1,stats['seen']):4.1f}%) repaired={stats['repaired']:5d} "
            f"gate_rej={stats['gate_rej']:6d} judge_rej={stats['judge_rej']:5d} "
            f"dup={stats['dup']:4d} err={stats['err']:4d} | "
            f"{rate:5.0f} rows/h ETA {eta:4.1f}h",
            flush=True)
        if gate_rejects:
            print("        gates:", dict(gate_rejects.most_common(8)), flush=True)
        if judge_rejects:
            print("        judge:", dict(judge_rejects.most_common(6)), flush=True)

    def handle(premise, text):
        """Gate -> repair -> gate -> judge -> dedup. Returns True if accepted."""
        stats["seen"] += 1
        r = check_story(text, premise=premise.user_prompt)
        story = text
        if not r.ok:
            msgs = P.repair_messages(text, r.failures, r.stats)
            if msgs:
                try:
                    fixed = T.chat(tcfg, msgs)
                    r2 = check_story(fixed, premise=premise.user_prompt)
                    if r2.ok:
                        stats["repaired"] += 1
                        story, r = fixed, r2
                except Exception:
                    pass
        if not r.ok:
            stats["gate_rej"] += 1
            for g in dict.fromkeys(r.failures):
                gate_rejects[g] += 1
            rej_w.write({"premise_id": premise.id, "stage": "gates",
                         "failures": sorted(set(r.failures)),
                         "detail": r.detail, "stats": r.stats})
            return False

        if (why := dedup.check(story)):
            stats["dup"] += 1
            rej_w.write({"premise_id": premise.id, "stage": "dedup", "why": why})
            return False

        judged = None
        if jcfg:
            try:
                judged = score_story(jcfg, premise.user_prompt, story,
                                     order_seed=rng.randrange(7))
            except Exception:
                judged = None
            if judged is None:
                stats["judge_unparsed"] += 1
            elif not judged.ok:
                stats["judge_rej"] += 1
                for why in judged.reasons:
                    judge_rejects[why.split("=")[0]] += 1
                rej_w.write({"premise_id": premise.id, "stage": "judge",
                             "reasons": judged.reasons, "scores": judged.scores})
                return False

        stats["accepted"] += 1
        acc_w.write({
            "premise_id": premise.id,
            "premise": premise.user_prompt,
            "register": premise.register,
            "is_human_premise": premise.is_human_premise,
            "twist_architecture": premise.facets.get("twist"),
            "story": story,
            "gate_stats": r.stats,
            "judge": judged.scores if judged else None,
            "n_twists": judged.n_real_twists if judged else None,
            "prompt_version": prompt_v,
            "messages": P.training_messages(premise.user_prompt, story),
        })
        return True

    # Generate in waves so accepted rows can enter the few-shot pool and the
    # dedup index stays a single-threaded structure.
    wave = max(args.concurrency * 4, 64)
    idx = 0
    while stats["accepted"] < args.target and idx < len(premises):
        batch = premises[idx:idx + wave]
        idx += wave
        block = idx // max(1, args.fewshot_block)
        shots = block_fewshots(pool, block, args.fewshot, args.seed)
        tasks = [
            (p.id, P.teacher_messages(p.user_prompt, p.brief, shots), {})
            for p in batch
        ]
        by_id = {p.id: p for p in batch}
        results: dict[str, str] = {}

        def collect(rec, st, el):
            if rec.get("text"):
                results[rec["premise_id"]] = rec["text"]

        st = T.run_batch(tcfg, tasks, raw_w, collect)
        stats["err"] += st["failed"]
        for pid, text in results.items():
            handle(by_id[pid], text)
            if stats["seen"] % args.stats_every == 0:
                report()
        # Refresh the few-shot pool with newly accepted, judged stories.
        pool = fewshot_pool([root / "data" / "gold" / "seeds.jsonl",
                             out / "accepted.jsonl"])

    raw_w.close(); acc_w.close(); rej_w.close()
    report()
    summary = {
        "target": args.target, "accepted": stats["accepted"],
        "seen": stats["seen"], "repaired": stats["repaired"],
        "gate_rejects": dict(gate_rejects), "judge_rejects": dict(judge_rejects),
        "dedup_rejects": dict(dedup.rejects), "errors": stats["err"],
        "prompt_version": prompt_v, "elapsed_h": round((time.time()-t0)/3600, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
