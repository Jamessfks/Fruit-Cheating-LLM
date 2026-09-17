#!/usr/bin/env python3
"""Second pass: judge the gate-accepted corpus and write the final SFT set.

Run as a separate pass rather than inline with generation, for three reasons:

* **Model independence.** The judge must not be the model that wrote the story,
  or it rewards its own diction. Generation loads the teacher; judging loads a
  different family. One at a time keeps the memory budget simple.
* **Inspectability.** The corpus can be read and the rubric sanity-checked
  before spending judge hours on it.
* **Re-thresholding is free.** Every score is stored, so changing the admission
  bar costs a re-read instead of a re-judge.

Usage:
  judge_corpus.py --accepted data/run1/accepted.jsonl \
      --judge-url http://127.0.0.1:8000 --judge-model judge --concurrency 16
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fruitdrama import prompts as P     # noqa: E402
from fruitdrama import teacher as T     # noqa: E402
from fruitdrama.judge import score_story  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", default="data/run1/accepted.jsonl")
    ap.add_argument("--out", default="data/run1/judged.jsonl")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8000")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--stats-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = T.GenConfig(base_url=args.judge_url, model=args.judge_model,
                      concurrency=args.concurrency, max_tokens=900,
                      temperature=0.0, top_p=1.0)
    if not T.server_ready(cfg):
        print(f"FATAL: no judge at {args.judge_url}", file=sys.stderr)
        return 2

    src = pathlib.Path(args.accepted)
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    writer = T.ResumableWriter(args.out)
    pending = [r for r in rows if r["premise_id"] not in writer.done]
    print(f"{len(rows)} rows, {len(pending)} to judge "
          f"({len(rows) - len(pending)} already done)", flush=True)

    rng = random.Random(11)
    stats = collections.Counter()
    reasons = collections.Counter()
    dims: dict[str, list[float]] = collections.defaultdict(list)
    t0 = time.time()

    def work(r):
        # Rotate rubric item order per story to blunt position bias.
        j = score_story(cfg, r["premise"], r["story"],
                        order_seed=rng.randrange(7))
        return r, j

    def report():
        el = time.time() - t0
        done = stats["pass"] + stats["fail"] + stats["unparsed"]
        rate = done / el * 3600 if el else 0
        print(f"[{el/60:6.1f}m] judged={done:6d} pass={stats['pass']:6d} "
              f"({100*stats['pass']/max(1,done):4.1f}%) fail={stats['fail']:5d} "
              f"unparsed={stats['unparsed']:4d} | {rate:5.0f}/h "
              f"ETA {(len(pending)-done)/max(1,rate):4.1f}h", flush=True)
        if reasons:
            print("        top reasons:", dict(reasons.most_common(6)), flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(work, r) for r in pending]
        for fut in as_completed(futs):
            try:
                r, j = fut.result()
            except Exception as e:  # noqa: BLE001
                stats["error"] += 1
                continue
            if j is None:
                stats["unparsed"] += 1
                writer.write({"premise_id": r["premise_id"], "judge_ok": None,
                              "error": "unparseable"})
            else:
                stats["pass" if j.ok else "fail"] += 1
                if not j.ok:
                    for why in j.reasons:
                        reasons[why.split("=")[0]] += 1
                for k, _ in P.JUDGE_DIMENSIONS:
                    v = j.scores.get(k)
                    if isinstance(v, (int, float)):
                        dims[k].append(v)
                writer.write({
                    "premise_id": r["premise_id"], "judge_ok": j.ok,
                    "n_twists": j.n_real_twists, "reasons": j.reasons,
                    "scores": j.scores,
                })
            done = stats["pass"] + stats["fail"] + stats["unparsed"]
            if done % args.stats_every == 0:
                report()
    writer.close()
    report()

    summary = {
        "judged": stats["pass"] + stats["fail"] + stats["unparsed"],
        "passed": stats["pass"], "failed": stats["fail"],
        "unparsed": stats["unparsed"], "errors": stats["error"],
        "judge_model": args.judge_model,
        "dimension_means": {k: round(sum(v) / len(v), 2) for k, v in dims.items() if v},
        "top_fail_reasons": dict(reasons.most_common(10)),
        "elapsed_h": round((time.time() - t0) / 3600, 2),
    }
    pathlib.Path(args.out).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
