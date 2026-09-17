#!/usr/bin/env python3
"""Score a checkpoint or endpoint over the frozen held-out premise set.

Writes a run directory with everything needed to defend or reproduce the
number: the config (endpoint, model, prompt version, premise-set hash), the raw
generations, deterministic metrics, judge scores, and a pass/fail gate report.

The premise set is drawn with PremiseEngine(mode="eval"), which uses only the
facet values reserved in data/eval_holdout.json. Those values never appear in
training, so contamination is structural rather than a similarity threshold.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import random
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from eval.metrics import story_metrics            # noqa: E402
from fruitdrama import prompts as P               # noqa: E402
from fruitdrama import teacher as T               # noqa: E402
from fruitdrama.judge import score_story          # noqa: E402
from fruitdrama.premises import PremiseEngine     # noqa: E402

# Ship gate. A checkpoint that misses any of these does not ship; the miss is
# reported with its actual value rather than rounded away.
GATE = {
    "storyboard_leakage_rate": ("<=", 0.0),
    "words.in_band_rate": (">=", 0.80),
    "emoji.in_range_rate": (">=", 0.85),
    "gate_pass_rate": (">=", 0.70),
    "unique_openings_rate": (">=", 0.90),
    "mean_pairwise_jaccard": ("<=", 0.12),
    "judge.twists_ge3_rate": (">=", 0.85),
    "judge.fun_camp_voice_mean": (">=", 4.0),
    "judge.pg13_fail_count": ("<=", 0.0),
}


def dig(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="fruit")
    ap.add_argument("--tag", required=True, help="checkpoint / artifact label")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--premise-seed", type=int, default=99)
    ap.add_argument("--judge-url", default="")
    ap.add_argument("--judge-model", default="judge")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out-root", default="eval/runs")
    args = ap.parse_args()

    cfg = T.GenConfig(base_url=args.endpoint, model=args.model,
                      concurrency=args.concurrency, max_tokens=1600)
    if not T.server_ready(cfg):
        print(f"FATAL: no server at {args.endpoint}", file=sys.stderr)
        return 2

    premises = list(PremiseEngine(seed=args.premise_seed, mode="eval").generate(args.n))
    pset_hash = hashlib.sha256(
        "\n".join(p.user_prompt for p in premises).encode()).hexdigest()[:16]

    stamp = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out = pathlib.Path(args.out_root) / f"{stamp}__{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    # Generate through the SAME serve-time prompt the product uses -- not the
    # teacher brief. Otherwise the eval measures a prompt the user never sends.
    tasks = [(p.id, [{"role": "system", "content": P.STORY_SYSTEM},
                     {"role": "user", "content": p.user_prompt}], {})
             for p in premises]
    w = T.ResumableWriter(str(out / "raw.jsonl"))
    t0 = time.time()
    st = T.run_batch(cfg, tasks, w, None)
    w.close()
    rows = {json.loads(l)["premise_id"]: json.loads(l)
            for l in (out / "raw.jsonl").read_text().splitlines() if l.strip()}
    pairs = [(p.user_prompt, rows[p.id].get("text", ""))
             for p in premises if p.id in rows and rows[p.id].get("text")]
    prompts_l, stories = [x[0] for x in pairs], [x[1] for x in pairs]

    metrics = story_metrics(stories, prompts_l)

    judge_block = {}
    if args.judge_url:
        jcfg = T.GenConfig(base_url=args.judge_url, model=args.judge_model,
                           concurrency=max(4, args.concurrency // 2), max_tokens=900)
        if T.server_ready(jcfg):
            rng = random.Random(7)
            jl = (out / "judge.jsonl").open("w", encoding="utf-8")
            twists, fun, pg13_fail, dims = [], [], 0, {}
            for prm, story in pairs:
                j = None
                try:
                    j = score_story(jcfg, prm, story, order_seed=rng.randrange(7))
                except Exception:
                    pass
                if j is None:
                    continue
                jl.write(json.dumps({"premise": prm, "scores": j.scores,
                                      "n_twists": j.n_real_twists,
                                      "accept": j.ok, "reasons": j.reasons},
                                     ensure_ascii=False) + "\n")
                twists.append(j.n_real_twists)
                if not j.scores.get("pg13", True):
                    pg13_fail += 1
                for k, _ in P.JUDGE_DIMENSIONS:
                    v = j.scores.get(k)
                    if isinstance(v, (int, float)):
                        dims.setdefault(k, []).append(v)
            jl.close()
            n = max(1, len(twists))
            judge_block = {
                "n_judged": len(twists),
                "twists_ge3_rate": round(sum(t >= 3 for t in twists) / n, 3),
                "twists_mean": round(sum(twists) / n, 2),
                "pg13_fail_count": pg13_fail,
                **{f"{k}_mean": round(sum(v) / len(v), 2) for k, v in dims.items()},
            }
        else:
            print(f"WARN: judge unreachable at {args.judge_url}", file=sys.stderr)

    report = {"metrics": metrics, "judge": judge_block}
    gate_rows, passed = [], True
    for path, (op, thr) in GATE.items():
        val = dig(report, path)
        if val is None:
            gate_rows.append({"metric": path, "value": None, "threshold": thr,
                              "op": op, "status": "MISSING"})
            continue
        ok = val <= thr if op == "<=" else val >= thr
        passed = passed and ok
        gate_rows.append({"metric": path, "value": val, "op": op,
                          "threshold": thr, "status": "pass" if ok else "FAIL"})

    (out / "config.json").write_text(json.dumps({
        "tag": args.tag, "endpoint": args.endpoint, "model": args.model,
        "prompt_version": P.prompt_version(), "premise_set_sha256": pset_hash,
        "n_premises": args.n, "premise_seed": args.premise_seed,
        "judge_model": args.judge_model if judge_block else None,
        "generated": len(stories), "gen_errors": st["failed"],
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "gate.json").write_text(json.dumps(
        {"pass": passed, "checks": gate_rows}, indent=2))

    print(json.dumps(report, indent=1))
    print("\nSHIP GATE:", "PASS" if passed else "FAIL")
    for row in gate_rows:
        if row["status"] != "pass":
            print(f"   {row['status']:7s} {row['metric']} = {row['value']} "
                  f"(need {row['op']} {row['threshold']})")
    print(f"\nrun dir: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
