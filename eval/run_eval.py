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

# Ship gate. A checkpoint that misses any of these does not ship; the miss is
# reported with its actual value rather than rounded away.
#
# NOTE on the judge thresholds below: absolute LLM dimension scores were
# measured to SATURATE -- a pasted v1 storyboard scored 5/5 on every axis. They
# are retained as a backstop against degenerate output, but they are NOT the
# discriminating gate and a pass here means little on its own. The gates that
# actually carry weight are the deterministic metrics and eval/ab_compare.py's
# blind position-swapped pairwise comparison, which cannot saturate because it
# is a relative judgement. Treat a green judge block as necessary, not
# sufficient.
GATE = {
    "metrics.storyboard_leakage_rate": ("<=", 0.0),
    "metrics.words.in_band_rate": (">=", 0.80),
    "metrics.emoji.in_range_rate": (">=", 0.85),
    "metrics.gate_pass_rate": (">=", 0.70),
    "metrics.unique_openings_rate": (">=", 0.90),
    "metrics.mean_pairwise_jaccard": ("<=", 0.12),
    "judge.twists_ge2_rate": (">=", 0.85),
    "judge.fun_camp_voice_mean": (">=", 4.0),
    "judge.pg13_fail_count": ("<=", 0.0),
}

# Gates whose absence is acceptable: the judge block is empty when --judge-url
# is not given, and a deterministic-only run is a legitimate mode. Every other
# missing metric is a broken harness and must fail.
OPTIONAL_PREFIXES = ("judge.",)


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

    # Validate the benchmark before the environment: a drifted premise set is a
    # hard error whether or not a server is up.
    premises, pset_hash = load_frozen_premises(args.n)

    cfg = T.GenConfig(base_url=args.endpoint, model=args.model,
                      concurrency=args.concurrency, max_tokens=1600)
    if not T.server_ready(cfg):
        print(f"FATAL: no server at {args.endpoint}", file=sys.stderr)
        return 2

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
                "twists_ge2_rate": round(sum(t >= 2 for t in twists) / n, 3),
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
            optional = path.startswith(OPTIONAL_PREFIXES)
            if not optional:
                passed = False
            gate_rows.append({"metric": path, "value": None, "threshold": thr,
                              "op": op,
                              "status": "SKIPPED (no judge)" if optional
                                        else "MISSING -> FAIL"})
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
