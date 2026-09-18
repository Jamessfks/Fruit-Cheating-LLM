"""Deterministic eval metrics. No model calls, no judge, runs in seconds.

These are the metrics that already discriminate a tuned checkpoint from a base
one on format, length, emoji and diversity -- so they are worth running on every
checkpoint, long before the judge is involved.

The highest-signal metric here is `unique_openings`. A model that has collapsed
onto one favourite first sentence still scores acceptably on length, emoji and
format, and only a corpus-level opening check notices.
"""

from __future__ import annotations

import collections
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fruitdrama import contract as C      # noqa: E402
from fruitdrama import textstats as T     # noqa: E402
from fruitdrama.gates import check_story  # noqa: E402


def _pct(values, q):
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def story_metrics(stories: list[str], premises: list[str] | None = None) -> dict:
    premises = premises or [None] * len(stories)
    words, emoji, dens, paras, clause, segs, revs = [], [], [], [], [], [], []
    gate_fail = collections.Counter()
    n_pass = 0
    leakage = 0
    for s, p in zip(stories, premises):
        r = check_story(s, premise=p)
        if r.ok:
            n_pass += 1
        else:
            for g in dict.fromkeys(r.failures):
                gate_fail[g] += 1
            if "G2" in r.failures:
                leakage += 1
        st = r.stats
        words.append(st.get("words", 0))
        emoji.append(st.get("emoji", 0))
        dens.append(st.get("emoji_per_100w", 0))
        paras.append(st.get("paragraphs", 0))
        clause.append(st.get("clause_final_frac", 0))
        segs.append(st.get("segments_with_emoji", 0))
        revs.append(st.get("reversal_hits", 0))

    n = max(1, len(stories))
    openings = {T.opening_ngram(s, 6) for s in stories}
    closings = {T.closing_ngram(s, 6) for s in stories}
    in_band = sum(C.WORD_MIN <= w <= C.WORD_MAX for w in words) / n
    return {
        "n": len(stories),
        "gate_pass_rate": round(n_pass / n, 3),
        "gate_failures": dict(gate_fail.most_common()),
        "storyboard_leakage_rate": round(leakage / n, 4),
        "words": {"p10": _pct(words, .10), "p50": _pct(words, .50),
                  "p90": _pct(words, .90), "in_band_rate": round(in_band, 3)},
        "read_minutes_p50": round(_pct(words, .50) / C.WORDS_PER_MINUTE, 2),
        "emoji": {"p50": _pct(emoji, .50),
                  "density_p50": _pct(dens, .50),
                  "in_range_rate": round(
                      sum(C.EMOJI_PER_100W_MIN <= d <= C.EMOJI_PER_100W_MAX
                          for d in dens) / n, 3)},
        "emoji_clause_final_p50": _pct(clause, .50),
        "emoji_segments_p50": _pct(segs, .50),
        "paragraphs_p50": _pct(paras, .50),
        "reversal_hits_p50": _pct(revs, .50),
        # Diversity: the metrics that answer "50 stories, or one story with the
        # names swapped?"
        "unique_openings_rate": round(len(openings) / n, 3),
        "unique_closings_rate": round(len(closings) / n, 3),
        # NOTE: distinct-n is scale-dependent -- total n-grams grow faster than
        # unique ones, so a larger corpus always scores lower. Only compare
        # equal-sized samples. Measured on this corpus: 0.82 within any 500-row
        # window but 0.66 across 3,017 rows, and that drop is arithmetic, not
        # degradation (the first-500 and last-500 windows were 0.819 vs 0.810).
        "distinct_3": round(T.distinct_n(stories, 3), 4),
        "distinct_4": round(T.distinct_n(stories, 4), 4),
        "mean_pairwise_jaccard": round(_mean_pairwise_jaccard(stories), 4),
    }


def _mean_pairwise_jaccard(stories: list[str], shingle: int = 5, cap: int = 60) -> float:
    """Mean pairwise 5-gram Jaccard. Capped sample: it is O(n^2)."""
    subset = stories[:cap]
    sets = []
    for s in subset:
        toks = [w.lower() for w in T.words(s)]
        sets.append({" ".join(toks[i:i + shingle]) for i in range(len(toks) - shingle + 1)})
    if len(sets) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if u:
                total += len(sets[i] & sets[j]) / len(u)
                pairs += 1
    return total / pairs if pairs else 0.0
