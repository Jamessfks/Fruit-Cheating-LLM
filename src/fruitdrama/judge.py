"""LLM-judge driver: score candidate stories, store every score.

Two deliberate choices:

* The twist count is never asked for as a number. The rubric requires each
  twist to be quoted verbatim from the story and flagged for whether it
  reframes earlier text; the count is derived here from the list length. An
  unanchored ``{"twists": 4}`` is worthless, and models will happily produce one.
* Every score is written to disk, and admission thresholds are applied
  afterwards from contract/prompts. Re-thresholding a 15k-row corpus then costs
  seconds instead of re-judging for hours.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import prompts as P
from . import teacher as T

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.M)


def parse_judge(raw: str) -> dict | None:
    """Best-effort JSON extraction from a judge response.

    Judges wrap JSON in prose or code fences often enough that a bare
    json.loads() throws away a meaningful fraction of otherwise-valid scores.
    """
    if not raw:
        return None
    text = _FENCE.sub("", raw).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to the outermost balanced brace span.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


@dataclass
class Judgement:
    ok: bool
    reasons: list[str]
    scores: dict
    n_real_twists: int


def score_story(
    cfg: T.GenConfig,
    user_prompt: str,
    story: str,
    order_seed: int = 0,
) -> Judgement | None:
    """One judged story. Temperature 0 so the gate is reproducible."""
    msgs = P.judge_messages(user_prompt, story, order_seed=order_seed)
    raw = T.chat(cfg, msgs, temperature=0.0, top_p=1.0, max_tokens=900)
    scores = parse_judge(raw)
    if scores is None:
        return None
    ok, reasons = P.judge_accept(scores)
    real = [
        t for t in scores.get("twists", [])
        if isinstance(t, dict) and t.get("recontextualizes")
        and (t.get("genuineness") or 0) >= 3
    ]
    # A quote the judge invented is not evidence of a twist. Verify each anchor
    # actually occurs in the story before it counts.
    verified = [t for t in real if _quote_present(t.get("quote", ""), story)]
    dropped = len(real) - len(verified)
    if dropped:
        reasons.append(f"{dropped} twist quote(s) not found in story")
    # Unverifiable quotes are tolerated only while enough verified twists remain.
    accept = ok and len(verified) >= P.JUDGE_MIN_VERIFIED_TWISTS
    return Judgement(accept, reasons, scores, len(verified))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())


def _quote_present(quote: str, story: str) -> bool:
    q = " ".join(_norm(quote).split())
    if len(q) < 8:
        return False
    hay = " ".join(_norm(story).split())
    if q in hay:
        return True
    # Tolerate light paraphrase at the edges: require a solid contiguous run.
    words = q.split()
    for n in (6, 5, 4):
        if len(words) >= n and any(
            " ".join(words[i:i + n]) in hay for i in range(len(words) - n + 1)
        ):
            return True
    return False
