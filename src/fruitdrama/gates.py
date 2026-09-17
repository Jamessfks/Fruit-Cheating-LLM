"""Deterministic gates for a candidate story. Stdlib only, no model calls.

Runs before the LLM judge so the judge only ever sees plausible candidates.
Every rejection carries a gate id and a human reason, because the accept rate
per gate is itself the most useful debugging signal when a generation run goes
wrong -- a sudden collapse in G1 means the teacher stopped honouring the length
ask, and that is worth knowing at row 500 rather than row 20,000.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

from . import contract as C
from . import filters as F
from . import textstats as T

_BIBLE_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "fruit_bible.json"
_bible_cache: list[dict] | None = None


def bible() -> list[dict]:
    global _bible_cache
    if _bible_cache is None:
        _bible_cache = json.loads(_BIBLE_PATH.read_text())
    return _bible_cache


def bible_names() -> set[str]:
    return {e["name"].lower() for e in bible()}


def cast_index() -> list[tuple[str, str]]:
    """(fruit name, given name) for each of the 64 characters.

    Counting bare fruit words punishes good prose: a writer introduces "Star
    Strawberry" once and then says "Star". Requiring the fruit surname three
    times would reward name-spamming and reject natural narration, so presence
    is measured across both forms.
    """
    out = []
    for e in bible():
        given = (e.get("example") or "").split()
        out.append((e["name"], given[0] if given else ""))
    return out


def character_mentions(text: str) -> dict[str, int]:
    """Mentions per character, combining fruit name and given name.

    Fruit names match case-insensitively; given names match case-sensitively
    with word boundaries, because a few ("Star", "Pearl", "Fern") are also
    ordinary words and only the capitalised form implies the character.
    """
    low = text.lower()
    counts: dict[str, int] = {}
    for fruit, given in cast_index():
        n = low.count(fruit.lower())
        if given:
            n += len(re.findall(r"\b" + re.escape(given) + r"\b", text))
        if n:
            counts[fruit] = n
    return counts


@dataclass
class GateResult:
    ok: bool
    failures: list[str] = field(default_factory=list)
    detail: dict[str, str] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)

    def fail(self, gate: str, reason: str) -> None:
        self.ok = False
        self.failures.append(gate)
        self.detail[gate] = reason


_TITLE_MAX_CHARS = 90


def split_title(text: str) -> tuple[str | None, str]:
    """Separate an optional title line from the story body.

    A title is a short opening line that does not end like a sentence. Keeping
    it out of the paragraph and density accounting matters: counted as a
    paragraph it skews emoji coverage, and counted as prose it skews the
    short-line ratio.
    """
    text = text.strip()
    parts = re.split(r"\n\s*\n+", text, maxsplit=1)
    if len(parts) == 2:
        head = parts[0].strip()
        if (
            "\n" not in head
            and len(head) <= _TITLE_MAX_CHARS
            and not head.endswith((".", "!", "?", "…", ",", ";", ":"))
            and T.word_count(head) <= 12
        ):
            return head, parts[1].strip()
    return None, text


def check_story(
    text: str,
    premise: str | None = None,
    required_fruits: list[str] | None = None,
) -> GateResult:
    r = GateResult(ok=True)
    text = F.normalize_story(text or "")
    if not text:
        r.fail("G0", "empty")
        return r

    title, body = split_title(text)
    wc = T.word_count(text)
    sents = T.sentences(body)
    paras = T.paragraphs(body)
    n_emoji = T.count_emoji(text)

    r.stats.update(
        words=wc,
        read_min=round(C.read_minutes(wc), 2),
        sentences=len(sents),
        paragraphs=len(paras),
        emoji=n_emoji,
        distinct_emoji=len(T.distinct_emoji(text)),
        emoji_per_sentence=round(n_emoji / len(sents), 3) if sents else 0.0,
        midsentence_emoji=round(T.midsentence_emoji_fraction(text), 3),
        para_emoji_coverage=round(T.paragraphs_with_emoji_fraction(body), 3),
        prose_ascii=round(T.prose_ascii_ratio(text), 4),
        has_title=1.0 if title else 0.0,
    )

    # G1 -- length. Rows failing only this are repairable, not discardable.
    if wc < C.WORD_GATE_MIN:
        r.fail("G1", f"too short: {wc}w < {C.WORD_GATE_MIN}")
    elif wc > C.WORD_GATE_MAX:
        r.fail("G1", f"too long: {wc}w > {C.WORD_GATE_MAX}")

    # G2 -- storyboard / scaffolding leakage. The failure mode the rebuild exists
    # to remove, so zero tolerance.
    m = C.LEAKAGE_RE.search(text)
    if m:
        r.fail("G2", f"format leakage: {m.group(0)[:40]!r}")
    for para in paras[1:]:
        # Strip trailing emoji and quotes first: a short line of dialogue ending
        # in a glyph is prose, not a heading.
        tail = T.strip_emoji(para).rstrip().rstrip("\"'”’)»")
        if ("\n" not in para
                and T.word_count(para) <= C.MID_HEADER_MAX_WORDS
                and not tail.endswith((".", "!", "?", "…", ",", ";", ":", "—", "-"))):
            r.fail("G2", f"mid-story section header: {para[:40]!r}")
            break
    n_ast = text.count("*")
    r.stats["asterisks"] = n_ast
    if n_ast > C.MAX_ASTERISKS:
        r.fail("G2", f"markdown emphasis: {n_ast} asterisks")

    # G3 -- emoji density per 100 words (style-stable; see contract.py).
    if wc:
        per100 = 100.0 * n_emoji / wc
        r.stats["emoji_per_100w"] = round(per100, 2)
        if per100 < C.EMOJI_PER_100W_MIN:
            r.fail("G3", f"too few emoji: {n_emoji} in {wc}w ({per100:.1f}/100w)")
        elif per100 > C.EMOJI_PER_100W_MAX:
            r.fail("G3", f"too many emoji: {n_emoji} in {wc}w ({per100:.1f}/100w)")

    # G4 -- spread, not clustering.
    if (mx := T.max_emoji_in_sentence(body)) > C.MAX_EMOJI_PER_SENTENCE:
        r.fail("G4", f"{mx} emoji in one sentence")
    segs = T.segments_with_emoji(body, C.N_SEGMENTS)
    r.stats["segments_with_emoji"] = segs
    if segs < C.MIN_SEGMENTS_WITH_EMOJI:
        r.fail("G4", f"emoji reach only {segs}/{C.N_SEGMENTS} parts of the story")
    if T.adjacent_emoji_pairs(body) > C.MAX_ADJACENT_EMOJI_PAIRS:
        r.fail("G4", "emoji runs")

    # G5 -- woven, not bolted on.
    if n_emoji:
        floating, clause_final = T.emoji_position_stats(text)
        r.stats["floating_emoji"] = floating
        r.stats["clause_final_frac"] = round(clause_final, 3)
        frac_floating = floating / n_emoji
        r.stats["floating_frac"] = round(frac_floating, 3)
        # One effective cap, not two that disagree: whichever of the absolute
        # floor and the fraction is more permissive at this emoji count.
        allowed = max(C.MAX_FLOATING_EMOJI,
                      int(C.MAX_FLOATING_FRACTION * n_emoji))
        if floating > allowed:
            r.fail("G5", f"{floating} of {n_emoji} emoji float between "
                          f"sentences (allowed {allowed})")
        if clause_final > C.MAX_CLAUSE_FINAL_EMOJI_FRACTION:
            r.fail("G5", f"{clause_final:.0%} of emoji just end a clause")

    # G6 -- PG-13.
    if not F.is_sfw(text):
        r.fail("G6", "tier-1 blocklist")
    if (m := C.EXPLICIT_TIER2_RE.search(text)):
        r.fail("G6", f"explicit: {m.group(0)[:30]!r}")
    if (m := C.GORE_TIER2_RE.search(text)):
        r.fail("G6", f"gore: {m.group(0)[:30]!r}")

    # G7 -- repetition.
    toks = [w.lower() for w in T.words(text)]
    ttr = len(set(toks)) / len(toks) if toks else 0.0
    r.stats["ttr"] = round(ttr, 3)
    if ttr < C.MIN_TTR:
        r.fail("G7", f"low lexical variety: ttr={ttr:.3f}")
    if T.has_repeated_ngram(text, C.MAX_NGRAM_REPEAT_N):
        r.fail("G7", f"repeated {C.MAX_NGRAM_REPEAT_N}-gram")
    if (ov := T.max_sentence_overlap(body)) > C.MAX_SENTENCE_OVERLAP:
        r.fail("G7", f"near-duplicate sentences: {ov:.2f}")

    # G8 -- charset. NOTE: filters.ascii_ratio() cannot be used here; emoji are
    # non-ASCII by design, so its 0.85 threshold would reject every good story.
    if T.prose_ascii_ratio(text) < C.MIN_PROSE_ASCII:
        r.fail("G8", "non-ASCII prose beyond emoji and typography")
    if text.rstrip().endswith("...") or text.rstrip().endswith("……"):
        r.fail("G8", "looks truncated mid-phrase")

    # G9 -- twist screen. A cheap filter, not a count; the judge does the count
    # and must quote each twist verbatim.
    hits = [m.start() for m in C.REVERSAL_RE.finditer(body)]
    late_cut = len(body) * C.REVERSAL_LATE_FRACTION
    late = sum(1 for h in hits if h >= late_cut)
    r.stats.update(reversal_hits=len(hits), reversal_late=late)
    if len(hits) < C.REVERSAL_MIN_HITS:
        r.fail("G9", f"only {len(hits)} reversal markers")
    elif late < C.REVERSAL_MIN_LATE_HITS:
        r.fail("G9", f"twists front-loaded: {late} late markers")

    # G10 -- ending must not deflate.
    tail = " ".join(sents[-2:]) if sents else body[-200:]
    # Strip closing quotes before testing terminal punctuation: a story that
    # ends on dialogue ("Whose is this?") ends with a quotation mark, and a
    # naive endswith() reads that as a story with no hook at all.
    tail_end = tail.rstrip().rstrip("\"'”’)»")
    if C.RESOLVED_ENDING_RE.search(tail):
        r.fail("G10", "resolved ending, no cliffhanger")
    else:
        last = sents[-1] if sents else tail
        hook = (
            tail_end.endswith(("?", "…"))
            or C.REVERSAL_RE.search(tail)
            # A short, punchy, unresolved closing line is a cliffhanger in its
            # own right; requiring "?" or a lexicon hit rejected real hooks.
            or T.word_count(last) <= 14
        )
        if not hook:
            r.fail("G10", "ending lands no reveal or hook")

    # G11 -- premise fidelity: a fruit the user named must actually be a lead.
    mentions = character_mentions(text)
    if required_fruits:
        for fruit in required_fruits:
            if mentions.get(fruit, 0) < C.MIN_NAMED_FRUIT_MENTIONS:
                r.fail("G11", f"named fruit underused: {fruit}")
                break
    present = [n for n, c in mentions.items() if c >= C.MIN_NAMED_FRUIT_MENTIONS]
    r.stats["cast_present"] = len(present)
    if len(present) < C.MIN_DISTINCT_CAST:
        r.fail("G11", f"only {len(present)} fruit characters carry the story")

    # G12 -- prose shape.
    if not (C.PARA_MIN <= len(paras) <= C.PARA_MAX):
        r.fail("G12", f"{len(paras)} paragraphs outside {C.PARA_MIN}-{C.PARA_MAX}")
    if any(T.word_count(p) > C.MAX_PARA_WORDS for p in paras):
        r.fail("G12", "a paragraph runs over length")
    lines = [ln for ln in body.split("\n") if ln.strip()]
    if lines:
        short = sum(1 for ln in lines if T.word_count(ln) < C.SHORT_LINE_WORDS)
        frac = short / len(lines)
        r.stats["short_line_frac"] = round(frac, 3)
        if frac > C.MAX_SHORT_LINE_FRACTION:
            r.fail("G12", f"drifted into lines/script: {frac:.2f} short lines")

    return r


def repairable(r: GateResult) -> bool:
    """True when length is the only problem, so one targeted rewrite recovers it.

    Worth distinguishing: length-only misses are ~a third of all rejections and
    cost 1x a generation to fix instead of 1x to throw away and 1x to replace.
    """
    return not r.ok and set(r.failures) == {"G1"}


class CorpusDedup:
    """Cross-story near-duplicate rejection, plus opening/closing collapse.

    The opening-line counter is the one that earns its keep: a teacher that has
    settled into one favourite first sentence still passes every per-story gate,
    and only a corpus-level counter notices.
    """

    def __init__(self) -> None:
        self._exact: set[str] = set()
        self._sigs: list[tuple[int, ...]] = []
        self._bands: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        self._open: dict[str, int] = {}
        self._close: dict[str, int] = {}
        self.rejects: dict[str, int] = {}
        self.band_rows = 4

    def _band_keys(self, sig: tuple[int, ...]):
        return [
            (i, sig[i * self.band_rows:(i + 1) * self.band_rows])
            for i in range(len(sig) // self.band_rows)
        ]

    def check(self, text: str) -> str | None:
        key = F.norm_key(text)
        if key in self._exact:
            self.rejects["exact"] = self.rejects.get("exact", 0) + 1
            return "exact duplicate"

        op = T.opening_ngram(text, C.OPENING_NGRAM)
        if op and self._open.get(op, 0) >= C.MAX_SHARED_OPENING:
            self.rejects["opening"] = self.rejects.get("opening", 0) + 1
            return f"opening line reused: {op!r}"
        cl = T.closing_ngram(text, C.OPENING_NGRAM)
        if cl and self._close.get(cl, 0) >= C.MAX_SHARED_CLOSING:
            self.rejects["closing"] = self.rejects.get("closing", 0) + 1
            return f"closing line reused: {cl!r}"

        sig = T.minhash(text, C.NGRAM_SHINGLE)
        # Banding keeps this O(1)-ish instead of comparing against every
        # accepted row, which would be 300M comparisons at corpus scale.
        candidates: set[int] = set()
        for bk in self._band_keys(sig):
            candidates.update(self._bands.get(bk, ()))
        for idx in candidates:
            j = T.minhash_jaccard(sig, self._sigs[idx])
            if j >= C.MINHASH_REJECT_JACCARD:
                self.rejects["near"] = self.rejects.get("near", 0) + 1
                return f"near-duplicate of row {idx} (J={j:.2f})"

        idx = len(self._sigs)
        self._sigs.append(sig)
        for bk in self._band_keys(sig):
            self._bands.setdefault(bk, []).append(idx)
        self._exact.add(key)
        self._open[op] = self._open.get(op, 0) + 1
        self._close[cl] = self._close.get(cl, 0) + 1
        return None
