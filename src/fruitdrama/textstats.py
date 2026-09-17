"""Text measurement for the story contract. Stdlib only.

The emoji logic is the load-bearing part. A naive codepoint count treats
"👩‍❤️‍👨" as three emoji and "🍓🏽" as two, which inflates every density
metric and would make the gates reject good stories while passing spammy ones.
Everything here counts *graphemes* -- one user-perceived emoji, one unit.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# ---------------------------------------------------------------- emoji
# Base pictographs. Deliberately excludes the arrow blocks (U+2190-21FF) and
# ©/®/™, which appear in ordinary typography and would be false positives.
_BASE = (
    "\U0001F300-\U0001FAFF"   # pictographs, emoticons, transport, supplemental
    "\U0001F000-\U0001F0FF"   # mahjong, dominoes, playing cards
    "☀-➿"           # misc symbols + dingbats (includes ❤ ✨ ☀)
    "⬀-⯿"           # misc symbols and arrows (includes ⭐)
    "←-⇿"           # excluded below; listed for documentation only
)
_BASE_CLASS = (
    "[\U0001F300-\U0001FAFF\U0001F000-\U0001F0FF☀-➿⬀-⯿]"
)
_RI = "[\U0001F1E6-\U0001F1FF]"                 # regional indicator (flags)
_MOD = "[\U0001F3FB-\U0001F3FF️⃣‍]"
_SKIN = "[\U0001F3FB-\U0001F3FF]"
_ZWJ = "‍"

# Order matters: keycaps and flag pairs must be tried before the general
# sequence, or their components get counted individually.
EMOJI_RE = re.compile(
    "(?:"
    r"[0-9#*]️?⃣"                               # keycap: 3️⃣
    f"|{_RI}{_RI}"                                        # flag: 🇺🇸
    f"|{_BASE_CLASS}(?:{_SKIN}|️)*"                  # base + modifiers
    f"(?:{_ZWJ}{_BASE_CLASS}(?:{_SKIN}|️)*)*"        # ZWJ continuations
    ")"
)

# Typographic characters that are non-ASCII but stylistically wanted in good
# prose. Stripped before measuring ASCII purity so curly quotes and em dashes
# are not mistaken for encoding damage.
_TYPOGRAPHY = "‘’“”–—… ′″éèíñüöá"

_SENT_SPLIT = re.compile(r"(?<=[.!?…])[\"'”’)]*\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9'’\-]+")


def emoji_spans(text: str) -> list[tuple[int, int, str]]:
    """Every emoji grapheme as (start, end, glyph)."""
    return [(m.start(), m.end(), m.group()) for m in EMOJI_RE.finditer(text)]


def count_emoji(text: str) -> int:
    return len(emoji_spans(text))


def strip_emoji(text: str) -> str:
    return EMOJI_RE.sub("", text)


def distinct_emoji(text: str) -> set[str]:
    return {g for _, _, g in emoji_spans(text)}


def sentences(text: str) -> list[str]:
    """Sentence split tolerant of emoji, quotes and ellipses."""
    flat = re.sub(r"\s*\n+\s*", " ", text.strip())
    parts = [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]
    return parts or ([flat] if flat else [])


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]


def words(text: str) -> list[str]:
    return _WORD_RE.findall(strip_emoji(text))


def word_count(text: str) -> int:
    return len(words(text))


def trailing_emoji_fraction(text: str) -> float:
    """Fraction of emoji that decorate a boundary rather than sit inside a clause.

    An emoji is decorative if a sentence terminator sits immediately on either
    side of it. Checking only what follows is not enough: the most common
    bolted-on pattern is "She left. \U0001F353 He lied." -- the emoji trails the
    previous sentence while *looking* mid-stream to a forward-only check.
    """
    spans = emoji_spans(text)
    if not spans:
        return 0.0
    terminators = ".!?\u2026\n"
    quotes = "\"'\u201d\u2019)"
    trailing = 0
    for start, end, _ in spans:
        # Look backward, skipping spaces, quotes and any adjacent emoji run.
        before = text[:start].rstrip(" \t")
        while True:
            m = EMOJI_RE.search(before)
            if m and m.end() == len(before):
                before = before[:m.start()].rstrip(" \t")
            else:
                break
        before = before.rstrip(quotes)
        preceded = before == "" or before[-1] in terminators

        # Look forward, skipping spaces and any adjacent emoji run.
        after = text[end:].lstrip(" \t")
        while True:
            m = EMOJI_RE.match(after)
            if not m:
                break
            after = after[m.end():].lstrip(" \t")
        after = after.lstrip(quotes)
        followed = after == "" or after[0] in terminators

        if preceded or followed:
            trailing += 1
    return trailing / len(spans)


def midsentence_emoji_fraction(text: str) -> float:
    return 1.0 - trailing_emoji_fraction(text)


def adjacent_emoji_pairs(text: str) -> int:
    """Count runs of >=2 emoji separated by nothing but spaces."""
    spans = emoji_spans(text)
    pairs = 0
    for i in range(len(spans) - 1):
        between = text[spans[i][1]:spans[i + 1][0]]
        if between.strip(" \t") == "":
            pairs += 1
    return pairs


def max_emoji_in_sentence(text: str) -> int:
    return max((count_emoji(s) for s in sentences(text)), default=0)


def paragraphs_with_emoji_fraction(text: str) -> float:
    paras = paragraphs(text)
    if not paras:
        return 0.0
    return sum(1 for p in paras if count_emoji(p)) / len(paras)


def segments_with_emoji(text: str, n: int = 5) -> int:
    """How many of n equal-length slices of the text contain an emoji."""
    if not text:
        return 0
    size = max(1, len(text) // n)
    return sum(1 for i in range(n) if count_emoji(text[i * size:(i + 1) * size]))


def prose_ascii_ratio(text: str) -> float:
    """ASCII purity of the prose, ignoring emoji and wanted typography.

    filters.ascii_ratio() cannot be used directly on these stories: emoji are
    non-ASCII by construction, so a 0.85 threshold on the raw text would reject
    every correctly-formatted story. Strip emoji and accepted typography first.
    """
    body = strip_emoji(text)
    body = "".join(c for c in body if c not in _TYPOGRAPHY)
    if not body:
        return 0.0
    return sum(1 for c in body if ord(c) < 128) / len(body)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def has_repeated_ngram(text: str, n: int) -> bool:
    toks = [w.lower() for w in words(text)]
    seen: set[tuple[str, ...]] = set()
    for g in ngrams(toks, n):
        if g in seen:
            return True
        seen.add(g)
    return False


def max_sentence_overlap(text: str) -> float:
    """Highest Jaccard overlap between any two sentences in the story."""
    sents = [set(w.lower() for w in words(s)) for s in sentences(text)]
    sents = [s for s in sents if len(s) >= 5]
    worst = 0.0
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            union = sents[i] | sents[j]
            if not union:
                continue
            worst = max(worst, len(sents[i] & sents[j]) / len(union))
    return worst


def distinct_n(texts: list[str], n: int = 3) -> float:
    """Corpus-level diversity: unique n-grams / total n-grams across texts.

    This is the metric that answers 'did 50 premises yield 50 stories, or one
    story with the names swapped?'
    """
    total = 0
    uniq: set[tuple[str, ...]] = set()
    for t in texts:
        g = ngrams([w.lower() for w in words(t)], n)
        total += len(g)
        uniq.update(g)
    return len(uniq) / total if total else 0.0


# ---------------------------------------------------------------- MinHash
_MASK = (1 << 64) - 1


def _shingles(text: str, k: int) -> set[str]:
    toks = [w.lower() for w in words(text)]
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash(text: str, k: int = 5, perms: int = 64) -> tuple[int, ...]:
    """64-permutation MinHash signature. Stdlib only, no datasketch.

    Uses one md5 per shingle plus cheap XOR-multiply permutations, which is
    accurate enough for near-duplicate rejection and fast enough for 26k rows.
    """
    sh = _shingles(text, k)
    if not sh:
        return tuple([0] * perms)
    base = [int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big") for s in sh]
    sig = []
    for p in range(perms):
        a = (p * 0x9E3779B97F4A7C15 + 0x165667B19E3779F9) & _MASK
        sig.append(min(((h ^ a) * 0xFF51AFD7ED558CCD) & _MASK for h in base))
    return tuple(sig)


def minhash_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def opening_ngram(text: str, n: int = 6) -> str:
    toks = [w.lower() for w in words(text)][:n]
    return " ".join(toks)


def closing_ngram(text: str, n: int = 6) -> str:
    toks = [w.lower() for w in words(text)][-n:]
    return " ".join(toks)
