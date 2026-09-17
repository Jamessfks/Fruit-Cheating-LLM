"""Text cleaning, safety (SFW), quality, and dedup helpers.

Everything here is provider-agnostic and dependency-free (stdlib only) so it can
run before any ML libraries are installed.
"""
import re
import hashlib

_WS = re.compile(r"\s+")


def _safe(t: str) -> str:
    """Drop lone surrogates / invalid code points so text is always UTF-8 writable."""
    if not t:
        return t
    return t.encode("utf-8", "ignore").decode("utf-8", "ignore")

# --- detokenization (Reddit WritingPrompts etc. are whitespace-tokenized) ---
_NEWLINE_TOK = re.compile(r"\s*<\s*newline\s*>\s*", re.I)
_TAG = re.compile(r"^\s*\[\s*[A-Z]{2,3}\s*\]\s*")          # leading [ WP ] [ EU ] [ CW ] ...
_SP_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:%\)\]\}])")
_SP_AFTER_OPEN = re.compile(r"([\(\[\{])\s+")
_CONTRACTION = re.compile(r"\s+'(s|re|ve|ll|d|m)\b")
_NOT = re.compile(r"\s+n't\b")


def detokenize(text: str) -> str:
    if not text:
        return ""
    t = text.replace("``", '"').replace("''", '"')
    t = _NEWLINE_TOK.sub("\n", t)
    t = _TAG.sub("", t)
    t = _SP_BEFORE_PUNCT.sub(r"\1", t)
    t = _SP_AFTER_OPEN.sub(r"\1", t)
    t = _NOT.sub("n't", t)
    t = _CONTRACTION.sub(r"'\1", t)
    t = re.sub(r'\s+"\s*', ' "', t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return _safe(t.strip())


def clean(text: str) -> str:
    """Light normalization applied to every field."""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # An emoji abutting a word ("<glyph>The air hung thick") renders badly and
    # is purely a spacing artifact.
    t = _EMOJI_ABUT.sub(r"\1 \2", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return _safe(t.strip())


# --- safety: platform-safe (SFW) filter ---
# Exact words: matched with word boundaries on both sides.
_BLOCK_WORDS = [
    "cock", "cocks", "cum", "cumming", "blowjob", "handjob", "cunt", "pussy",
    "anal", "creampie", "deepthroat", "gangbang", "bukkake", "fellatio",
    "cunnilingus", "dildo", "porn", "pornhub", "xxx", "nsfw", "titties",
    "boobs", "fuck", "fucking", "fucked", "fucker", "motherfucker",
    "nigger", "nigga", "faggot", "retard", "kike", "tranny", "coon",
]
# Stems: matched from a word boundary and allowed to run on
# ("masturbat" -> masturbating, masturbation).
_BLOCK_STEMS = ["masturbat", "orgasm", "ejaculat", "clitor"]

# Stays blocked as a slur, but exempts the two fixed idioms ("a chink of
# light", "a chink in the armour") rather than dropping the term entirely.
_BLOCK_CONTEXTUAL = [r"\bchink\b(?!\s+(?:of|in)\b)"]

# NOTE: v1 joined these with re.escape() and no boundaries, despite a comment
# claiming they were "word-boundaried below". The stem `cum` therefore matched
# inside "documents", "circumstance", "accumulated" and "Cucumber" -- and
# Cucumber is one of the 64 bible characters, so every Cucumber story was
# silently discarded. Boundaries are not cosmetic here.
_BLOCK_RE = re.compile(
    "|".join(
        [r"\b" + re.escape(w) + r"\b" for w in _BLOCK_WORDS]
        + [r"\b" + re.escape(w) + r"\w*" for w in _BLOCK_STEMS]
        + _BLOCK_CONTEXTUAL
    ),
    re.I,
)


def is_sfw(*texts: str) -> bool:
    return not _BLOCK_RE.search("\n".join(t for t in texts if t))


# --- quality ---
def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def quality_ok(text: str, min_len: int = 120, max_len: int = 12000,
               min_ascii: float = 0.85, min_ttr: float = 0.28) -> bool:
    """Coarse English-prose quality gate."""
    if not text:
        return False
    n = len(text)
    if n < min_len or n > max_len:
        return False
    if ascii_ratio(text) < min_ascii:
        return False
    if sum(c.isalpha() for c in text) / n < 0.5:
        return False
    words = text.split()
    if len(words) < 20:
        return False
    if len(set(w.lower() for w in words)) / len(words) < min_ttr:  # anti-repetition
        return False
    return True


# --- dedup ---
def norm_key(text: str) -> str:
    t = _WS.sub(" ", (text or "").lower()).strip()
    return hashlib.md5(t[:500].encode("utf-8")).hexdigest()


class Dedup:
    """Exact + normalized near-dup guard keyed on the assistant text."""

    def __init__(self):
        self.seen = set()

    def is_new(self, text: str) -> bool:
        k = norm_key(text)
        if k in self.seen:
            return False
        self.seen.add(k)
        return True


# keywords used to bias generic story sources toward soap-opera drama / twists
DRAMA_KW = re.compile(
    r"\b(betray\w*|affair|cheat\w*|mistress|lover|adulter\w*|secret\w*|revenge|"
    r"vengean\w*|marriage|married|divorce|widow\w*|pregnan\w*|inherit\w*|"
    r"scandal|confess\w*|jealous\w*|deceiv\w*|lie[sd]?|betrothed|scheme\w*|"
    r"twist|reveal\w*|blackmail|funeral|murder\w*|wedding|husband|wife|"
    r"fianc\w*|heir\w*|estate|dynasty|family)\b",
    re.I,
)


def is_dramatic(text: str, min_hits: int = 2) -> bool:
    return len(DRAMA_KW.findall(text or "")) >= min_hits


# --- story normalisation ---
_EMPH = re.compile(r"(?<!\w)(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1(?!\w)", re.S)
_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?|here(?:'s| is)[^\n]{0,60}|certainly[,!.]?)\s*\n+",
    re.I,
)
_BULLET = re.compile(r"^[ \t]*[-*+\u2022][ \t]+", re.M)
_EMOJI_ABUT = re.compile(
    "([\U0001F300-\U0001FAFF\U0001F000-\U0001F0FF\u2600-\u27BF\u2B00-\u2BFF])"
    "([A-Za-z])")


def normalize_story(text: str) -> str:
    """Strip removable formatting artifacts before gating and storage.

    Deliberately conservative: it removes markdown emphasis, a leading
    conversational preamble, and bullet markers, because those are artifacts of
    the generator rather than properties of the prose. It does NOT touch emoji
    placement, runs, length or endings -- those are genuine quality signals and
    silently repairing them would hide a teacher that is drifting.
    """
    if not text:
        return ""
    t = _safe(text).strip()
    t = _PREAMBLE.sub("", t)
    for _ in range(3):  # nested **_x_** needs more than one pass
        new = _EMPH.sub(r"\2", t)
        if new == t:
            break
        t = new
    t = _BULLET.sub("", t)
    # An emoji abutting a word ("<glyph>The air hung thick") renders badly and
    # is purely a spacing artifact.
    t = _EMOJI_ABUT.sub(r"\1 \2", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
