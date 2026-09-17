"""The story format contract — one source of truth.

Every component reads its numbers from here: the teacher brief that asks for
stories, the deterministic gates that reject them, the judge rubric that scores
them, the eval harness that decides whether a checkpoint ships, and the
serve-time system prompt. If a threshold lives in two places it will drift, and
a corpus built against a drifted target is silently wrong.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- length
# 2-3 minute read. 250 wpm is the standard silent-reading rate for easy prose.
WORDS_PER_MINUTE = 250
WORD_MIN = 500
WORD_MAX = 750
WORD_TARGET = 620

# Gate band is slightly wider than the spec; the judge arbitrates the edges.
WORD_GATE_MIN = 480
WORD_GATE_MAX = 780

# What we ASK the generator for, as distinct from what the gate ENFORCES.
# Measured: with an ask of 620 and an explicit "do not exceed 750", gemma's
# length failures were 97 too-long and 0 too-short, median 813 words -- a
# systematic +80..+120 overshoot, because models cannot count words. Asking for
# a lower number moves the distribution into the band without touching the
# product spec, which stays WORD_MIN..WORD_MAX and is what the gate checks.
PROMPT_WORD_TARGET = 540
PROMPT_WORD_MAX = 650

# ---------------------------------------------------------------- emoji
# Measured per 100 words, NOT per sentence. Real generations run 8.2 words per
# sentence in this genre (short punchy dramatic lines), so a per-sentence target
# demanded 21-28 emoji for a 600-word story. The hand-authored seed sits at 2.9
# per 100 words and reads right; per-100-words is stable against sentence-length
# style, per-sentence is not.
EMOJI_PER_100W_MIN = 1.2   # ~7-8 emoji per 620 words; calibrated by reading
EMOJI_PER_100W_MAX = 5.0   # generations, not by copying the gold seeds
EMOJI_PER_SENTENCE_MIN = 0.33          # retained for reporting only
EMOJI_PER_SENTENCE_MAX = 1.00
MAX_EMOJI_PER_SENTENCE = 2             # 3+ in one sentence reads as spam
MAX_ADJACENT_EMOJI_PAIRS = 1           # allow one 🍓💔 flourish per story
MIN_PARAGRAPHS_WITH_EMOJI = 0.25       # reported; spread is enforced by segments
N_SEGMENTS = 5                         # equal text spans
MIN_SEGMENTS_WITH_EMOJI = 4            # of N_SEGMENTS -- style-stable spread test
# The gate that enforces "woven through the prose", not decoration: at least
# half of all emoji must sit mid-sentence rather than trailing a clause.
# Two distinct decorative patterns, measured separately:
#   * an emoji whose PRECEDING non-space char ends a sentence is floating
#     between sentences -- the worst form, capped hard.
#   * an emoji immediately BEFORE a terminator ends a clause, which reads fine
#     in moderation but becomes a tic if every sentence does it.
# Raised from 1 after reading a story rejected at 3: 614 words, 9 emoji at
# 1.47/100w, each attached to a concrete noun, genuinely good prose. Two or
# three boundary emoji in a 9-20 emoji story is a flourish; the pattern only
# becomes a tic when floating emoji DOMINATE, which the fraction cap catches
# and an absolute cap of 1 does not distinguish.
MAX_FLOATING_EMOJI = 3
MAX_FLOATING_FRACTION = 0.40
# Raised from 0.60 after reading a story at 0.615 that reads perfectly well.
# The metric conflates two different things: an emoji attached to a concrete
# noun that happens to END a sentence ("her ice sculptures [ice]." ) is
# well-placed, and it is only incidental that the noun is sentence-final.
# What is actually bad is an emoji attached to nothing -- and that is the
# `floating` check above, which is the real discriminator. This cap is kept
# only to catch the degenerate case where literally every sentence ends in one.
MAX_CLAUSE_FINAL_EMOJI_FRACTION = 0.85

# ---------------------------------------------------------------- shape
# Measured paragraph counts across 12 real generations: 11-39, median 17.
# Dialogue-heavy telenovela prose naturally runs long here, and it reads better
# than the 5-10 the first draft assumed. The short-line fraction below is what
# actually guards against drift into script format.
PARA_MIN = 5
PARA_MAX = 30
MAX_PARA_WORDS = 160
MAX_SHORT_LINE_FRACTION = 0.60         # catches drift into lists / script cues
SHORT_LINE_WORDS = 15

# ---------------------------------------------------------------- twists
TWIST_MIN = 3
TWIST_MAX = 5
REVERSAL_MIN_HITS = 8                  # cheap lexical screen, not a count
REVERSAL_LATE_FRACTION = 0.40          # >=3 hits must land in the last 60%
REVERSAL_MIN_LATE_HITS = 4

# ---------------------------------------------------------------- quality
MIN_TTR = 0.28                         # type/token ratio, anti-repetition
MAX_NGRAM_REPEAT_N = 8                 # no repeated 8-gram within one story
MAX_SENTENCE_OVERLAP = 0.80            # near-duplicate sentences
MIN_PROSE_ASCII = 0.95                 # computed AFTER emoji+typography strip
MAX_ASTERISKS = 4                      # the teacher emits 26-42 per story
MID_HEADER_MAX_WORDS = 8               # a short unpunctuated mid-story line is
                                       # a section header, not prose
MINHASH_REJECT_JACCARD = 0.50          # cross-story near-duplicate
MAX_SHARED_OPENING = 3                 # stories sharing a 6-gram opening
MAX_SHARED_CLOSING = 3
NGRAM_SHINGLE = 5
OPENING_NGRAM = 6

# ---------------------------------------------------------------- premise fidelity
MIN_NAMED_FRUIT_MENTIONS = 3           # a fruit the user named must be a lead
MIN_DISTINCT_CAST = 2

# ---------------------------------------------------------------- leakage
# The v1 model emitted 6-scene video storyboards. These patterns are the exact
# surface forms of that format plus the usual LLM scaffolding drift. Any hit is
# a hard reject: this is the failure mode the whole rebuild exists to remove.
LEAKAGE_PATTERNS = [
    r"^\s*(NARRATION|VISUAL|SCENE|EPISODE|TITLE|LOGLINE|CAST|SETTING)\s*[:\-—]",
    r"^\s*(INT|EXT)\.\s",
    r"^\s*Scene\s*(\d+|[—\-])",
    r"^\s*EPISODE\s+\d+",
    r"^\s*PART\s+\d+",
    r"^\s*#{1,6}\s",               # markdown headers
    r"^\s*[-*+]\s",                # bullets
    r"^\s*\d+[.)]\s",              # numbered outline
    r"^\s*-{3,}\s*$",              # horizontal rules
    r"^\s*\*{2}[^*\n]{1,80}\*{2}\s*$",   # bold-only line used as a header
    r"^[A-Z][A-Z \.'’\-]{3,}:",    # ALL-CAPS screenplay character cue
    r"^\s*\(.*\)\s*$",             # parenthetical stage direction
    r"\bV\.O\.\b",                     # voiceover
    # Episode-recap scaffolding. Slips through in TITLE position, where
    # split_title() removes it before the mid-story header check can see it.
    r"^\s*[^\w\n]{0,3}\s*previously\s+on\b",
    r"^\s*[^\w\n]{0,3}\s*(next time|to be continued|last (?:time|episode))\b",
    r"^\s*(FADE (IN|OUT)|CUT TO|DISSOLVE)\b",
]
LEAKAGE_RE = re.compile("|".join(LEAKAGE_PATTERNS), re.M | re.I)

# Endings that deflate a cliffhanger.
RESOLVED_ENDING_RE = re.compile(
    r"\b(happily ever after|at last at peace|the end|lived happily|"
    r"finally at peace|and so it ended|all was well|forgave (?:him|her|them) "
    r"completely)\b",
    re.I,
)

# ---------------------------------------------------------------- twist lexicon
# A *screen*, not a counter. Cheap enough to run on every generation; the real
# twist count comes from the judge, which must quote each twist verbatim.
REVERSAL_PHRASES = [
    "but what nobody knew", "what nobody knew", "but nobody knew", "little did",
    "had never been", "was never", "wasn't even", "was not even", "had always been",
    "all along", "the whole time", "turned out", "turns out", "as it turned out",
    "in truth", "the truth was", "the real", "the actual", "not his", "not hers",
    "not her", "not theirs", "never her", "never his", "except for one thing",
    "one problem", "there was just one", "months earlier", "years earlier",
    "weeks earlier", "the night before", "that same morning", "unbeknownst",
    "what she didn't know", "what he didn't know", "what they didn't know",
    "she had no idea", "he had no idea", "they had no idea", "had lied",
    "had been lying", "was lying", "the lie", "a lie", "faked", "had faked",
    "staged", "had staged", "planned it", "planned the whole", "orchestrated",
    "set (?:him|her|them) up", "had set", "it was she who", "it was he who",
    "it had been", "was actually", "actually the", "secretly", "in secret",
    "the secret", "her secret", "his secret", "their secret", "revealed",
    "the reveal", "confessed", "confession", "admitted", "the truth came",
    "came out", "found out", "discovered", "uncovered", "the test", "dna",
    "paternity", "the results", "the will", "rewrote", "forged", "forgery",
    "identical", "twin", "switched", "swapped", "impostor", "imposter",
    "the same", "recognized", "recognised", "remembered", "photograph",
    "the photo", "the video", "the recording", "recorded", "the letter",
    "the message", "the text", "the voicemail", "blackmail", "blackmailed",
    "threatened", "betrayed", "betrayal", "double-crossed", "revenge",
    "for years", "since the beginning", "from the start", "this whole time",
    "never loved", "never really", "didn't love", "wasn't hers", "wasn't his",
    "belonged to", "the father", "the mother", "pregnant", "still alive",
    "not dead", "alive", "returned", "came back", "back from", "had survived",
    "poisoned", "the poison", "the knife", "the gun", "insurance",
    "the money was", "bankrupt", "broke", "stole", "had stolen", "embezzled",
    "had been told", "had refused", "had known", "had not known", "had stopped",
    "were not the only", "was not the only", "neither of those", "neither of them",
    "had agreed", "had waited", "had counted on", "had decided", "had taken",
    "had walked into", "had hired", "had paid", "had arranged", "had kept",
    "had hidden", "had signed", "had rewritten", "had always", "had never",
]
# Past perfect ("had lied", "had never been told") and negation reframing
# ("was not his", "neither of them") are the grammar of the reveal.
REVERSAL_STRUCTURAL = [
    r"\bhad (?:never |always |not |already |long |been |only )*[a-z]+(?:ed|en|wn|pt|lt|ld|de|ne)\b",
    r"\b(?:was|were|had|has|is|are)(?:n\u2019t|n't| not| never)\b",
    r"\bneither\b",
    r"\bnot (?:his|her|hers|theirs|mine|yours|ours|the)\b",
    r"\bnever (?:been|had|loved|wanted|meant|intended|once)\b",
]

REVERSAL_RE = re.compile(
    "|".join(
        [r"\b" + p if "(" not in p else p for p in REVERSAL_PHRASES]
        + REVERSAL_STRUCTURAL
    ),
    re.I,
)

# ---------------------------------------------------------------- content bar
# PG-13 telenovela camp. src.filters.is_sfw() covers slurs and the crudest
# sexual vocabulary; this is the second tier it misses -- explicit acts, and
# the gore that campy betrayal drama does not need.
EXPLICIT_TIER2 = [
    r"\bthrust(?:ing|s)?\b", r"\bpenetrat", r"\bgenital", r"\borgasm",
    r"\bclimax(?:ed|ing)?\b", r"\baroused?\b", r"\bnipple", r"\bgroin\b",
    r"\bnaked bod", r"\bstrip(?:ped)? (?:her|him|them) (?:bare|naked)\b",
    r"\bmounted (?:her|him)\b", r"\bmoan(?:ed|ing|s)?\b", r"\bgasp(?:ed|ing)? "
    r"with pleasure\b", r"\bwrithe?(?:d|ing)?\b", r"\bfondl", r"\bcaress(?:ed|ing)? "
    r"(?:her|his) (?:breast|thigh)", r"\bundress",
]
GORE_TIER2 = [
    r"\bdismember", r"\bdisembowel", r"\bentrails\b", r"\bpool of blood\b",
    r"\bblood(?:y)? pulp\b", r"\bskull (?:cracked|caved)", r"\bsevered (?:head|limb)",
    r"\bgutted (?:her|him|them)\b", r"\bmutilat",
]
EXPLICIT_TIER2_RE = re.compile("|".join(EXPLICIT_TIER2), re.I)
GORE_TIER2_RE = re.compile("|".join(GORE_TIER2), re.I)


def read_minutes(n_words: int) -> float:
    """Reading time in minutes for a word count."""
    return n_words / WORDS_PER_MINUTE


def emoji_budget(n_words: int) -> tuple[int, int]:
    """Advisory emoji count range for a target length, for the teacher brief.

    Derived from the per-sentence bounds via a ~15-word mean sentence so the
    brief can state a concrete number -- models hit stated numbers far more
    reliably than implied ones.
    """
    sentences = max(1, round(n_words / 15))
    return (
        max(1, int(sentences * EMOJI_PER_SENTENCE_MIN)),
        int(sentences * EMOJI_PER_SENTENCE_MAX),
    )
