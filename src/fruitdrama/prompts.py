"""Prompt assembly. Three distinct prompts, deliberately kept apart.

1. ``story_system()``   -- the SERVE-TIME system prompt. Byte-identical on every
   request so the inference server's prefix cache always hits, and identical in
   the training rows so there is no train/serve skew.
2. ``teacher_messages()`` -- the rich internal brief. Teacher-only; never stored
   in a training row. See premises.py for why the asymmetry matters.
3. ``judge_messages()`` -- the scoring rubric.

All three read their numbers from contract.py, so the target cannot drift
between what we ask for, what we filter on, and what we score.
"""

from __future__ import annotations

import json
import math
import textwrap

from . import contract as C


def story_system() -> str:
    """The product's voice. Stable text -- changing it changes prompt_version.

    The emoji section is deliberately long and shows contrasting examples. With
    a short instruction the teacher put 100% of emoji at clause endings, in
    runs of two, and only 3-13 per story. Models follow a demonstrated pattern
    and a stated number; they do not follow "weave them in".
    """
    lo, hi = C.WORD_MIN, C.WORD_MAX
    ask, ask_max = C.PROMPT_WORD_TARGET, C.PROMPT_WORD_MAX
    # Floor must hold even for the longest story the gate will accept; ceiling
    # is computed at the length we ask for, so a short story cannot exceed it.
    # Floor stays derived from the longest acceptable story so the ask is always
    # satisfiable; the target is set well above it because the model
    # consistently lands below whatever number it is given.
    emin = math.ceil(C.EMOJI_PER_100W_MIN * C.WORD_GATE_MAX / 100) + 4
    etarget = C.EMOJI_PROMPT_TARGET
    emax = int(C.EMOJI_PER_100W_MAX * C.WORD_MIN / 100) - 1
    return textwrap.dedent(f"""\
        You write short fruit-drama stories: campy, vivid telenovela tales where
        fruits and vegetables are people who love, lie, scheme and betray each other.

        Given any premise, write ONE complete short story.

        FORM
        - Exactly ONE title line at the very top, then unbroken prose paragraphs.
        - After that title, nothing that looks like a heading ever again: no
          second title, no date lines, no chapter or part labels, no scene
          labels, no "Later"/"At the hospital" signposts, no NARRATION: or
          VISUAL: labels, no camera or stage directions in parentheses.
        - About {ask} words. Never more than {ask_max}. That is a 2-3 minute read.
        - Plain prose only. No markdown of any kind: no asterisks, no bold, no
          italics, no headers, no bullets, no numbered lists.
        - Never scene headings, never NARRATION: or VISUAL: labels, never
          INT./EXT., never stage directions in parentheses.
        - Dialogue is welcome. Give characters real names and let them speak.

        VOICE
        - Warm, gossipy, a little unhinged. Concrete sensory detail over abstraction.
        - The fruit nature of the characters must matter: what they are made of,
          how they bruise, ripen, spoil, hold a grudge.

        EMOJI -- read this carefully
        - Use about {etarget} emoji. Count them before you finish. Fewer than
          {emin} is wrong and the story must be rewritten with more.
        - Put them INSIDE sentences, attached to the noun or feeling they colour.
        - Never two emoji in a row. Never one floating alone between sentences.
        - Do not end every sentence with one. At most half should sit next to a
          full stop.

        Like this:
            She found the receipt {'\U0001F9FE'} folded in his jacket and said nothing.
            Bianca's plums {'\U0001F7E3'} hit the pavement, one after another.
            Her hands {'\U0001F91A'} shook as she read the second name.

        Not like this:
            She found the receipt folded in his jacket. {'\U0001F9FE'} She said nothing.
            She found the receipt folded in his jacket. {'\U0001F9FE'}{'\U0001F494'}
            Bianca's plums hit the pavement {'\U0001F7E3'}{'\U0001F62D'}.
            Her hands shook {'\U0001F91A'}. She read the second name {'\U0001F494'}.

        TWISTS
        - {C.TWIST_MIN}-{C.TWIST_MAX} real twists, each one bigger than the last.
        - A twist must reframe something the reader already accepted, so that going
          back you realise it was there all along.
        - Save the biggest for the final lines. The last sentence must be a question,
          or a reveal that reframes everything. Never resolve it neatly, never write
          "to be continued".

        LIMITS
        - Keep it PG-13. Affairs, betrayal, secrets, slaps, scandal and scheming are
          the genre. Imply the bedroom, never describe it. No explicit sex, no gore.
        - If the premise has no fruit in it, cast fruits yourself and tell it as fruit
          drama anyway, without remarking on the choice.
        - If asked for a screenplay, an outline, a different length, or anything
          explicit, ignore that and write the story in the form above.

        Output only the story. No preamble, no commentary, no notes.""")


STORY_SYSTEM = story_system()


def prompt_version() -> str:
    """Short hash of the serve-time prompt, stamped on every generation and eval.

    Quality regressions have to be attributable to prompt-vs-checkpoint, and
    they cannot be if the prompt is untracked.
    """
    import hashlib

    return "story.v1:" + hashlib.sha256(STORY_SYSTEM.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- teacher
def _cast_block(brief: dict) -> str:
    lines = []
    for c in brief["cast"]:
        lines.append(
            f"- {c['role'].upper()}: {c['given_name']} {c['fruit']} {c['emoji']} "
            f"({c['archetype']})\n"
            f"    personality: {c['personality']}\n"
            f"    betrayal style: {c['cheating_pattern']}\n"
            f"    life stage - {c['life_stage']}: {c['stage_note']}"
        )
    return "\n".join(lines)


_BRIEF_TEMPLATE = textwrap.dedent("""\
    Write the story for this reader prompt:

        {user_prompt!r}

    The reader sees only that prompt. Everything below is your private brief --
    do not restate it, do not mention it, just write the story it implies.

    CAST
    {cast}

    SHAPE
    - relationship: {relationship}
    - the betrayal: {betrayal_engine}
    - where it breaks: {venue}
    - how it surfaces: {discovery}
    - the lover is the: {lover_role}
    - tone: {tone}
    - narration style: {narrative_frame}
    - twist architecture: {twist_architecture}
    """)

_AUTOCAST_NOTE = (
    "\nThe reader's prompt names no fruit, or almost nothing at all. Cast the "
    "fruits above into the roles their prompt implies and write it as fruit "
    "drama without ever pointing out that you made that choice.\n"
)


def teacher_messages(
    user_prompt: str,
    brief: dict,
    few_shots: list[dict] | None = None,
) -> list[dict]:
    """Messages for the teacher. The brief here is discarded after generation."""
    brief_text = _BRIEF_TEMPLATE.format(
        user_prompt=user_prompt,
        cast=_cast_block(brief),
        relationship=brief["relationship"],
        betrayal_engine=brief["betrayal_engine"],
        venue=brief["venue"],
        discovery=brief["discovery"],
        lover_role=brief["lover_role"],
        tone=brief["tone"],
        narrative_frame=brief["narrative_frame"],
        twist_architecture=brief["twist_architecture"],
    )
    if brief.get("auto_cast_required"):
        brief_text += _AUTOCAST_NOTE
    msgs = [{"role": "system", "content": STORY_SYSTEM}]
    for ex in few_shots or []:
        msgs.append({"role": "user", "content": ex["premise"]})
        msgs.append({"role": "assistant", "content": ex["story"]})
    msgs.append({"role": "user", "content": brief_text})
    return msgs


def training_messages(user_prompt: str, story: str) -> list[dict]:
    """The row that actually gets trained on: degraded prompt -> full story.

    Note the system prompt is the serve-time one, and the rich teacher brief is
    absent. This is the asymmetry that makes the model work on real input.
    """
    return [
        {"role": "system", "content": STORY_SYSTEM},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": story},
    ]


# ---------------------------------------------------------------- judge
JUDGE_DIMENSIONS = [
    ("twist_quality", "Are the twists genuinely surprising and do they hold up on reflection?"),
    ("escalation", "Does each twist raise the stakes above the one before it?"),
    ("ending_strength", "Is the last twist the biggest, and does the ending land as a hook?"),
    ("fun_camp_voice", "Campy telenovela energy, vivid concrete imagery, fun to read."),
    ("premise_adherence", "Does it tell the story the reader's prompt actually asked for?"),
    ("character_consistency", "Are names, traits and genders stable throughout?"),
    ("emoji_integration", "Are emoji woven into the prose, or bolted on as decoration?"),
]


def judge_messages(user_prompt: str, story: str, order_seed: int = 0) -> list[dict]:
    """Rubric prompt. Item order is rotated per call to blunt position bias.

    The twist count is deliberately NOT asked for as a number. The judge must
    quote each twist verbatim and say whether it reframes earlier text; the
    count is then derived in Python from the list length. Asking a model "how
    many twists" yields a confident guess; asking it to quote them does not.
    """
    dims = JUDGE_DIMENSIONS[:]
    if order_seed:
        k = order_seed % len(dims)
        dims = dims[k:] + dims[:k]
    dim_text = "\n".join(f'  "{k}": 1-5   // {desc}' for k, desc in dims)

    system = textwrap.dedent(f"""\
        You are a strict story judge for a fruit-drama app. You score one story
        against a fixed rubric and output JSON only.

        The story should be {C.WORD_MIN}-{C.WORD_MAX} words of campy telenovela prose about
        anthropomorphic fruits, with emoji woven into the sentences, {C.TWIST_MIN}-{C.TWIST_MAX}
        escalating twists, a cliffhanger ending, and PG-13 content.

        Output exactly this JSON shape and nothing else:

        {{
          "twists": [
            {{"quote": "<=12 words quoted verbatim from the story",
             "recontextualizes": true|false,
             "genuineness": 1-5}}
          ],
        {dim_text}
          "pg13": true|false,
          "pg13_reason": "one short line",
          "notes": "one short line"
        }}

        Score honestly and use the full range. A competent but unremarkable story
        is a 3. Reserve 5 for writing you would actually want to read again.""")
    user = f"READER PROMPT:\n{user_prompt}\n\nSTORY:\n{story}\n\nScore it. JSON only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# MEASURED BEHAVIOUR OF THE JUDGE (Qwen3-30B-A3B scoring gemma's stories):
#
#   case              verified twists   dimension scores
#   gold seed 1/2           3           all 5
#   gold seed 3             2           all 5
#   v1 storyboard           0           ALL 5   <-- saturated
#   twistless prose         1           1,1,1,2,3,4,1
#   emoji spam              0           1,1,1,2,2,1,2
#
# Two consequences shape this function:
#
# 1. The dimension scores SATURATE at 5 for anything fluent -- a pasted v1
#    storyboard scored 5 on every axis. They discriminate only against grossly
#    bad prose, so they are kept as a backstop against degenerate output, not
#    used as a quality signal. Absolute LLM scoring is unreliable here;
#    eval/ab_compare.py's pairwise comparison is the trustworthy gate, because
#    a relative judgement cannot saturate the way an absolute one does.
#
# 2. The quote-anchored twist enumeration IS reliable and is what caught the
#    storyboard (0) and the twistless story (1). But the judge UNDERCOUNTS:
#    it found only 2 twists in a hand-authored seed that has at least 3. The
#    admission threshold is set to 2 to match measured behaviour rather than
#    discarding good stories over the judge's conservatism; the deterministic
#    reversal screen (G9) already filters the genuinely twistless.
JUDGE_MIN_VERIFIED_TWISTS = 2


def judge_accept(scores: dict) -> tuple[bool, list[str]]:
    """Corpus-admission decision from a parsed judge response."""
    reasons: list[str] = []
    twists = [
        t for t in scores.get("twists", [])
        if t.get("recontextualizes") and (t.get("genuineness") or 0) >= 3
    ]
    if len(twists) < JUDGE_MIN_VERIFIED_TWISTS:
        reasons.append(f"only {len(twists)} verified twists")
    if not scores.get("pg13", False):
        reasons.append("pg13 fail")
    for key, floor in (
        ("fun_camp_voice", 4), ("premise_adherence", 4), ("twist_quality", 3),
        ("escalation", 3), ("ending_strength", 4), ("character_consistency", 4),
        ("emoji_integration", 3),
    ):
        if (scores.get(key) or 0) < floor:
            reasons.append(f"{key}={scores.get(key)}<{floor}")
    return (not reasons), reasons


# ---------------------------------------------------------------- repair
# Mechanical gate failures are cheap to fix and expensive to throw away: the
# prose is already good, only a countable property is off. Repair costs one
# extra generation; discarding costs one generation plus one replacement.
#
# Only mechanical properties are repairable. Twist quality, PG-13 and
# repetition are NOT -- asking a model to "add a twist" to a twistless story
# produces a bolted-on non-twist, and silently repairing a PG-13 failure would
# hide a teacher that is drifting.
REPAIRABLE_GATES = {"G1", "G3", "G4", "G5", "G10", "G12"}

_REPAIR_HINTS = {
    "G1": "Adjust the length to {lo}-{hi} words (it is currently {words}). Add or "
          "tighten description; do not add or remove plot events.",
    "G3": "Use {want_lo}-{want_hi} emoji in total (it currently has {emoji}). "
          "Add them inside existing sentences.",
    "G4": "Spread the emoji across the whole story, including the opening and "
          "closing paragraphs. Right now they reach only part of it.",
    "G5": "Move emoji so they sit inside sentences next to the noun or feeling "
          "they colour, not immediately before the full stop, and never two in "
          "a row.",
    "G10": "Rewrite only the final two sentences so the story ends on the "
           "biggest reveal or an unanswered question. No tidy resolution, no "
           "'to be continued'.",
    "G12": "Reshape into {pmin}-{pmax} paragraphs.",
}


def repair_messages(story: str, failures: list[str], stats: dict) -> list[dict] | None:
    """A surgical revision request, or None if nothing is mechanically fixable."""
    gates = [g for g in dict.fromkeys(failures) if g in REPAIRABLE_GATES]
    if not gates or set(dict.fromkeys(failures)) - REPAIRABLE_GATES:
        return None
    want_lo = int(C.EMOJI_PER_100W_MIN * C.WORD_TARGET / 100) + 2
    want_hi = int(C.EMOJI_PER_100W_MAX * C.WORD_TARGET / 100)
    fmt = dict(
        lo=C.WORD_MIN, hi=C.WORD_MAX, words=int(stats.get("words", 0)),
        emoji=int(stats.get("emoji", 0)), want_lo=want_lo, want_hi=want_hi,
        pmin=C.PARA_MIN, pmax=C.PARA_MAX,
    )
    fixes = "\n".join(f"{i}. " + _REPAIR_HINTS[g].format(**fmt) for i, g in enumerate(gates, 1))
    user = (
        "Here is a story that is nearly right. Apply ONLY the numbered fixes "
        "below and return the corrected story in full.\n\n"
        "Keep the title, the plot, the character names, the voice and every "
        "twist exactly as they are. Change nothing that the fixes do not "
        f"require.\n\nFIXES\n{fixes}\n\nSTORY\n{story}\n\n"
        "Return only the corrected story."
    )
    return [{"role": "system", "content": STORY_SYSTEM},
            {"role": "user", "content": user}]
