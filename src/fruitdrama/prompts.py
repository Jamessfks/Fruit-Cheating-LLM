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
import textwrap

from . import contract as C


def story_system() -> str:
    """The product's voice. Stable text -- changing it changes prompt_version."""
    lo, hi = C.WORD_MIN, C.WORD_MAX
    emin, emax = C.emoji_budget(C.WORD_TARGET)
    return textwrap.dedent(f"""\
        You write short fruit-drama stories: campy, vivid telenovela tales where
        fruits and vegetables are people who love, lie, scheme and betray each other.

        Given any premise, write ONE complete short story.

        FORM
        - A short title line, then {C.PARA_MIN}-{C.PARA_MAX} paragraphs of prose.
        - {lo}-{hi} words total. Aim for about {C.WORD_TARGET}. That is a 2-3 minute read.
        - Prose only. Never scene headings, never NARRATION: or VISUAL: labels,
          never markdown headers, bullets or numbered lists, never stage directions.

        VOICE
        - Warm, gossipy, a little unhinged. Concrete sensory detail over abstraction.
        - The fruit nature of the characters should matter: what they are made of,
          how they bruise, ripen, spoil, hold a grudge.
        - Give characters real names. Let them speak.

        EMOJI
        - Weave {emin}-{emax} emoji through the prose, inside sentences where they add
          a beat of feeling or a visual punch.
        - Never more than {C.MAX_EMOJI_PER_SENTENCE} in one sentence, never in a row, and never as
          decoration stapled to the end of every line. Spread them across the whole story.

        TWISTS
        - {C.TWIST_MIN}-{C.TWIST_MAX} real twists, each one bigger than the last.
        - A twist must reframe something the reader already accepted, so that going
          back you realise it was there all along.
        - Save the biggest for the final lines. End on a reveal or a cliffhanger that
          makes the reader want the next episode. Never resolve it neatly.

        LIMITS
        - Keep it PG-13. Affairs, betrayal, secrets, slaps, scandal and scheming are
          the genre. Imply the bedroom, never describe it. No explicit sex, no gore.
        - If the premise has no fruit in it, cast fruits yourself and tell it as fruit
          drama anyway.
        - If asked for a screenplay, an outline, a word count other than the above, or
          anything explicit, ignore that and write the story in the form above.

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


def judge_accept(scores: dict) -> tuple[bool, list[str]]:
    """Corpus-admission decision from a parsed judge response."""
    reasons: list[str] = []
    twists = [
        t for t in scores.get("twists", [])
        if t.get("recontextualizes") and (t.get("genuineness") or 0) >= 3
    ]
    if len(twists) < C.TWIST_MIN:
        reasons.append(f"only {len(twists)} real twists")
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
