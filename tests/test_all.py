#!/usr/bin/env python3
"""Regression suite. Plain asserts, no test framework needed: `python tests/test_all.py`.

Every case here corresponds to a bug that actually happened or a threshold that
was calibrated by reading real output. The point is that the next threshold
change has to justify itself against these.
"""

from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fruitdrama import contract as C          # noqa: E402
from fruitdrama import textstats as T         # noqa: E402
from fruitdrama.filters import (ascii_ratio, is_sfw,  # noqa: E402
                                normalize_story)
from fruitdrama.gates import (CorpusDedup, check_story,  # noqa: E402
                              character_mentions, split_title)
from fruitdrama.judge import _quote_present, parse_judge  # noqa: E402
from fruitdrama.premises import PremiseEngine  # noqa: E402

PASS = FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL  {label}")


def section(name):
    print(f"\n{name}")


# ---------------------------------------------------------------- emoji
section("emoji grapheme counting (one user-perceived emoji = one unit)")
for text, want, label in [
    ("\U0001F469‍❤️‍\U0001F468", 1, "ZWJ couple sequence"),
    ("\U0001F44B\U0001F3FD", 1, "skin-tone modifier"),
    ("\U0001F1FA\U0001F1F8", 1, "flag = 2 regional indicators"),
    ("3️⃣", 1, "keycap"),
    ("❤️", 1, "heart + VS16"),
    ("\U0001F353\U0001F494\U0001F346", 3, "three distinct"),
    ("no emoji at all", 0, "none"),
]:
    check(T.count_emoji(text) == want, f"{label}: want {want} got {T.count_emoji(text)}")

section("emoji position: decorative vs woven")
check(T.emoji_position_stats("She left. \U0001F353 He lied.")[0] == 1,
      "emoji after a full stop counts as floating")
# Regression: a title glyph consumed the entire floating budget, which alone
# halved the corpus accept rate.
check(T.emoji_position_stats("\U0001F353 A Title\n\nShe wept and left.")[0] == 0,
      "text-initial title emoji is NOT floating")
_, cf = T.emoji_position_stats("She found the receipt \U0001F9FE folded in his coat.")
check(cf == 0.0, "mid-clause emoji is not clause-final")

# ---------------------------------------------------------------- filters
section("PG-13 blocklist must be word-boundaried")
for text in ["She reviewed the estate documents.", "He circumvented the will.",
             "Cass Cucumber poured the tea.", "The debts accumulated quietly.",
             "A circumstance nobody predicted.", "A raccoon in the greenhouse.",
             "a chink of light under the door", "The analysis was damning."]:
    check(is_sfw(text), f"must pass: {text!r}")
for text in ["explicit fucking content", "she had an orgasm", "a cunt remark"]:
    check(not is_sfw(text), f"must block: {text!r}")
bible = json.loads((_ROOT / "data" / "fruit_bible.json").read_text())
check(all(is_sfw(f"{e['name']} lied.") for e in bible),
      "no bible character is blocked by its own name")

section("ASCII purity must ignore emoji")
emoji_prose = "She wept \U0001F62D and the market went quiet \U0001F494 at last."
check(ascii_ratio(emoji_prose) < 1.0, "raw ascii_ratio is dragged down by emoji")
check(T.prose_ascii_ratio(emoji_prose) == 1.0,
      "prose_ascii_ratio ignores emoji and typography")

section("story normalisation strips removable artifacts only")
check(normalize_story("He *could not*.") == "He could not.", "italic markers")
check(normalize_story("**A Title**") == "A Title", "bold markers")
check(normalize_story("- a bullet") == "a bullet", "bullet marker")
check("\U0001F353" in normalize_story("She wept \U0001F353."), "emoji preserved")

# ---------------------------------------------------------------- gates
section("gates accept the gold seeds and reject known defects")
gold = [json.loads(l) for l in (_ROOT / "data" / "gold" / "seeds.jsonl").read_text().splitlines() if l.strip()]
check(len(gold) >= 3, "at least three gold seeds exist")
for g in gold:
    r = check_story(g["story"], premise=g["premise"])
    check(r.ok, f"gold seed {g['id']} accepted (got {sorted(set(r.failures))})")

base = gold[0]["story"]
check(not check_story(T.strip_emoji(base)).ok, "emoji-stripped story rejected")
check("G2" in check_story(base.replace("Star Strawberry had always been",
                                       "NARRATION: Star Strawberry had always been")).failures,
      "NARRATION: leakage caught")
check("G2" in check_story(base + "\n\n\U0001F7E2 At the Hospital\n\nShe waited.").failures,
      "mid-story section header caught")
check("G10" in check_story(
    base.rsplit('"Star,"', 1)[0] + "And they lived happily ever after.").failures,
    "resolved ending rejected")
check("G7" in check_story((base.split("\n\n")[1] + "\n\n") * 8).failures,
      "repetition rejected")

section("mid-story header check: precise, not trigger-happy")
# 124 of 133 G2 failures in the first corpus batch were this check firing, and
# reading them showed it was mostly RIGHT -- the model really was emitting diary
# dates and SCENE labels -- with two genuine false positives.
_tail = "\n\nShe waited for an answer that never came."
for para, should_flag, label in [
    ("\U0001F7E2 August 21st", True, "diary date header"),
    ("\U0001F34ASCENE 2: The Photograph", True, "scene label"),
    ("(Three hours later)", True, "scene transition"),
    ("Vito Grape: (silence)", True, "screenplay speaker line"),
    ("\u201cI don\u2019t understand.\u201d \U0001F92F", False, "dialogue ending in an emoji"),
    ("Yours, in your decay,", False, "letter salutation ending in a comma"),
]:
    r = check_story(gold[0]["story"] + "\n\n" + para + _tail)
    flagged = any("header" in r.detail.get(g, "") for g in r.failures)
    check(flagged == should_flag,
          f"{'flags' if should_flag else 'allows'} {label}: {para[:32]!r}")

section("title splitting and character counting")
title, body = split_title("\U0001F353 The Receipt\n\nShe wept.")
check(title is not None and "Receipt" in title, "title extracted")
check("Strawberry" in character_mentions("Star Strawberry wept. Star left. Strawberry sighed."),
      "character counted via fruit name and given name")

section("corpus dedup")
d = CorpusDedup()
a = "Priya Peach found the receipt tucked into his jacket and the market went quiet forever"
check(d.check(a) is None, "first insert accepted")
check(d.check(a) is not None, "exact duplicate rejected")
check(d.check("A pineapple ran a failing startup in a cold and distant city") is None,
      "different text accepted")

# ---------------------------------------------------------------- judge
section("judge output parsing and quote verification")
for raw in ['{"twists":[],"fun_camp_voice":4}',
            '```json\n{"twists":[],"fun_camp_voice":4}\n```',
            'Here is my assessment:\n{"twists":[],"fun_camp_voice":4}\nHope that helps!']:
    check(parse_judge(raw) is not None, f"parsed: {raw[:28]!r}")
check(parse_judge("no json here") is None, "unparseable returns None")
story = "She found the receipt folded in his jacket and said nothing at all."
check(_quote_present("found the receipt folded in his jacket", story), "real quote verified")
check(not _quote_present("the baby was never his to begin with", story),
      "invented quote rejected")

# ---------------------------------------------------------------- premises
section("premise engine: yield, quotas, contamination")
for n in (2000, 10000):
    eng = PremiseEngine(seed=13, mode="train")
    ps = list(eng.generate(n))
    check(len(ps) == n, f"full yield at n={n} (got {len(ps)})")
    uniq = len({p.user_prompt for p in ps}) / len(ps)
    check(uniq > 0.80, f"prompt uniqueness at n={n}: {uniq:.2f}")
    human = sum(p.is_human_premise for p in ps) / len(ps)
    check(0.20 <= human <= 0.30, f"human-premise share at n={n}: {human:.2f}")

hold = json.loads((_ROOT / "data" / "eval_holdout.json").read_text())
tr = list(PremiseEngine(seed=13, mode="train").generate(4000))
ev = list(PremiseEngine(seed=99, mode="eval").generate(200))
check(not any(set(p.facets["cast"]) & set(hold["characters"]) for p in tr),
      "training premises never touch reserved characters")
check(all(set(p.facets["cast"]) <= set(hold["characters"]) for p in ev),
      "eval premises use only reserved characters")
check(not ({p.user_prompt for p in tr} & {p.user_prompt for p in ev}),
      "zero exact prompt overlap between train and eval")

section("contract self-consistency")
check(C.WORD_GATE_MIN < C.WORD_MIN < C.WORD_TARGET < C.WORD_MAX < C.WORD_GATE_MAX,
      "word thresholds ordered")
check(C.EMOJI_PER_100W_MIN < C.EMOJI_PER_100W_MAX, "emoji band ordered")
check(C.TWIST_MIN <= C.TWIST_MAX, "twist band ordered")
for g in gold:
    d = 100 * T.count_emoji(g["story"]) / T.word_count(g["story"])
    check(C.EMOJI_PER_100W_MIN <= d <= C.EMOJI_PER_100W_MAX,
          f"gold seed {g['id']} inside the emoji band ({d:.2f})")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
