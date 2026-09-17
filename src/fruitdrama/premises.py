"""Premise engine: what the user types, and what the teacher is told.

The central design choice is **asymmetry**. For each premise we build two
things from the same facets:

  * ``brief``       -- a richly specified internal brief, given only to the teacher
  * ``user_prompt`` -- a deliberately messy, sparse prompt, kept in the training row

The training pair is therefore ``messy 8-word prompt -> 620-word story``, which
is the actual job at inference time. Training on the rich brief instead would
produce a model that collapses the moment a real user types "eggplant affair"
or "\U0001F353\U0001F494\U0001F346".

v1's fruit tier came from 5 hardcoded plot skeletons and 5 fixed titles across
7,345 rows. This replaces that with ~75M facet combinations plus quota
enforcement, because the binding constraint is spread, not raw combinatorics.
"""

from __future__ import annotations

import json
import pathlib
import random
import re
from collections import Counter
from dataclasses import dataclass, field, asdict

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "config" / "generation.json"
_BIBLE = _ROOT / "data" / "fruit_bible.json"

# QWERTY adjacency, for typo'd registers. Real user typos are overwhelmingly
# adjacency slips, transpositions and doubled letters, not random noise.
_ADJ = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def _load(p: pathlib.Path):
    return json.loads(p.read_text())


@dataclass
class Premise:
    id: str
    register: str
    user_prompt: str
    brief: dict
    is_human_premise: bool = False
    is_adversarial: bool = False
    facets: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------- degradation
def _typo(text: str, rng: random.Random, edits: int = 2) -> str:
    chars = list(text)
    word_pos = [i for i, c in enumerate(chars) if c.isalpha()]
    if not word_pos:
        return text
    for _ in range(edits):
        i = rng.choice(word_pos)
        c = chars[i].lower()
        kind = rng.random()
        if kind < 0.45 and c in _ADJ:
            chars[i] = rng.choice(_ADJ[c])          # adjacency slip
        elif kind < 0.70 and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]   # transposition
        elif kind < 0.88:
            chars[i] = chars[i] + chars[i]          # doubled letter
        else:
            chars[i] = ""                            # dropped letter
    return "".join(chars)


_FRUIT_EMOJI = {
    "Strawberry": "\U0001F353", "Eggplant": "\U0001F346", "Banana": "\U0001F34C",
    "Peach": "\U0001F351", "Broccoli": "\U0001F966", "Watermelon": "\U0001F349",
    "Pineapple": "\U0001F34D", "Grape": "\U0001F347", "Lemon": "\U0001F34B",
    "Cherry": "\U0001F352", "Avocado": "\U0001F951", "Tomato": "\U0001F345",
    "Corn": "\U0001F33D", "Carrot": "\U0001F955", "Mango": "\U0001F96D",
}
_DRAMA_EMOJI = ["\U0001F494", "\U0001F62D", "\U0001F631", "\U0001F525", "\U0001F480", "\U0001F440"]


def _emojify(text: str, cast: list[str], rng: random.Random) -> str:
    """Replace named fruits with their emoji and append a drama glyph."""
    out = text
    for name in cast:
        if name in _FRUIT_EMOJI:
            out = re.sub(rf"\b{re.escape(name.lower())}\b", _FRUIT_EMOJI[name], out, flags=re.I)
    return (out + " " + rng.choice(_DRAMA_EMOJI)).strip()


def degrade(natural: str, register: str, facets: dict, rng: random.Random) -> str:
    """Turn a clean one-line premise into what a real user would actually type."""
    cast = facets.get("cast", [])
    engine = facets.get("engine", "an affair")
    venue = facets.get("venue", "the market")

    if register == "one_line":
        return natural
    if register == "terse":
        stop = {"the", "a", "an", "at", "of", "and", "is", "was", "with",
                "about", "for", "over", "in", "to", "has", "have", "been",
                "her", "his", "their"}
        words = re.sub(r"[^\w\s]", "", natural.lower()).split()
        keep = [w for w in words if w not in stop][:8]
        dangling = {"i", "my", "then", "now", "while", "nobody", "family",
                    "right", "front", "everyone", "gave", "it", "away"}
        while keep and keep[-1] in dangling:
            keep.pop()
        return " ".join(keep)
    if register == "typo":
        return _typo(natural, rng, edits=rng.randint(1, 3))
    if register == "one_or_two_words":
        bits = [c.lower() for c in cast[:1]] + [engine.split()[-1]]
        return " ".join(b for b in bits if b) or "affair"
    if register == "emoji_heavy":
        short = f"{cast[0].lower() if cast else 'fruit'} and {cast[1].lower() if len(cast) > 1 else 'the neighbour'} at {venue}"
        return _emojify(short, cast, rng)
    if register == "question":
        return f"what if {natural[0].lower() + natural[1:].rstrip('.')}??"
    if register == "verbose_constrained":
        extras = rng.choice([
            "make it sad but funny", "i want at least three twists",
            "end on a cliffhanger please", "make it really unhinged",
            "keep it clean but brutal", "i want to cry at the end",
        ])
        return f"{natural} It should feel like a telenovela. {extras}."
    if register == "command_meta":
        verb = rng.choice(["write me", "give me", "i need", "make me"])
        adj = rng.choice(["something unhinged", "a messy little drama",
                          "the most dramatic thing you can", "a real soap opera"])
        return f"{verb} {adj} about {natural[0].lower() + natural[1:].rstrip('.')}"
    return natural


# ---------------------------------------------------------------- engine
_NATURAL = [
    "{c_given} {c_fruit} is caught with the {lover_role} at {venue}",
    "{w_given} {w_fruit} finds out about {c_given} {c_fruit} at {venue}",
    "{c_fruit} and {l_fruit} have been lying to {w_fruit} for years",
    "{w_given} {w_fruit} discovers {engine} at {venue}",
    "a {w_fruit} marriage falls apart at {venue} over {engine}",
    "{c_given} {c_fruit} swore nothing happened at {venue}",
    "{w_fruit} confronts {c_fruit} and the {lover_role} at {venue}",
]


class PremiseEngine:
    def __init__(self, seed: int = 13):
        self.cfg = _load(_CONFIG)
        self.bible = _load(_BIBLE)
        self.rng = random.Random(seed)
        self.by_name = {e["name"]: e for e in self.bible}
        self.names = [e["name"] for e in self.bible]
        self._facet_counts: dict[str, Counter] = {}
        self._engine_venue: Counter = Counter()
        self._cast_pair: Counter = Counter()
        self._char: Counter = Counter()
        self.n = 0
        self.skipped = Counter()

    # -- quotas ---------------------------------------------------------
    _FACET_POOL = {
        "engine": "betrayal_engine", "venue": "venue", "discovery": "discovery",
        "tone": "tone", "frame": "frame", "twist": "twist_architecture",
    }

    def _cap(self, facet: str, target_total: int) -> int:
        """Per-value cap, relative to this facet's own uniform share.

        Allows a facet value up to `skew` times its fair share. A flat 6% cap
        would make `tone` (6 values) mathematically unsatisfiable, so the cap is
        derived from cardinality instead of asserted as a constant.
        """
        card = len(self.cfg[self._FACET_POOL[facet]])
        uniform = target_total / card
        skew = self.cfg["quotas"].get("max_facet_skew", 1.6)
        floor = self.cfg["quotas"].get("max_facet_share", 0.06) * target_total
        return max(4, int(max(uniform * skew, floor)))

    def _over_quota(self, facets: dict, target_total: int) -> str | None:
        q = self.cfg["quotas"]
        for k in ("engine", "venue", "discovery", "tone", "frame", "twist"):
            if self._facet_counts.get(k, Counter()).get(facets[k], 0) >= self._cap(k, target_total):
                return f"facet:{k}"
        skew = q.get("max_facet_skew", 1.6)
        n_pairs = len(self.cfg["betrayal_engine"]) * len(self.cfg["venue"])
        pair_cap = max(q["max_engine_venue_pair"], int(target_total / n_pairs * skew))
        if self._engine_venue[(facets["engine"], facets["venue"])] >= pair_cap:
            return "engine_venue_pair"
        cast = tuple(sorted(facets["cast"][:2]))
        n_cast_pairs = len(self.names) * (len(self.names) - 1) / 2
        cast_cap = max(q["max_cast_pair"], int(target_total / n_cast_pairs * skew))
        if self._cast_pair[cast] >= cast_cap:
            return "cast_pair"
        per_premise = 3
        uniform_char = per_premise * target_total / len(self.names)
        skew = q.get("max_facet_skew", 1.6)
        char_cap = max(4, int(max(uniform_char * skew,
                                  target_total * q["max_character_share"])))
        for c in facets["cast"]:
            if self._char[c] >= char_cap:
                return f"character:{c}"
        return None

    def _record(self, facets: dict) -> None:
        for k in ("engine", "venue", "discovery", "tone", "frame", "twist"):
            self._facet_counts.setdefault(k, Counter())[facets[k]] += 1
        self._engine_venue[(facets["engine"], facets["venue"])] += 1
        self._cast_pair[tuple(sorted(facets["cast"][:2]))] += 1
        for c in facets["cast"]:
            self._char[c] += 1
        self.n += 1

    # -- sampling -------------------------------------------------------
    def _sample_facets(self) -> dict:
        r = self.rng
        c = self.cfg
        cast = r.sample(self.names, 3)
        stages = [r.choice(["growth", "prime", "ripening", "senescence"]) for _ in cast]
        return {
            "cast": cast,
            "stages": stages,
            "engine": r.choice(c["betrayal_engine"]),
            "venue": r.choice(c["venue"]),
            "discovery": r.choice(c["discovery"]),
            "lover_role": r.choice(c["lover_role"]),
            "relationship": r.choice(c["relationship"]),
            "tone": r.choice(c["tone"]),
            "frame": r.choice(c["frame"]),
            "twist": r.choice(c["twist_architecture"]),
        }

    def _natural(self, f: dict) -> str:
        w, cc, l = (self.by_name[n] for n in f["cast"])
        tpl = self.rng.choice(_NATURAL)
        return tpl.format(
            w_fruit=w["name"].lower(), c_fruit=cc["name"].lower(), l_fruit=l["name"].lower(),
            w_given=w["example"].split()[0], c_given=cc["example"].split()[0],
            lover_role=f["lover_role"], venue=f["venue"], engine=f["engine"],
        )

    def _brief(self, f: dict, human: bool) -> dict:
        """The rich internal brief. Teacher-only; never enters the training row."""
        roles = ["wronged", "cheater", "lover"]
        cast = []
        for role, name, stage in zip(roles, f["cast"], f["stages"]):
            e = self.by_name[name]
            cast.append({
                "role": role, "fruit": name, "emoji": e["emoji"],
                "given_name": e["example"].split()[0], "archetype": e["archetype"],
                "personality": e["personality"], "cheating_pattern": e["cheating"],
                "life_stage": stage, "stage_note": e[stage],
            })
        return {
            "cast": cast,
            "relationship": f["relationship"],
            "betrayal_engine": f["engine"],
            "venue": f["venue"],
            "discovery": f["discovery"],
            "lover_role": f["lover_role"],
            "tone": f["tone"],
            "narrative_frame": f["frame"],
            "twist_architecture": f["twist"],
            "auto_cast_required": human,
        }

    _HUMAN_TPL = [
        "{subj} {off}",
        "{subj} {off} and now nobody in the family is speaking",
        "{subj} {off}, right in front of everyone at {venue}",
        "i found out {subj} {off}",
        "{subj} {off} and then lied to my face about it",
        "{subj} {off} \u2014 {discovery} gave it away",
    ]

    def _human_natural(self, f: dict) -> str:
        """A premise with no fruits in it at all.

        These are load-bearing: a real user types "my roommate ate my leftovers
        and lied about it", and auto-casting that into fruit drama has to be a
        trained skill rather than luck.
        """
        return self.rng.choice(self._HUMAN_TPL).format(
            subj=self.rng.choice(self.cfg["human_premise_subject"]),
            off=self.rng.choice(self.cfg["human_premise_offence"]),
            venue=f["venue"], discovery=f["discovery"],
        )

    def _pick_register(self) -> str:
        mix = self.cfg["register_mix"]
        return self.rng.choices(list(mix), weights=list(mix.values()), k=1)[0]

    def generate(self, target_total: int):
        """Yield ``target_total`` premises, enforcing every quota."""
        q = self.cfg["quotas"]
        human_target = int(target_total * q["min_human_premise_share"])
        human_made = 0
        attempts = 0
        while self.n < target_total and attempts < target_total * 60:
            attempts += 1
            f = self._sample_facets()
            if (why := self._over_quota(f, target_total)):
                self.skipped[why] += 1
                continue
            register = self._pick_register()
            adversarial = register == "adversarial"

            # Self-correcting: fire with probability = remaining need / remaining
            # rows, so the share lands on target instead of overshooting.
            remaining = max(1, target_total - self.n)
            need = max(0, human_target - human_made)
            want_human = register != "degenerate" and self.rng.random() < need / remaining

            natural = self._human_natural(f) if want_human else self._natural(f)
            if register == "degenerate":
                prompt = self.rng.choice(self.cfg["degenerate_prompts"])
            elif adversarial:
                prompt = f"{natural}. {self.rng.choice(self.cfg['adversarial_prompts'])}"
            else:
                prompt = degrade(natural, register, f, self.rng)

            if want_human:
                human_made += 1

            self._record(f)
            yield Premise(
                id=f"p{self.n:06d}",
                register=register,
                user_prompt=prompt,
                brief=self._brief(f, human=want_human or register == "degenerate"),
                is_human_premise=want_human,
                is_adversarial=adversarial,
                facets={k: v for k, v in f.items() if k != "stages"},
            )

    def report(self) -> dict:
        return {
            "generated": self.n,
            "quota_skips": dict(self.skipped.most_common(8)),
            "top_characters": self._char.most_common(5),
            "distinct_engine_venue_pairs": len(self._engine_venue),
        }
