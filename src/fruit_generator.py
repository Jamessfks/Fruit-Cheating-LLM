"""Synthetic 'AI fruit drama' episode generator.

Produces on-target training rows in the exact 6-scene schema described by the
Flashloop soap-opera formula (Scene 1 recap -> 2-3 rising -> 4-5 peak -> 6
cliffhanger; NARRATION + VISUAL per scene). Deterministic given a seed; fully
offline. These rows carry the FRUIT system prompt.
"""
import random

# archetype -> list of (appearance, signature item)
CAST = {
    "matriarch": [
        ("an elderly broccoli woman", "a silk ivory robe and a pearl necklace"),
        ("a silver-haired broccoli matriarch", "a lace shawl and heirloom pearls"),
        ("a stern broccoli grandmother", "a high-collared black gown and a jade brooch"),
    ],
    "villain": [
        ("a tall grape man", "a tailored black suit and a gold signet ring"),
        ("a sleek eggplant man", "a dark velvet blazer and a diamond cufflink"),
        ("a sharp-eyed grape woman", "a charcoal power suit and blood-red nails"),
    ],
    "temptress": [
        ("a young strawberry woman", "a red silk dress and crimson lipstick"),
        ("a glamorous cherry woman", "a scarlet gown and ruby earrings"),
        ("a radiant strawberry woman", "a plunging wine-colored dress and gold hoops"),
    ],
    "innocent": [
        ("a gentle banana man", "a plain linen shirt and a simple wristband"),
        ("a soft-spoken corn woman", "a cream cardigan and a thin silver locket"),
        ("a kind-eyed banana man", "a beige sweater and round glasses"),
    ],
    "heir": [
        ("a young avocado man", "a sharp navy suit and a gold watch"),
        ("an ambitious bell-pepper woman", "a tailored red blazer and a slim briefcase"),
        ("a brooding avocado man", "an unbuttoned tuxedo and a silver signet ring"),
    ],
}

SETTINGS = [
    ("a luxurious penthouse living room", "at night", "warm lamplight and long shadows"),
    ("an elegant marble bathroom", "in the early morning", "soft golden light"),
    ("an upscale restaurant booth", "late at night", "flickering candlelight"),
    ("a sprawling vineyard estate", "at dusk", "amber sunset haze"),
    ("a sterile hospital corridor", "after midnight", "cold blue fluorescent light"),
    ("a grand ballroom mid-wedding", "in the evening", "glittering chandelier light"),
    ("a glass-walled corporate boardroom", "at night", "harsh overhead light"),
    ("a moonlit garden gala", "at night", "silver moonlight and string lights"),
    ("a rain-streaked limousine", "at night", "passing streetlights"),
    ("a family mansion's grand staircase", "at dusk", "dim golden sconces"),
]

MOODS = ["tense", "suspicious", "heartbroken", "furious", "intimate and secretive",
         "cold and confrontational", "devastated", "triumphant and cruel"]


def _ref(entry):
    return entry[0]


def _full(entry):
    return f"{entry[0]} in {entry[1]}"


def _visual(feat, setting, extra=""):
    place, tod, light = setting
    who = " and ".join(_full(f) for f in feat)
    extra = (" " + extra) if extra else ""
    return (f"Dramatic 9:16 vertical scene in {place} {tod}. {who}.{extra} "
            f"{light}, {random.choice(MOODS)} atmosphere.")


def _trim(line, hard=115):
    if len(line) <= hard:
        return line
    cut = line[:hard].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "..."


# Each plot returns (title, logline, [(label, narration, visual), ...x6])
def _infidelity(rng):
    betrayed = rng.choice(CAST["innocent"] + CAST["matriarch"])
    cheater = rng.choice(CAST["temptress"])
    lover = rng.choice(CAST["villain"] + CAST["heir"])
    s = rng.choice(SETTINGS)
    twist = rng.choice([
        f"the {lover[0].split(' in ')[0]} was secretly recording every word",
        "a paternity letter was already in the mail",
        f"the betrayed {betrayed[0]} had known for months and said nothing",
        "the affair had been arranged by the family matriarch herself",
    ])
    title = "The Affair at Midnight"
    logline = (f"{cheater[0].capitalize()}'s affair with {lover[0]} is uncovered by "
               f"{betrayed[0]} — and no one leaves unscathed.")
    beats = [
        ("Recap", f"Last time, {betrayed[0]} still believed the marriage was untouchable.",
         _visual([betrayed], s, "Looking calmly out a rain-blurred window.")),
        ("Rising Tension", f"But {cheater[0]} kept slipping away at midnight, again and again.",
         _visual([cheater], rng.choice(SETTINGS), "Glancing over one shoulder, phone hidden.")),
        ("Rising Tension", "A whispered call. A deleted message. The truth clawed toward the surface.",
         _visual([cheater, lover], rng.choice(SETTINGS), "Leaning close, whispering.")),
        ("Dramatic Peak", f"Then the door opened — and {betrayed[0]} saw everything.",
         _visual([betrayed, cheater, lover], s, "Frozen in the doorway, betrayal on every face.")),
        ("Dramatic Peak", f"'How long?' {betrayed[0]} whispered. The silence answered for them.",
         _visual([betrayed, cheater], s, "Tears and trembling hands, faces inches apart.")),
        ("Cliffhanger", f"But no one noticed {twist} — and everything is about to shatter.",
         _visual([lover], s, "A slow, knowing half-smile in the shadows.")),
    ]
    return title, logline, beats


def _pregnancy(rng):
    mother = rng.choice(CAST["temptress"] + CAST["innocent"])
    father = rng.choice(CAST["villain"] + CAST["heir"])
    matri = rng.choice(CAST["matriarch"])
    s = rng.choice(SETTINGS)
    twist = rng.choice([
        f"the real father was {rng.choice(CAST['heir'])[0]}",
        "there were two heartbeats, not one",
        "the matriarch had swapped the test results",
        "the pregnancy would decide who inherits everything",
    ])
    title = "The Test in the Drawer"
    logline = (f"{mother[0].capitalize()} hides a pregnancy that could topple the "
               f"family — but the secret refuses to stay buried.")
    beats = [
        ("Recap", f"Last time, {mother[0]} swore the family name would stay spotless.",
         _visual([mother], s, "Standing very still, one hand on her stomach.")),
        ("Rising Tension", f"But {mother[0]} hid a pregnancy test deep in the drawer.",
         _visual([mother], SETTINGS[1], "Holding a pregnancy test with tears in her eyes.")),
        ("Rising Tension", f"{matri[0].capitalize()} sensed a secret and started asking questions.",
         _visual([matri, mother], s, "An accusing stare meeting a guilty one.")),
        ("Dramatic Peak", f"'You're pregnant.' The words dropped like glass in front of everyone.",
         _visual([mother, father, matri], s, "A whole room turning to stare.")),
        ("Dramatic Peak", f"{father[0].capitalize()} went pale — because he knew it couldn't be his.",
         _visual([father], s, "Draining of color, gripping the back of a chair.")),
        ("Cliffhanger", f"What no one knew: {twist}.",
         _visual([mother], s, "A single hand resting on her stomach, a faint secret smile.")),
    ]
    return title, logline, beats


def _dynasty(rng):
    matri = rng.choice(CAST["matriarch"])
    heir1 = rng.choice(CAST["heir"])
    heir2 = rng.choice(CAST["villain"])
    s = rng.choice(SETTINGS)
    twist = rng.choice([
        "the will had been rewritten one hour before the reading",
        f"the estate was already secretly sold to {rng.choice(CAST['villain'])[0]}",
        "there was a third heir no one had ever met",
        "the matriarch was still very much alive",
    ])
    title = "Blood and Vineyards"
    logline = (f"When {matri[0]} announces the heir to the family empire, "
               f"{heir1[0]} and {heir2[0]} go to war.")
    beats = [
        ("Recap", f"Last time, {matri[0]} promised the empire to only one child.",
         _visual([matri], s, "Seated at the head of a long table, unreadable.")),
        ("Rising Tension", f"{heir1[0].capitalize()} and {heir2[0]} circled the inheritance like wolves.",
         _visual([heir1, heir2], s, "Two rivals sizing each other up across the room.")),
        ("Rising Tension", "A forged signature. A missing ledger. The empire began to crack.",
         _visual([heir2], SETTINGS[6], "Sliding a document into a briefcase, alone.")),
        ("Dramatic Peak", f"At the will's reading, {matri[0]} named the heir — and the room erupted.",
         _visual([matri, heir1, heir2], s, "A dropped envelope, shouting, chaos.")),
        ("Dramatic Peak", f"'You were never family,' {heir2[0]} hissed at {heir1[0]}.",
         _visual([heir1, heir2], s, "Faces inches apart, pure venom.")),
        ("Cliffhanger", f"But hidden in the fine print: {twist}.",
         _visual([matri], s, "A close-up on a signature and a knowing eye.")),
    ]
    return title, logline, beats


def _revenge(rng):
    wronged = rng.choice(CAST["innocent"] + CAST["heir"])
    target = rng.choice(CAST["villain"] + CAST["temptress"])
    s = rng.choice(SETTINGS)
    twist = rng.choice([
        "the revenge had ruined the wrong person entirely",
        f"the target had already planned a counter-move with {rng.choice(CAST['villain'])[0]}",
        "the wronged one had a partner nobody suspected",
        "the whole scheme was being watched from the start",
    ])
    title = "A Cold Dish"
    logline = (f"After years of humiliation, {wronged[0]} finally moves against "
               f"{target[0]} — but revenge has a price.")
    beats = [
        ("Recap", f"Last time, {wronged[0]} vowed that one day the debt would be paid.",
         _visual([wronged], s, "Staring at an old photograph, jaw tight.")),
        ("Rising Tension", f"Quietly, {wronged[0]} gathered every secret {target[0]} had buried.",
         _visual([wronged], SETTINGS[6], "Spreading documents across a desk at night.")),
        ("Rising Tension", "One phone call would end an empire. A finger hovered over 'send'.",
         _visual([wronged], s, "A trembling hand over a glowing phone screen.")),
        ("Dramatic Peak", f"At the gala, the truth about {target[0]} played on every screen.",
         _visual([target], SETTINGS[5], "Horror spreading across a face under bright lights.")),
        ("Dramatic Peak", f"'You did this,' {target[0]} breathed. {wronged[0]} only smiled.",
         _visual([wronged, target], s, "One calm face, one crumbling face.")),
        ("Cliffhanger", f"What the crowd never saw: {twist}.",
         _visual([target], s, "A slow turn toward the camera, eyes hardening.")),
    ]
    return title, logline, beats


def _secret_identity(rng):
    stranger = rng.choice(CAST["temptress"] + CAST["heir"])
    family = rng.choice(CAST["matriarch"])
    foil = rng.choice(CAST["villain"])
    s = rng.choice(SETTINGS)
    twist = rng.choice([
        f"the stranger was the long-lost child of {family[0]}",
        "the new arrival had come to take back everything",
        f"the foil had known the true identity all along",
        "the family had faked a death years ago to hide it",
    ])
    title = "The Stranger at the Gate"
    logline = (f"A charming newcomer, {stranger[0]}, arrives at the estate — and "
               f"nothing about their story adds up.")
    beats = [
        ("Recap", f"Last time, {family[0]} welcomed a stranger into the family home.",
         _visual([family, stranger], s, "A polite handshake with wary eyes.")),
        ("Rising Tension", f"But {stranger[0]} knew things no outsider ever should.",
         _visual([stranger], s, "Tracing a finger over an old family portrait.")),
        ("Rising Tension", f"{foil[0].capitalize()} started digging into the newcomer's past.",
         _visual([foil], SETTINGS[6], "Studying a file under a single desk lamp.")),
        ("Dramatic Peak", f"'I know who you really are,' {foil[0]} said. The room went silent.",
         _visual([foil, stranger, family], s, "Every head turning at once.")),
        ("Dramatic Peak", f"{stranger[0].capitalize()} pulled out a photograph — and {family[0]} gasped.",
         _visual([family], s, "A hand flying to the mouth, eyes wide.")),
        ("Cliffhanger", f"The secret they'd all been hiding: {twist}.",
         _visual([stranger], s, "A slow, unreadable smile in the doorway.")),
    ]
    return title, logline, beats


PLOTS = [_infidelity, _pregnancy, _dynasty, _revenge, _secret_identity]

USER_TEMPLATES = [
    "Write episode {n} of an AI fruit drama. Logline: {logline}",
    "New fruit-drama episode ({n}). {logline} End on a cliffhanger.",
    "Give me a 6-scene AI fruit drama script. {logline}",
    "Episode {n}, please. Premise: {logline} Make it as shareable as possible.",
    "Fruit drama, episode {n}. {logline} Keep it platform-safe but scandalous.",
]


def _render(title, n, beats):
    out = [f"EPISODE {n}: {title}", ""]
    for label, narr, vis in beats:
        out.append(f"Scene — {label}")
        out.append(f"NARRATION: {_trim(narr)}")
        out.append(f"VISUAL: {_trim(vis, 240)}")
        out.append("")
    return "\n".join(out).strip()


def generate(n_rows, seed=13):
    """Yield {'user','assistant','meta'} dicts, deduped, until n_rows produced."""
    rng = random.Random(seed)
    seen = set()
    produced = 0
    attempts = 0
    while produced < n_rows and attempts < n_rows * 40:
        attempts += 1
        plot = rng.choice(PLOTS)
        title, logline, beats = plot(rng)
        ep = rng.randint(1, 9)
        assistant = _render(title, ep, beats)
        key = assistant[:400]
        if key in seen:
            continue
        seen.add(key)
        user = rng.choice(USER_TEMPLATES).format(n=ep, logline=logline)
        produced += 1
        yield {
            "user": user,
            "assistant": assistant,
            "meta": {"source": "synthetic-fruit", "slice": "fruit_synth",
                     "license": "ours", "plot": plot.__name__.lstrip("_")},
        }


if __name__ == "__main__":
    for i, r in enumerate(generate(2)):
        print("USER:", r["user"])
        print(r["assistant"])
        print("=" * 60)
