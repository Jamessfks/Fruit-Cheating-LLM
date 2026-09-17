"""Build the 'Fruit Drama Bible' Excel workbook from data/fruit_bible.json.

Sheets: Fruit Bible (master reference) | Life Stages & Legend | Roles & Dispositions
| Story Builder (example). No formulas (pure reference), so no recalc needed.
"""
import os
import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import json as _json
import pathlib as _pathlib

# Single source of truth; these lists were previously duplicated verbatim
# between this file and the deleted src/build_interface.py.
_TAX = _json.loads((_pathlib.Path(__file__).resolve().parents[1]
                    / "data" / "taxonomy.json").read_text())
ROLES = _TAX["roles"]
DISPOSITIONS = _TAX["dispositions"]
CHARACTERS = _TAX["characters"]


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open(os.path.join(ROOT, "data", "fruit_bible.json"), encoding="utf-8"))
OUT_DIR = os.path.join(ROOT, "deliverables")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "Fruit_Drama_Bible.xlsx")

FONT = "Arial"
PLUM = "5B2A86"
PLUM_LT = "F3EEF9"
WHITE = "FFFFFF"
STAGE_FILLS = {  # green -> gold -> orange -> brown : growth -> prime -> ripening -> senescence
    "growth": "2E7D32", "prime": "F9A825", "ripening": "EF6C00", "senescence": "6D4C41",
}
thin = Side(style="thin", color="D9D2E9")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

wb = Workbook()

# ------------------------------------------------------------ Sheet 1: Bible
ws = wb.active
ws.title = "Fruit Bible"
COLS = [
    ("#", 4, None), ("Fruit", 15, None), ("Emoji", 7, None),
    ("Dominant Gender", 15, None), ("Example Name", 17, None),
    ("Drama Archetype", 24, None), ("Personality", 34, None),
    ("Character", 40, None), ("Cheating / Betrayal Pattern", 40, None),
    ("Key Properties", 40, None),
    ("Stage 1 — Growth (Cell Expansion)", 40, "growth"),
    ("Stage 2 — Prime (Maturation)", 40, "prime"),
    ("Stage 3 — Ripening (Starts to Fall)", 40, "ripening"),
    ("Stage 4 — Senescence (Failure)", 40, "senescence"),
]
KEYS = ["name", "emoji", "gender", "example", "archetype", "personality",
        "character", "cheating", "properties", "growth", "prime", "ripening", "senescence"]

for c, (title, width, stage) in enumerate(COLS, start=1):
    cell = ws.cell(row=1, column=c, value=title)
    fill = STAGE_FILLS[stage] if stage else PLUM
    cell.fill = PatternFill("solid", fgColor=fill)
    cell.font = Font(name=FONT, size=10, bold=True, color=WHITE)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = BORDER
    ws.column_dimensions[get_column_letter(c)].width = width

for i, fruit in enumerate(DATA):
    r = i + 2
    row_fill = PLUM_LT if i % 2 else None
    ws.cell(row=r, column=1, value=i + 1)
    for c, key in enumerate(KEYS, start=2):
        cell = ws.cell(row=r, column=c, value=fruit.get(key, ""))
        cell.font = Font(name=FONT, size=10)
        cell.alignment = Alignment(vertical="top", wrap_text=True,
                                   horizontal="center" if key == "emoji" else "left")
        cell.border = BORDER
        if row_fill:
            cell.fill = PatternFill("solid", fgColor=row_fill)
    ws.cell(row=r, column=1).alignment = Alignment(horizontal="center", vertical="top")
    if row_fill:
        ws.cell(row=r, column=1).fill = PatternFill("solid", fgColor=row_fill)
    # bold the fruit name
    ws.cell(row=r, column=2).font = Font(name=FONT, size=10, bold=True)

ws.freeze_panes = "C2"
ws.auto_filter.ref = f"A1:{get_column_letter(len(COLS))}{len(DATA)+1}"
ws.sheet_view.showGridLines = False


def title_block(ws, text, span=2, size=14):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=span)
    c = ws.cell(row=1, column=1, value=text)
    c.font = Font(name=FONT, size=size, bold=True, color=WHITE)
    c.fill = PatternFill("solid", fgColor=PLUM)
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 24


def kv(ws, start, pairs, w0=26, w1=90):
    ws.column_dimensions["A"].width = w0
    ws.column_dimensions["B"].width = w1
    r = start
    for k, v in pairs:
        a = ws.cell(row=r, column=1, value=k)
        b = ws.cell(row=r, column=2, value=v)
        a.font = Font(name=FONT, size=10, bold=True)
        b.font = Font(name=FONT, size=10)
        a.alignment = Alignment(vertical="top", wrap_text=True)
        b.alignment = Alignment(vertical="top", wrap_text=True)
        r += 1
    return r


# ------------------------------------------------- Sheet 2: Legend
ws2 = wb.create_sheet("Life Stages & Legend")
ws2.sheet_view.showGridLines = False
title_block(ws2, "The Fruit Drama Bible — How It Works", span=2)
rows = [
    ("", ""),
    ("PURPOSE", "A reference for the fine-tuned LLM and the input interface. Every fruit is a "
                "character with a fixed personality, a cheating pattern, and a 4-stage life arc. "
                "Pick fruits, assign a stage/role/disposition, add context, and the LLM weaves "
                "them into a mind-blowing drama-opera episode."),
    ("", ""),
    ("THE FOUR LIFE STAGES", "Each fruit behaves differently at each stage — choose the stage to set who the character is right now."),
    ("1. Growth (Cell Expansion)", "Green / young / unripe. Insecure, weak, forming. The 'before' self."),
    ("2. Prime (Maturation)", "Ripe / peak. Confident, powerful, magnetic — often doing what they once feared or hated."),
    ("3. Ripening (Starts to Fall)", "Over-ripe. The turn: betrayal, exposure, decline begins. The most dramatic stage."),
    ("4. Senescence (Failure)", "Rotten / spent. Collapse, regret, downfall — the tragic 'after'."),
    ("", ""),
    ("GENDER", "Every fruit can be male or female, but most are DOMINANT in one gender "
               "(e.g. Peach = Female-dominant, Banana = Male-dominant, Lemon = Balanced). "
               "The dominant gender is the default; override it in the interface for any character."),
    ("", ""),
    ("EXAMPLE (Peter Banana)", "Male. Green = shy and weak. Prime = bulky and confident, doing the reckless things "
                               "he hated when green. Ripening = friends betray him, he becomes self-hating. "
                               "Senescence = collapses into regret."),
    ("", ""),
    ("ARCHETYPE", "The character's default dramatic role (Matriarch, Villain/Seducer, Temptress, Innocent, "
                  "Heir, Underdog, etc.). A starting point — the interface lets you reassign it per story."),
    ("", ""),
    ("HOW THE LLM USES THIS", "The interface pulls each chosen fruit's stage-specific traits from this bible, "
                              "combines them with your role/disposition/character and your context prompt, and "
                              "produces the full system+user prompt for the fruit-drama LLM (a 6-scene episode "
                              "with NARRATION + VISUAL per scene, ending on a jaw-dropping twist)."),
    ("", ""),
    ("CLICKABLE VERSION", "Open interface/index.html in a browser for a point-and-click builder."),
    ("NOTE ON EMOJI", "Some fruits have no dedicated Unicode emoji; the closest color/shape stand-in is used "
                      "(e.g. plum = purple circle, lime = green circle)."),
]
kv(ws2, 3, rows)

# ------------------------------------------------- Sheet 3: Roles & Dispositions
ws3 = wb.create_sheet("Roles & Dispositions")
ws3.sheet_view.showGridLines = False
for i, row in enumerate(example):
    for c, v in enumerate(row, start=1):
        cell = ws4.cell(row=i + 3, column=c, value=v)
        cell.font = Font(name=FONT, size=10)
        cell.alignment = Alignment(vertical="top", wrap_text=True,
                                   horizontal="center" if c in (1, 3, 4, 5) else "left")
        if i % 2:
            cell.fill = PatternFill("solid", fgColor=PLUM_LT)
ctx_r = len(example) + 4
c = ws4.cell(row=ctx_r, column=1, value="Context / location / setting:")
c.font = Font(name=FONT, size=10, bold=True)
ws4.merge_cells(start_row=ctx_r, start_column=2, end_row=ctx_r, end_column=9)
cv = ws4.cell(row=ctx_r, column=2,
              value="A luxury wellness retreat in the hills; the annual charity gala; old rivalries "
                    "resurface and a paternity secret is about to detonate.")
cv.font = Font(name=FONT, size=10)
cv.alignment = Alignment(vertical="top", wrap_text=True)
note = ws4.cell(row=ctx_r + 2, column=1,
                value="Tip: open interface/index.html for a clickable version that auto-writes the full LLM prompt.")
note.font = Font(name=FONT, size=10, italic=True, color="6D4C41")
ws4.merge_cells(start_row=ctx_r + 2, start_column=1, end_row=ctx_r + 2, end_column=9)

wb.save(OUT)
print("wrote", OUT)
print("sheets:", wb.sheetnames)
print("fruits:", len(DATA))
