"""Measure real *unique* yield per source to set sane caps."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PQ_TMP", os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pq_tmp"))
os.makedirs(os.environ["PQ_TMP"], exist_ok=True)
import adapters as A
from fruitdrama import filters as F

GEN = ['drama', 'romance', 'psychological', 'thriller', 'mystery', 'crime',
       'noir', 'historical', 'biopic', 'war', 'horror']
VC = ['dialogue_craft', 'ensemble_scene', 'scene_writing', 'monologue',
      'exposition_scene', 'closing_image', 'opening_image',
      'character_generation', 'transition_scene']
SC = ['storyline_generation', 'structure_analysis', 'act_break',
      'series_bible', 'pitch_document', 'tv_pilot', 'script_coverage']

uniq = set(); y = 0
for r in A.adapt_screenplay(10**7, 10**7, GEN, VC, SC, 120000):
    y += 1; uniq.add(hash(r["assistant"]))
print(f"SCREENPLAY: yielded={y} unique={len(uniq)}", flush=True)

u2 = set(); y2 = 0
for r in A.adapt_multichar(10**7, 10**7):
    y2 += 1; u2.add(hash(r["assistant"]))
print(f"MULTICHAR: yielded={y2} unique={len(u2)}", flush=True)

c2 = c3 = n = kept = 0
seen = set()
for row in A._iter_parquet("euclaise/writingprompts", "train", "default",
                           columns=["prompt", "story"]):
    n += 1
    if n > 140000:
        break
    p = F.detokenize(row.get("prompt") or "")
    s = F.detokenize(row.get("story") or "")
    if not p or not s:
        continue
    if not F.is_sfw(p, s) or not F.quality_ok(s, min_len=400, max_len=8000):
        continue
    blob = p + " " + s[:1500]
    if F.is_dramatic(blob, 2):
        c2 += 1
        if hash(s) not in seen:
            seen.add(hash(s)); kept += 1
    if F.is_dramatic(blob, 3):
        c3 += 1
print(f"WRITINGPROMPTS: scanned={n} min2={c2} (unique~{kept}) min3={c3}", flush=True)
print("MEASURE DONE", flush=True)
