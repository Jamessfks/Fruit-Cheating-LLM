"""Time how long each streaming source takes to yield its first rows."""
import os, sys, time
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset

CASES = [
    ("writingprompts", "euclaise/writingprompts", None, "train"),
    ("multichar", "agentlans/multi-character-dialogue", None, "train"),
    ("dramabench", "FutureMa/DramaBench", "full", "train"),
    ("screenplay", "Atum09/screenplay-dataset", None, "train"),
    ("gutenberg_en", "manu/project_gutenberg", None, "en"),
]

for name, ds, cfg, split in CASES:
    t = time.time()
    try:
        it = load_dataset(ds, cfg, split=split, streaming=True) if cfg else \
             load_dataset(ds, split=split, streaming=True)
        t_load = time.time() - t
        n = 0
        keys = None
        for row in it:
            if keys is None:
                keys = list(row.keys())
            n += 1
            if n >= 20:
                break
        print(f"{name:14} load={t_load:5.1f}s  first20={time.time()-t:5.1f}s  keys={keys}",
              flush=True)
    except Exception as e:
        print(f"{name:14} ERROR after {time.time()-t:5.1f}s: {type(e).__name__}: {e}",
              flush=True)
print("PROBE DONE", flush=True)
