"""Offline smoke test: generate fruit-drama episodes straight from base + LoRA.

Useful to sanity-check a freshly trained adapter without standing up a server.

  python infer.py
"""
import os
import json
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel                                    # noqa: E402

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
ADAPTER = os.environ.get("ADAPTER_DIR", "./qwen3-8b-fruit-lora")
TRAIN_JSONL = os.environ.get("TRAIN_JSONL", "data/fruit_drama_sft.train.jsonl")

# Use the exact fruit system prompt the model was trained with.
SYS = None
for line in open(TRAIN_JSONL, encoding="utf-8"):
    r = json.loads(line)
    if r.get("meta", {}).get("slice") == "fruit_synth":
        SYS = r["messages"][0]["content"]
        break
assert SYS, "fruit system prompt not found in training data"

PROMPTS = [
    "Write episode 4 of an AI fruit drama. Logline: A strawberry temptress betrays the "
    "banana heir with his own brother, the plantain, while the broccoli matriarch secretly "
    "rewrites her will. End on a jaw-dropping twist.",
    ("Write ONE 6-scene AI fruit-drama episode (Episode 1) that intertwines the cast below "
     "into a single, mind-blowing storyline with a shocking final twist.\n\nCAST:\n"
     "1. Peach (\"Priya Peach\") - Female, Prime. Femme fatale, hard pit no one cracks. "
     "Role: Yoga instructor. Disposition: Hot and arrogant. Function: Temptress.\n"
     "2. Banana (\"Peter Banana\") - Male, Prime. Golden and bulky. Role: Gym bro. "
     "Disposition: Disciplined; secretly rich. Function: The Golden child.\n"
     "3. Watermelon - Male, Ripening. Gentle giant cracking under pressure. Role: Retired "
     "pro athlete. Disposition: Haunted by the past. Function: The Underdog.\n\n"
     "CONTEXT: A luxury wellness retreat during the annual charity gala; old rivalries "
     "resurface and a paternity secret is about to detonate."),
]

print("loading base + adapter ...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("loaded.", flush=True)

for i, p in enumerate(PROMPTS, 1):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": p}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=700, do_sample=True, temperature=0.8,
                             top_p=0.9, repetition_penalty=1.05,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n" + "=" * 72, flush=True)
    print(f"PROMPT {i}: {p[:90]}...")
    print("-" * 72)
    print(gen)
print("\nINFER_DONE", flush=True)
