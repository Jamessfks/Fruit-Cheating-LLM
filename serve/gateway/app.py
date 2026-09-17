"""Fruit Drama gateway: the only thing the browser talks to.

Two properties drive the design.

**The client cannot supply a system prompt.** `POST /api/story` has no
`messages` field at all -- only a premise and an optional cast drawn from closed
enumerations. That is the hardening: you cannot inject through a schema with no
slot for it. v1's UI posted a full `messages` array containing its own system
prompt, so any device on the tailnet could rewrite the product's voice.

**Same-origin.** The gateway serves the web app itself, so the app calls a
relative `/api/story` and there is no endpoint to configure, no CORS wildcard,
and no `file://` Origin: null. v1 hardcoded a Tailscale IP into the HTML at
build time, and that IP is already stale.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import pathlib
import re
import sys
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from fruitdrama import contract as C        # noqa: E402
from fruitdrama import prompts as P         # noqa: E402
from fruitdrama import textstats as T       # noqa: E402
from fruitdrama.filters import is_sfw       # noqa: E402

UPSTREAM = os.environ.get("FRUIT_UPSTREAM", "http://127.0.0.1:8000")
MODEL = os.environ.get("FRUIT_MODEL", "fruit")
SLOTS = int(os.environ.get("FRUIT_SLOTS", "4"))
RATE_PER_MIN = int(os.environ.get("FRUIT_RATE_PER_MIN", "6"))
# Generous, because it covers queueing behind other in-flight requests, not
# just generation. A story itself streams its first token in well under a second
# on an idle box.
FIRST_TOKEN_TIMEOUT = float(os.environ.get("FRUIT_FIRST_TOKEN_TIMEOUT", "180"))
STATIC = pathlib.Path(__file__).parent / "static"

BIBLE = json.loads((_ROOT / "data" / "fruit_bible.json").read_text())
FRUIT_NAMES = {e["name"] for e in BIBLE}
STAGES = {"growth", "prime", "ripening", "senescence"}
TONES = {"campy", "weepy", "vicious", "comic", "gothic", "glamorous", "absurd"}

# Prompt-injection canaries. Cheap, deterministic, and they run before any
# model call so a hostile premise costs nothing.
INJECTION = re.compile(
    r"(ignore (all )?previous|disregard (the )?above|you are now|system\s*:|"
    r"</?s>|<\|im_(start|end)\|>|reveal your (system )?prompt|"
    r"print your (system )?prompt)", re.I)

app = FastAPI(title="Fruit Drama", docs_url=None, redoc_url=None)
_sem = asyncio.Semaphore(SLOTS)
_buckets: dict[str, collections.deque] = collections.defaultdict(collections.deque)
_stories: dict[str, dict] = {}
_ring: collections.deque = collections.deque(maxlen=64)
_metrics = collections.Counter()


class CastMember(BaseModel):
    fruit: str
    stage: str = "prime"

    @field_validator("fruit")
    @classmethod
    def known_fruit(cls, v: str) -> str:
        if v not in FRUIT_NAMES:
            raise ValueError("unknown fruit")
        return v

    @field_validator("stage")
    @classmethod
    def known_stage(cls, v: str) -> str:
        if v not in STAGES:
            raise ValueError("unknown life stage")
        return v


class StoryRequest(BaseModel):
    # Note the absence of a `messages` field. That is deliberate and is the
    # whole hardening story; do not add one.
    premise: str = Field(min_length=1, max_length=280)
    tone: str = "campy"
    cast: list[CastMember] = Field(default_factory=list, max_length=5)

    @field_validator("tone")
    @classmethod
    def known_tone(cls, v: str) -> str:
        return v if v in TONES else "campy"


def client_ip(req: Request) -> str:
    return req.client.host if req.client else "unknown"


def rate_limited(ip: str) -> bool:
    now = time.time()
    q = _buckets[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        return True
    q.append(now)
    return False


def screen_premise(text: str) -> str | None:
    """Return a refusal reason, or None if the premise is fine."""
    flat = " ".join(text.split())
    if not flat:
        return "Give me something to work with."
    if INJECTION.search(flat):
        return "That looks like an instruction rather than a premise."
    if re.search(r"https?://|www\.", flat):
        return "Links are not premises."
    letters = sum(c.isalpha() or c.isspace() for c in flat)
    if len(flat) > 12 and letters / len(flat) < 0.5 and T.count_emoji(flat) == 0:
        return "I could not find a story in that."
    if not is_sfw(flat):
        return "Let us keep this PG-13."
    return None


def build_messages(req: StoryRequest) -> list[dict]:
    """Assemble the prompt server-side. The system prompt is byte-identical on
    every request, which is what keeps the upstream prefix cache warm."""
    user = req.premise.strip()
    if req.cast:
        by_name = {e["name"]: e for e in BIBLE}
        lines = []
        for m in req.cast:
            e = by_name[m.fruit]
            lines.append(f"- {e['example']} {e['emoji']} ({m.stage}): {e['archetype']}")
        user += "\n\nCast I want in it:\n" + "\n".join(lines)
    if req.tone != "campy":
        user += f"\n\nTone: {req.tone}."
    return [{"role": "system", "content": P.STORY_SYSTEM},
            {"role": "user", "content": user}]


async def upstream_busy() -> bool:
    """True when every model-server slot is already generating."""
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(f"{UPSTREAM}/slots")
        if r.status_code == 200:
            slots = r.json()
            return bool(slots) and all(s.get("is_processing") for s in slots)
    except Exception:
        pass
    return False


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@app.get("/healthz")
async def healthz():
    """Liveness only. Never touches upstream, so it stays green while the model
    loads and the app can show a warming state instead of a connection error."""
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{UPSTREAM}/health")
        if r.status_code == 200:
            return {"ready": True}
    except Exception:
        pass
    return JSONResponse({"ready": False, "reason": "model_loading"}, status_code=503)


@app.get("/metrics")
async def metrics():
    return dict(_metrics)


@app.get("/api/cast")
async def cast():
    return {"fruits": [{"name": e["name"], "emoji": e["emoji"],
                        "given_name": e["example"].split()[0],
                        "archetype": e["archetype"], "gender": e["gender"]}
                       for e in BIBLE],
            "stages": sorted(STAGES), "tones": sorted(TONES)}


@app.get("/api/examples")
async def examples():
    return {"examples": [
        "strawberry catches her husband with the eggplant at the farmers market",
        "my roommate ate my leftovers and lied about it",
        "two fruits, one affair, one suspicious baby",
        "the broccoli matriarch rewrites the will at the gender-reveal party",
        "\U0001F345 \U0001F494 \U0001F346",
        "what if the pineapple faked his own death??",
        "a lemon funeral where the widow is the suspect",
        "my business partner forged my signature and then lied to my face",
    ]}


@app.get("/api/story/{story_id}")
async def get_story(story_id: str):
    rec = _stories.get(story_id)
    if not rec:
        raise HTTPException(404, "unknown story")
    return rec


@app.post("/api/story")
async def story(req: StoryRequest, request: Request):
    ip = client_ip(request)
    if rate_limited(ip):
        _metrics["rate_limited"] += 1
        return JSONResponse({"error": "rate_limited",
                             "message": "The writers' room is full. Try again shortly."},
                            status_code=429, headers={"Retry-After": "20"})
    reason = screen_premise(req.premise)
    story_id = "st_" + uuid.uuid4().hex[:12]

    async def gen():
        if reason:
            _metrics["policy_rejected"] += 1
            yield sse("policy", {"code": "input_rejected", "message": reason})
            return
        yield sse("meta", {"story_id": story_id, "model": MODEL,
                           "prompt_version": P.prompt_version(),
                           "target_words": [C.WORD_MIN, C.WORD_MAX]})
        if await upstream_busy():
            yield sse("waiting", {"message": "All the writers are mid-story. "
                                             "Yours is next in line."})
        payload = {
            "model": MODEL, "messages": build_messages(req),
            "temperature": 0.9, "top_p": 0.95, "max_tokens": 1500,
            "stream": True, "chat_template_kwargs": {"enable_thinking": False},
        }
        buf, sent, t0, first = "", 0, time.time(), None
        whole: list[str] = []
        waiting_sent = False
        async with _sem:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(
                        connect=3.0, read=FIRST_TOKEN_TIMEOUT, write=10.0,
                        pool=5.0)) as c:
                    async with c.stream("POST", f"{UPSTREAM}/v1/chat/completions",
                                        json=payload) as resp:
                        if resp.status_code != 200:
                            yield sse("error", {"code": "upstream_error",
                                                "retryable": True})
                            return
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            chunk = line[6:].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                delta = json.loads(chunk)["choices"][0]["delta"]
                            except Exception:
                                continue
                            piece = delta.get("content") or ""
                            if not piece:
                                continue
                            if first is None:
                                first = time.time() - t0
                            buf += piece
                            # Emit on sentence/paragraph boundaries so the
                            # reader never sees a half-formed clause, and so a
                            # short lookahead exists for the leak check below.
                            while (m := re.search(r"[.!?…]['\"”’]?\s|\n\n", buf)):
                                out, buf = buf[:m.end()], buf[m.end():]
                                if C.LEAKAGE_RE.search(out):
                                    _metrics["leak_stopped"] += 1
                                    yield sse("policy", {"code": "output_stopped"})
                                    return
                                whole.append(out)
                                sent += len(out)
                                yield sse("delta", {"i": sent, "t": out})
            except (httpx.ReadTimeout, httpx.ConnectError):
                _metrics["upstream_timeout"] += 1
                yield sse("error", {"code": "upstream_timeout",
                                    "message": "The writer went quiet.",
                                    "retryable": True})
                return
        if buf:
            whole.append(buf)
            sent += len(buf)
            yield sse("delta", {"i": sent, "t": buf})

        full = "".join(whole)
        words = T.word_count(full)
        _metrics["stories"] += 1
        rec = {"story_id": story_id, "premise": req.premise,
               "words": words,
               "read_min": round(C.read_minutes(words), 1),
               "emoji": T.count_emoji(full),
               "ms_ttft": int((first or 0) * 1000),
               "ms_total": int((time.time() - t0) * 1000)}
        # Keep the text for the permalink, but bound the map so a long-running
        # process cannot grow without limit.
        _stories[story_id] = {**rec, "story": full}
        _ring.append(story_id)
        while len(_stories) > _ring.maxlen:
            oldest = _ring.popleft() if _ring else None
            if oldest:
                _stories.pop(oldest, None)
            else:
                break
        yield sse("done", rec)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if STATIC.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
