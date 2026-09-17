"""Bulk generation client for a local OpenAI-compatible server. Stdlib only.

Designed for a job measured in tens of hours that WILL be interrupted:
append-only output keyed on premise id, resume by reading back what is already
there, and fsync often enough that a hard kill costs seconds of work rather
than hours.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_LEADING_THINK = re.compile(r"^.*?</think>\s*", re.S)


@dataclass
class GenConfig:
    base_url: str = "http://127.0.0.1:8000"
    model: str = "fruit"
    temperature: float = 0.95
    top_p: float = 0.95
    min_p: float = 0.03
    max_tokens: int = 1400
    presence_penalty: float = 0.3
    repeat_penalty: float = 1.05
    timeout: float = 600.0
    concurrency: int = 16
    retries: int = 2


def chat(cfg: GenConfig, messages: list[dict], **overrides) -> str:
    """One blocking chat completion. Raises on persistent failure."""
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "min_p": cfg.min_p,
        "max_tokens": cfg.max_tokens,
        "presence_penalty": cfg.presence_penalty,
        "repeat_penalty": cfg.repeat_penalty,
        "stream": False,
        # Belt and braces for hybrid-reasoning models: the server flag alone can
        # regress on a version bump, so the request asks too and the response is
        # scrubbed below.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(overrides)
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(cfg.retries + 1):
        try:
            req = urllib.request.Request(
                f"{cfg.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"] or ""
            text = _THINK.sub("", text)
            if "</think>" in text:
                text = _LEADING_THINK.sub("", text)
            return text.strip()
        except Exception as e:  # noqa: BLE001 - retry on anything transient
            last = e
            if attempt < cfg.retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"generation failed after {cfg.retries + 1} tries: {last}")


def server_ready(cfg: GenConfig, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{cfg.base_url}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class ResumableWriter:
    """Append-only JSONL with an id index, so a restart skips finished work."""

    def __init__(self, path: str, id_key: str = "premise_id", fsync_every: int = 25):
        self.path = path
        self.id_key = id_key
        self.fsync_every = fsync_every
        self._lock = threading.Lock()
        self._n = 0
        self.done: set[str] = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        self.done.add(json.loads(line)[id_key])
                    except Exception:
                        continue
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, rec: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._n += 1
            if self._n % self.fsync_every == 0:
                self._fh.flush()
                os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()


def run_batch(cfg: GenConfig, tasks: list[tuple[str, list[dict], dict]],
              writer: ResumableWriter, on_result=None) -> dict:
    """Generate for each (id, messages, meta), skipping ids already written."""
    pending = [t for t in tasks if t[0] not in writer.done]
    stats = {"total": len(tasks), "skipped": len(tasks) - len(pending),
             "ok": 0, "failed": 0}
    t0 = time.time()

    def work(task):
        tid, messages, meta = task
        text = chat(cfg, messages)
        return tid, text, meta

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futs = {pool.submit(work, t): t[0] for t in pending}
        for fut in as_completed(futs):
            try:
                tid, text, meta = fut.result()
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                writer.write({writer.id_key: futs[fut], "error": str(e)[:300]})
                continue
            stats["ok"] += 1
            rec = {writer.id_key: tid, "text": text, **meta}
            writer.write(rec)
            if on_result:
                on_result(rec, stats, time.time() - t0)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats
