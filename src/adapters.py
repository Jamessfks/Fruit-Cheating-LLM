"""Per-source adapters: pull a permissive HF dataset and yield unified rows.

Each adapter yields dicts: {"user": str, "assistant": str, "meta": {...}}.

Ingestion strategy: HF row-by-row streaming is far too slow here (rate-limited
to a few rows/sec unauthenticated), so we bulk-download parquet SHARDS and parse
them locally with pyarrow (fast), one shard at a time then delete it, to respect
a small disk budget. Tiny/unconverted datasets are downloaded in full.
"""
import os
import gzip
import json
import tempfile
import urllib.request

import filters as F

_UA = {"User-Agent": "fruit-cheating-llm/1.0"}
_PQ_TMP = os.environ.get("PQ_TMP", tempfile.gettempdir())


def _parquet_files(dataset, split, config=None):
    url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as r:
        data = json.load(r)
    out = []
    for f in data.get("parquet_files", []):
        if f.get("split") != split:
            continue
        if config and f.get("config") != config:
            continue
        out.append(f["url"])
    return out


def _download(url, path, timeout=180):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _iter_parquet(dataset, split, config=None, columns=None):
    """Yield row dicts by downloading each parquet shard, parsing, then deleting."""
    import pyarrow.parquet as pq
    files = _parquet_files(dataset, split, config)
    if not files:
        raise RuntimeError(f"no parquet files listed for {dataset} [{config}/{split}]")
    for i, url in enumerate(files):
        tmp = os.path.join(_PQ_TMP, f"_pq_{abs(hash(dataset))%99999}_{i}.parquet")
        try:
            _download(url, tmp)
            pf = pq.ParquetFile(tmp)
            for g in range(pf.num_row_groups):
                tbl = pf.read_row_group(g, columns=columns)
                for row in tbl.to_pylist():
                    yield row
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _repo_parquet_files(dataset, branch="main"):
    url = f"https://huggingface.co/api/datasets/{dataset}/tree/{branch}?recursive=1"
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as r:
        tree = json.load(r)
    return [f"https://huggingface.co/datasets/{dataset}/resolve/{branch}/{e['path']}"
            for e in tree if e.get("type") == "file"
            and e.get("path", "").endswith(".parquet")]


def _iter_repo_parquet(dataset, columns=None, branch="main"):
    """Read a repo's own parquet shards directly (each shard read independently,
    so per-shard schema differences are fine)."""
    import pyarrow.parquet as pq
    files = _repo_parquet_files(dataset, branch)
    if not files:
        raise RuntimeError(f"no parquet files in repo {dataset}@{branch}")
    for i, url in enumerate(files):
        tmp = os.path.join(_PQ_TMP, f"_repo_{abs(hash(dataset)) % 99999}_{i}.parquet")
        try:
            _download(url, tmp)
            pf = pq.ParquetFile(tmp)
            avail = set(pf.schema_arrow.names)
            cols = [c for c in columns if c in avail] if columns else None
            for g in range(pf.num_row_groups):
                for row in pf.read_row_group(g, columns=cols).to_pylist():
                    yield row
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _iter_repo_jsonl(dataset, filename, branch="main"):
    """Download a repo's (optionally gzipped) JSONL data file and yield rows."""
    url = f"https://huggingface.co/datasets/{dataset}/resolve/{branch}/{filename}"
    tmp = os.path.join(_PQ_TMP, f"_jsonl_{abs(hash(dataset)) % 99999}")
    try:
        _download(url, tmp)
        opener = gzip.open if filename.endswith(".gz") else open
        with opener(tmp, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _iter_rows_api(dataset, split, config="default", max_rows=None):
    """Paginate the datasets-server /rows API (pre-decoded rows, 100/page).

    Used for small datasets whose shards have inconsistent schemas and so cannot
    be unified by `datasets.load_dataset`.
    """
    offset, page = 0, 100
    while True:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
               f"&config={config}&split={split}&offset={offset}&length={page}")
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as r:
            rows = json.load(r).get("rows", [])
        if not rows:
            break
        for item in rows:
            yield item["row"]
        offset += len(rows)
        if len(rows) < page or (max_rows and offset >= max_rows):
            break


def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            j = json.loads(v)
            return j if isinstance(j, list) else [j]
        except Exception:
            return []
    return []


def _as_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            j = json.loads(v)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------- screenplay
def adapt_screenplay(voice_cap, structure_cap, genres, voice_cats,
                     structure_cats, max_scan):
    genres = set(genres)
    voice_cats, structure_cats = set(voice_cats), set(structure_cats)
    v = s = scanned = 0
    try:
        it = _iter_parquet("Atum09/screenplay-dataset", "train", "default",
                           columns=["conversations", "category", "genre"])
    except Exception as e:
        print(f"  [screenplay] load failed: {e}")
        return
    for row in it:
        if scanned >= max_scan or (v >= voice_cap and s >= structure_cap):
            break
        scanned += 1
        if (row.get("genre") or "").lower() not in genres:
            continue
        cat = (row.get("category") or "").lower()
        if cat in voice_cats and v < voice_cap:
            slice_name = "screenplay_voice"
        elif cat in structure_cats and s < structure_cap:
            slice_name = "screenplay_structure"
        else:
            continue
        user = asst = None
        for turn in _as_list(row.get("conversations")):
            frm = (turn.get("from") or "").lower()
            val = turn.get("value") or ""
            if user is None and frm in ("human", "user", "prompter"):
                user = val
            elif asst is None and frm in ("gpt", "assistant", "chatgpt", "bot"):
                asst = val
        if not user or not asst:
            continue
        user, asst = F.clean(user), F.clean(asst)
        if not F.is_sfw(user, asst) or not F.quality_ok(asst, max_len=9000):
            continue
        if slice_name == "screenplay_voice":
            v += 1
        else:
            s += 1
        yield {"user": user, "assistant": asst,
               "meta": {"source": "Atum09/screenplay-dataset", "slice": slice_name,
                        "license": "mit", "genre": row.get("genre"), "category": cat}}
    print(f"  [screenplay] scanned={scanned} voice={v} structure={s}")


# ------------------------------------------------------------ multi-character
def adapt_multichar(cap, max_scan):
    n = scanned = 0
    try:
        it = _iter_repo_jsonl("agentlans/multi-character-dialogue", "train.jsonl.gz")
    except Exception as e:
        print(f"  [multichar] load failed: {e}")
        return
    for row in it:
        if n >= cap or scanned >= max_scan:
            break
        scanned += 1
        setting = F.clean(row.get("setting") or "")
        conv = _as_list(row.get("conversation"))
        if not setting or len(conv) < 3:
            continue
        chars = _as_dict(row.get("characters"))
        names = ", ".join(list(chars.keys())[:8]) if chars else ""
        lines = []
        for t in conv:  # turns are dicts in some shards, plain strings in others
            if isinstance(t, dict):
                who = t.get("from") or t.get("speaker") or ""
                msg = t.get("message") or t.get("value") or ""
            else:
                who, msg = "", str(t)
            msg = F.clean(msg)
            if msg:
                lines.append(f"{who}: {msg}" if who else msg)
        asst = "\n".join(lines)
        user = ("Write a dramatic multi-character scene as screenplay dialogue.\n"
                f"Setting: {setting}")
        if names:
            user += f"\nCharacters: {names}"
        if not F.is_sfw(user, asst) or not F.quality_ok(asst, max_len=9000):
            continue
        n += 1
        yield {"user": user, "assistant": asst,
               "meta": {"source": "agentlans/multi-character-dialogue",
                        "slice": "multichar", "license": "cc-by-4.0"}}
    print(f"  [multichar] scanned={scanned} kept={n}")


# --------------------------------------------------------------- drama bench
def adapt_dramabench(cap, max_scan):
    n = scanned = 0
    try:
        it = _iter_parquet("FutureMa/DramaBench", "train", "full",
                           columns=["title", "description", "context", "continuation"])
    except Exception as e:
        print(f"  [dramabench] load failed: {e}")
        return
    for row in it:
        if n >= cap or scanned >= max_scan:
            break
        scanned += 1
        context = F.clean(row.get("context") or "")
        cont = F.clean(row.get("continuation") or "")
        if not context or not cont:
            continue
        title = F.clean(row.get("title") or "")
        desc = F.clean(row.get("description") or "")
        head = "Continue this dramatic script in the same voice."
        if title:
            head += f"\nTitle: {title}"
        if desc:
            head += f"\nPremise: {desc[:400]}"
        user = f"{head}\n\n---\n{context[-4000:]}"
        if not F.is_sfw(user, cont) or not F.quality_ok(cont, max_len=9000):
            continue
        n += 1
        yield {"user": user, "assistant": cont,
               "meta": {"source": "FutureMa/DramaBench", "slice": "dramabench",
                        "license": "mit"}}
    print(f"  [dramabench] scanned={scanned} kept={n}")


# ----------------------------------------------------------------- gutenberg
def adapt_gutenberg(cap_chunks, max_books, chunks_per_book=4):
    """Public-domain melodrama. DISABLED by default (see config): the full en
    corpus is too large to pull quickly. Kept for use with a curated PD subset."""
    n = books = 0
    instr = ("Write an emotionally charged dramatic scene in the style of "
             "classic melodrama, rich with dialogue and rising tension.")
    try:
        it = _iter_parquet("manu/project_gutenberg", "en")
    except Exception as e:
        print(f"  [gutenberg] load failed: {e}")
        return
    for row in it:
        if n >= cap_chunks or books >= max_books:
            break
        books += 1
        text = row.get("text") or ""
        if len(text) < 4000:
            continue
        body = text.split("*** START", 1)[-1] if "*** START" in text else text
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        got, buf, buflen = 0, [], 0
        for p in paras:
            buf.append(p)
            buflen += len(p)
            if buflen >= 1200:
                chunk = F.clean("\n\n".join(buf))
                buf, buflen = [], 0
                if got >= chunks_per_book or n >= cap_chunks:
                    break
                if '"' not in chunk and "'" not in chunk:
                    continue
                if not F.is_dramatic(chunk, 2) or not F.is_sfw(chunk) \
                        or not F.quality_ok(chunk, max_len=4000):
                    continue
                got += 1
                n += 1
                yield {"user": instr, "assistant": chunk,
                       "meta": {"source": "manu/project_gutenberg",
                                "slice": "gutenberg_melodrama",
                                "license": "public-domain", "book_id": row.get("id")}}
    print(f"  [gutenberg] books_scanned={books} chunks={n}")


# ------------------------------------------------------------ writing prompts
def adapt_writingprompts(cap, max_scan):
    n = scanned = 0
    try:
        it = _iter_parquet("euclaise/writingprompts", "train", "default",
                           columns=["prompt", "story"])
    except Exception as e:
        print(f"  [writingprompts] load failed: {e}")
        return
    for row in it:
        if n >= cap or scanned >= max_scan:
            break
        scanned += 1
        prompt = F.detokenize(row.get("prompt") or "")
        story = F.detokenize(row.get("story") or "")
        if not prompt or not story:
            continue
        if not F.is_dramatic(prompt + " " + story[:1500], 2):
            continue
        if not F.is_sfw(prompt, story) or not F.quality_ok(story, min_len=400, max_len=8000):
            continue
        user = ("Write a dramatic short story with a shocking plot twist based on "
                f"this prompt:\n\n{prompt}")
        n += 1
        yield {"user": user, "assistant": story,
               "meta": {"source": "euclaise/writingprompts",
                        "slice": "writingprompts_twist", "license": "mit"}}
    print(f"  [writingprompts] scanned={scanned} kept={n}")
