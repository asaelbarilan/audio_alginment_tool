"""A browser tool for hand-marking Hebrew word boundaries, to build the alignment gold set.

There is no Hebrew corpus with human word timings, so the gold set has to be made by hand,
and hand-marking is the expensive step. Praat can do it but costs a lot of friction per
boundary. This is built around the observation that makes the job cheap:

    most boundaries are already right in one of the aligners

So each word shows both proposals, and the common case is one keypress to accept the better
one. Dragging is the fallback, not the default.

    python -m hebrew_training.align_tag_server \\
        --a A_hebrew.jsonl --b B_mms.jsonl --out gold.jsonl

then open http://localhost:8080. Clips are served worst-disagreement-first, because that is
where a human judgement is worth the most; agreement regions teach nothing.

Saves after every clip, so it can be closed and reopened. Standard library plus soundfile —
no web framework, no CDN, works offline.

The output has the same schema as the input manifests, with `words` carrying the human
times, so `alignment_disagreement.py` can score any aligner against it directly.
"""

from __future__ import annotations

import argparse
import io
import os
import json
import re
import unicodedata
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

_WS = re.compile(r"\s+")
LOCK = threading.Lock()
# Set in main() when the host provides a database. Marks then live there instead of
# on the container disk, which does not survive a redeploy.
STORE = None
PAD = 1.0  # seconds of context served either side, so a boundary at the very edge of a
#           clip is still judgeable by what comes before and after it


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", type=Path, required=True, help="First aligned manifest.")
    parser.add_argument("--b", type=Path, help="Second aligned manifest, shown for comparison.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Gold manifest. With several annotators this becomes a directory: "
        "each writes <out>/<name>.jsonl, so nobody overwrites anyone.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="0.0.0.0 to accept connections from other machines.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("TAG_TOKEN", ""),
        help="If set, every request must carry ?t=<token>. Not real auth -- it "
        "just stops a stray crawler writing to your gold set.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Several annotators: ask each for a name and keep their marks in "
        "separate files, so the same clips can be marked twice and the "
        "agreement between people measured.",
    )
    parser.add_argument(
        "--clips-root",
        type=Path,
        help="Look every clip up by file name in this directory instead of the absolute "
        "path in the manifest. Needed to host: the manifests carry the Windows paths "
        "they were built with, which resolve nowhere on a Linux server.",
    )
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--limit", type=int, default=100, help="How many clips to serve.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    return parser.parse_args()


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def load(path: Path) -> dict[tuple[str, float], dict]:
    rows = {}
    if path is None or not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(row["path"], round(float(row.get("start", 0.0)), 3))] = row
    return rows


def timed(row: dict) -> list[dict]:
    return [
        w
        for w in (row.get("words") or [])
        if w.get("start") is not None and w.get("end") is not None
    ]


def disagreement(a: dict, b: dict | None) -> float:
    """How far apart the two aligners are on this clip, used to order the work."""
    if not b:
        return 0.0
    wa, wb = timed(a), timed(b)
    if len(wa) != len(wb) or not wa:
        return 0.0
    ends = sorted(abs(float(x["end"]) - float(y["end"])) for x, y in zip(wa, wb))
    return ends[int(0.9 * (len(ends) - 1))]


def build_clips(args) -> list[dict]:
    """Only clips whose word list reproduces the transcript exactly.

    A clip whose words are a subset of what is spoken is worse than useless here: the
    annotator hears seven words, sees four, and has nowhere to put the boundaries for the
    missing three. That happens whenever a word was dropped upstream for having no
    duration -- which is 11-13% of words in this corpus -- so it has to be excluded rather
    than trusted.
    """
    rows_a, rows_b = load(args.a), load(args.b)
    clips = []
    skipped = 0
    for key, row in rows_a.items():
        words = timed(row)
        if len(words) < 2:
            continue
        transcript = clean(row.get("transcript", ""))
        if transcript and " ".join(clean(w["word"]) for w in words) != transcript:
            skipped += 1
            continue
        other = rows_b.get(key)
        clips.append(
            {
                "path": row["path"],
                "start": key[1],
                "duration": float(row["duration"]),
                "transcript": row.get("transcript", ""),
                "a": [
                    {"word": clean(w["word"]), "start": float(w["start"]), "end": float(w["end"])}
                    for w in words
                ],
                "b": (
                    [
                        {
                            "word": clean(w["word"]),
                            "start": float(w["start"]),
                            "end": float(w["end"]),
                        }
                        for w in timed(other)
                    ]
                    if other and len(timed(other)) == len(words)
                    else None
                ),
                "score": disagreement(row, other),
            }
        )
    if skipped:
        print(f"skipped {skipped} clips whose words do not reproduce the transcript")
    clips.sort(key=lambda c: c["score"], reverse=True)
    return clips[: args.limit]


def resolve(path_value: str, root: Path | None) -> Path:
    """The clip file, rebased onto --clips-root when one is given."""
    # PurePath cannot split a Windows path on Linux, so take the basename by hand.
    name = str(path_value).replace("\\", "/").rsplit("/", 1)[-1]
    return (root / name) if root else Path(path_value)


def clip_wav(clip: dict, root: Path | None = None) -> tuple[bytes, float]:
    """The clip's audio as wav bytes, with PAD seconds of context either side.

    Prefers plain soundfile, because that keeps the tool deployable: reading a window out of
    a multi-hour source needs `training.dataloader`, which drags in torch and the whole
    training package. Once the clips have been cut to their own wav files -- which is what
    `data/gold_set/clips` already is -- the dependency is just soundfile and numpy, small
    enough to host anywhere.
    """
    import soundfile

    begin = max(0.0, clip["start"] - PAD)
    lead = clip["start"] - begin
    want = clip["duration"] + lead + PAD
    path = resolve(clip["path"], root)

    if path.suffix.lower() == ".wav" and path.exists():
        with soundfile.SoundFile(path) as handle:
            rate = handle.samplerate
            handle.seek(int(begin * rate))
            wav = handle.read(int(want * rate), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
    else:
        from training.dataloader import _load_window

        rate = 24000
        wav = _load_window(str(path), begin, want, rate)

    buffer = io.BytesIO()
    soundfile.write(buffer, wav, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), lead


PAGE_FILE = Path(__file__).with_name("align_tag_page.html")


def page() -> bytes:
    """Read the UI from disk on every request, so editing the page needs no restart."""
    return PAGE_FILE.read_bytes()


_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class PostgresStore:
    """Marks in Postgres, for hosts whose container disk does not survive a redeploy.

    One row per annotator holding their whole mark set. At 134 clips that is a few KB, so
    rewriting the blob per save costs nothing and removes every question about partial
    writes -- the same reasoning as rewriting the jsonl file in full.
    """

    def __init__(self, url: str):
        import psycopg

        self.psycopg = psycopg
        self.url = url
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS marks ("
                "  annotator TEXT PRIMARY KEY,"
                "  payload   JSONB NOT NULL,"
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )

    def connect(self):
        return self.psycopg.connect(self.url, autocommit=True)

    def load(self, who: str, clips: list[dict]) -> dict[int, list]:
        with self.connect() as conn:
            row = conn.execute("SELECT payload FROM marks WHERE annotator = %s", (who,)).fetchone()
        if not row:
            return {}
        return index_rows(row[0], clips)

    def save(self, who: str, clips: list[dict], saved: dict[int, list]) -> None:
        payload = [gold_row(clips[i], saved[i]) for i in sorted(saved)]
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO marks (annotator, payload, updated_at)"
                " VALUES (%s, %s, now())"
                " ON CONFLICT (annotator) DO UPDATE"
                "   SET payload = EXCLUDED.payload, updated_at = now()",
                (who, json.dumps(payload, ensure_ascii=False)),
            )


def gold_row(clip: dict, words: list) -> dict:
    """One output record. Same schema as the input manifests, so a gold file can be scored
    against any aligner with no conversion."""
    return {
        "path": clip["path"],
        "start": clip["start"],
        "duration": clip["duration"],
        "transcript": clip["transcript"],
        "words": [
            {
                "word": w["word"],
                "start": round(float(w["start"]), 4),
                "end": round(float(w["end"]), 4),
            }
            for w in words
        ],
    }


def index_rows(rows: list[dict], clips: list[dict]) -> dict[int, list]:
    """Saved records -> {clip index: words}, matched on (path, start)."""
    by_key = {(r["path"], round(float(r["start"]), 3)): r["words"] for r in rows}
    out = {}
    for index, clip in enumerate(clips):
        hit = by_key.get((clip["path"], round(clip["start"], 3)))
        if hit:
            out[index] = hit
    return out


def gold_path(args, who: str | None) -> Path:
    """One file per annotator when several are marking, otherwise the single --out file."""
    if not args.multi:
        return args.out
    args.out.mkdir(parents=True, exist_ok=True)
    return args.out / f"{who or 'anon'}.jsonl"


def load_saved(path: Path, clips: list[dict]) -> dict[int, list]:
    if not path.exists():
        return {}
    by_key = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_key[(row["path"], round(float(row["start"]), 3))] = row["words"]
    out = {}
    for index, clip in enumerate(clips):
        hit = by_key.get((clip["path"], round(clip["start"], 3)))
        if hit:
            out[index] = hit
    return out


def make_handler(args, clips, saved):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: A003 -- quiet; progress is in the page
            pass

        def send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def who(self):
            from urllib.parse import parse_qs

            name = (parse_qs(urlparse(self.path).query).get("who") or [""])[0]
            return name if _NAME.match(name) else None

        def authorised(self):
            from urllib.parse import parse_qs

            if not args.token:
                return True
            return (parse_qs(urlparse(self.path).query).get("t") or [""])[0] == args.token

        def do_GET(self):
            route = urlparse(self.path).path
            # The page itself is served without a token: a managed host probes GET / for a
            # 2xx to decide the app is alive, and a 403 there reads as a dead app. Nothing
            # is exposed by this -- the HTML carries no clips and no marks, and every /api
            # route below still demands the token.
            if route == "/":
                return self.send(200, page(), "text/html; charset=utf-8")
            if not self.authorised():
                return self.send(403, b"bad or missing token", "text/plain")
            if route == "/api/meta":
                meta = {"multi": bool(args.multi), "clips": len(clips)}
                return self.send(200, json.dumps(meta).encode("utf-8"), "application/json")
            if route == "/api/clips":
                if STORE is not None:
                    mine = STORE.load(self.who() or "anon", clips)
                elif args.multi:
                    mine = load_saved(gold_path(args, self.who()), clips)
                else:
                    mine = saved
                payload = []
                for index, clip in enumerate(clips):
                    entry = dict(clip)
                    entry["saved"] = mine.get(index)
                    payload.append(entry)
                return self.send(
                    200,
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            if route.startswith("/api/audio/"):
                index = int(route.rsplit("/", 1)[1])
                try:
                    data, lead = clip_wav(clips[index], args.clips_root)
                except Exception as exc:  # noqa: BLE001 -- a bad clip must not kill the server
                    return self.send(500, str(exc).encode(), "text/plain")
                headers = {"X-Lead": f"{lead:.4f}", "Accept-Ranges": "bytes"}
                # An <audio> element cannot seek without byte ranges: setting currentTime on
                # a response served as a plain 200 is silently ignored and playback stays
                # wherever it was. Every "play this word" then started from the top.
                span = self.headers.get("Range")
                if span and span.startswith("bytes="):
                    first, _, last = span[6:].partition("-")
                    begin = int(first) if first else 0
                    end = int(last) if last else len(data) - 1
                    end = min(end, len(data) - 1)
                    if begin > end:
                        return self.send(416, b"", "audio/wav", headers)
                    chunk = data[begin : end + 1]
                    headers["Content-Range"] = f"bytes {begin}-{end}/{len(data)}"
                    return self.send(206, chunk, "audio/wav", headers)
                return self.send(200, data, "audio/wav", headers)
            return self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.authorised():
                return self.send(403, b"bad or missing token", "text/plain")
            if urlparse(self.path).path != "/api/gold":
                return self.send(404, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            who = self.who() or "anon"
            # Re-read before writing. Two annotators sharing a store would otherwise each
            # hold a stale copy and the second save would drop the first one's work.
            with LOCK:
                if STORE is not None:
                    mine = STORE.load(who, clips)
                    mine[int(body["index"])] = body["words"]
                    STORE.save(who, clips, mine)
                else:
                    path = gold_path(args, self.who())
                    mine = load_saved(path, clips)
                    mine[int(body["index"])] = body["words"]
                    write_gold(path, clips, mine)
            return self.send(200, b'{"ok":true}', "application/json")

    return Handler


def write_gold(out: Path, clips: list[dict], saved: dict[int, list]) -> None:
    """Rewritten in full after each clip: 100 rows is nothing, and a partial append that
    crashed mid-write would be worse than the rewrite cost."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for index in sorted(saved):
            clip = clips[index]
            handle.write(
                json.dumps(
                    {
                        "path": clip["path"],
                        "start": clip["start"],
                        "duration": clip["duration"],
                        "transcript": clip["transcript"],
                        "words": [
                            {
                                "word": w["word"],
                                "start": round(float(w["start"]), 4),
                                "end": round(float(w["end"]), 4),
                            }
                            for w in saved[index]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    global STORE
    args = parse_args()
    database = os.environ.get("DATABASE_URL", "")
    if database:
        STORE = PostgresStore(database)
        print("marks -> Postgres (DATABASE_URL)")
    clips = build_clips(args)
    if not clips:
        raise SystemExit(f"no usable clips in {args.a}")

    # In --multi each annotator has their own file, loaded per request from their name;
    # there is no single shared state to resume into here.
    saved: dict[int, list] = {} if args.multi else load_saved(args.out, clips)
    if saved:
        print(f"resuming: {len(saved)} clips already marked")

    missing = [c for c in clips if not resolve(c["path"], args.clips_root).exists()]
    if missing:
        hint = " (try --clips-root)" if not args.clips_root else ""
        first = resolve(missing[0]["path"], args.clips_root)
        print(
            f"{len(missing)} of {len(clips)} clip files are not where the "
            f"manifest says they are{hint}."
        )
        raise SystemExit(f"  first missing: {first}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args, clips, saved))
    words = sum(len(c["a"]) for c in clips)
    print(f"{len(clips)} clips, {words} boundaries to check")
    where = "localhost" if args.host in ("127.0.0.1", "localhost") else args.host
    link = f"http://{where}:{args.port}/"
    if args.token:
        link += f"?t={args.token}"
    print(f"open {link}   (ctrl-c to stop; progress is saved per clip)")
    if args.multi:
        print(f"several annotators: each gets their own file under {args.out}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. {len(saved)} clips marked -> {args.out}")


if __name__ == "__main__":
    main()
