"""A browser tool for hand-marking Hebrew word boundaries, to build the alignment gold set.

There is no Hebrew corpus with human word timings, so the gold set has to be made by hand,
and hand-marking is the expensive step. Praat can do it but costs a lot of friction per
boundary. This is built around the observation that makes the job cheap:

    most boundaries are already right in one of the aligners

So each word shows both proposals, and the common case is one keypress to accept the better
one. Dragging is the fallback, not the default.

    python -m hebrew_training.align_tag_server \\
        --datasets-folder data/datasets --dataset plenum --out gold.jsonl

then open http://localhost:8080. Clips are served worst-disagreement-first, because that is
where a human judgement is worth the most; agreement regions teach nothing.

Saves after every clip, so it can be closed and reopened. Standard library plus soundfile —
no web framework, no CDN, works offline.

Datasets live under a folder or a bucket (see --datasets-folder / --datasets-bucket): each
top-level entry is one dataset, holding a metadata.json, a manifest.jsonl (one row per clip,
each carrying a `labels` array -- one entry per annotation source, e.g. two forced aligners),
and an audio/ folder. The `labels[0]` entry seeds the marks; a second label, when present, is
only used to rank clips by disagreement.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# When a .env file is present - load that
load_dotenv()


_WS = re.compile(r"\s+")
LOCK = threading.Lock()
# Set in main() when the host provides a database. Marks then live there instead of
# on the container disk, which does not survive a redeploy.
STORE = None
CLAIMS = None
BLOCK = 10  # clips a person holds at once; refilled as they work
STALE_SECONDS = (
    24 * 3600
)  # an unmarked claim this old goes back in the pool, so a person
#                            who opens the page and wanders off does not strand their share


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--datasets-folder",
        type=Path,
        help="Local folder whose top-level subfolders are datasets. Ignored when "
        "--datasets-bucket is also given.",
    )
    parser.add_argument(
        "--datasets-bucket",
        help="Read datasets from this S3 bucket instead of the filesystem -- same layout, "
        "one top-level key prefix per dataset. Takes priority over --datasets-folder. "
        "Endpoint and credentials come from S3_ENDPOINT / S3_REGION / S3_ACCESS_KEY_ID / "
        "S3_SECRET_ACCESS_KEY.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Which dataset to serve -- a top-level folder name under --datasets-folder / "
        "--datasets-bucket.",
    )
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
        "--auth",
        choices=["xhost"],
        help="Require Google sign-in, verifying xhostd's signed cookie. Identity then comes "
        "from the cookie and ?who= is ignored, which is what stops an annotator writing as "
        "someone else. Needs a database for the name bindings.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Hand each annotator their own clips, so nobody marks the same sentence "
        "twice. Without it everyone walks the same order, which is what measures how "
        "far two humans sit apart on the same boundary.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Several annotators: ask each for a name and keep their marks in "
        "separate files, so the same clips can be marked twice and the "
        "agreement between people measured.",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="How many clips to serve."
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    return parser.parse_args()


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


# ---- datasets --------------------------------------------------------------
#
# A dataset is a folder (local or in a bucket) of:
#   metadata.json     {"name": ...}
#   manifest.jsonl     one row per clip: {id, metadata, labels: [...]}
#   audio/<file>.wav    referenced by each label's `audio` field
#
# `labels` is one entry per annotation source (what used to be "the A file" and "the B
# file"), each with its own words -- the same clip, several people's or aligners' opinions
# of where the boundaries are.


class Dataset:
    name: str

    def manifest(self) -> list[dict]:
        raise NotImplementedError

    def metadata(self) -> dict:
        raise NotImplementedError

    def read_bytes(self, relpath: str) -> bytes:
        raise NotImplementedError

    def audio_exists(self, relpath: str) -> bool | None:
        """True/False when checkable up front, None when it can only be known by trying
        (a bucket read costs a round trip, so it is not worth doing 100 times at startup)."""
        return None


def _read_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class LocalDataset(Dataset):
    def __init__(self, root: Path, name: str):
        self.name = name
        self.dir = root / name
        if not self.dir.is_dir():
            available = (
                sorted(p.name for p in root.iterdir() if p.is_dir())
                if root.is_dir()
                else []
            )
            raise SystemExit(
                f"no dataset {name!r} under {root}"
                + (f" (found: {', '.join(available)})" if available else "")
            )

    def manifest(self) -> list[dict]:
        path = self.dir / "manifest.jsonl"
        if not path.exists():
            raise SystemExit(f"missing manifest.jsonl in {self.dir}")
        return _read_jsonl(path.read_text(encoding="utf-8"))

    def metadata(self) -> dict:
        path = self.dir / "metadata.json"
        if not path.exists():
            raise SystemExit(f"missing metadata.json in {self.dir}")
        return json.loads(path.read_text(encoding="utf-8"))

    def read_bytes(self, relpath: str) -> bytes:
        return (self.dir / relpath).read_bytes()

    def audio_exists(self, relpath: str) -> bool | None:
        return (self.dir / relpath).exists()


class BucketDataset(Dataset):
    def __init__(self, bucket: str, name: str):
        self.name = name
        self.bucket = bucket
        self.prefix = f"{name}/"

    def manifest(self) -> list[dict]:
        return _read_jsonl(
            s3_bytes(self.bucket, self.prefix + "manifest.jsonl").decode("utf-8")
        )

    def metadata(self) -> dict:
        return json.loads(
            s3_bytes(self.bucket, self.prefix + "metadata.json").decode("utf-8")
        )

    def read_bytes(self, relpath: str) -> bytes:
        return s3_bytes(self.bucket, self.prefix + relpath)


def resolve_dataset(args) -> Dataset:
    if args.datasets_bucket:
        return BucketDataset(args.datasets_bucket, args.dataset)
    if args.datasets_folder:
        return LocalDataset(args.datasets_folder, args.dataset)
    raise SystemExit("one of --datasets-folder or --datasets-bucket is required")


def timed(label: dict) -> list[dict]:
    return [
        w
        for w in (label.get("words") or [])
        if w.get("start") is not None and w.get("end") is not None
    ]


def disagreement(a: dict, b: dict | None) -> float:
    """How far apart two label sources are on this clip, used to order the work."""
    if not b:
        return 0.0
    wa, wb = timed(a), timed(b)
    if len(wa) != len(wb) or not wa:
        return 0.0
    ends = sorted(abs(float(x["end"]) - float(y["end"])) for x, y in zip(wa, wb))
    return ends[int(0.9 * (len(ends) - 1))]


def build_clips(rows: list[dict], limit: int) -> list[dict]:
    """Only clips whose first label's words reproduce its transcript exactly.

    A clip whose words are a subset of what is spoken is worse than useless here: the
    annotator hears seven words, sees four, and has nowhere to put the boundaries for the
    missing three. A second label, when the row has one, is used only to rank the work --
    it is shown for comparison but never has to pass this check itself.
    """
    clips = []
    skipped = 0
    for row in rows:
        labels = row.get("labels") or []
        if not labels:
            continue
        primary = labels[0]
        words = timed(primary)
        if len(words) < 2:
            continue
        text = clean(row.get("text", ""))
        if text and " ".join(clean(w["word"]) for w in words) != text:
            skipped += 1
            continue
        secondary = labels[1] if len(labels) > 1 else None
        other_words = timed(secondary) if secondary else []
        clips.append(
            {
                "id": row["id"],
                "metadata": row.get("metadata") or {},
                "duration": float(row["duration"]),
                "text": row.get("text", ""),
                "audio": row["audio"],
                "labels": [
                    {
                        "source": label.get("source", ""),
                        "words": [
                            {
                                "word": clean(w["word"]),
                                "start": float(w["start"]),
                                "end": float(w["end"]),
                            }
                            for w in timed(label)
                        ],
                    }
                    for label in labels
                ],
                "a": [
                    {
                        "word": clean(w["word"]),
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                    }
                    for w in words
                ],
                "b": (
                    [
                        {
                            "word": clean(w["word"]),
                            "start": float(w["start"]),
                            "end": float(w["end"]),
                        }
                        for w in other_words
                    ]
                    if secondary and len(other_words) == len(words)
                    else None
                ),
                "score": disagreement(primary, secondary),
            }
        )
    if skipped:
        print(f"skipped {skipped} clips whose words do not reproduce the transcript")
    clips.sort(key=lambda c: c["score"], reverse=True)
    return clips[:limit]


_S3_CACHE: dict[str, bytes] = {}
_S3_ORDER: list[str] = []
S3_CACHE_MAX = (
    48  # ~15 MB of wav; enough that a person working through a block re-reads
)
#                    nothing, small enough not to hold the whole corpus in memory


_S3 = None


def s3_client():
    """Built from the injected environment only. S3_ENDPOINT inside the container is a
    platform-internal address, so constructing it from the hostname would point at the
    wrong place.

    Made once: botocore builds a signer and loads service models per client, which is not
    something to repeat on every clip.
    """
    global _S3
    if _S3 is not None:
        return _S3
    import boto3

    _S3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    )
    return _S3


def s3_bytes(bucket: str, key: str) -> bytes:
    cache_key = f"{bucket}/{key}"
    hit = _S3_CACHE.get(cache_key)
    if hit is not None:
        return hit
    body = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    _S3_CACHE[cache_key] = body
    _S3_ORDER.append(cache_key)
    while len(_S3_ORDER) > S3_CACHE_MAX:
        _S3_CACHE.pop(_S3_ORDER.pop(0), None)
    return body


def clip_wav(clip: dict, dataset: Dataset) -> tuple[bytes, float]:
    """The clip's audio as wav bytes.

    The dataset's audio already is exactly the clip -- no more context pad either side, that
    was a serving-time convenience of the old per-clip files, not part of the clip itself.
    Read through soundfile regardless of source, so a non-wav or stereo file still comes out
    as the mono 16-bit wav the page's <audio> element expects.
    """
    import soundfile

    raw = dataset.read_bytes(clip["audio"])
    with soundfile.SoundFile(io.BytesIO(raw)) as handle:
        rate = handle.samplerate
        wav = handle.read(dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    buffer = io.BytesIO()
    soundfile.write(buffer, wav, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), 0.0


PAGE_FILE = Path(__file__).with_name("align_tag_page.html")


def page() -> bytes:
    """Read the UI from disk on every request, so editing the page needs no restart."""
    return PAGE_FILE.read_bytes()


_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

AUTH_ISSUER = "https://auth.xhostd.com"
JWKS_URL = "https://auth.xhostd.com/xhost-auth/jwks"
COOKIE = "__Host-xhost_id"
_JWKS: dict = {"keys": [], "fetched": 0.0}


def jwks() -> dict:
    """The signing keys, cached for an hour. Refetched on an unknown kid so a rotation does
    not lock everybody out until the cache expires."""
    import urllib.request

    if _JWKS["keys"] and time.time() - _JWKS["fetched"] < 3600:
        return _JWKS
    with urllib.request.urlopen(JWKS_URL, timeout=10) as handle:
        _JWKS["keys"] = json.loads(handle.read().decode("utf-8")).get("keys", [])
    _JWKS["fetched"] = time.time()
    return _JWKS


def identity(cookie_header: str, host: str) -> dict | None:
    """The signed-in user, or None.

    The token is verified properly -- RS256 pinned, issuer and audience checked, signature
    against the published keys. A decode-only read would accept anything a caller cared to
    forge, and the whole point of this is that an annotator cannot write as someone else.
    """
    import jwt
    from jwt import PyJWKSet

    token = ""
    for part in (cookie_header or "").split(";"):
        name, _, value = part.strip().partition("=")
        if name == COOKIE:
            token = value
            break
    if not token:
        return None

    try:
        kid = jwt.get_unverified_header(token).get("kid")
        keys = PyJWKSet.from_dict(jwks())
        signing = next((k for k in keys.keys if k.key_id == kid), None)
        if signing is None:
            _JWKS["fetched"] = 0.0  # a rotated key; refetch once before giving up
            keys = PyJWKSet.from_dict(jwks())
            signing = next((k for k in keys.keys if k.key_id == kid), None)
        if signing is None:
            return None
        claims = jwt.decode(
            token,
            signing.key,
            algorithms=["RS256"],
            issuer=AUTH_ISSUER,
            audience=host,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except Exception:  # noqa: BLE001 -- any failure is simply "not signed in"
        return None
    return {
        "sub": claims["sub"],
        "email": claims.get("email", ""),
        "display": claims.get("name") or claims.get("email", ""),
    }


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
        self.pending_migration = False
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS identities ("
                "  sub     TEXT PRIMARY KEY,"
                "  name    TEXT NOT NULL UNIQUE,"
                "  email   TEXT,"
                "  display TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS claims ("
                "  clip_key   TEXT PRIMARY KEY,"
                "  annotator  TEXT NOT NULL,"
                "  claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                "  done       BOOLEAN NOT NULL DEFAULT false)"
            )
            # marks: one row per (annotator, clip), so a save only rewrites the clip that
            # changed. A pre-existing table without the clip_key column is the old
            # schema (one JSONB blob per annotator) and is served read-only until a user
            # confirms the migration -- see migrate().
            if conn.execute("SELECT to_regclass('marks')").fetchone()[0] is None:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS marks ("
                    "  annotator  TEXT NOT NULL,"
                    "  clip_key   TEXT NOT NULL,"
                    "  payload    JSONB NOT NULL,"
                    "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                    "  PRIMARY KEY (annotator, clip_key))"
                )
            else:
                column = conn.execute(
                    "SELECT column_name FROM information_schema.columns"
                    " WHERE table_name = 'marks' AND column_name = 'clip_key'"
                ).fetchone()
                if column is None:
                    self.pending_migration = True

    def connect(self):
        return self.psycopg.connect(self.url, autocommit=True)

    def load(self, who: str, clips: list[dict]) -> dict[int, list]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM marks WHERE annotator = %s", (who,)
            ).fetchall()
        if not rows:
            return {}
        return index_rows([r[0] for r in rows], clips)

    def claims(self) -> dict[str, tuple[str, float]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT clip_key, annotator, EXTRACT(EPOCH FROM claimed_at) FROM claims"
            ).fetchall()
        return {r[0]: (r[1], float(r[2])) for r in rows}

    def claim(self, who: str, keys: list[str]) -> None:
        """Take these clips for `who`, skipping any another annotator still holds.

        ON CONFLICT is what makes this safe: two people refilling at the same instant race
        for the same row, and the loser takes none of it rather than both walking away
        believing they own the clip. A claim is only stealable once it has gone stale
        without being marked.
        """
        if not keys:
            return
        cutoff = time.time() - STALE_SECONDS
        with self.connect() as conn:
            for key in keys:
                conn.execute(
                    "INSERT INTO claims (clip_key, annotator) VALUES (%s, %s)"
                    " ON CONFLICT (clip_key) DO UPDATE"
                    "   SET annotator = EXCLUDED.annotator, claimed_at = now(), done = false"
                    " WHERE claims.annotator = %s"
                    "    OR (claims.done = false AND claims.claimed_at < to_timestamp(%s))",
                    (key, who, who, cutoff),
                )

    def finish(self, who: str, key: str) -> None:
        """Mark a claim done, so it is never handed to anyone else."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE claims SET done = true WHERE clip_key = %s AND annotator = %s",
                (key, who),
            )

    def binding(self, sub: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT name FROM identities WHERE sub = %s", (sub,)
            ).fetchone()
        return row[0] if row else None

    def bind(self, sub: str, name: str, email: str, display: str) -> str | None:
        """Tie a Google account to an annotator name. Returns the bound name, or None when
        the name already belongs to somebody else.

        The existing marks are filed under short names chosen before there was any sign-in,
        so the first login has to be able to claim one -- otherwise turning auth on orphans
        everybody's work.
        """
        with self.connect() as conn:
            held = conn.execute(
                "SELECT sub FROM identities WHERE name = %s", (name,)
            ).fetchone()
            if held and held[0] != sub:
                return None
            conn.execute(
                "INSERT INTO identities (sub, name, email, display) VALUES (%s,%s,%s,%s)"
                " ON CONFLICT (sub) DO UPDATE SET name = EXCLUDED.name,"
                "   email = EXCLUDED.email, display = EXCLUDED.display",
                (sub, name, email, display),
            )
        return name

    def migrate(self) -> None:
        """Split the old one-blob-per-annotator marks into one row per clip.

        Runs only after a user confirms on the page; the table is left untouched until then.
        clip_key is rebuilt from each stored record exactly as it was computed when that
        record was saved (path/start under the legacy per-source dataset), so the new rows
        line up with the claims table made under that same scheme.
        """
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS marks_new ("
                "  annotator  TEXT NOT NULL,"
                "  clip_key   TEXT NOT NULL,"
                "  payload    JSONB NOT NULL,"
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                "  PRIMARY KEY (annotator, clip_key))"
            )
            old = conn.execute(
                "SELECT annotator, payload, updated_at FROM marks ORDER BY annotator"
            ).fetchall()
            for annotator, payload, updated_at in old:
                rows = payload if isinstance(payload, list) else [payload]
                for row in rows:
                    if "id" in row:
                        key = row["id"]
                    else:
                        name = str(row["path"]).replace("\\", "/").rsplit("/", 1)[-1]
                        key = f"{name}@{round(float(row['start']), 3)}"
                    conn.execute(
                        "INSERT INTO marks_new (annotator, clip_key, payload, updated_at)"
                        " VALUES (%s, %s, %s, %s)",
                        (
                            annotator,
                            key,
                            json.dumps(row, ensure_ascii=False),
                            updated_at,
                        ),
                    )
            conn.execute("DROP TABLE marks")
            conn.execute("ALTER TABLE marks_new RENAME TO marks")
        self.pending_migration = False

    def everything(self) -> list[tuple[str, list]]:
        """Every annotator's marks, for export."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT annotator, jsonb_agg(payload ORDER BY clip_key)"
                "  FROM marks GROUP BY annotator"
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def progress(self) -> list[dict]:
        """Who has marked what. With the tool open to more than one person there is
        otherwise no way to see that anyone has been working, or who."""
        with self.connect() as conn:
            marks = conn.execute(
                "SELECT annotator, count(*),"
                "       EXTRACT(EPOCH FROM max(updated_at))"
                "  FROM marks GROUP BY annotator"
            ).fetchall()
            held = conn.execute(
                "SELECT annotator, count(*) FROM claims WHERE done = false GROUP BY annotator"
            ).fetchall()
        holding = {r[0]: int(r[1]) for r in held}
        rows = [
            {
                "name": r[0],
                "marked": int(r[1]),
                "holding": holding.get(r[0], 0),
                "last": float(r[2]),
            }
            for r in marks
        ]
        for name, count in holding.items():
            if not any(r["name"] == name for r in rows):
                rows.append({"name": name, "marked": 0, "holding": count, "last": None})
        return sorted(rows, key=lambda r: r["last"] or 0, reverse=True)

    def save(self, who: str, clips: list[dict], saved: dict[int, list]) -> None:
        """Upsert one row per saved clip, so a save touches only what changed."""
        with self.connect() as conn:
            for index in sorted(saved):
                conn.execute(
                    "INSERT INTO marks (annotator, clip_key, payload, updated_at)"
                    " VALUES (%s, %s, %s, now())"
                    " ON CONFLICT (annotator, clip_key) DO UPDATE"
                    "   SET payload = EXCLUDED.payload, updated_at = now()",
                    (
                        who,
                        clip_key(clips[index]),
                        json.dumps(
                            gold_row(clips[index], saved[index]), ensure_ascii=False
                        ),
                    ),
                )


class FileClaims:
    """The same claim bookkeeping as PostgresStore, in a json file next to the marks.

    Only used when there is no database, i.e. running locally. One process, so the module
    LOCK is all the mutual exclusion needed.
    """

    def __init__(self, directory: Path):
        self.path = directory / "_claims.json"

    def read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def claims(self) -> dict[str, tuple[str, float]]:
        return {k: (v["annotator"], v["claimed_at"]) for k, v in self.read().items()}

    def claim(self, who: str, keys: list[str]) -> None:
        if not keys:
            return
        cutoff = time.time() - STALE_SECONDS
        rows = self.read()
        for key in keys:
            held = rows.get(key)
            if held and held["annotator"] != who:
                if held.get("done") or held["claimed_at"] >= cutoff:
                    continue
            rows[key] = {"annotator": who, "claimed_at": time.time(), "done": False}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def finish(self, who: str, key: str) -> None:
        rows = self.read()
        if key in rows and rows[key]["annotator"] == who:
            rows[key]["done"] = True
            self.path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def clip_key(clip: dict) -> str:
    """Stable id for a clip: the dataset's own id, generated once at conversion time.

    Simpler than it used to be -- the legacy manifests carried no id of their own, so a key
    was built from the clip's file name and offset. A dataset row has an id already.
    """
    return clip["id"]


def assign(claimer, who: str, clips: list[dict], marked: set[int]) -> set[int]:
    """Indices this annotator owns, topping their block up from the unclaimed pool.

    Everyone works the same worst-disagreement-first order, so the pool hands out the most
    valuable unclaimed clip next regardless of who asks.
    """
    held = claimer.claims()
    cutoff = time.time() - STALE_SECONDS
    # Work you have already marked is yours whatever the claim table says. Claims can be
    # absent for marks that arrived another way -- imported from a local session, say --
    # and dropping them from your list would look like the work had been lost.
    mine, free = set(marked), []
    for index, clip in enumerate(clips):
        key = clip_key(clip)
        owner = held.get(key)
        if index in marked:
            continue  # already yours; never offer it as fresh work
        if owner is None:
            free.append((index, key))
        elif owner[0] == who:
            mine.add(index)
        elif owner[1] < cutoff and index not in marked:
            free.append((index, key))
    outstanding = len(mine - marked)
    if outstanding < BLOCK and free:
        take = free[: BLOCK - outstanding]
        claimer.claim(who, [key for _, key in take])
        mine.update(index for index, _ in take)
    return mine


def gold_row(clip: dict, words: list) -> dict:
    """One output record: the clip's id plus the human words, so a gold file can be scored
    against any dataset export with no path/offset guessing."""
    return {
        "id": clip["id"],
        "text": clip["text"],
        "duration": clip["duration"],
        "words": [
            {
                "word": w["word"],
                "start": round(float(w["start"]), 4),
                "end": round(float(w["end"]), 4),
                # Present only where the annotator corrected the ASR text. Downstream has to
                # be able to separate a corrected clip from one that was right already.
                **({"was": w["was"]} if w.get("was") is not None else {}),
                # A word the transcript never had. A word it had and should not have is
                # simply absent -- the clip keeps its original `text`, so a deletion
                # stays recoverable without a flag of its own.
                **({"added": True} if w.get("added") else {}),
            }
            for w in words
        ],
    }


def index_rows(rows: list[dict], clips: list[dict]) -> dict[int, list]:
    """Saved records -> {clip index: words}, matched on id."""
    by_key = {r["id"]: r["words"] for r in rows if "id" in r}
    out = {}
    for index, clip in enumerate(clips):
        hit = by_key.get(clip["id"])
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
        if "id" in row:
            by_key[row["id"]] = row["words"]
    out = {}
    for index, clip in enumerate(clips):
        hit = by_key.get(clip["id"])
        if hit:
            out[index] = hit
    return out


def make_handler(args, dataset: Dataset, clips, saved):
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

        def signed_in(self):
            """The verified Google account, when auth is on."""
            if not args.auth:
                return None
            host = self.headers.get("Host", "").split(":")[0]
            return identity(self.headers.get("Cookie", ""), host)

        def who(self):
            """The annotator whose marks this request touches.

            With auth on this comes from the verified cookie, so `?who=` is ignored -- it is
            what let anyone write as anyone.
            """
            if args.auth:
                person = self.signed_in()
                if not person or STORE is None:
                    return None
                return STORE.binding(person["sub"])
            from urllib.parse import parse_qs

            name = (parse_qs(urlparse(self.path).query).get("who") or [""])[0]
            return name if _NAME.match(name) else None

        def authorised(self):
            from urllib.parse import parse_qs

            if not args.token:
                return True
            return (parse_qs(urlparse(self.path).query).get("t") or [""])[
                0
            ] == args.token

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
            if route == "/api/me":
                person = self.signed_in()
                body = {
                    "auth": bool(args.auth),
                    "logged_in": bool(person),
                    "migrate": bool(STORE is not None and STORE.pending_migration),
                    "login_url": "/xhost-auth/login?return_to=/",
                    "logout_url": "/xhost-auth/logout?return_to=/",
                }
                if person:
                    body.update(
                        {
                            "display": person["display"],
                            "email": person["email"],
                            "name": STORE.binding(person["sub"]) if STORE else None,
                        }
                    )
                return self.send(
                    200,
                    json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            if STORE is not None and STORE.pending_migration:
                # The old marks schema is not writable. Nothing is lost by refusing:
                # the page shows the migration gate, and until a user confirms there is
                # no safe way to write a fix without overwriting another annotator's blob.
                return self.send(
                    503, b'{"error":"database migration required"}', "application/json"
                )
            if args.auth and route.startswith("/api/") and route != "/api/progress":
                if not self.signed_in():
                    return self.send(401, b'{"error":"sign in"}', "application/json")
                if not self.who():
                    return self.send(
                        409, b'{"error":"pick a name"}', "application/json"
                    )
            if route == "/api/meta":
                meta = {
                    "multi": bool(args.multi),
                    "clips": len(clips),
                    "dataset": dataset.name,
                }
                return self.send(
                    200, json.dumps(meta).encode("utf-8"), "application/json"
                )
            if route == "/api/export":
                # The marks live in Postgres once hosted, but the rest of the pipeline reads
                # jsonl keyed on clip id. So export in exactly that shape, with the
                # annotator added, and nothing else: a file that needs converting before it
                # can be scored is a file that will be scored wrong.
                lines = []
                if STORE is not None:
                    for name, payload in STORE.everything():
                        for row in payload:
                            lines.append(
                                json.dumps(
                                    {**row, "annotator": name}, ensure_ascii=False
                                )
                            )
                else:
                    directory = args.out if args.multi else args.out.parent
                    for f in sorted(directory.glob("*.jsonl")):
                        for line in f.read_text(encoding="utf-8").splitlines():
                            if line.strip():
                                lines.append(
                                    json.dumps(
                                        {**json.loads(line), "annotator": f.stem},
                                        ensure_ascii=False,
                                    )
                                )
                blob = ("\n".join(lines) + "\n").encode("utf-8")
                return self.send(
                    200,
                    blob,
                    "application/x-ndjson; charset=utf-8",
                    {"Content-Disposition": 'attachment; filename="gold.jsonl"'},
                )
            if route == "/api/progress":
                if STORE is not None:
                    rows = STORE.progress()
                else:
                    rows = []
                    directory = args.out if args.multi else args.out.parent
                    for f in sorted(directory.glob("*.jsonl")):
                        lines = [
                            ln
                            for ln in f.read_text(encoding="utf-8").splitlines()
                            if ln.strip()
                        ]
                        rows.append(
                            {
                                "name": f.stem,
                                "marked": len(lines),
                                "holding": 0,
                                "last": f.stat().st_mtime,
                            }
                        )
                body = {
                    "clips": len(clips),
                    "annotators": rows,
                    "marked_total": sum(r["marked"] for r in rows),
                }
                # Where the audio is actually coming from. Without this there is no way to
                # tell from outside which dataset source is live.
                if isinstance(dataset, BucketDataset):
                    body["audio"] = {
                        "source": "bucket",
                        "bucket": dataset.bucket,
                        "dataset": dataset.name,
                    }
                else:
                    body["audio"] = {"source": "folder", "dataset": dataset.name}
                return self.send(
                    200,
                    json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            if route == "/api/clips":
                who = self.who() or "anon"
                if STORE is not None:
                    mine = STORE.load(who, clips)
                elif args.multi:
                    mine = load_saved(gold_path(args, self.who()), clips)
                else:
                    mine = saved
                # With --split each annotator is handed their own clips, so two people never
                # mark the same sentence. Without it everyone walks the same order, which is
                # what measures how far two humans sit apart.
                owned = None
                if args.split and CLAIMS is not None:
                    with LOCK:
                        owned = assign(CLAIMS, who, clips, set(mine))
                payload = []
                for index, clip in enumerate(clips):
                    if owned is not None and index not in owned:
                        continue
                    payload.append(
                        {
                            "id": clip["id"],
                            "key": clip["id"],
                            "metadata": clip["metadata"],
                            "text": clip["text"],
                            "duration": clip["duration"],
                            "words": clip["a"],
                            "labels": clip["labels"],
                            "saved": mine.get(index),
                        }
                    )
                return self.send(
                    200,
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            if route.startswith("/api/audio/"):
                index = int(route.rsplit("/", 1)[1])
                try:
                    data, lead = clip_wav(clips[index], dataset)
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
            route = urlparse(self.path).path
            if route == "/api/claim-name":
                person = self.signed_in()
                if not person or STORE is None:
                    return self.send(401, b'{"error":"sign in"}', "application/json")
                length = int(self.headers.get("Content-Length", 0))
                wanted = json.loads(self.rfile.read(length).decode("utf-8")).get(
                    "name", ""
                )
                if not _NAME.match(wanted):
                    return self.send(400, b'{"error":"bad name"}', "application/json")
                bound = STORE.bind(
                    person["sub"], wanted, person["email"], person["display"]
                )
                if bound is None:
                    return self.send(409, b'{"error":"taken"}', "application/json")
                return self.send(
                    200, json.dumps({"name": bound}).encode("utf-8"), "application/json"
                )
            if route == "/api/migrate":
                # Confirmed by any user on the page. Until this runs the whole store is
                # read-only, so there is no window where one blob is half-split.
                if STORE is None or not STORE.pending_migration:
                    return self.send(
                        409, b'{"error":"no migration pending"}', "application/json"
                    )
                try:
                    STORE.migrate()
                except Exception as exc:  # noqa: BLE001 -- surface whatever failed
                    return self.send(
                        500,
                        json.dumps({"error": str(exc)}).encode(),
                        "application/json",
                    )
                return self.send(200, b'{"ok":true}', "application/json")
            if STORE is not None and STORE.pending_migration:
                return self.send(
                    503, b'{"error":"database migration required"}', "application/json"
                )
            if args.auth and not self.who():
                return self.send(401, b'{"error":"sign in"}', "application/json")
            if route != "/api/gold":
                return self.send(404, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            # The page numbers clips by their position in what it was served, which under
            # --split is a subset. Resolve by clip key so a save cannot land on someone
            # else's sentence.
            if body.get("key"):
                match = [i for i, c in enumerate(clips) if clip_key(c) == body["key"]]
                if not match:
                    return self.send(400, b"unknown clip", "text/plain")
                body["index"] = match[0]
            who = self.who() or "anon"
            # Re-read before writing. Two annotators sharing a store would otherwise each
            # hold a stale copy and the second save would drop the first one's work.
            with LOCK:
                index = int(body["index"])
                if STORE is not None:
                    mine = STORE.load(who, clips)
                    mine[index] = body["words"]
                    STORE.save(who, clips, mine)
                else:
                    path = gold_path(args, self.who())
                    mine = load_saved(path, clips)
                    mine[index] = body["words"]
                    write_gold(path, clips, mine)
                if CLAIMS is not None:
                    CLAIMS.finish(who, clip_key(clips[index]))
            return self.send(200, b'{"ok":true}', "application/json")

    return Handler


def write_gold(out: Path, clips: list[dict], saved: dict[int, list]) -> None:
    """Rewritten in full after each clip: 100 rows is nothing, and a partial append that
    crashed mid-write would be worse than the rewrite cost.

    Shares gold_row with the Postgres store so the two writers cannot drift -- they had
    separate copies of the record shape, and only one of them learned about corrected text.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for index in sorted(saved):
            row = gold_row(clips[index], saved[index])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    global STORE, CLAIMS
    args = parse_args()
    if not args.datasets_bucket and not args.datasets_folder:
        raise SystemExit("one of --datasets-folder or --datasets-bucket is required")

    database = os.environ.get("DATABASE_URL", "")
    if database:
        STORE = PostgresStore(database)
        print("marks -> Postgres (DATABASE_URL)")
    if args.auth and STORE is None:
        raise SystemExit(
            "--auth needs DATABASE_URL: the name bindings live in the database"
        )
    if args.auth:
        print("google sign-in required; ?who= ignored")
    if args.split:
        CLAIMS = STORE if STORE is not None else FileClaims(args.out)
        print(f"split mode: each annotator gets their own clips, {BLOCK} at a time")

    dataset = resolve_dataset(args)
    meta = dataset.metadata()
    kind = "bucket" if isinstance(dataset, BucketDataset) else "folder"
    print(f"dataset: {meta.get('name', dataset.name)} ({kind})")

    clips = build_clips(dataset.manifest(), args.limit)
    if not clips:
        raise SystemExit(f"no usable clips in dataset {args.dataset!r}")

    # In --multi each annotator has their own file, loaded per request from their name;
    # there is no single shared state to resume into here.
    saved: dict[int, list] = {} if args.multi else load_saved(args.out, clips)
    if saved:
        print(f"resuming: {len(saved)} clips already marked")

    if isinstance(dataset, LocalDataset):
        missing = [c for c in clips if dataset.audio_exists(c["audio"]) is False]
        if missing:
            first = dataset.dir / missing[0]["audio"]
            raise SystemExit(
                f"{len(missing)} of {len(clips)} clip audio files are not where the "
                f"manifest says they are; first missing: {first}"
            )

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args, dataset, clips, saved)
    )
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
