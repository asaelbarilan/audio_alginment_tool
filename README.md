# Hebrew alignment tagging tool

A browser tool for hand-marking Hebrew word boundaries, used to build a gold set that says
which forced aligner to trust. Deployed on xhostd; this repo is the source of truth.

- `hebrew_training/align_tag_server.py` — stdlib HTTP server: serves clips, stores marks
- `hebrew_training/align_tag_page.html` — the marking UI
- `data/gold_set/*.jsonl` — the aligner proposals each clip starts from

The audio is **not** in this repo. It lives in the channel's S3-compatible bucket; the
container copies any missing clip up from `--clips-root` at startup, which is how the
original 35 MB of wav got out of the image.

Marks go to Postgres (`DATABASE_URL`), one row per annotator. `GET /api/export` writes them
back out as jsonl in the same shape as the input manifests, plus an `annotator` field.

Sign-in is Google via xhostd, verified server-side from the signed cookie.
