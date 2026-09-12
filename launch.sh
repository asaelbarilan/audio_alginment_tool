#!/bin/sh
set -eu
# Not uvicorn: this is a stdlib ThreadingHTTPServer, which is plenty for a handful of
# annotators and keeps the dependencies small.
#
# Clips live in the channel's object store, not in the image: baking them in meant every
# deploy rebuilt and shipped 35 MB of audio. --clips-root stays as the source for the
# one-time copy up to the bucket, and as the fallback when there is no object store.
#
# Marks go to Postgres via DATABASE_URL, so --out is only the fallback for a host
# without a database.
# install.sh's uv sync built .venv on CPython 3.11; run from it so the pinned
# interpreter and dependencies win, not whatever system python is default.
exec .venv/bin/python -m hebrew_training.align_tag_server \
  --a data/gold_set/A_hebrew.jsonl \
  --b data/gold_set/B_mms.jsonl \
  --clips-s3 clips/ \
  --clips-root data/gold_set/clips \
  --out /tmp/marks --multi --split --limit 134 \
  ${TAG_AUTH:+--auth "$TAG_AUTH"} \
  --host 0.0.0.0 --port "$XHOST_HTTP_PORT"
