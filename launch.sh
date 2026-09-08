#!/bin/sh
set -eu
# Not uvicorn: this is a stdlib ThreadingHTTPServer, which is plenty for a handful of
# annotators and keeps the dependency list to soundfile and numpy.
#
# --clips-root is what makes the manifests portable -- the paths inside them are the Windows
# paths they were built with. Marks go to Postgres via DATABASE_URL, so --out is only the
# fallback for a host without a database.
exec python -m hebrew_training.align_tag_server \
  --a data/gold_set/A_hebrew.jsonl \
  --b data/gold_set/B_mms.jsonl \
  --clips-root data/gold_set/clips \
  --out /tmp/marks --multi --limit 134 \
  --host 0.0.0.0 --port "$XHOST_HTTP_PORT"
