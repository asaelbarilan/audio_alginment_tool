#!/bin/sh
set -eu
# Not uvicorn: this is a stdlib ThreadingHTTPServer, which is plenty for a handful of
# annotators and keeps the dependencies small.
#
# Datasets live in the channel's object store, not in the image: baking them in meant
# every deploy rebuilt and shipped 35 MB of audio. DATASETS_BUCKET picks that up when the
# host provides one; --datasets-folder is the fallback for a host without one.
#
# Marks go to Postgres via DATABASE_URL, so --out is only the fallback for a host
# without a database.
# install.sh's uv sync built .venv on CPython 3.11; run from it so the pinned
# interpreter and dependencies win, not whatever system python is default.
BUCKET="${DATASETS_BUCKET:-${S3_BUCKET:-}}"
exec .venv/bin/python -m hebrew_training.align_tag_server \
  ${BUCKET:+--datasets-bucket "$BUCKET"} \
  --datasets-folder data/datasets \
  --dataset "${DATASET:-plenum}" \
  --out /tmp/marks --multi --split --limit 134 \
  ${TAG_AUTH:+--auth "$TAG_AUTH"} \
  --host 0.0.0.0 --port "$XHOST_HTTP_PORT"
