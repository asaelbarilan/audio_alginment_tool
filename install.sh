#!/bin/sh
set -eu
# soundfile is a binding, not an implementation: without libsndfile it imports fine and then
# fails on the first clip.
apt-get update && apt-get install -y --no-install-recommends libsndfile1
# .python-version picks CPython 3.11; --locked fails if pyproject.toml drifts from uv.lock.
uv sync --locked --no-cache
