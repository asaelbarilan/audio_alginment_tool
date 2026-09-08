#!/bin/sh
set -eu
# soundfile is a binding, not an implementation: without libsndfile it imports fine and then
# fails on the first clip.
apt-get update && apt-get install -y --no-install-recommends libsndfile1
uv pip install --system --no-cache -r requirements.txt
