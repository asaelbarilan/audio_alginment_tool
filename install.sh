#!/bin/sh
set -eu
# soundfile is a binding, not an implementation: without libsndfile it imports fine and then
# fails on the first clip.
apt-get update && apt-get install -y --no-install-recommends libsndfile1
# .python-version picks CPython 3.11; --locked fails if pyproject.toml drifts from uv.lock.
export UV_PYTHON_INSTALL_DIR=/opt/uv/python
uv sync --locked --no-cache
chmod -R a+rX /opt/uv
chmod 755 /root
