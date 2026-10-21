#!/usr/bin/env bash
# Start the web UI (default http://0.0.0.0:8090). Extra arguments go to web/app.py, e.g. --port 9000.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python web/app.py "$@"
