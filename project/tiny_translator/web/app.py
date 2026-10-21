#!/usr/bin/env python3
"""Run the translator web UI on its own. The Deep_learning dashboard also mounts it at /translator/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.webui import bp  # noqa: E402

app = Flask(__name__)
app.json.ensure_ascii = False
app.register_blueprint(bp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
