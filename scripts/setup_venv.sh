#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
