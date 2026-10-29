#!/usr/bin/env bash
# 端到端冒烟（复用 sibling news-commentary，1 epoch 小模型）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

PY="${PYTHON:-python3}"
if [[ -x /nproject/project/conda_env/bilingual_mt/bin/python ]]; then
  PY=/nproject/project/conda_env/bilingual_mt/bin/python
fi

$PY -m mt.cli prepare -c configs/smoke.yaml
$PY -m mt.cli train -c configs/smoke.yaml --no-resume
$PY -m mt.cli eval smoke --split val --beam 1 --max-examples 64
$PY -m mt.cli translate smoke -d en2zh -t "Hello, how are you?"
$PY -m mt.cli translate smoke -d zh2en -t "你好吗？"
echo "smoke OK"
