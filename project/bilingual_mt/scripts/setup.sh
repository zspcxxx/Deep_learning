#!/usr/bin/env bash
# Install deps into .venv (if python3-venv available) or conda env bilingual_mt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

install_pip_deps() {
  local py="$1"
  "$py" -m pip install -U pip wheel
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "检测到 NVIDIA GPU → 安装 CUDA 版 torch（若失败可改用 CPU）"
    "$py" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || \
      "$py" -m pip install torch
  else
    echo "未检测到 GPU → 安装 CPU 版 torch"
    "$py" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || \
      "$py" -m pip install torch
  fi
  "$py" -m pip install -r requirements.txt
}

if command -v conda >/dev/null 2>&1; then
  ENV_DIR="${CONDA_ENV_DIR:-/nproject/project/conda_env/bilingual_mt}"
  if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    conda create -y -p "$ENV_DIR" python=3.10 pip
  fi
  install_pip_deps "$ENV_DIR/bin/python"
  echo "完成。用法："
  echo "  export PATH=$ENV_DIR/bin:\$PATH"
  echo "  cd $ROOT && PYTHONPATH=. python -m mt.cli --help"
  exit 0
fi

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -m venv --help >/dev/null 2>&1; then
  echo "需要 python3-venv 或 conda。例如: sudo apt install python3.10-venv"
  exit 1
fi
if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
install_pip_deps python
echo "完成。用法："
echo "  source .venv/bin/activate"
echo "  PYTHONPATH=. python -m mt.cli --help"
