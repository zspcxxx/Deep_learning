#!/usr/bin/env bash
# Create .venv and install PyTorch + Flask. Picks the CUDA 13 wheels on Jetson Thor (aarch64, CUDA 13),
# the default CUDA wheels on other NVIDIA machines, and the CPU wheels otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip

if [ "$(uname -m)" = "aarch64" ] && [ -d /usr/local/cuda-13 ]; then
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
elif command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/pip install torch
else
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/pip install "flask>=3.0" numpy zhconv

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
