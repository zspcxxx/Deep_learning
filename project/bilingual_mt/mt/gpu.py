"""Read current GPU utilization for the web UI (Jetson sysfs, then nvidia-smi)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Jetson Orin / Thor expose load as 0–1000 (tenths of a percent).
_JETSON_LOAD_PATHS = (
    Path("/sys/devices/platform/gpu.0/load"),
    Path("/sys/devices/gpu.0/load"),
)


def _jetson_util() -> int | None:
    for path in _JETSON_LOAD_PATHS:
        try:
            raw = path.read_text(encoding="utf-8").strip()
            digits = "".join(ch for ch in raw if ch.isdigit())
            if not digits:
                continue
            return max(0, min(100, round(int(digits) / 10)))
        except OSError:
            continue
    return None


def _nvidia_smi() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    # One GPU is enough for this lab; take the first line.
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 4:
        return None

    def _num(text: str) -> int | None:
        try:
            return int(float(text))
        except ValueError:
            return None

    util = _num(parts[0])
    return {
        "util": max(0, min(100, util)) if util is not None else None,
        "mem_used_mb": _num(parts[1]),
        "mem_total_mb": _num(parts[2]),
        "name": parts[3] or None,
        "source": "nvidia-smi",
    }


def read_gpu() -> dict:
    """Return a JSON-friendly snapshot. util is 0–100, or null if unknown."""
    smi = _nvidia_smi()
    jetson = _jetson_util()
    if smi is not None:
        # On Jetson, nvidia-smi util is often stuck at 0; prefer the sysfs load counter.
        if jetson is not None:
            return {**smi, "util": jetson, "source": "jetson+nvidia-smi"}
        return smi
    if jetson is not None:
        return {"util": jetson, "mem_used_mb": None, "mem_total_mb": None, "name": "Jetson GPU", "source": "jetson"}
    return {"util": None, "mem_used_mb": None, "mem_total_mb": None, "name": None, "source": None}
