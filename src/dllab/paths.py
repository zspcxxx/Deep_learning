from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"


def ensure_artifacts(*parts: str) -> Path:
    path = ARTIFACTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_data(*parts: str) -> Path:
    path = DATA_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
