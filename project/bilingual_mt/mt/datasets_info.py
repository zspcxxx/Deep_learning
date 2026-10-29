"""List / download datasets for the web UI."""
from __future__ import annotations

from pathlib import Path

from .config import DATA_DIR, LEGACY_DATA_DIR
from .data import BUILTINS, download_builtin


def _count_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            n += 1
    return n


def _resolve_tsv(name: str) -> Path | None:
    for root in (DATA_DIR, LEGACY_DATA_DIR):
        cand = root / f"{name}.tsv"
        if cand.is_file():
            return cand
    return None


def list_datasets() -> list[dict]:
    """Return dataset cards for the config tab."""
    names = set(BUILTINS)
    for root in (DATA_DIR, LEGACY_DATA_DIR):
        if not root.is_dir():
            continue
        for p in root.glob("*.tsv"):
            names.add(p.stem)

    out = []
    for name in sorted(names):
        path = _resolve_tsv(name)
        ready = path is not None
        meta = BUILTINS.get(name, {})
        out.append(
            {
                "name": name,
                "label": meta.get("label", name),
                "ready": ready,
                "downloadable": name in BUILTINS,
                "pairs": _count_lines(path) if ready else 0,
                "path": str(path) if ready else None,
            }
        )
    return out


def ensure_dataset(name: str, log=None) -> Path:
    path = _resolve_tsv(name)
    if path is not None:
        return path
    if name in BUILTINS:
        if log:
            log(f"下载数据集 {name}…")
        return download_builtin(name)
    raise FileNotFoundError(f"找不到数据集 {DATA_DIR}/{name}.tsv")
