"""List / download datasets for the web UI."""
from __future__ import annotations

from pathlib import Path

from .config import DATA_DIR, SIBLING_DATA
from .data import BUILTINS, download_builtin


def _count_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            n += 1
    return n


def list_datasets() -> list[dict]:
    """Return dataset cards for the config tab."""
    names = set(BUILTINS)
    for root in (DATA_DIR, SIBLING_DATA):
        if not root.is_dir():
            continue
        for p in root.glob("*.tsv"):
            names.add(p.stem)

    out = []
    for name in sorted(names):
        path = DATA_DIR / f"{name}.tsv"
        if not path.is_file():
            sib = SIBLING_DATA / f"{name}.tsv"
            if sib.is_file():
                path = sib
        ready = path.is_file()
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
    path = DATA_DIR / f"{name}.tsv"
    if path.is_file():
        return path
    if (SIBLING_DATA / f"{name}.tsv").is_file():
        return download_builtin(name)  # will symlink
    if name in BUILTINS:
        if log:
            log(f"下载数据集 {name}…")
        return download_builtin(name)
    raise FileNotFoundError(f"找不到数据集 data/{name}.tsv")
