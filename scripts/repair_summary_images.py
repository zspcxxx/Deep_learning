#!/usr/bin/env python3
"""Ensure each lesson entry in artifacts/summary.json has metrics.images."""
from __future__ import annotations

import json
from pathlib import Path

from dllab.paths import ARTIFACTS_DIR


def main() -> None:
    path = ARTIFACTS_DIR / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    for entry in summary.get("lessons", []):
        metrics = entry.setdefault("metrics", {})
        top_images = entry.pop("images", None)
        if top_images and not metrics.get("images"):
            metrics["images"] = top_images
        if not metrics.get("images"):
            name = entry.get("name", "")
            candidates = [ARTIFACTS_DIR / name / "metrics.json"]
            if name.startswith("03"):
                candidates.append(ARTIFACTS_DIR / "03_backprop" / "metrics.json")
            for mp in candidates:
                if not mp.exists():
                    continue
                data = json.loads(mp.read_text(encoding="utf-8"))
                if data.get("images"):
                    metrics["images"] = data["images"]
                    break
        n = len(metrics.get("images") or [])
        print(f"{entry.get('name')}: {n} images")
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
