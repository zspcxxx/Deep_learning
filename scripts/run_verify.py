#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import import_module

from dllab.catalog import LESSONS as CATALOG
from dllab.io_utils import dump_metrics
from dllab.paths import ARTIFACTS_DIR, PROJECT_ROOT, ensure_artifacts

LESSONS = [(item["name"], item["module"]) for item in CATALOG]


def run_all() -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    ensure_artifacts()
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "lessons": [],
        "passed": True,
    }
    for name, module_name in LESSONS:
        print("\n" + "#" * 80)
        print(f"# {name}")
        print("#" * 80)
        t0 = time.time()
        entry = {"name": name, "status": "running"}
        try:
            mod = import_module(module_name)
            metrics = mod.run()
            elapsed = time.time() - t0
            passed = bool(metrics.get("passed", True))
            entry.update({
                "status": "passed" if passed else "failed",
                "elapsed_sec": round(elapsed, 2),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {k: v for k, v in metrics.items() if k not in {"history_mlp", "history_cnn", "table"}},
            })
            if not passed:
                summary["passed"] = False
            print(f"[OK] {name} in {elapsed:.1f}s  passed={passed}")
        except Exception as exc:
            elapsed = time.time() - t0
            entry.update({
                "status": "error",
                "elapsed_sec": round(elapsed, 2),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            summary["passed"] = False
            print(f"[ERROR] {name}: {exc}")
        summary["lessons"].append(entry)
        dump_metrics(ARTIFACTS_DIR / "summary.json", summary)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    dump_metrics(ARTIFACTS_DIR / "summary.json", summary)
    (ARTIFACTS_DIR / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "updated_at"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all deep-learning lab verifications")
    parser.parse_args()
    result = run_all()
    sys.exit(0 if result["passed"] else 1)
