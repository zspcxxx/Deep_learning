#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

from dllab.io_utils import dump_metrics
from dllab.paths import ARTIFACTS_DIR, PROJECT_ROOT, ensure_artifacts

LESSONS = [
    ("01_perceptron", "lessons.01_perceptron"),
    ("02_mlp", "lessons.02_mlp"),
    ("03_backprop_visual", "lessons.03_backprop_visual"),
    ("03_backpropagation", "lessons.03_backpropagation"),
    ("04_activations", "lessons.04_activations"),
    ("05_learning_rate", "lessons.05_learning_rate"),
    ("06_weight_init", "lessons.06_weight_init"),
    ("07_normalization", "lessons.07_normalization"),
    ("08_regularization", "lessons.08_regularization"),
    ("09_mnist", "lessons.09_mnist"),
]


def run_all(mnist_epochs: int = 10) -> dict:
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
            if name == "09_mnist":
                metrics = mod.run(epochs=mnist_epochs)
            else:
                metrics = mod.run()
            elapsed = time.time() - t0
            passed = bool(metrics.get("passed", True))
            entry.update({
                "status": "passed" if passed else "failed",
                "elapsed_sec": round(elapsed, 2),
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
    parser.add_argument("--mnist-epochs", type=int, default=3)
    args = parser.parse_args()
    result = run_all(mnist_epochs=args.mnist_epochs)
    sys.exit(0 if result["passed"] else 1)
