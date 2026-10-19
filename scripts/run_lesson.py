#!/usr/bin/env python3
"""单独运行某一个教学实验。"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

from dllab.catalog import LESSONS, find_lesson
from dllab.io_utils import dump_metrics
from dllab.paths import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(
        description="单独运行 Deep Learning Lab 中的一个实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python scripts/run_lesson.py --list\n  python scripts/run_lesson.py 01\n  python scripts/run_lesson.py 09",
    )
    parser.add_argument("lesson", nargs="?", help="实验编号或名称，如 01 / lr / 09")
    parser.add_argument("--list", action="store_true", help="列出全部实验")
    parser.add_argument("--result-json", type=Path, help="把 run() 返回的指标写到该 JSON 文件（供看板读取）")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))

    if args.list or not args.lesson:
        print("可单独运行的实验：\n")
        for item in LESSONS:
            print(f"  {item['id']:<12s}  {item['title']}")
            print(f"               {item['help']}")
            print(f"               python scripts/run_lesson.py {item['id']}\n")
        if not args.lesson:
            return 0

    try:
        item = find_lesson(args.lesson)
    except KeyError as exc:
        print(exc)
        return 2

    print(f"\n=== {item['title']} ===")
    print(f"原理介绍: docs/{item['doc']}")
    print("单独运行: python scripts/run_lesson.py " + item["id"])
    print()
    mod = import_module(item["module"])
    metrics = mod.run()
    if args.result_json:
        dump_metrics(args.result_json, metrics)
    passed = bool(metrics.get("passed", True))
    print("\n" + ("通过" if passed else "未通过"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
