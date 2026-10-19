"""Run lessons as subprocesses and record an annotated event stream for the web UI.

Event types (each has a monotonically increasing "i"):
  log   {text, level}               one stdout/stderr line of the lesson
  note  {title, body, level, line}  annotation triggered by log line `line`
  step  {index}                     lesson step `index` reached
  done  {status, returncode, elapsed_sec}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dllab.catalog import LESSONS, find_lesson
from dllab.io_utils import dump_metrics
from dllab.notes import Annotator, explain_result, get_notes, image_note
from dllab.paths import ARTIFACTS_DIR, PROJECT_ROOT

RUNS_DIR = ARTIFACTS_DIR / "runs"
SUMMARY_PATH = ARTIFACTS_DIR / "summary.json"


class BusyError(RuntimeError):
    def __init__(self, active: "Run"):
        super().__init__(f"{active.lesson['id']} is running")
        self.active = active


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _line_level(text: str) -> str:
    if text.startswith("Traceback") or "Error" in text.split(":")[0]:
        return "error"
    if "✗" in text or "未通过" in text:
        return "error"
    if "✓" in text or text.strip() in {"通过"} or "[OK]" in text:
        return "ok"
    if "Warning" in text or "Early stop" in text:
        return "warn"
    return "info"


class Run:
    def __init__(self, lesson: dict):
        self.id = uuid.uuid4().hex[:12]
        self.lesson = lesson
        self.status = "running"
        self.started_at = _now()
        self.finished_at: str | None = None
        self.elapsed_sec: float | None = None
        self.returncode: int | None = None
        self.events: list[dict] = []
        self.result: dict | None = None
        self.findings: list[dict] = []
        self.images: list[dict] = []
        self.cond = threading.Condition()
        self.proc: subprocess.Popen | None = None
        self.cancel_requested = False

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def emit(self, type_: str, **payload) -> int:
        with self.cond:
            i = len(self.events)
            self.events.append({"i": i, "type": type_, **payload})
            self.cond.notify_all()
            return i

    def to_dict(self, include_events: bool = True) -> dict:
        d = {
            "id": self.id,
            "lesson_id": self.lesson["id"],
            "name": self.lesson["name"],
            "title": self.lesson["title"],
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_sec": self.elapsed_sec,
            "returncode": self.returncode,
            "result": self.result,
            "findings": self.findings,
            "images": self.images,
        }
        if include_events:
            d["events"] = list(self.events)
        return d


class RunManager:
    """One lesson at a time: the lessons are CPU-bound and would only slow each other down."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active: Run | None = None
        self._runs: dict[str, Run] = {}
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> Run | None:
        run = self._active
        return run if run and not run.finished else None

    def start(self, lesson_id: str) -> Run:
        lesson = find_lesson(lesson_id)
        with self._lock:
            if self.active:
                raise BusyError(self.active)
            run = Run(lesson)
            self._runs[run.id] = run
            self._active = run
        threading.Thread(target=self._work, args=(run,), daemon=True).start()
        return run

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if not run or run.finished or not run.proc:
            return False
        run.cancel_requested = True
        run.proc.terminate()
        return True

    def latest(self, lesson_id: str) -> dict | None:
        """The running run, else the last completed (persisted) run, else e.g. a cancelled one."""
        lesson = find_lesson(lesson_id)
        mine = [r for r in self._runs.values() if r.lesson["id"] == lesson["id"]]
        if mine and not mine[-1].finished:
            return mine[-1].to_dict()
        path = RUNS_DIR / f"{lesson['name']}.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("result"):
                # Re-derive from the stored metrics so edits to dllab/notes show up without re-running.
                rec["findings"] = explain_result(lesson["id"], rec["result"])
                rec["images"] = [{"path": p, "note": image_note(lesson["id"], p)} for p in rec["result"].get("images", [])]
            return rec
        return mine[-1].to_dict() if mine else None

    def _work(self, run: Run) -> None:
        lesson = run.lesson
        notes = get_notes(lesson["id"])
        annotator = Annotator(notes)
        cmd = [sys.executable, "-u", "scripts/run_lesson.py", lesson["id"]]
        fd, result_path = tempfile.mkstemp(prefix=f"{lesson['name']}_", suffix=".json")
        os.close(fd)
        cmd += ["--result-json", result_path]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
        src = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        run.emit("log", text=f"$ python scripts/run_lesson.py {lesson['id']}", level="cmd")
        t0 = time.time()
        try:
            run.proc = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            for raw in run.proc.stdout:
                text = raw.rstrip("\n")
                line_i = run.emit("log", text=text, level=_line_level(text))
                for note in annotator.feed(text):
                    if note["step"] is not None:
                        run.emit("step", index=note["step"])
                    if note["body"]:
                        run.emit("note", title=note["title"], body=note["body"], level=note["level"], line=line_i)
            run.returncode = run.proc.wait()
        except Exception as exc:
            run.emit("log", text=f"runner error: {exc}", level="error")
            run.returncode = -1

        run.elapsed_sec = round(time.time() - t0, 2)
        try:
            text = Path(result_path).read_text(encoding="utf-8")
            run.result = json.loads(text) if text.strip() else None
        except (OSError, json.JSONDecodeError):
            run.result = None
        finally:
            Path(result_path).unlink(missing_ok=True)

        if run.cancel_requested:
            status = "cancelled"
        elif run.returncode == 0:
            status = "passed"
        elif run.result is not None:
            status = "failed"
        else:
            status = "error"
        if run.result is not None:
            run.findings = explain_result(lesson["id"], run.result)
            run.images = [{"path": p, "note": image_note(lesson["id"], p)} for p in run.result.get("images", [])]
            if len(notes.steps):
                run.emit("step", index=len(notes.steps))
        run.finished_at = _now()

        if status != "cancelled":
            record = {**run.to_dict(), "status": status}
            (RUNS_DIR / f"{lesson['name']}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        if status in ("passed", "failed"):
            update_summary(lesson["name"], status, run.elapsed_sec, run.result)
        # Atomic with the status flip, so a stream that sees `finished` has always received "done".
        with run.cond:
            run.status = status
            run.emit("done", status=status, returncode=run.returncode, elapsed_sec=run.elapsed_sec)


def update_summary(name: str, status: str, elapsed: float, metrics: dict) -> None:
    """Insert / replace one lesson entry in artifacts/summary.json, keeping catalog order."""
    summary = {"lessons": []}
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    entries = {e["name"]: e for e in summary.get("lessons", [])}
    entries[name] = {
        "name": name,
        "status": status,
        "elapsed_sec": elapsed,
        "metrics": {k: v for k, v in metrics.items() if k not in {"table", "updated_at"}},
        "source": "web",
        "finished_at": _now(),
    }
    order = [item["name"] for item in LESSONS]
    lessons = [entries[n] for n in order if n in entries]
    summary.update({
        "lessons": lessons,
        "passed": all(e["status"] == "passed" for e in lessons) and len(lessons) == len(order),
        "finished_at": _now(),
    })
    summary.setdefault("started_at", summary["finished_at"])
    dump_metrics(SUMMARY_PATH, summary)
