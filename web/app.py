#!/usr/bin/env python3
"""Lesson runner dashboard: run each lesson from the browser and read annotated logs and results."""
from __future__ import annotations

import json

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

from dllab.catalog import LESSONS, find_lesson
from dllab.notes import get_notes
from dllab.paths import ARTIFACTS_DIR, PROJECT_ROOT
from dllab.runner import RUNS_DIR, SUMMARY_PATH, BusyError, RunManager

app = Flask(__name__)
app.json.ensure_ascii = False
DOCS = PROJECT_ROOT / "docs"
manager = RunManager()

DOC_FILES = {
    "01": "01_perceptron.html",
    "02": "02_mlp.html",
    "03": "03_backprop.html",
    "03-visual": "03_backprop.html",
    "04": "04_activations.html",
    "05": "05_learning_rate.html",
    "06": "06_weight_init.html",
    "07": "07_normalization.html",
    "08": "08_regularization.html",
    "09": "09_full_pipeline.html",
    "10": "10_cnn.html",
    "11": "11_residual.html",
    "12": "12_rnn.html",
}


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return {"passed": False, "lessons": [], "started_at": None, "finished_at": None}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def lesson_status(item: dict, summary_entries: dict) -> dict:
    """Latest known state of one lesson: whichever is newer of its last web run and its summary entry."""
    path = RUNS_DIR / f"{item['name']}.json"
    rec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    entry = summary_entries.get(item["name"])
    if entry and (rec is None or (entry.get("finished_at") or "") > (rec.get("finished_at") or "")):
        return {"status": entry.get("status"), "elapsed_sec": entry.get("elapsed_sec"),
                "finished_at": entry.get("finished_at"), "source": entry.get("source", "verify")}
    if rec:
        return {"status": rec["status"], "elapsed_sec": rec.get("elapsed_sec"), "finished_at": rec.get("finished_at"),
                "source": "web"}
    return {"status": None, "elapsed_sec": None, "finished_at": None, "source": None}


def lesson_list() -> list[dict]:
    entries = {e["name"]: e for e in load_summary().get("lessons", [])}
    active = manager.active
    out = []
    for item in LESSONS:
        state = lesson_status(item, entries)
        if active and active.lesson["id"] == item["id"]:
            state = {**state, "status": "running"}
        out.append({
            "id": item["id"], "name": item["name"], "title": item["title"], "help": item["help"],
            "doc": item["doc"], **state,
        })
    return out


def run_brief(run) -> dict | None:
    return run.to_dict(include_events=False) if run else None


# ---------------------------------------------------------------- pages

@app.get("/")
def index():
    return render_template("index.html", lessons=lesson_list())


@app.get("/lesson/<lesson_id>")
def lesson_page(lesson_id: str):
    try:
        item = find_lesson(lesson_id)
    except KeyError:
        abort(404)
    ids = [x["id"] for x in LESSONS]
    pos = ids.index(item["id"])
    notes = get_notes(item["id"])
    meta = {
        "id": item["id"], "name": item["name"], "title": item["title"], "help": item["help"], "doc": item["doc"],
        "intro": notes.intro, "steps": notes.steps,
    }
    return render_template(
        "lesson.html",
        lesson=meta,
        prev=LESSONS[pos - 1] if pos > 0 else None,
        next=LESSONS[pos + 1] if pos + 1 < len(LESSONS) else None,
        autorun=request.args.get("autorun") == "1",
    )


# ---------------------------------------------------------------- API

@app.get("/api/summary")
def api_summary():
    return load_summary()


@app.get("/api/lessons")
def api_lessons():
    return jsonify(lesson_list())


@app.get("/api/lessons/<lesson_id>/latest")
def api_latest(lesson_id: str):
    try:
        return jsonify(manager.latest(lesson_id))
    except KeyError:
        abort(404)


@app.post("/api/lessons/<lesson_id>/run")
def api_run(lesson_id: str):
    try:
        run = manager.start(lesson_id)
    except KeyError:
        abort(404)
    except BusyError as exc:
        return jsonify({"error": "busy", "active": run_brief(exc.active)}), 409
    return jsonify(run_brief(run)), 202


@app.get("/api/active")
def api_active():
    return jsonify(run_brief(manager.active))


@app.get("/api/runs/<run_id>")
def api_run_detail(run_id: str):
    run = manager.get(run_id)
    if not run:
        abort(404)
    return jsonify(run.to_dict())


@app.post("/api/runs/<run_id>/cancel")
def api_cancel(run_id: str):
    return jsonify({"cancelled": manager.cancel(run_id)})


@app.get("/api/runs/<run_id>/events")
def api_events(run_id: str):
    run = manager.get(run_id)
    if not run:
        abort(404)
    last = request.headers.get("Last-Event-ID") or request.args.get("since", "-1")
    start = int(last) + 1

    def stream():
        i = start
        while True:
            with run.cond:
                if i >= len(run.events) and not run.finished:
                    run.cond.wait(timeout=15)
                batch = run.events[i:]
                finished = run.finished
            if not batch:
                if finished:
                    return
                yield ": keepalive\n\n"
                continue
            for ev in batch:
                yield f"id: {ev['i']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            i += len(batch)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- static docs & artifacts

@app.get("/learn.css")
@app.get("/learn/learn.css")
def learn_css():
    return send_from_directory(DOCS, "learn.css")


@app.get("/learn/")
@app.get("/learn")
def learn_index():
    return send_from_directory(DOCS, "index.html")


@app.get("/learn/<key>")
def learn_page(key: str):
    # Support both short keys (/learn/01) and relative filenames
    # (/learn/01_perceptron.html) so docs work via Flask and file://.
    if key.endswith((".html", ".css")):
        name = key
    else:
        name = DOC_FILES.get(key)
        if name is None:
            try:
                name = find_lesson(key)["doc"]
            except KeyError:
                abort(404)
    path = (DOCS / name).resolve()
    if DOCS.resolve() not in path.parents and path != DOCS.resolve():
        abort(403)
    if not path.exists() or not path.is_file():
        abort(404)
    return send_from_directory(DOCS, name)


@app.get("/file/<path:rel>")
def artifact_file(rel: str):
    target = (PROJECT_ROOT / rel).resolve()
    if ARTIFACTS_DIR.resolve() not in target.parents:
        abort(403)
    if not target.exists():
        abort(404)
    return send_from_directory(target.parent, target.name)


def main():
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)


if __name__ == "__main__":
    main()
