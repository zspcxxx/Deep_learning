"""Web UI as a Flask blueprint: mounted at / by web/app.py, or under a prefix by a host app."""
from __future__ import annotations

import threading

from flask import Blueprint, jsonify, render_template, request

from . import runs
from .config import schema, validate
from .data import ROOT, download_dataset, list_datasets, make_batches, prepare
from .gpu import read_gpu
from .model import TinyTransformer
from .trainer import JobManager

bp = Blueprint(
    "translator",
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
    static_url_path="/static",
)
manager = JobManager()
_download_lock = threading.Lock()


def error(message: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": message, **extra}), status


def dataset_names() -> list[str]:
    return [d["name"] for d in list_datasets()]


@bp.get("/")
def index():
    return render_template("translator/index.html")


# ---------------------------------------------------------------- configuration

@bp.get("/api/config/schema")
def config_schema():
    return jsonify(schema(list_datasets()))


@bp.get("/api/datasets")
def datasets():
    return jsonify(list_datasets())


@bp.post("/api/datasets/<name>/download")
def dataset_download(name: str):
    if not _download_lock.acquire(blocking=False):
        return error("已有下载在进行")
    try:
        pairs = download_dataset(name)
    except ValueError as exc:
        return error(str(exc))
    except OSError as exc:
        return error(f"下载失败：{exc}", 502)
    finally:
        _download_lock.release()
    return jsonify({"ok": True, "pairs": pairs, "datasets": list_datasets()})


@bp.post("/api/config/check")
def config_check():
    cfg, errors = validate(request.get_json(force=True) or {}, dataset_names())
    if errors:
        return error("配置有误", fields=errors)
    ready = {d["name"]: d["ready"] for d in list_datasets()}
    if not ready.get(cfg.dataset):
        return error("数据集尚未下载", need_download=True)
    try:
        data = prepare(cfg, log=lambda _: None)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return error(str(exc))
    model = TinyTransformer(len(data.src_vocab), len(data.tgt_vocab), cfg)
    steps = len(make_batches(data.train, cfg.batch_size))
    return jsonify({"ok": True, "config": cfg.to_dict(), "stats": data.stats, "params": model.num_params(),
                    "steps_per_epoch": steps, "total_steps": steps * cfg.epochs})


# ---------------------------------------------------------------- training

@bp.post("/api/train/start")
def train_start():
    cfg, errors = validate(request.get_json(force=True) or {}, dataset_names())
    if errors:
        return error("配置有误", fields=errors)
    try:
        job = manager.start(cfg)
    except RuntimeError as exc:
        return error(str(exc), 409)
    return jsonify({"ok": True, "run_id": job.run_id})


@bp.post("/api/train/stop")
def train_stop():
    job = manager.job
    if job is None or not job.alive:
        return error("当前没有进行中的训练")
    job.stop()
    return jsonify({"ok": True})


@bp.get("/api/train/status")
def train_status():
    job = manager.job
    gpu = read_gpu()
    if job is None:
        return jsonify({"state": None, "events": [], "next": 0, "gpu": gpu})
    since = request.args.get("since", default=0, type=int)
    snap = job.snapshot(max(0, since))
    snap["config"] = job.cfg.to_dict() if since == 0 else None
    snap["gpu"] = gpu
    return jsonify(snap)


@bp.get("/api/system/gpu")
def system_gpu():
    """Current GPU utilization for the live meter (training and evaluation)."""
    return jsonify({"ok": True, "gpu": read_gpu()})


# ---------------------------------------------------------------- runs & evaluation

@bp.get("/api/runs")
def runs_list():
    return jsonify(runs.list_runs(manager.active_id()))


@bp.get("/api/runs/<run_id>")
def run_get(run_id: str):
    try:
        return jsonify(runs.run_detail(run_id))
    except KeyError:
        return error("找不到该运行", 404)


@bp.delete("/api/runs/<run_id>")
def run_delete(run_id: str):
    if run_id == manager.active_id():
        return error("该运行正在训练，不能删除", 409)
    try:
        runs.delete_run(run_id)
    except KeyError:
        return error("找不到该运行", 404)
    return jsonify({"ok": True})


@bp.post("/api/runs/<run_id>/translate")
def run_translate(run_id: str):
    body = request.get_json(force=True) or {}
    beam = max(1, min(int(body.get("beam", 1)), 10))
    try:
        result = runs.load_translator(run_id, body.get("checkpoint", "best")).translate(str(body.get("text", "")), beam)
    except KeyError:
        return error("找不到该运行或检查点", 404)
    except (FileNotFoundError, ValueError) as exc:
        return error(str(exc))
    return jsonify({"ok": True, **result})


@bp.post("/api/runs/<run_id>/evaluate")
def run_evaluate(run_id: str):
    body = request.get_json(force=True) or {}
    split = body.get("split", "test")
    if split not in {"test", "val"}:
        return error("split 只能是 test 或 val")
    beam = max(1, min(int(body.get("beam", 1)), 10))
    limit = max(0, int(body.get("limit", 0)))
    try:
        result = runs.load_translator(run_id, body.get("checkpoint", "best")).evaluate(split, limit, beam)
    except KeyError:
        return error("找不到该运行或检查点", 404)
    except FileNotFoundError as exc:
        return error(str(exc))
    return jsonify({"ok": True, **result})
