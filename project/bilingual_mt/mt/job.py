"""Background training job for the web UI (one at a time, event stream)."""
from __future__ import annotations

import json
import math
import re
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import RUNS_DIR, TrainConfig, resolve_device
from .data import (
    expand_directions,
    load_en_zh_tsv,
    split_pairs,
    write_split_tsv,
)
from .dataset import MTDataset, collate_batch
from .datasets_info import ensure_dataset
from .decode import greedy_decode, ids_to_text
from .metrics import bleu_for_direction
from .model import build_model
from .train import _label_smoothed_nll, make_scheduler
from .vocab import Vocab, train_sentencepiece

STEP_EVERY = 10
VAL_BLEU_LIMIT = 200
N_SAMPLES = 6


class Stopped(Exception):
    pass


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def new_run_dir(name: str) -> Path:
    safe = re.sub(r"[^\w-]+", "-", name).strip("-")[:40] or "run"
    path = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class TrainingJob:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.dir = new_run_dir(cfg.name)
        self.run_id = self.dir.name
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.events: list[dict] = []
        self.state = {
            "run_id": self.run_id,
            "name": cfg.name,
            "status": "preparing",
            "message": "",
            "epoch": 0,
            "epochs": cfg.epochs,
            "step": 0,
            "total_steps": 0,
            "started_at": time.time(),
            "finished_at": None,
            "device": None,
            "params": None,
            "stats": None,
            "best_val_bleu": None,
            "best_epoch": None,
            "test_bleu": None,
        }
        write_json(self.dir / "config.json", cfg.to_dict())
        self._write_summary()
        self.thread = threading.Thread(target=self._run, name=f"bilingual-{self.run_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self.state["status"] in {"preparing", "running"}:
                self.state["status"] = "stopping"

    def _set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    def emit(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append({"seq": len(self.events), "kind": kind, "t": round(time.time(), 2), **data})

    def log(self, text: str) -> None:
        self.emit("log", text=text)
        with (self.dir / "train.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {text}\n")

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            state = dict(self.state)
            end = state["finished_at"] or time.time()
            state["elapsed"] = round(end - state["started_at"], 1)
            return {"state": state, "events": self.events[since:], "next": len(self.events)}

    def _write_summary(self, **extra) -> None:
        with self._lock:
            s = self.state
            summary = {
                "status": s["status"],
                "message": s["message"],
                "started_at": s["started_at"],
                "finished_at": s["finished_at"],
                "epochs_done": s["epoch"],
                "steps": s["step"],
                "device": s["device"],
                "params": s["params"],
                "stats": s["stats"],
                "best_val_bleu": s["best_val_bleu"],
                "best_epoch": s["best_epoch"],
                "test_bleu": s["test_bleu"],
                **extra,
            }
        write_json(self.dir / "summary.json", summary)

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise Stopped

    def _run(self) -> None:
        try:
            self._train()
            status, message = ("stopped", "已手动停止") if self._stop.is_set() else ("finished", "训练完成")
        except Stopped:
            status, message = "stopped", "已手动停止"
        except Exception as exc:  # noqa: BLE001
            self.log(traceback.format_exc())
            status, message = "failed", f"{type(exc).__name__}: {exc}"
        self._set(status=status, message=message, finished_at=time.time())
        self.log(message)
        self.emit("done", status=status)
        self._write_summary()

    def _train(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        device_str = resolve_device(cfg.device)
        device = torch.device(device_str)
        self._set(
            device=device_str
            + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
        )

        tsv = ensure_dataset(cfg.dataset, log=self.log)
        self._check_stop()
        en_zh = load_en_zh_tsv(
            tsv, max_pairs=cfg.max_pairs, simplified=cfg.zh_simplified, seed=cfg.seed
        )
        pairs = expand_directions(en_zh, cfg.direction)
        train_p, val_p, test_p = split_pairs(
            pairs, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, seed=cfg.seed
        )
        write_split_tsv(self.dir / "train.tsv", train_p)
        write_split_tsv(self.dir / "val.tsv", val_p)
        write_split_tsv(self.dir / "test.tsv", test_p)
        self.log(f"方向 {cfg.direction}：train/val/test = {len(train_p)}/{len(val_p)}/{len(test_p)}")

        texts = [t for en, zh in en_zh for t in (en, zh)]
        spm_path = train_sentencepiece(texts, self.dir / "spm", vocab_size=cfg.vocab_size)
        cfg.spm_model = str(spm_path)
        write_json(self.dir / "config.json", cfg.to_dict())
        vocab = Vocab(spm_path)
        self._check_stop()

        train_ds = MTDataset(train_p, vocab, cfg.max_src_len, cfg.max_tgt_len)
        val_ds = MTDataset(val_p, vocab, cfg.max_src_len, cfg.max_tgt_len)
        test_ds = MTDataset(test_p, vocab, cfg.max_src_len, cfg.max_tgt_len)
        collate = partial(collate_batch, pad_id=vocab.pad_id)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=min(64, cfg.batch_size), shuffle=False, collate_fn=collate
        )

        stats = {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
            "src_vocab": len(vocab),
            "tgt_vocab": len(vocab),
            "direction": cfg.direction,
            "raw_pairs": len(en_zh),
        }
        model = build_model(cfg, len(vocab), vocab.pad_id).to(device)
        params = model.num_parameters()
        steps_per_epoch = max(1, len(train_loader))
        total = steps_per_epoch * cfg.epochs
        self._set(params=params, stats=stats, total_steps=total, status="running")
        self._write_summary()
        self.log(f"参数 {params:,}；每轮 {steps_per_epoch} 步，共 {total} 步；词表 {len(vocab)}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay
        )
        scheduler = make_scheduler(optimizer, cfg, total)
        use_amp = bool(cfg.amp and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        history: dict = {"epochs": [], "steps": []}
        step, best, bad_epochs = 0, None, 0

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            t0 = time.time()
            win_loss = win_tok = ep_loss = ep_tok = 0.0
            win_t = time.time()
            stopped = False
            for batch in train_loader:
                if self._stop.is_set():
                    stopped = True
                    break
                src = batch["src"].to(device)
                tgt_in = batch["tgt_in"].to(device)
                tgt_out = batch["tgt_out"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(src, tgt_in)
                    loss = _label_smoothed_nll(
                        logits, tgt_out, pad_id=vocab.pad_id, smoothing=cfg.label_smoothing
                    )
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                ntok = int(tgt_out.ne(vocab.pad_id).sum())
                lv = float(loss.item())
                win_loss += lv * ntok
                win_tok += ntok
                ep_loss += lv * ntok
                ep_tok += ntok
                if step % STEP_EVERY == 0 or step == total:
                    now = time.time()
                    point = {
                        "step": step,
                        "epoch": epoch,
                        "loss": round(win_loss / max(1, win_tok), 4),
                        "lr": optimizer.param_groups[0]["lr"],
                        "tok_s": round(win_tok / max(1e-6, now - win_t)),
                    }
                    history["steps"].append(point)
                    self.emit("step", **point)
                    win_loss = win_tok = 0.0
                    win_t = now
                self._set(step=step, epoch=epoch)

            if stopped:
                ckpt = {
                    "model": model.state_dict(),
                    "epoch": epoch - 1,
                    "val_bleu": best,
                    "spm_model": str(spm_path),
                    "cfg": cfg.to_dict(),
                }
                torch.save(ckpt, self.dir / "last.pt")
                if best is None:
                    torch.save(ckpt, self.dir / "best.pt")
                self.log(f"第 {epoch} 轮进行到第 {step} 步时停止")
                break

            metrics, sample_rows = self._validate(
                model, val_loader, val_ds, vocab, device, cfg
            )
            improved = best is None or metrics["bleu"] > best
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_bleu": metrics["bleu"],
                "val_loss": metrics["loss"],
                "spm_model": str(spm_path),
                "cfg": cfg.to_dict(),
            }
            torch.save(ckpt, self.dir / "last.pt")
            if improved:
                best, bad_epochs = metrics["bleu"], 0
                torch.save(ckpt, self.dir / "best.pt")
                self._set(best_val_bleu=best, best_epoch=epoch)
            else:
                bad_epochs += 1

            record = {
                "epoch": epoch,
                "step": step,
                "train_loss": round(ep_loss / max(1, ep_tok), 4),
                "val_loss": round(metrics["loss"], 4),
                "val_ppl": round(metrics["ppl"], 2),
                "val_bleu": metrics["bleu"],
                "bleu_en2zh": metrics.get("bleu_en2zh"),
                "bleu_zh2en": metrics.get("bleu_zh2en"),
                "seconds": round(time.time() - t0, 1),
                "best": improved,
                "samples": sample_rows,
            }
            history["epochs"].append(record)
            write_json(self.dir / "history.json", history)
            self.emit("epoch", **record)
            self.log(
                f"第 {epoch} 轮：train {record['train_loss']}  val_ppl {record['val_ppl']}  "
                f"BLEU {metrics['bleu']}"
                + (f" en2zh={metrics.get('bleu_en2zh')} zh2en={metrics.get('bleu_zh2en')}" if cfg.direction == "both" else "")
                + (" ★" if improved else "")
                + f"  {record['seconds']}s"
            )
            self._write_summary()
            if cfg.patience and bad_epochs >= cfg.patience:
                self.log(f"验证 BLEU 连续 {cfg.patience} 轮未提升，提前停止")
                break
            if self._stop.is_set():
                break

        self._final_test(model, test_ds, vocab, device, cfg)

    @torch.no_grad()
    def _validate(self, model, loader, val_ds, vocab, device, cfg):
        model.eval()
        total_loss = n_tok = 0.0
        hyps_by: dict[str, list[str]] = {"en2zh": [], "zh2en": []}
        refs_by: dict[str, list[str]] = {"en2zh": [], "zh2en": []}
        samples: list[dict] = []
        n_decoded = 0
        for batch in loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            logits = model(src, tgt_in)
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
            mask = tgt_out.ne(vocab.pad_id)
            total_loss += float((nll * mask).sum())
            n_tok += int(mask.sum())

            if n_decoded < VAL_BLEU_LIMIT:
                pred = greedy_decode(model, src, vocab, cfg.max_decode_len)
                for ids, src_t, ref, direction in zip(
                    pred, batch["src_text"], batch["tgt_text"], batch["direction"]
                ):
                    if n_decoded >= VAL_BLEU_LIMIT:
                        break
                    hyp = ids_to_text(ids, vocab)
                    hyps_by[direction].append(hyp)
                    refs_by[direction].append(ref)
                    if len(samples) < N_SAMPLES:
                        samples.append(
                            {
                                "en": src_t if direction == "en2zh" else ref,
                                "src": src_t,
                                "ref": ref,
                                "hyp": hyp,
                                "direction": direction,
                            }
                        )
                    n_decoded += 1

        loss = total_loss / max(1, n_tok)
        bleus = []
        out = {"loss": loss, "ppl": math.exp(min(loss, 20))}
        for d in ("en2zh", "zh2en"):
            if hyps_by[d]:
                b = bleu_for_direction(hyps_by[d], refs_by[d], d)
                out[f"bleu_{d}"] = round(b, 2)
                bleus.append(b)
        out["bleu"] = round(sum(bleus) / len(bleus), 2) if bleus else 0.0
        return out, samples

    @torch.no_grad()
    def _final_test(self, model, test_ds, vocab, device, cfg) -> None:
        best_path = self.dir / "best.pt"
        if not best_path.is_file():
            return
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        collate = partial(collate_batch, pad_id=vocab.pad_id)
        loader = DataLoader(
            test_ds, batch_size=min(64, cfg.batch_size), shuffle=False, collate_fn=collate
        )
        hyps_by: dict[str, list[str]] = {"en2zh": [], "zh2en": []}
        refs_by: dict[str, list[str]] = {"en2zh": [], "zh2en": []}
        for batch in loader:
            src = batch["src"].to(device)
            pred = greedy_decode(model, src, vocab, cfg.max_decode_len)
            for ids, ref, direction in zip(pred, batch["tgt_text"], batch["direction"]):
                hyps_by[direction].append(ids_to_text(ids, vocab))
                refs_by[direction].append(ref)
        bleus = []
        detail = {}
        for d in ("en2zh", "zh2en"):
            if hyps_by[d]:
                b = bleu_for_direction(hyps_by[d], refs_by[d], d)
                detail[f"bleu_{d}"] = round(b, 2)
                bleus.append(b)
        score = round(sum(bleus) / len(bleus), 2) if bleus else 0.0
        self._set(test_bleu=score)
        self.log(f"测试集 BLEU {score} {detail}")
        self._write_summary(test_detail=detail)


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.job: TrainingJob | None = None

    def start(self, cfg: TrainConfig) -> TrainingJob:
        with self._lock:
            if self.job is not None and self.job.alive:
                raise RuntimeError("已有训练在进行，请先停止或等待完成")
            self.job = TrainingJob(cfg)
            self.job.start()
            return self.job

    def active_id(self) -> str | None:
        job = self.job
        return job.run_id if job is not None and job.alive else None
