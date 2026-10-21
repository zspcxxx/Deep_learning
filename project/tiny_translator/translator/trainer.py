"""Background training job: one run at a time, progress exposed as an event list for the web UI."""
from __future__ import annotations

import math
import random
import threading
import time
import traceback

import torch
from torch import nn

from .config import TrainConfig
from .data import (BUILTIN, PAD, collate, dataset_path, detokenize_zh, download_dataset, make_batches,
                   prepare, tokenize_zh, write_pairs)
from .decode import translate_ids
from .metrics import corpus_bleu
from .model import TinyTransformer
from .runs import new_run_dir, resolve_device, write_json

STEP_EVERY = 10
VAL_BLEU_LIMIT = 500
N_SAMPLES = 6


class Stopped(Exception):
    pass


def lr_factor(step: int, cfg: TrainConfig, total: int) -> float:
    s = step + 1
    if cfg.warmup_steps and s <= cfg.warmup_steps:
        return s / cfg.warmup_steps
    if cfg.schedule == "constant":
        return 1.0
    if cfg.schedule == "inverse_sqrt":
        return math.sqrt(max(1, cfg.warmup_steps) / s)
    progress = min(1.0, (s - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.dim() < 2 or "emb" in name else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class TrainingJob:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.dir = new_run_dir(cfg.name)
        self.run_id = self.dir.name
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.events: list[dict] = []
        self.state = {
            "run_id": self.run_id, "name": cfg.name, "status": "preparing", "message": "",
            "epoch": 0, "epochs": cfg.epochs, "step": 0, "total_steps": 0,
            "started_at": time.time(), "finished_at": None, "device": None, "params": None,
            "stats": None, "best_val_bleu": None, "best_epoch": None, "test_bleu": None,
        }
        write_json(self.dir / "config.json", cfg.to_dict())
        self._write_summary()
        self.thread = threading.Thread(target=self._run, name=f"train-{self.run_id}", daemon=True)

    # -------------------------------------------------------------- bookkeeping

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
                "status": s["status"], "message": s["message"], "started_at": s["started_at"],
                "finished_at": s["finished_at"], "epochs_done": s["epoch"], "steps": s["step"],
                "device": s["device"], "params": s["params"], "stats": s["stats"],
                "best_val_bleu": s["best_val_bleu"], "best_epoch": s["best_epoch"], "test_bleu": s["test_bleu"],
                **extra,
            }
        write_json(self.dir / "summary.json", summary)

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise Stopped

    # -------------------------------------------------------------- training

    def _run(self) -> None:
        try:
            self._train()
            status, message = ("stopped", "已手动停止") if self._stop.is_set() else ("finished", "训练完成")
        except Stopped:
            status, message = "stopped", "已手动停止"
        except Exception as exc:  # noqa: BLE001 - surface every failure to the UI
            self.log(traceback.format_exc())
            status, message = "failed", f"{type(exc).__name__}: {exc}"
        self._set(status=status, message=message, finished_at=time.time())
        self.log(message)
        self.emit("done", status=status)
        self._write_summary()

    def _train(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = random.Random(cfg.seed)
        device = resolve_device(cfg.device)
        self._set(device=str(device) + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

        if not dataset_path(cfg.dataset).is_file():
            if cfg.dataset not in BUILTIN:
                raise FileNotFoundError(f"找不到数据集 data/{cfg.dataset}.tsv")
            download_dataset(cfg.dataset, log=self.log)
        self._check_stop()
        data = prepare(cfg, log=self.log)
        write_json(self.dir / "vocab_src.json", data.src_vocab.to_list())
        write_json(self.dir / "vocab_tgt.json", data.tgt_vocab.to_list())
        write_pairs(self.dir / "val.tsv", [(e.en, e.zh) for e in data.val])
        write_pairs(self.dir / "test.tsv", [(e.en, e.zh) for e in data.test])

        model = TinyTransformer(len(data.src_vocab), len(data.tgt_vocab), cfg).to(device)
        params = model.num_params()
        steps_per_epoch = len(make_batches(data.train, cfg.batch_size))
        total = steps_per_epoch * cfg.epochs
        self._set(params=params, stats=data.stats, total_steps=total, status="running")
        self._write_summary()
        self.log(f"模型参数 {params:,}；每轮 {steps_per_epoch} 步，共 {total} 步；设备 {self.state['device']}")

        opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg, total))
        train_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)
        eval_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
        tgt_vocab = data.tgt_vocab
        val_bleu_set = data.val[:VAL_BLEU_LIMIT]
        samples = data.val[:N_SAMPLES]
        history = {"epochs": [], "steps": []}
        max_len = cfg.max_tgt_len + 10

        step, best, bad_epochs = 0, None, 0
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            t0 = time.time()
            win_loss = win_tok = ep_loss = ep_tok = 0.0
            win_t = time.time()
            stopped = False
            for idx in make_batches(data.train, cfg.batch_size, rng):
                if self._stop.is_set():
                    stopped = True
                    break
                src, tgt_in, tgt_out = collate(data.train, idx, device)
                logits = model(src, tgt_in)
                loss = train_loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                step += 1
                ntok = int((tgt_out != PAD).sum())
                lv = float(loss)
                win_loss += lv * ntok
                win_tok += ntok
                ep_loss += lv * ntok
                ep_tok += ntok
                if step % STEP_EVERY == 0 or step == total:
                    now = time.time()
                    point = {"step": step, "epoch": epoch, "loss": round(win_loss / win_tok, 4),
                             "lr": opt.param_groups[0]["lr"], "tok_s": round(win_tok / max(1e-6, now - win_t))}
                    history["steps"].append(point)
                    self.emit("step", **point)
                    win_loss = win_tok = 0.0
                    win_t = now
                self._set(step=step, epoch=epoch)

            if stopped:
                # Keep the half-trained weights so a stopped run can still be inspected.
                ckpt = {"model": model.state_dict(), "epoch": epoch - 1, "val_bleu": best, "val_loss": None}
                torch.save(ckpt, self.dir / "last.pt")
                if best is None:
                    torch.save(ckpt, self.dir / "best.pt")
                self.log(f"第 {epoch} 轮进行到第 {step} 步时停止，已保存当前权重")
                break

            # ---- end of epoch: validation
            model.eval()
            val_loss_sum = val_tok = 0.0
            with torch.no_grad():
                for idx in make_batches(data.val, 256):
                    src, tgt_in, tgt_out = collate(data.val, idx, device)
                    logits = model(src, tgt_in)
                    val_loss_sum += float(eval_loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)))
                    val_tok += int((tgt_out != PAD).sum())
            val_loss = val_loss_sum / max(1, val_tok)
            outs = translate_ids(model, [e.src for e in val_bleu_set], device, max_len)
            hyps = [tgt_vocab.decode(o) for o in outs]
            bleu = corpus_bleu(hyps, [tokenize_zh(e.zh) for e in val_bleu_set])["bleu"]
            sample_rows = [{"en": e.en, "ref": e.zh, "hyp": detokenize_zh(h)}
                           for e, h in zip(samples, hyps[:N_SAMPLES])]

            improved = best is None or bleu > best
            ckpt = {"model": model.state_dict(), "epoch": epoch, "val_bleu": bleu, "val_loss": val_loss}
            torch.save(ckpt, self.dir / "last.pt")
            if improved:
                best, bad_epochs = bleu, 0
                torch.save(ckpt, self.dir / "best.pt")
                self._set(best_val_bleu=bleu, best_epoch=epoch)
            else:
                bad_epochs += 1

            record = {"epoch": epoch, "step": step, "train_loss": round(ep_loss / max(1, ep_tok), 4),
                      "val_loss": round(val_loss, 4), "val_ppl": round(math.exp(min(val_loss, 20)), 2),
                      "val_bleu": bleu, "seconds": round(time.time() - t0, 1), "best": improved,
                      "samples": sample_rows}
            history["epochs"].append(record)
            write_json(self.dir / "history.json", history)
            self.emit("epoch", **record)
            self.log(f"第 {epoch} 轮：训练损失 {record['train_loss']}，验证损失 {record['val_loss']}，"
                     f"验证 BLEU {bleu}{'（最佳）' if improved else ''}，用时 {record['seconds']}s")
            self._write_summary()

            if cfg.patience and bad_epochs >= cfg.patience:
                self.log(f"验证 BLEU 连续 {cfg.patience} 轮未提升，提前停止")
                break
            if self._stop.is_set():
                break

        self._final_test(model, data, device, max_len)

    def _final_test(self, model, data, device, max_len) -> None:
        ckpt = torch.load(self.dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        outs = translate_ids(model, [e.src for e in data.test], device, max_len)
        hyps = [data.tgt_vocab.decode(o) for o in outs]
        score = corpus_bleu(hyps, [tokenize_zh(e.zh) for e in data.test])
        self._set(test_bleu=score["bleu"])
        self.log(f"测试集（{len(data.test)} 句，第 {ckpt['epoch']} 轮最佳模型，贪心解码）BLEU {score['bleu']}")
        self._write_summary(test_detail=score)


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
