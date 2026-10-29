"""Training loop with AMP, cosine/inverse-sqrt schedule, early stopping."""
from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RUNS_DIR, TrainConfig, load_config, resolve_device
from .data import read_split_tsv
from .dataset import MTDataset, collate_batch
from .decode import greedy_decode, ids_to_text
from .metrics import bleu_for_direction
from .model import build_model
from .vocab import Vocab


def make_scheduler(optimizer, cfg: TrainConfig, total_steps: int):
    """LambdaLR multipliers; optimizer base lr is cfg.lr."""
    if cfg.schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    if cfg.schedule == "inverse_sqrt":
        warm = max(1, cfg.warmup_steps)
        # Peak multiplier = 1.0 at step == warm (Vaswani-style, renormalized).
        peak = float(cfg.d_model**-0.5) * (warm**-0.5)

        def lr_lambda(step: int) -> float:
            step = max(1, step)
            raw = float(cfg.d_model**-0.5) * min(step**-0.5, step * warm**-1.5)
            return raw / peak

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    warm = max(1, cfg.warmup_steps)
    total = max(warm + 1, total_steps)

    def lr_lambda(step: int) -> float:
        if step < warm:
            return float(step) / float(warm)
        progress = (step - warm) / float(max(1, total - warm))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _label_smoothed_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pad_id: int,
    smoothing: float,
) -> torch.Tensor:
    """logits (B,T,V), target (B,T)."""
    V = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    loss = (1.0 - smoothing) * nll + smoothing * smooth
    mask = target.ne(pad_id)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    vocab: Vocab,
    device: str,
    *,
    max_decode_len: int,
    compute_bleu: bool = True,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_tok = 0
    hyps_by_dir: dict[str, list[str]] = {"en2zh": [], "zh2en": []}
    refs_by_dir: dict[str, list[str]] = {"en2zh": [], "zh2en": []}

    for batch in loader:
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        logits = model(src, tgt_in)
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)
        mask = tgt_out.ne(vocab.pad_id)
        total_loss += float((nll * mask).sum().item())
        n_tok += int(mask.sum().item())

        if compute_bleu:
            pred_ids = greedy_decode(model, src, vocab, max_decode_len)
            for ids, ref, direction in zip(pred_ids, batch["tgt_text"], batch["direction"]):
                hyp = ids_to_text(ids, vocab)
                hyps_by_dir[direction].append(hyp)
                refs_by_dir[direction].append(ref)

    ppl = math.exp(total_loss / max(1, n_tok))
    out: dict = {"loss": total_loss / max(1, n_tok), "ppl": ppl}
    bleus = []
    for d in ("en2zh", "zh2en"):
        if hyps_by_dir[d]:
            b = bleu_for_direction(hyps_by_dir[d], refs_by_dir[d], d)
            out[f"bleu_{d}"] = b
            bleus.append(b)
    if bleus:
        out["bleu"] = sum(bleus) / len(bleus)
    else:
        out["bleu"] = 0.0
    return out


def train(cfg: TrainConfig, run_dir: Path | None = None, resume: bool = True) -> Path:
    run_dir = run_dir or (RUNS_DIR / cfg.name)
    device = resolve_device(cfg.device)
    print(f"device={device}  run={run_dir}")

    spm_path = Path(cfg.spm_model) if cfg.spm_model else run_dir / "spm.model"
    if not spm_path.is_file():
        raise FileNotFoundError(f"缺少词表 {spm_path}，请先运行 prepare")
    vocab = Vocab(spm_path)

    train_pairs = read_split_tsv(run_dir / "train.tsv")
    val_pairs = read_split_tsv(run_dir / "val.tsv")
    train_ds = MTDataset(train_pairs, vocab, cfg.max_src_len, cfg.max_tgt_len)
    val_ds = MTDataset(val_pairs, vocab, cfg.max_src_len, cfg.max_tgt_len)
    collate = partial(collate_batch, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )

    model = build_model(cfg, len(vocab), vocab.pad_id).to(device)
    print(f"参数量：{model.num_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay
    )
    total_steps = max(1, cfg.epochs * len(train_loader))
    scheduler = make_scheduler(optimizer, cfg, total_steps)
    use_amp = bool(cfg.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = run_dir / "checkpoint.pt"
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.json"
    start_epoch = 0
    global_step = 0
    best_bleu = -1.0
    bad_epochs = 0
    history: list[dict] = []

    if resume and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state.get("epoch", 0) + 1
        global_step = state.get("global_step", 0)
        best_bleu = state.get("best_bleu", -1.0)
        bad_epochs = state.get("bad_epochs", 0)
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"从 epoch {start_epoch} 恢复 (best_bleu={best_bleu:.2f})")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n_tok = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False)
        for step, batch in enumerate(pbar):
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
            global_step += 1

            with torch.no_grad():
                mask = tgt_out.ne(vocab.pad_id)
                running += float(loss.item() * mask.sum().item())
                n_tok += int(mask.sum().item())
            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running / max(1, n_tok)
        metrics = evaluate(
            model, val_loader, vocab, device, max_decode_len=cfg.max_decode_len, compute_bleu=True
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_ppl": metrics["ppl"],
            "bleu": metrics.get("bleu", 0.0),
            "bleu_en2zh": metrics.get("bleu_en2zh"),
            "bleu_zh2en": metrics.get("bleu_zh2en"),
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        history.append(row)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"epoch {epoch+1}: train_loss={train_loss:.3f}  "
            f"val_ppl={metrics['ppl']:.2f}  bleu={metrics.get('bleu', 0):.2f}  "
            f"en2zh={metrics.get('bleu_en2zh')}  zh2en={metrics.get('bleu_zh2en')}"
        )

        improved = metrics.get("bleu", 0.0) > best_bleu + 1e-4
        if improved:
            best_bleu = metrics["bleu"]
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.to_dict(),
                    "spm_model": str(spm_path),
                    "epoch": epoch,
                    "best_bleu": best_bleu,
                },
                best_path,
            )
            print(f"  ✓ 新最佳 BLEU={best_bleu:.2f} → {best_path}")
        else:
            bad_epochs += 1

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_bleu": best_bleu,
                "bad_epochs": bad_epochs,
                "cfg": cfg.to_dict(),
                "spm_model": str(spm_path),
            },
            ckpt_path,
        )

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"早停：连续 {bad_epochs} 个 epoch BLEU 未提升")
            break

    return run_dir


def train_from_config(config_path: str, run_name: str | None = None, resume: bool = True) -> Path:
    cfg = load_config(config_path)
    if run_name:
        cfg.name = run_name
    # Prefer prepared config.json if present (has spm_model filled).
    run_dir = RUNS_DIR / cfg.name
    prepared = run_dir / "config.json"
    if prepared.is_file():
        saved = json.loads(prepared.read_text(encoding="utf-8"))
        for k, v in saved.items():
            if hasattr(cfg, k) and v not in (None, ""):
                setattr(cfg, k, v)
    return train(cfg, run_dir=run_dir, resume=resume)
