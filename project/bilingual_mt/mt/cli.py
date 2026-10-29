"""CLI: prepare | train | eval | translate | download."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python -m mt.cli` from project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bilingual-mt",
        description="中英互译 Transformer 训练程序（prepare → train → eval → translate）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="下载内置平行语料到 data/")
    d.add_argument("dataset", choices=["cmn-eng", "news-commentary", "ted2020", "all"])
    d.add_argument("--force", action="store_true")

    pr = sub.add_parser("prepare", help="清洗语料、划分、训练 SentencePiece")
    pr.add_argument("-c", "--config", required=True, help="configs/*.yaml 或路径")
    pr.add_argument("--name", default=None, help="覆盖 run 名称")
    pr.add_argument("--force-download", action="store_true")

    tr = sub.add_parser("train", help="训练编码器–解码器 Transformer")
    tr.add_argument("-c", "--config", required=True)
    tr.add_argument("--name", default=None)
    tr.add_argument("--no-resume", action="store_true")

    ev = sub.add_parser("eval", help="在 val/test 上计算 BLEU")
    ev.add_argument("run", help="runs/<name> 或路径")
    ev.add_argument("--split", default="test", choices=["val", "test"])
    ev.add_argument("--beam", type=int, default=None)
    ev.add_argument("--device", default="auto")
    ev.add_argument("--max-examples", type=int, default=0)

    trn = sub.add_parser("translate", help="单句或文件翻译")
    trn.add_argument("run", help="runs/<name> 或路径")
    trn.add_argument("-d", "--direction", required=True, choices=["en2zh", "zh2en"])
    trn.add_argument("-t", "--text", default=None, help="待译文本")
    trn.add_argument("-f", "--file", default=None, help="每行一句的输入文件")
    trn.add_argument("-o", "--output", default=None)
    trn.add_argument("--beam", type=int, default=None)
    trn.add_argument("--device", default="auto")

    info = sub.add_parser("info", help="打印 run 配置与模型规模")
    info.add_argument("run")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "download":
        from .data import BUILTINS, download_builtin

        names = list(BUILTINS) if args.dataset == "all" else [args.dataset]
        for n in names:
            download_builtin(n, force=args.force)
        return 0

    if args.cmd == "prepare":
        from .prepare import prepare_from_config

        prepare_from_config(args.config, run_name=args.name, force=args.force_download)
        return 0

    if args.cmd == "train":
        from .train import train_from_config

        train_from_config(args.config, run_name=args.name, resume=not args.no_resume)
        return 0

    if args.cmd == "eval":
        from .eval import evaluate_split
        from .translate import resolve_run

        evaluate_split(
            resolve_run(args.run),
            split=args.split,
            beam_size=args.beam,
            device=args.device,
            max_examples=args.max_examples,
        )
        return 0

    if args.cmd == "translate":
        from .translate import resolve_run, translate_file, translate_text

        run = resolve_run(args.run)
        if args.text:
            print(translate_text(args.text, args.direction, run_dir=run, beam_size=args.beam, device=args.device))
        elif args.file:
            translate_file(
                Path(args.file),
                args.direction,
                run_dir=run,
                out_path=Path(args.output) if args.output else None,
                beam_size=args.beam,
                device=args.device,
            )
        else:
            print("交互模式：输入句子，空行退出。方向 =", args.direction)
            while True:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line:
                    break
                print(
                    translate_text(
                        line, args.direction, run_dir=run, beam_size=args.beam, device=args.device
                    )
                )
        return 0

    if args.cmd == "info":
        import json

        from .model import build_model
        from .translate import resolve_run
        from .vocab import Vocab
        from .config import TrainConfig

        run = resolve_run(args.run)
        cfg = TrainConfig(**json.loads((run / "config.json").read_text(encoding="utf-8")))
        spm = Path(cfg.spm_model) if cfg.spm_model else run / "spm.model"
        vocab = Vocab(spm)
        model = build_model(cfg, len(vocab), vocab.pad_id)
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
        print(f"vocab_size={len(vocab)}  params={model.num_parameters():,}")
        best = run / "best.pt"
        ckpt = run / "checkpoint.pt"
        print(f"best.pt: {'yes' if best.is_file() else 'no'}  checkpoint.pt: {'yes' if ckpt.is_file() else 'no'}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
