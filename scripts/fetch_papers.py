#!/usr/bin/env python3
"""Download the Transformer papers referenced by lesson 16 into docs/papers/.

The PDFs are not committed (arXiv licences usually do not allow redistribution); this script
fetches them from arXiv and writes docs/papers/manifest.json with size and SHA-256. Run again to
fetch missing files; existing files are checked against the manifest and kept.

    python scripts/fetch_papers.py            # download missing, verify existing
    python scripts/fetch_papers.py --verify   # only verify, no network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parents[1] / "docs" / "papers"
MANIFEST = PAPERS_DIR / "manifest.json"

# (arXiv id, short file name, title, year)
PAPERS = [
    ("1409.0473", "bahdanau_attention", "Neural Machine Translation by Jointly Learning to Align and Translate", 2014),
    ("1706.03762", "transformer", "Attention Is All You Need", 2017),
    ("1810.04805", "bert", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2018),
    ("2005.14165", "gpt3", "Language Models are Few-Shot Learners", 2020),
    ("1910.10683", "t5", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", 2019),
    ("2010.11929", "vit", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", 2020),
    ("2302.13971", "llama", "LLaMA: Open and Efficient Foundation Language Models", 2023),
    ("1512.03385", "resnet", "Deep Residual Learning for Image Recognition", 2015),
    ("1607.06450", "layer_norm", "Layer Normalization", 2016),
    ("2002.04745", "pre_ln", "On Layer Normalization in the Transformer Architecture", 2020),
    ("1910.07467", "rmsnorm", "Root Mean Square Layer Normalization", 2019),
    ("1606.08415", "gelu", "Gaussian Error Linear Units (GELUs)", 2016),
    ("2002.05202", "swiglu", "GLU Variants Improve Transformer", 2020),
    ("1803.02155", "relative_position", "Self-Attention with Relative Position Representations", 2018),
    ("2104.09864", "rope", "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021),
    ("2108.12409", "alibi", "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", 2021),
    ("1412.6980", "adam", "Adam: A Method for Stochastic Optimization", 2014),
    ("1711.05101", "adamw", "Decoupled Weight Decay Regularization", 2017),
    ("1512.00567", "label_smoothing", "Rethinking the Inception Architecture for Computer Vision", 2015),
    ("1508.07909", "bpe", "Neural Machine Translation of Rare Words with Subword Units", 2015),
    ("2001.08361", "scaling_laws", "Scaling Laws for Neural Language Models", 2020),
    ("1911.02150", "mqa", "Fast Transformer Decoding: One Write-Head is All You Need", 2019),
    ("2305.13245", "gqa", "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023),
    ("2205.14135", "flash_attention", "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022),
]


def file_name(arxiv_id: str, short: str) -> str:
    return f"{arxiv_id}_{short}.pdf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(arxiv_id: str, dest: Path) -> None:
    req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}",
                                 headers={"User-Agent": "deep-learning-lab/1.0 (course reading list)"})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
        f.write(resp.read())
    if tmp.read_bytes()[:5] != b"%PDF-":
        tmp.unlink()
        raise RuntimeError("response is not a PDF")
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="only check files against the manifest")
    args = parser.parse_args()

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    failed = 0
    for i, (arxiv_id, short, title, year) in enumerate(PAPERS, 1):
        name = file_name(arxiv_id, short)
        path = PAPERS_DIR / name
        tag = f"[{i:2d}/{len(PAPERS)}] {name}"
        if path.exists():
            digest = sha256(path)
            known = manifest.get(name, {}).get("sha256")
            if known and known != digest:
                print(f"{tag}  SHA-256 与清单不符，删掉后重新运行即可重新下载")
                failed += 1
                continue
            print(f"{tag}  已存在 {path.stat().st_size / 1e6:.1f} MB  {'校验通过' if known else '已记入清单'}")
        elif args.verify:
            print(f"{tag}  缺失")
            failed += 1
            continue
        else:
            try:
                download(arxiv_id, path)
            except Exception as exc:
                print(f"{tag}  下载失败：{exc}")
                failed += 1
                continue
            digest = sha256(path)
            print(f"{tag}  下载完成 {path.stat().st_size / 1e6:.1f} MB")
            time.sleep(3)
        manifest[name] = {"arxiv": arxiv_id, "title": title, "year": year,
                          "bytes": path.stat().st_size, "sha256": digest}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(PAPERS) - failed}/{len(PAPERS)} 篇就绪，清单：{MANIFEST}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
