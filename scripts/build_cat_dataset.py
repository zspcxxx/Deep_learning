#!/usr/bin/env python3
"""Build data/cat/cat_vs_noncat.npz (lesson 10) from CIFAR-10.

Source: the CIFAR-10 upload by its authors' group on Hugging Face (uoft-cs/cifar10), which mirrors
https://www.cs.toronto.edu/~kriz/cifar.html — the original host is very slow from here.
Cats are CIFAR-10 class 3; "not cat" images are drawn evenly from the other 9 classes.

Only this script needs the extra packages:  pip install pyarrow pillow
The downloaded parquet files (~143 MB) are cached outside the repo (default ~/.cache/dllab/).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np

from dllab.datasets import CAT_DATASET_PATH, CIFAR10_CLASSES

HF_BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
FILES = {
    "train": ("train-00000-of-00001.parquet", "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883"),
    "test": ("test-00000-of-00001.parquet", "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"),
}
CAT = CIFAR10_CLASSES.index("cat")


def fetch(name: str, sha256: str, cache: Path) -> Path:
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading {HF_BASE}/{name}", flush=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(f"{HF_BASE}/{name}", tmp)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"SHA-256 mismatch for {path}: {digest} (expected {sha256})")
    return path


def read_split(path: Path, expected_per_class: int):
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path).to_pydict()
    labels = np.asarray(table["label"], dtype=np.int64)
    images = np.stack([np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB")) for cell in table["img"]])
    counts = np.bincount(labels, minlength=10)
    if images.shape[1:] != (32, 32, 3) or not np.all(counts == expected_per_class):
        raise SystemExit(f"unexpected CIFAR-10 split {path.name}: shape {images.shape}, per-class {counts}")
    return images.astype(np.uint8), labels


def balanced_subset(x, y, n_per_class: int, rng):
    """n cats + n non-cats (non-cats split as evenly as possible over the 9 other classes)."""
    cat_idx = rng.choice(np.where(y == CAT)[0], size=n_per_class, replace=False)
    others = [c for c in range(10) if c != CAT]
    counts = [n_per_class // 9 + (1 if i < n_per_class % 9 else 0) for i in range(9)]
    neg_idx = np.concatenate([rng.choice(np.where(y == c)[0], size=k, replace=False) for c, k in zip(others, counts)])
    idx = rng.permutation(np.concatenate([cat_idx, neg_idx]))
    return x[idx], (y[idx] == CAT).astype(np.int64), y[idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=Path.home() / ".cache/dllab")
    p.add_argument("--train-per-class", type=int, default=3000)
    p.add_argument("--test-per-class", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    x_tr, y_tr = read_split(fetch(*FILES["train"], args.cache), 5000)
    x_te, y_te = read_split(fetch(*FILES["test"], args.cache), 1000)
    train = balanced_subset(x_tr, y_tr, args.train_per_class, rng)
    test = balanced_subset(x_te, y_te, args.test_per_class, rng)

    CAT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CAT_DATASET_PATH,
        x_train=train[0], y_train=train[1], source_train=train[2],
        x_test=test[0], y_test=test[1], source_test=test[2],
        class_names=np.array(CIFAR10_CLASSES),
    )
    size_mb = CAT_DATASET_PATH.stat().st_size / 1e6
    print(f"wrote {CAT_DATASET_PATH} ({size_mb:.1f} MB): train {len(train[1])} (cats {train[1].sum()}), "
          f"test {len(test[1])} (cats {test[1].sum()})")


if __name__ == "__main__":
    main()
