from __future__ import annotations

from pathlib import Path
import urllib.request

import numpy as np

from dllab.paths import PROJECT_ROOT


def truncated_gaussian_2d(n_samples: int, sigma: float, half_width: float, rng, max_batches: int = 200) -> np.ndarray:
    samples = np.empty((0, 2))
    for _ in range(max_batches):
        need = n_samples - len(samples)
        if need <= 0:
            break
        batch = rng.randn(int(need * 1.5) + 10, 2) * sigma
        mask = np.all(np.abs(batch) <= half_width, axis=1)
        samples = np.vstack([samples, batch[mask]])
    return samples[:n_samples]


def generate_logic_gate_dataset(
    X_base, y_base, n_per_point: int = 50, sigma: float = 0.30, half_width: float = 0.49, seed: int = 1,
):
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for xi, yi in zip(X_base, y_base):
        local = truncated_gaussian_2d(n_per_point, sigma, half_width, rng)
        X_list.append(xi + local)
        y_list.append(np.full(n_per_point, yi, dtype=float))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def generate_2d_dataset(n_per_class: int = 300, center: float = 10.0, sigma: float = 3.5, half_width: float = 9.9, seed: int = 42):
    rng = np.random.RandomState(seed)
    pos_local = truncated_gaussian_2d(n_per_class, sigma, half_width, rng)
    neg_local = truncated_gaussian_2d(n_per_class, sigma, half_width, rng)
    X = np.vstack([pos_local + np.array([center, center]), neg_local + np.array([-center, -center])])
    y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def generate_xor_dataset(n_per_quadrant: int = 100, noise: float = 0.15, seed: int = 0):
    rng = np.random.RandomState(seed)

    def cluster(cx, cy, label):
        X = rng.randn(n_per_quadrant, 2) * noise + np.array([cx, cy])
        y = np.full(n_per_quadrant, label)
        return X, y

    parts = [cluster(0, 0, 0), cluster(0, 1, 1), cluster(1, 0, 1), cluster(1, 1, 0)]
    X = np.vstack([p[0] for p in parts])
    y = np.hstack([p[1] for p in parts])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def generate_moons_dataset(n_per_class: int = 200, noise: float = 0.10, seed: int = 1):
    rng = np.random.RandomState(seed)
    theta1 = rng.uniform(0, np.pi, n_per_class)
    X1 = np.column_stack([np.cos(theta1), np.sin(theta1)]) + rng.randn(n_per_class, 2) * noise
    theta2 = rng.uniform(0, np.pi, n_per_class)
    X2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5]) + rng.randn(n_per_class, 2) * noise
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


MNIST_URLS = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
)

# 7-segment bitmasks (a,b,c,d,e,f,g) for offline fallback 28x28 synthesis
_DIGIT_SEGMENTS = {
    0: (1, 1, 1, 1, 1, 1, 0),
    1: (0, 1, 1, 0, 0, 0, 0),
    2: (1, 1, 0, 1, 1, 0, 1),
    3: (1, 1, 1, 1, 0, 0, 1),
    4: (0, 1, 1, 0, 0, 1, 1),
    5: (1, 0, 1, 1, 0, 1, 1),
    6: (1, 0, 1, 1, 1, 1, 1),
    7: (1, 1, 1, 0, 0, 0, 0),
    8: (1, 1, 1, 1, 1, 1, 1),
    9: (1, 1, 1, 1, 0, 1, 1),
}


def _synthesize_mnist_fallback(n_total: int, seed: int = 42):
    """Offline fallback: generate realistic 28x28 handwritten-style digit images."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:28, 0:28].astype(np.float32)
    # Segment endpoints in 28x28 coordinates: (x0, y0, x1, y1)
    seg_coords = [
        (8.0, 5.0, 20.0, 5.0),    # a: top
        (20.0, 5.0, 20.0, 14.0),  # b: top-right
        (20.0, 14.0, 20.0, 23.0), # c: bottom-right
        (8.0, 23.0, 20.0, 23.0),  # d: bottom
        (8.0, 14.0, 8.0, 23.0),   # e: bottom-left
        (8.0, 5.0, 8.0, 14.0),    # f: top-left
        (8.0, 14.0, 20.0, 14.0),  # g: middle
    ]
    images = np.zeros((n_total, 28, 28), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int64)
    for idx in range(n_total):
        d = idx % 10
        labels[idx] = d
        dx = rng.uniform(-2.2, 2.2)
        dy = rng.uniform(-2.2, 2.2)
        shear = rng.uniform(-0.18, 0.18)
        scale = rng.uniform(0.86, 1.14)
        sigma = rng.uniform(1.25, 1.85)
        x_t = (xx - 14.0 - dx + shear * (yy - 14.0)) / scale + 14.0
        y_t = (yy - 14.0 - dy) / scale + 14.0
        canvas = np.zeros((28, 28), dtype=np.float32)
        mask = _DIGIT_SEGMENTS[d]
        for seg_on, (x0, y0, x1, y1) in zip(mask, seg_coords):
            if not seg_on:
                continue
            vx, vy = x1 - x0, y1 - y0
            seg_len2 = vx * vx + vy * vy + 1e-8
            t = np.clip(((x_t - x0) * vx + (y_t - y0) * vy) / seg_len2, 0.0, 1.0)
            px = x0 + t * vx
            py = y0 + t * vy
            dist2 = (x_t - px) ** 2 + (y_t - py) ** 2
            stroke = np.exp(-dist2 / (2.0 * sigma * sigma))
            canvas = np.maximum(canvas, stroke)
        noise = rng.randn(28, 28).astype(np.float32) * 0.06
        images[idx] = np.clip((canvas + noise) * 255.0, 0.0, 255.0)
    perm = rng.permutation(n_total)
    return images[perm], labels[perm]


def load_mnist_dataset(
    n_train: int = 4000,
    n_val: int = 800,
    n_test: int = 1200,
    seed: int = 42,
    cache_dir: Path | None = None,
):
    """Load MNIST (28x28 flattened to 784) as float32 in [0, 255] and int64 labels 0..9.

    Downloads and caches `data/mnist.npz` on first use; falls back to synthetic 28x28
    digits only if the network is completely unreachable and no cache exists.
    """
    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "data"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "mnist.npz"

    source = "mnist_npz"
    if not npz_path.exists():
        for url in MNIST_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                tmp_path = npz_path.with_suffix(".npz.tmp")
                tmp_path.write_bytes(data)
                tmp_path.replace(npz_path)
                break
            except Exception:
                continue

    rng = np.random.RandomState(seed)
    if npz_path.exists():
        with np.load(npz_path) as data:
            x_tr_raw = data["x_train"].astype(np.float32)
            y_tr_raw = data["y_train"].astype(np.int64)
            x_te_raw = data["x_test"].astype(np.float32)
            y_te_raw = data["y_test"].astype(np.int64)

        tr_idx = rng.permutation(len(y_tr_raw))[: n_train + n_val]
        te_idx = rng.permutation(len(y_te_raw))[:n_test]

        X_tr_all = x_tr_raw[tr_idx].reshape(-1, 784)
        y_tr_all = y_tr_raw[tr_idx]
        X_train, y_train = X_tr_all[:n_train], y_tr_all[:n_train]
        X_val, y_val = X_tr_all[n_train : n_train + n_val], y_tr_all[n_train : n_train + n_val]
        X_test = x_te_raw[te_idx].reshape(-1, 784)
        y_test = y_te_raw[te_idx]
    else:
        source = "synthetic_fallback"
        total = n_train + n_val + n_test
        imgs, lbls = _synthesize_mnist_fallback(total, seed=seed)
        X_all = imgs.reshape(-1, 784)
        X_train, y_train = X_all[:n_train], lbls[:n_train]
        X_val, y_val = X_all[n_train : n_train + n_val], lbls[n_train : n_train + n_val]
        X_test, y_test = X_all[n_train + n_val :], lbls[n_train + n_val :]

    return X_train, y_train, X_val, y_val, X_test, y_test, source
