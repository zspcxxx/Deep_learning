from __future__ import annotations

import numpy as np


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
