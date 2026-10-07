from __future__ import annotations

import numpy as np


class Perceptron:
    """Single-layer perceptron with labels in {+1, -1}."""

    def __init__(self, learning_rate: float = 0.1, n_iters: int = 30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.errors_: list[int] = []
        self.accuracy_: list[float] = []

    def fit(self, X, y, verbose: bool = False, print_every: int = 20):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_iters):
            errors = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1

            self.errors_.append(errors)
            log_this = verbose and ((epoch + 1) % print_every == 0 or errors == 0)
            if log_this:
                acc = float(np.mean(self.predict(X) == y) * 100)
                self.accuracy_.append(acc)
                w_str = ", ".join(f"{wi:+.4f}" for wi in self.w)
                print(
                    f"    Epoch {epoch + 1:5d} | errors = {errors:4d} | "
                    f"accuracy = {acc:6.2f}% | w = [{w_str}] | b = {self.b:+.4f}"
                )
            if errors == 0:
                break
        return self

    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)
