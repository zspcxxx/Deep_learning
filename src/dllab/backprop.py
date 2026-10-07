from __future__ import annotations

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


class BackpropNN:
    """Two-layer network with handwritten forward/backward pass."""

    def __init__(self, n_in: int = 2, n_hidden: int = 4, learning_rate: float = 0.5, n_iters: int = 5000, activation: str = "sigmoid", seed: int = 42):
        self.n_in, self.n_hidden = n_in, n_hidden
        self.lr, self.n_iters = learning_rate, n_iters
        self.activation_name = activation
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) * np.sqrt(1.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, 1) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros(1)
        self.loss_: list[float] = []
        self.accuracy_: list[float] = []

    def _activate(self, z):
        return sigmoid(z) if self.activation_name == "sigmoid" else relu(z)

    def _activate_grad(self, z):
        return sigmoid_grad(z) if self.activation_name == "sigmoid" else relu_grad(z)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._activate(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)
        return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}

    @staticmethod
    def compute_loss(a2, y):
        y = y.reshape(-1, 1)
        eps = 1e-12
        return float(-np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps)))

    def backward(self, X, y, cache):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        dz2 = (cache["a2"] - y) / m
        dW2 = cache["a1"].T @ dz2
        db2 = np.sum(dz2, axis=0)
        dz1 = (dz2 @ self.W2.T) * self._activate_grad(cache["z1"])
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_params(self, grads):
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]

    def fit(self, X, y, verbose: bool = False, print_every: int = 500):
        for epoch in range(self.n_iters):
            cache = self.forward(X)
            loss = self.compute_loss(cache["a2"], y)
            self.loss_.append(loss)
            acc = float(np.mean((cache["a2"].ravel() >= 0.5).astype(int) == y) * 100)
            self.accuracy_.append(acc)
            self.update_params(self.backward(X, y, cache))
            if verbose and (epoch + 1) % print_every == 0:
                print(f"    Epoch {epoch + 1:5d} | loss = {loss:.6f} | accuracy = {acc:6.2f}%")
        return self

    def predict_proba(self, X):
        return self.forward(X)["a2"].ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def gradient_check(model: BackpropNN, X, y, eps: float = 1e-5):
    cache = model.forward(X)
    grads = model.backward(X, y, cache)
    results = {}
    for name in ["W1", "b1", "W2", "b2"]:
        param = getattr(model, name)
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = param[idx]
            param[idx] = old + eps
            loss_plus = model.compute_loss(model.forward(X)["a2"], y)
            param[idx] = old - eps
            loss_minus = model.compute_loss(model.forward(X)["a2"], y)
            param[idx] = old
            num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        ana_grad = grads["d" + name]
        rel_err = np.abs(num_grad - ana_grad) / (np.abs(num_grad) + np.abs(ana_grad) + 1e-8)
        results[name] = {
            "num": num_grad,
            "ana": ana_grad,
            "rel_err": rel_err,
            "max_rel_err": float(np.max(rel_err)),
        }
    return results
