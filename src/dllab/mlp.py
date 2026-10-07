from __future__ import annotations

import numpy as np


class MLP:
    """From-scratch multilayer perceptron for binary classification."""

    def __init__(self, layer_sizes, learning_rate: float = 0.1, n_iters: int = 5000, activation: str = "relu", seed: int = 42):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_name = activation
        self.seed = seed
        self.loss_: list[float] = []
        self.accuracy_: list[float] = []
        self._init_params()

    def _init_params(self):
        rng = np.random.RandomState(self.seed)
        self.W, self.b = [], []
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            scale = np.sqrt(2.0 / n_in) if self.activation_name == "relu" else np.sqrt(1.0 / n_in)
            self.W.append(rng.randn(n_in, n_out) * scale)
            self.b.append(np.zeros(n_out))

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_grad(z):
        return (z > 0).astype(float)

    @staticmethod
    def _tanh(z):
        return np.tanh(z)

    @staticmethod
    def _tanh_grad(z):
        return 1.0 - np.tanh(z) ** 2

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _activate(self, z):
        return self._relu(z) if self.activation_name == "relu" else self._tanh(z)

    def _activate_grad(self, z):
        return self._relu_grad(z) if self.activation_name == "relu" else self._tanh_grad(z)

    def _forward(self, X):
        activations, pre_acts, a = [X], [], X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            pre_acts.append(z)
            a = self._activate(z)
            activations.append(a)
        z_out = a @ self.W[-1] + self.b[-1]
        pre_acts.append(z_out)
        activations.append(self._sigmoid(z_out))
        return activations, pre_acts

    def _backward(self, X, y, activations, pre_acts):
        n_layers, m = len(self.W), X.shape[0]
        dW, db = [None] * n_layers, [None] * n_layers
        delta = (activations[-1] - y.reshape(-1, 1)) / m
        dW[-1] = activations[-2].T @ delta
        db[-1] = np.sum(delta, axis=0)
        for i in range(n_layers - 2, -1, -1):
            delta = (delta @ self.W[i + 1].T) * self._activate_grad(pre_acts[i])
            dW[i] = activations[i].T @ delta
            db[i] = np.sum(delta, axis=0)
        return dW, db

    def fit(self, X, y, verbose: bool = False, print_every: int = 500):
        for epoch in range(self.n_iters):
            activations, pre_acts = self._forward(X)
            a_out = activations[-1].ravel()
            eps = 1e-12
            loss = float(-np.mean(y * np.log(a_out + eps) + (1 - y) * np.log(1 - a_out + eps)))
            self.loss_.append(loss)
            acc = float(np.mean((a_out >= 0.5).astype(int) == y) * 100)
            self.accuracy_.append(acc)
            dW, db = self._backward(X, y, activations, pre_acts)
            for i in range(len(self.W)):
                self.W[i] -= self.lr * dW[i]
                self.b[i] -= self.lr * db[i]
            if verbose and (epoch + 1) % print_every == 0:
                print(f"    Epoch {epoch + 1:5d} | loss = {loss:.6f} | accuracy = {acc:6.2f}%")
        return self

    def predict_proba(self, X):
        return self._forward(X)[0][-1].ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
