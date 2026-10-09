from __future__ import annotations

import numpy as np


class MLP:
    """From-scratch multilayer perceptron for binary classification."""

    def __init__(
        self,
        layer_sizes,
        learning_rate: float = 0.1,
        n_iters: int = 5000,
        activation: str = "relu",
        seed: int = 42,
        init: str = "auto",
        batch_norm: bool = False,
        l2: float = 0.0,
        dropout: float = 0.0,
    ):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_name = activation
        self.seed = seed
        # auto | he | xavier | zeros | small | large
        self.init = init
        self.batch_norm = batch_norm
        self.l2 = float(l2)
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        self.loss_: list[float] = []
        self.accuracy_: list[float] = []
        self._bn_eps = 1e-5
        self._bn_momentum = 0.9
        self._rng = np.random.RandomState(self.seed + 17)
        self._init_params()

    def _init_params(self):
        rng = np.random.RandomState(self.seed)
        self.W, self.b = [], []
        self.gamma, self.beta = [], []
        self.running_mean, self.running_var = [], []
        scheme = self.init
        if scheme == "auto":
            scheme = "he" if self.activation_name == "relu" else "xavier"
        n_hidden = len(self.layer_sizes) - 2  # BN only on hidden layers
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            if scheme == "zeros":
                self.W.append(np.zeros((n_in, n_out)))
            elif scheme == "small":
                self.W.append(rng.randn(n_in, n_out) * 0.01)
            elif scheme == "large":
                self.W.append(rng.randn(n_in, n_out) * 2.0)
            elif scheme == "he":
                self.W.append(rng.randn(n_in, n_out) * np.sqrt(2.0 / n_in))
            elif scheme == "xavier":
                self.W.append(rng.randn(n_in, n_out) * np.sqrt(1.0 / n_in))
            else:
                raise ValueError(f"Unknown init scheme: {self.init!r}")
            self.b.append(np.zeros(n_out))
            if self.batch_norm and i < n_hidden:
                self.gamma.append(np.ones(n_out))
                self.beta.append(np.zeros(n_out))
                self.running_mean.append(np.zeros(n_out))
                self.running_var.append(np.ones(n_out))

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

    def _bn_forward(self, z, layer_idx: int, training: bool):
        eps = self._bn_eps
        if training:
            mu = np.mean(z, axis=0)
            var = np.var(z, axis=0)
            self.running_mean[layer_idx] = (
                self._bn_momentum * self.running_mean[layer_idx] + (1 - self._bn_momentum) * mu
            )
            self.running_var[layer_idx] = (
                self._bn_momentum * self.running_var[layer_idx] + (1 - self._bn_momentum) * var
            )
        else:
            mu = self.running_mean[layer_idx]
            var = self.running_var[layer_idx]
        inv_std = 1.0 / np.sqrt(var + eps)
        z_hat = (z - mu) * inv_std
        out = self.gamma[layer_idx] * z_hat + self.beta[layer_idx]
        cache = (z, z_hat, mu, var, inv_std)
        return out, cache

    @staticmethod
    def _bn_backward(dout, cache, gamma):
        z, z_hat, mu, var, inv_std = cache
        m = z.shape[0]
        dgamma = np.sum(dout * z_hat, axis=0)
        dbeta = np.sum(dout, axis=0)
        dz_hat = dout * gamma
        dz = (1.0 / m) * inv_std * (
            m * dz_hat
            - np.sum(dz_hat, axis=0)
            - z_hat * np.sum(dz_hat * z_hat, axis=0)
        )
        return dz, dgamma, dbeta

    def _forward(self, X, training: bool = True):
        activations, pre_acts, a = [X], [], X
        bn_caches, drop_masks = [], []
        n_hidden = len(self.W) - 1
        for i in range(n_hidden):
            z = a @ self.W[i] + self.b[i]
            if self.batch_norm:
                z, cache = self._bn_forward(z, i, training=training)
                bn_caches.append(cache)
            else:
                bn_caches.append(None)
            pre_acts.append(z)
            a = self._activate(z)
            if self.dropout > 0.0 and training:
                mask = (self._rng.rand(*a.shape) >= self.dropout).astype(a.dtype)
                a = a * mask / (1.0 - self.dropout)
                drop_masks.append(mask)
            else:
                drop_masks.append(None)
            activations.append(a)
        z_out = a @ self.W[-1] + self.b[-1]
        pre_acts.append(z_out)
        activations.append(self._sigmoid(z_out))
        return activations, pre_acts, bn_caches, drop_masks

    def _backward(self, X, y, activations, pre_acts, bn_caches, drop_masks):
        n_layers, m = len(self.W), X.shape[0]
        dW, db = [None] * n_layers, [None] * n_layers
        dgamma = [None] * len(self.gamma)
        dbeta = [None] * len(self.beta)
        delta = (activations[-1] - y.reshape(-1, 1)) / m
        dW[-1] = activations[-2].T @ delta
        db[-1] = np.sum(delta, axis=0)
        for i in range(n_layers - 2, -1, -1):
            delta = delta @ self.W[i + 1].T
            if drop_masks[i] is not None:
                delta = delta * drop_masks[i] / (1.0 - self.dropout)
            delta = delta * self._activate_grad(pre_acts[i])
            if self.batch_norm:
                delta, dgamma[i], dbeta[i] = self._bn_backward(delta, bn_caches[i], self.gamma[i])
            dW[i] = activations[i].T @ delta
            db[i] = np.sum(delta, axis=0)
        if self.l2 > 0.0:
            for i in range(n_layers):
                dW[i] = dW[i] + (self.l2 / m) * self.W[i]
        return dW, db, dgamma, dbeta

    def _sync_bn_stats(self, X):
        """Overwrite running mean/var using final weights (after last SGD step)."""
        a = X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            self.running_mean[i] = np.mean(z, axis=0)
            self.running_var[i] = np.var(z, axis=0)
            inv_std = 1.0 / np.sqrt(self.running_var[i] + self._bn_eps)
            out = self.gamma[i] * (z - self.running_mean[i]) * inv_std + self.beta[i]
            a = self._activate(out)

    def fit(self, X, y, verbose: bool = False, print_every: int = 500):
        for epoch in range(self.n_iters):
            activations, pre_acts, bn_caches, drop_masks = self._forward(X, training=True)
            a_out = activations[-1].ravel()
            eps = 1e-12
            data_loss = float(-np.mean(y * np.log(a_out + eps) + (1 - y) * np.log(1 - a_out + eps)))
            if self.l2 > 0.0:
                reg = 0.5 * self.l2 * sum(float(np.sum(w * w)) for w in self.W) / X.shape[0]
                loss = data_loss + reg
            else:
                loss = data_loss
            self.loss_.append(loss)
            acc = float(np.mean((a_out >= 0.5).astype(int) == y) * 100)
            self.accuracy_.append(acc)
            dW, db, dgamma, dbeta = self._backward(X, y, activations, pre_acts, bn_caches, drop_masks)
            for i in range(len(self.W)):
                self.W[i] -= self.lr * dW[i]
                self.b[i] -= self.lr * db[i]
            if self.batch_norm:
                for i in range(len(self.gamma)):
                    self.gamma[i] -= self.lr * dgamma[i]
                    self.beta[i] -= self.lr * dbeta[i]
            if verbose and (epoch + 1) % print_every == 0:
                print(f"    Epoch {epoch + 1:5d} | loss = {loss:.6f} | accuracy = {acc:6.2f}%")
        if self.batch_norm:
            self._sync_bn_stats(X)
        return self

    def predict_proba(self, X):
        return self._forward(X, training=False)[0][-1].ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
