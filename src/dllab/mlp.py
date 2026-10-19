from __future__ import annotations

from dllab.xp import (
    GPU, as_compute, as_device, as_param, asnumpy, cast_lists, gpu_enter, gpu_exit, host_rng, move_lists, np,
    prefer_cpu_if_small,
)

_MLP_LISTS = (
    "W", "b", "gamma", "beta", "running_mean", "running_var",
    "_mW", "_vW", "_mb", "_vb", "_mgamma", "_vgamma", "_mbeta", "_vbeta",
)


def _mlp_move(model, *, host: bool) -> None:
    move_lists(model, _MLP_LISTS, host=host)


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
        l1: float = 0.0,
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
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        self.loss_: list[float] = []
        self.accuracy_: list[float] = []
        self.val_loss_: list[float] = []
        self.best_epoch_: int | None = None
        self._bn_eps = 1e-5
        self._bn_momentum = 0.9
        self._rng = host_rng(self.seed + 17)
        self._init_params()

    def _init_params(self):
        rng = host_rng(self.seed)
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
                self.W.append(as_param(np.zeros((n_in, n_out))))
            elif scheme == "small":
                self.W.append(as_param(rng.randn(n_in, n_out) * 0.01))
            elif scheme == "large":
                self.W.append(as_param(rng.randn(n_in, n_out) * 2.0))
            elif scheme == "he":
                self.W.append(as_param(rng.randn(n_in, n_out) * (2.0 / n_in) ** 0.5))
            elif scheme == "xavier":
                self.W.append(as_param(rng.randn(n_in, n_out) * (1.0 / n_in) ** 0.5))
            else:
                raise ValueError(f"Unknown init scheme: {self.init!r}")
            self.b.append(as_param(np.zeros(n_out)))
            if self.batch_norm and i < n_hidden:
                self.gamma.append(as_param(np.ones(n_out)))
                self.beta.append(as_param(np.zeros(n_out)))
                self.running_mean.append(as_param(np.zeros(n_out)))
                self.running_var.append(as_param(np.ones(n_out)))

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
        X = as_compute(X)
        gpu_enter(self, lambda **kw: _mlp_move(self, **kw))
        try:
            return self._forward_body(X, training)
        finally:
            gpu_exit(self, lambda **kw: _mlp_move(self, **kw))

    def _forward_body(self, X, training: bool = True):
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
                mask = np.asarray(self._rng.rand(*a.shape) >= self.dropout, dtype=a.dtype)
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
        if self.l1 > 0.0:
            for i in range(n_layers):
                dW[i] = dW[i] + (self.l1 / m) * np.sign(self.W[i])
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

    def _data_loss(self, X, y) -> float:
        a_out = self._forward(X, training=False)[0][-1].ravel()
        eps = 1e-12
        return float(-np.mean(y * np.log(a_out + eps) + (1 - y) * np.log(1 - a_out + eps)))

    def _snapshot_weights(self):
        snap = {
            "W": [w.copy() for w in self.W],
            "b": [b.copy() for b in self.b],
            "gamma": [g.copy() for g in self.gamma],
            "beta": [b.copy() for b in self.beta],
            "running_mean": [m.copy() for m in self.running_mean],
            "running_var": [v.copy() for v in self.running_var],
        }
        return snap

    def _restore_weights(self, snap):
        self.W = [w.copy() for w in snap["W"]]
        self.b = [b.copy() for b in snap["b"]]
        self.gamma = [g.copy() for g in snap["gamma"]]
        self.beta = [b.copy() for b in snap["beta"]]
        self.running_mean = [m.copy() for m in snap["running_mean"]]
        self.running_var = [v.copy() for v in snap["running_var"]]

    def fit(
        self,
        X,
        y,
        verbose: bool = False,
        print_every: int = 500,
        X_val=None,
        y_val=None,
        early_stop_patience: int | None = None,
    ):
        with prefer_cpu_if_small(getattr(X, "size", 0)):
            return self._fit_small(X, y, verbose, print_every, X_val, y_val, early_stop_patience)

    def _fit_small(self, X, y, verbose, print_every, X_val, y_val, early_stop_patience):
        X, y = as_compute(X), as_compute(y)
        if X_val is not None:
            X_val, y_val = as_compute(X_val), as_compute(y_val)
        if GPU:
            self.W, self.b, self.gamma, self.beta, self.running_mean, self.running_var = cast_lists(
                self.W, self.b, self.gamma, self.beta, self.running_mean, self.running_var
            )
        gpu_enter(self, lambda **kw: _mlp_move(self, **kw))
        try:
            return self._fit_body(X, y, X_val, y_val, early_stop_patience, verbose, print_every)
        finally:
            gpu_exit(self, lambda **kw: _mlp_move(self, **kw))

    def _fit_body(self, X, y, X_val, y_val, early_stop_patience, verbose, print_every):
        best_val = float("inf")
        best_snap = None
        wait = 0
        self.val_loss_ = []
        self.best_epoch_ = None
        for epoch in range(self.n_iters):
            activations, pre_acts, bn_caches, drop_masks = self._forward(X, training=True)
            a_out = activations[-1].ravel()
            eps = 1e-12
            data_loss = float(-np.mean(y * np.log(a_out + eps) + (1 - y) * np.log(1 - a_out + eps)))
            reg = 0.0
            m = X.shape[0]
            if self.l2 > 0.0:
                reg += 0.5 * self.l2 * sum(float(np.sum(w * w)) for w in self.W) / m
            if self.l1 > 0.0:
                reg += self.l1 * sum(float(np.sum(np.abs(w))) for w in self.W) / m
            loss = data_loss + reg
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

            if X_val is not None and y_val is not None:
                vloss = self._data_loss(X_val, y_val)
                self.val_loss_.append(vloss)
                if early_stop_patience is not None:
                    if vloss < best_val - 1e-6:
                        best_val = vloss
                        best_snap = self._snapshot_weights()
                        self.best_epoch_ = epoch
                        wait = 0
                    else:
                        wait += 1
                        if wait >= early_stop_patience:
                            if verbose:
                                print(f"    Early stop at epoch {epoch + 1} (best={self.best_epoch_ + 1})")
                            break

            if verbose and (epoch + 1) % print_every == 0:
                extra = f" | val = {self.val_loss_[-1]:.6f}" if self.val_loss_ else ""
                print(f"    Epoch {epoch + 1:5d} | loss = {loss:.6f} | accuracy = {acc:6.2f}%{extra}")

        if best_snap is not None:
            self._restore_weights(best_snap)
        if self.batch_norm:
            self._sync_bn_stats(X)
        return self

    def predict_proba(self, X):
        return asnumpy(self._forward(as_compute(X), training=False)[0][-1].ravel())

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class MulticlassMLP:
    """Multi-class deep neural network implementing all 11 core techniques:
    - Parameter Init: He / Xavier (plus zeros / small / large for ablation)
    - Activations: ReLU / GELU / SiLU / Sigmoid / Tanh
    - Normalization: BatchNorm (BN) / LayerNorm (LN) / RMSNorm
    - Dropout: Inverted dropout on hidden activations
    - Loss Function: Cross-Entropy (Softmax) / MSE
    - L1 / L2 Regularization: Explicit penalty added to loss & gradients
    - Backpropagation: Exact analytical chain-rule through all layers/norms
    - Optimizers: SGD, SGD+Momentum, AdamW
    - LR Schedule: Warmup + Cosine Annealing (or constant)
    - Weight Decay: Decoupled weight decay in AdamW
    - Early Stopping: Validation loss monitoring with best-checkpoint restore
    """

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 3e-3,
        min_lr: float = 1e-4,
        n_epochs: int = 40,
        batch_size: int = 128,
        warmup_epochs: int = 5,
        lr_schedule: str = "warmup_cosine",  # "constant" | "warmup_cosine"
        activation: str = "gelu",            # "relu" | "gelu" | "silu" | "sigmoid" | "tanh"
        init: str = "he",                    # "he" | "xavier" | "zeros" | "small" | "large"
        norm: str | None = "rmsnorm",        # None | "bn" | "ln" | "rmsnorm"
        loss_type: str = "ce",               # "ce" (Cross-Entropy) | "mse"
        optimizer: str = "adamw",            # "sgd" | "momentum" | "adamw"
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 1e-2,          # AdamW decoupled weight decay
        l1: float = 0.0,
        l2: float = 0.0,
        dropout: float = 0.15,
        seed: int = 42,
    ):
        self.layer_sizes = list(layer_sizes)
        self.lr = float(learning_rate)
        self.min_lr = float(min_lr)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.warmup_epochs = int(warmup_epochs)
        self.lr_schedule = lr_schedule
        self.activation_name = activation.lower()
        self.init = init.lower()
        self.norm = norm.lower() if norm else None
        self.loss_type = loss_type.lower()
        self.optimizer = optimizer.lower()
        self.momentum = float(momentum)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.weight_decay = float(weight_decay)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.dropout = float(dropout)
        self.seed = int(seed)

        self.loss_: list[float] = []
        self.val_loss_: list[float] = []
        self.accuracy_: list[float] = []
        self.val_accuracy_: list[float] = []
        self.lr_history_: list[float] = []
        self.best_epoch_: int | None = None

        self._eps = 1e-5
        self._bn_momentum = 0.9
        self._rng = host_rng(self.seed + 19)
        self._step = 0
        self._init_params()

    def _init_params(self):
        rng = host_rng(self.seed)
        self.W, self.b = [], []
        self.gamma, self.beta = [], []
        self.running_mean, self.running_var = [], []
        n_hidden = len(self.layer_sizes) - 2

        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            # Use Xavier on the output classification head when hidden layers use He
            scheme = "xavier" if (i == n_hidden and self.init == "he") else self.init
            if scheme == "zeros":
                w = as_param(np.zeros((n_in, n_out)), np.float64)
            elif scheme == "small":
                w = as_param(rng.randn(n_in, n_out) * 0.01)
            elif scheme == "large":
                w = as_param(rng.randn(n_in, n_out) * 1.2)
            elif scheme == "he":
                w = as_param(rng.randn(n_in, n_out) * (2.0 / n_in) ** 0.5)
            elif scheme == "xavier":
                w = as_param(rng.randn(n_in, n_out) * (2.0 / (n_in + n_out)) ** 0.5)
            else:
                raise ValueError(f"Unknown init scheme: {self.init!r}")
            self.W.append(w)
            self.b.append(as_param(np.zeros(n_out), np.float64))

            if self.norm and i < n_hidden:
                self.gamma.append(as_param(np.ones(n_out), np.float64))
                self.beta.append(as_param(np.zeros(n_out), np.float64))
                if self.norm == "bn":
                    self.running_mean.append(as_param(np.zeros(n_out), np.float64))
                    self.running_var.append(as_param(np.ones(n_out), np.float64))

        # Optimizer state
        self._mW = [as_param(np.zeros_like(w)) for w in self.W]
        self._vW = [as_param(np.zeros_like(w)) for w in self.W]
        self._mb = [as_param(np.zeros_like(b)) for b in self.b]
        self._vb = [as_param(np.zeros_like(b)) for b in self.b]
        self._mgamma = [as_param(np.zeros_like(g)) for g in self.gamma]
        self._vgamma = [as_param(np.zeros_like(g)) for g in self.gamma]
        self._mbeta = [as_param(np.zeros_like(bt)) for bt in self.beta]
        self._vbeta = [as_param(np.zeros_like(bt)) for bt in self.beta]

    def _activate(self, z):
        name = self.activation_name
        if name == "relu":
            return np.maximum(0.0, z)
        if name == "gelu":
            c = np.sqrt(2.0 / np.pi)
            return 0.5 * z * (1.0 + np.tanh(c * (z + 0.044715 * z ** 3)))
        if name == "silu":
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
            return z * s
        if name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
        if name == "tanh":
            return np.tanh(z)
        raise ValueError(f"Unknown activation: {name!r}")

    def _activate_grad(self, z):
        name = self.activation_name
        if name == "relu":
            return (z > 0.0).astype(z.dtype)
        if name == "gelu":
            c = np.sqrt(2.0 / np.pi)
            inner = c * (z + 0.044715 * z ** 3)
            t = np.tanh(inner)
            cdf = 0.5 * (1.0 + t)
            pdf = 0.5 * z * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * z * z)
            return cdf + pdf
        if name == "silu":
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
            return s + z * s * (1.0 - s)
        if name == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
            return s * (1.0 - s)
        if name == "tanh":
            return 1.0 - np.tanh(z) ** 2
        raise ValueError(f"Unknown activation: {name!r}")

    @staticmethod
    def _softmax(z):
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(np.clip(z_shift, -500.0, 500.0))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _norm_forward(self, z, layer_idx: int, training: bool):
        eps = self._eps
        if self.norm == "bn":
            if training:
                mu = np.mean(z, axis=0)
                var = np.var(z, axis=0)
                self.running_mean[layer_idx] = (
                    self._bn_momentum * self.running_mean[layer_idx]
                    + (1.0 - self._bn_momentum) * mu
                )
                self.running_var[layer_idx] = (
                    self._bn_momentum * self.running_var[layer_idx]
                    + (1.0 - self._bn_momentum) * var
                )
            else:
                mu = self.running_mean[layer_idx]
                var = self.running_var[layer_idx]
            inv_std = 1.0 / np.sqrt(var + eps)
            z_hat = (z - mu) * inv_std
            out = self.gamma[layer_idx] * z_hat + self.beta[layer_idx]
            return out, ("bn", z_hat, inv_std)

        if self.norm == "ln":
            mu = np.mean(z, axis=1, keepdims=True)
            var = np.var(z, axis=1, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + eps)
            z_hat = (z - mu) * inv_std
            out = self.gamma[layer_idx] * z_hat + self.beta[layer_idx]
            return out, ("ln", z_hat, inv_std)

        if self.norm == "rmsnorm":
            ms = np.mean(z * z, axis=1, keepdims=True)
            inv_rms = 1.0 / np.sqrt(ms + eps)
            z_hat = z * inv_rms
            out = self.gamma[layer_idx] * z_hat
            return out, ("rmsnorm", z_hat, inv_rms)

        return z, None

    def _norm_backward(self, dout, cache, layer_idx: int):
        kind = cache[0]
        gamma = self.gamma[layer_idx]
        if kind == "bn":
            _, z_hat, inv_std = cache
            m = dout.shape[0]
            dgamma = np.sum(dout * z_hat, axis=0)
            dbeta = np.sum(dout, axis=0)
            dz_hat = dout * gamma
            dz = (1.0 / m) * inv_std * (
                m * dz_hat
                - np.sum(dz_hat, axis=0)
                - z_hat * np.sum(dz_hat * z_hat, axis=0)
            )
            return dz, dgamma, dbeta

        if kind == "ln":
            _, z_hat, inv_std = cache
            d = dout.shape[1]
            dgamma = np.sum(dout * z_hat, axis=0)
            dbeta = np.sum(dout, axis=0)
            dz_hat = dout * gamma
            dz = (1.0 / d) * inv_std * (
                d * dz_hat
                - np.sum(dz_hat, axis=1, keepdims=True)
                - z_hat * np.sum(dz_hat * z_hat, axis=1, keepdims=True)
            )
            return dz, dgamma, dbeta

        if kind == "rmsnorm":
            _, z_hat, inv_rms = cache
            dgamma = np.sum(dout * z_hat, axis=0)
            dbeta = np.zeros_like(gamma)
            dz_hat = dout * gamma
            dz = inv_rms * (
                dz_hat - z_hat * np.mean(dz_hat * z_hat, axis=1, keepdims=True)
            )
            return dz, dgamma, dbeta

        raise ValueError(f"Unknown norm cache kind: {kind!r}")

    def _forward(self, X, training: bool = True):
        gpu_enter(self, lambda **kw: _mlp_move(self, **kw))
        try:
            return self._forward_body(X, training)
        finally:
            gpu_exit(self, lambda **kw: _mlp_move(self, **kw))

    def _forward_body(self, X, training: bool = True):
        activations = [X]
        pre_acts = []
        norm_caches = []
        drop_masks = []
        a = X
        n_hidden = len(self.W) - 1

        for i in range(n_hidden):
            z = a @ self.W[i] + self.b[i]
            if self.norm:
                z, ncache = self._norm_forward(z, i, training=training)
                norm_caches.append(ncache)
            else:
                norm_caches.append(None)
            pre_acts.append(z)
            a = self._activate(z)
            if self.dropout > 0.0 and training:
                mask = np.asarray(self._rng.rand(*a.shape) >= self.dropout, dtype=a.dtype)
                a = a * mask / (1.0 - self.dropout)
                drop_masks.append(mask)
            else:
                drop_masks.append(None)
            activations.append(a)

        logits = a @ self.W[-1] + self.b[-1]
        pre_acts.append(logits)
        probs = self._softmax(logits)
        activations.append(probs)
        return activations, pre_acts, norm_caches, drop_masks

    def _compute_loss_and_delta(self, probs, y):
        m, k = probs.shape
        y_onehot = np.zeros((m, k), dtype=probs.dtype)
        y_onehot[np.arange(m), y] = 1.0

        if self.loss_type == "ce":
            eps = 1e-12
            data_loss = float(-np.mean(np.log(probs[np.arange(m), y] + eps)))
            delta = (probs - y_onehot) / m
        elif self.loss_type == "mse":
            diff = probs - y_onehot
            data_loss = float(0.5 * np.mean(np.sum(diff * diff, axis=1)))
            # dL/dz for softmax + MSE: p * ((p - y) - sum((p - y)*p)) / m
            dot = np.sum(diff * probs, axis=1, keepdims=True)
            delta = probs * (diff - dot) / m
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type!r}")

        reg = 0.0
        if self.l2 > 0.0:
            reg += 0.5 * self.l2 * sum(float(np.sum(w * w)) for w in self.W) / m
        if self.l1 > 0.0:
            reg += self.l1 * sum(float(np.sum(np.abs(w))) for w in self.W) / m
        return data_loss + reg, data_loss, delta

    def _backward(self, delta, activations, pre_acts, norm_caches, drop_masks):
        n_layers = len(self.W)
        m = activations[0].shape[0]
        dW = [None] * n_layers
        db = [None] * n_layers
        dgamma = [None] * len(self.gamma)
        dbeta = [None] * len(self.beta)

        dW[-1] = activations[-2].T @ delta
        db[-1] = np.sum(delta, axis=0)

        for i in range(n_layers - 2, -1, -1):
            delta = delta @ self.W[i + 1].T
            if drop_masks[i] is not None:
                delta = delta * drop_masks[i] / (1.0 - self.dropout)
            delta = delta * self._activate_grad(pre_acts[i])
            if self.norm:
                delta, dgamma[i], dbeta[i] = self._norm_backward(delta, norm_caches[i], i)
            dW[i] = activations[i].T @ delta
            db[i] = np.sum(delta, axis=0)

        if self.l2 > 0.0:
            for i in range(n_layers):
                dW[i] = dW[i] + (self.l2 / m) * self.W[i]
        if self.l1 > 0.0:
            for i in range(n_layers):
                dW[i] = dW[i] + (self.l1 / m) * np.sign(self.W[i])

        return dW, db, dgamma, dbeta

    def _get_lr(self, epoch: int) -> float:
        if self.lr_schedule != "warmup_cosine":
            return self.lr
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return self.lr * float(epoch + 1) / float(self.warmup_epochs)
        progress = float(epoch - self.warmup_epochs) / max(1.0, float(self.n_epochs - self.warmup_epochs))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr + (self.lr - self.min_lr) * float(cosine)

    def _apply_gradients(self, dW, db, dgamma, dbeta, lr_t: float):
        self._step += 1
        t = self._step
        eps = 1e-8

        if self.optimizer == "sgd":
            for i in range(len(self.W)):
                self.W[i] -= lr_t * dW[i]
                self.b[i] -= lr_t * db[i]
            for i in range(len(self.gamma)):
                self.gamma[i] -= lr_t * dgamma[i]
                self.beta[i] -= lr_t * dbeta[i]
            return

        if self.optimizer == "momentum":
            mu = self.momentum
            for i in range(len(self.W)):
                if self.weight_decay > 0.0:
                    self.W[i] *= (1.0 - lr_t * self.weight_decay)
                self._mW[i] = mu * self._mW[i] + dW[i]
                self._mb[i] = mu * self._mb[i] + db[i]
                self.W[i] -= lr_t * self._mW[i]
                self.b[i] -= lr_t * self._mb[i]
            for i in range(len(self.gamma)):
                self._mgamma[i] = mu * self._mgamma[i] + dgamma[i]
                self._mbeta[i] = mu * self._mbeta[i] + dbeta[i]
                self.gamma[i] -= lr_t * self._mgamma[i]
                self.beta[i] -= lr_t * self._mbeta[i]
            return

        if self.optimizer == "adamw":
            b1, b2 = self.beta1, self.beta2
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
            for i in range(len(self.W)):
                # Decoupled weight decay on weights (AdamW)
                if self.weight_decay > 0.0:
                    self.W[i] *= (1.0 - lr_t * self.weight_decay)
                self._mW[i] = b1 * self._mW[i] + (1.0 - b1) * dW[i]
                self._vW[i] = b2 * self._vW[i] + (1.0 - b2) * (dW[i] * dW[i])
                m_hat = self._mW[i] / bc1
                v_hat = self._vW[i] / bc2
                self.W[i] -= lr_t * m_hat / (np.sqrt(v_hat) + eps)

                self._mb[i] = b1 * self._mb[i] + (1.0 - b1) * db[i]
                self._vb[i] = b2 * self._vb[i] + (1.0 - b2) * (db[i] * db[i])
                mb_hat = self._mb[i] / bc1
                vb_hat = self._vb[i] / bc2
                self.b[i] -= lr_t * mb_hat / (np.sqrt(vb_hat) + eps)

            for i in range(len(self.gamma)):
                self._mgamma[i] = b1 * self._mgamma[i] + (1.0 - b1) * dgamma[i]
                self._vgamma[i] = b2 * self._vgamma[i] + (1.0 - b2) * (dgamma[i] * dgamma[i])
                mg_hat = self._mgamma[i] / bc1
                vg_hat = self._vgamma[i] / bc2
                self.gamma[i] -= lr_t * mg_hat / (np.sqrt(vg_hat) + eps)

                self._mbeta[i] = b1 * self._mbeta[i] + (1.0 - b1) * dbeta[i]
                self._vbeta[i] = b2 * self._vbeta[i] + (1.0 - b2) * (dbeta[i] * dbeta[i])
                mbt_hat = self._mbeta[i] / bc1
                vbt_hat = self._vbeta[i] / bc2
                self.beta[i] -= lr_t * mbt_hat / (np.sqrt(vbt_hat) + eps)
            return

        raise ValueError(f"Unknown optimizer: {self.optimizer!r}")

    def _snapshot(self):
        return {
            "W": [w.copy() for w in self.W],
            "b": [b.copy() for b in self.b],
            "gamma": [g.copy() for g in self.gamma],
            "beta": [bt.copy() for bt in self.beta],
            "running_mean": [m.copy() for m in self.running_mean],
            "running_var": [v.copy() for v in self.running_var],
        }

    def _restore(self, snap):
        self.W = [w.copy() for w in snap["W"]]
        self.b = [b.copy() for b in snap["b"]]
        self.gamma = [g.copy() for g in snap["gamma"]]
        self.beta = [bt.copy() for bt in snap["beta"]]
        self.running_mean = [m.copy() for m in snap["running_mean"]]
        self.running_var = [v.copy() for v in snap["running_var"]]

    def evaluate(self, X, y) -> tuple[float, float]:
        gpu_enter(self, lambda **kw: _mlp_move(self, **kw))
        try:
            probs = self._forward(as_compute(X), training=False)[0][-1]
            y = as_device(y)
            _, data_loss, _ = self._compute_loss_and_delta(probs, y)
            preds = np.argmax(probs, axis=1)
            acc = float(np.mean(preds == y) * 100.0)
            return data_loss, acc
        finally:
            gpu_exit(self, lambda **kw: _mlp_move(self, **kw))

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        early_stop_patience: int | None = None,
        verbose: bool = False,
        print_every: int = 10,
    ):
        with prefer_cpu_if_small(getattr(X, "size", 0)):
            return self._fit_small(X, y, X_val, y_val, early_stop_patience, verbose, print_every)

    def _fit_small(self, X, y, X_val, y_val, early_stop_patience, verbose, print_every):
        X = as_compute(X)
        y = as_device(y)
        if X_val is not None:
            X_val = as_compute(X_val)
            y_val = as_device(y_val)
        if GPU:
            self.W, self.b, self.gamma, self.beta, self.running_mean, self.running_var = cast_lists(
                self.W, self.b, self.gamma, self.beta, self.running_mean, self.running_var
            )
            self._mW, self._vW, self._mb, self._vb, self._mgamma, self._vgamma, self._mbeta, self._vbeta = cast_lists(
                self._mW, self._vW, self._mb, self._vb, self._mgamma, self._vgamma, self._mbeta, self._vbeta
            )
        gpu_enter(self, lambda **kw: _mlp_move(self, **kw))
        try:
            return self._fit_body(X, y, X_val, y_val, early_stop_patience, verbose, print_every)
        finally:
            gpu_exit(self, lambda **kw: _mlp_move(self, **kw))

    def _fit_body(self, X, y, X_val, y_val, early_stop_patience, verbose, print_every):
        n = X.shape[0]
        rng = host_rng(self.seed + 101)
        best_val = float("inf")
        best_snap = None
        wait = 0
        self.best_epoch_ = None

        for epoch in range(self.n_epochs):
            lr_t = self._get_lr(epoch)
            self.lr_history_.append(lr_t)
            perm = np.asarray(rng.permutation(n))
            epoch_loss = 0.0
            epoch_correct = 0

            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb, yb = X[idx], y[idx]
                activations, pre_acts, norm_caches, drop_masks = self._forward(xb, training=True)
                probs = activations[-1]
                loss, _, delta = self._compute_loss_and_delta(probs, yb)
                dW, db, dgamma, dbeta = self._backward(delta, activations, pre_acts, norm_caches, drop_masks)
                self._apply_gradients(dW, db, dgamma, dbeta, lr_t)

                bs = len(yb)
                epoch_loss += loss * bs
                epoch_correct += int(np.sum(np.argmax(probs, axis=1) == yb))

            train_loss = epoch_loss / n
            train_acc = (epoch_correct / n) * 100.0
            self.loss_.append(float(train_loss))
            self.accuracy_.append(float(train_acc))

            if X_val is not None and y_val is not None:
                vloss, vacc = self.evaluate(X_val, y_val)
                self.val_loss_.append(float(vloss))
                self.val_accuracy_.append(float(vacc))
                if early_stop_patience is not None:
                    if vloss < best_val - 1e-5:
                        best_val = vloss
                        best_snap = self._snapshot()
                        self.best_epoch_ = epoch
                        wait = 0
                    else:
                        wait += 1
                        if wait >= early_stop_patience:
                            if verbose:
                                print(
                                    f"    Early stop at epoch {epoch + 1} "
                                    f"(best={self.best_epoch_ + 1}, val_acc={self.val_accuracy_[self.best_epoch_]:.2f}%)"
                                )
                            break

            if verbose and ((epoch + 1) % print_every == 0 or epoch == 0):
                extra = ""
                if self.val_loss_:
                    extra = f" | val_loss = {self.val_loss_[-1]:.4f} | val_acc = {self.val_accuracy_[-1]:6.2f}%"
                print(
                    f"    Epoch {epoch + 1:3d}/{self.n_epochs} | lr = {lr_t:.5f} | "
                    f"loss = {train_loss:.4f} | train_acc = {train_acc:6.2f}%{extra}"
                )

        if best_snap is not None:
            self._restore(best_snap)
        return self

    def predict_proba(self, X):
        X = as_compute(X)
        return asnumpy(self._forward(X, training=False)[0][-1])

    def predict(self, X):
        return asnumpy(np.argmax(self._forward(as_compute(X), training=False)[0][-1], axis=1))
