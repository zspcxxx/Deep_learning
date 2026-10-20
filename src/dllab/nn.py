"""Small layer library with hand-written backward passes, plus a Sequential model and its training loop.

Every layer has forward(x, training) / backward(dout); layers with parameters list them in
`param_names` and store gradients as d<name> (e.g. W -> dW). Composite layers expose `children`.
"""
from __future__ import annotations

import numpy as _numpy

from dllab.xp import (
    as_device, as_param, asnumpy, cast_params, gpu_enter, gpu_exit, host_np, host_rng, move_params, np,
    on_gpu, prefer_cpu_if_small,
)


class ReLU:
    param_names = ()

    def forward(self, x, training: bool):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class Flatten:
    param_names = ()

    def forward(self, x, training: bool):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)


class Dense:
    param_names = ("W", "b")

    def __init__(self, n_in: int, n_out: int, rng, dtype, init: str = "he"):
        if init == "he":
            w = rng.randn(n_in, n_out) * (2.0 / n_in) ** 0.5
        elif init == "xavier":
            w = rng.randn(n_in, n_out) * (2.0 / (n_in + n_out)) ** 0.5
        elif init == "zeros":
            w = np.zeros((n_in, n_out))
        else:
            raise ValueError(f"unknown init {init!r}")
        self.W = as_param(w, dtype)
        self.b = as_param(host_np.zeros(n_out), dtype)

    def forward(self, x, training: bool):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T


class Dropout:
    param_names = ()

    def __init__(self, p: float, seed: int):
        self.p = float(p)
        self._rng = host_rng(seed)

    def forward(self, x, training: bool):
        if not training or self.p <= 0.0:
            self._mask = None
            return x
        self._mask = np.asarray(self._rng.rand(*x.shape) >= self.p, dtype=x.dtype) / (1.0 - self.p)
        return x * self._mask

    def backward(self, dout):
        return dout if self._mask is None else dout * self._mask


class Residual:
    """y = x + F(x), F = the `children` layers applied in order.

    Backward: dy/dx = I + dF/dx, so the incoming gradient reaches x unchanged through the identity
    path, plus whatever flows back through F.
    """

    param_names = ()

    def __init__(self, children: list[tuple[str, object]]):
        self.children = children

    def forward(self, x, training: bool):
        out = x
        for _, layer in self.children:
            out = layer.forward(out, training)
        return x + out

    def backward(self, dout):
        d = dout
        for _, layer in reversed(self.children):
            d = layer.backward(d)
        return dout + d


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Sequential:
    """Layers applied in order, trained with softmax cross-entropy.

    fit() uses mini-batches, linear warmup + cosine learning-rate decay, decoupled weight decay on
    weight matrices only, AdamW or SGD+momentum, and optional validation early stopping.
    """

    def __init__(self, layers: list, names: list[str], dtype=np.float32):
        self.layers, self.layer_names, self.dtype = layers, names, dtype
        self.loss_: list[float] = []
        self.accuracy_: list[float] = []
        self.val_loss_: list[float] = []
        self.val_accuracy_: list[float] = []
        self.lr_history_: list[float] = []
        self.best_epoch_: int | None = None
        self._gpu_depth = 0

    def _move(self, *, host: bool) -> None:
        move_params(self.parameters(), host=host)

    def parameters(self):
        def walk(prefix, layer):
            for p in layer.param_names:
                yield f"{prefix}.{p}", layer, p
            for cname, child in getattr(layer, "children", []):
                yield from walk(f"{prefix}.{cname}", child)

        for name, layer in zip(self.layer_names, self.layers):
            yield from walk(name, layer)

    @property
    def n_params(self) -> int:
        return int(sum(getattr(layer, p).size for _, layer, p in self.parameters()))

    def forward(self, x, training: bool = False):
        gpu_enter(self, self._move)
        try:
            a = np.asarray(x, dtype=self.dtype)
            for layer in self.layers:
                a = layer.forward(a, training)
            return a
        finally:
            gpu_exit(self, self._move)

    def backward(self, dlogits):
        d = dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d)

    def loss_and_backward(self, x, y):
        """Forward in training mode, softmax-CE loss, then backprop; returns (loss, probs)."""
        gpu_enter(self, self._move)
        try:
            probs = softmax(self.forward(x, training=True))
            m = len(y)
            loss = -np.mean(np.log(probs[np.arange(m), y] + 1e-12))
            d = probs.copy()
            d[np.arange(m), y] -= 1.0
            self.backward(d / m)
            return loss, probs
        finally:
            gpu_exit(self, self._move)

    def predict_proba(self, X, batch_size: int = 500):
        out = [softmax(self.forward(X[i : i + batch_size])) for i in range(0, len(X), batch_size)]
        return asnumpy(np.concatenate(out, axis=0))

    def predict(self, X):
        chunks = [self.forward(X[i : i + 500]) for i in range(0, len(X), 500)]
        return asnumpy(np.argmax(np.concatenate(chunks, axis=0), axis=1))

    def evaluate(self, X, y) -> tuple[float, float]:
        y = as_device(y)
        chunks = [softmax(self.forward(X[i : i + 500])) for i in range(0, len(y), 500)]
        probs = np.concatenate(chunks, axis=0)
        loss = float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))
        acc = float(np.mean(np.argmax(probs, axis=1) == y) * 100.0)
        return loss, acc

    def _snapshot(self):
        return {key: getattr(layer, p).copy() for key, layer, p in self.parameters()}

    def _restore(self, snap):
        for key, layer, p in self.parameters():
            getattr(layer, p)[...] = snap[key]

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        n_epochs: int = 12,
        batch_size: int = 64,
        lr: float = 2e-3,
        min_lr: float = 1e-4,
        warmup_epochs: int = 1,
        weight_decay: float = 1e-2,
        early_stop_patience: int | None = 4,
        optimizer: str = "adamw",
        momentum: float = 0.9,
        augment=None,
        seed: int = 0,
        verbose: bool = False,
        print_every: int = 1,
    ):
        """`augment(batch, rng) -> batch` is applied to each training batch (not to validation)."""
        if optimizer not in ("adamw", "sgd"):
            raise ValueError(f"unknown optimizer {optimizer!r}")
        with prefer_cpu_if_small(getattr(X, "size", 0)):
            return self._fit_on_device(
                X, y, X_val, y_val, n_epochs, batch_size, lr, min_lr, warmup_epochs,
                weight_decay, early_stop_patience, optimizer, momentum, augment, seed,
                verbose, print_every,
            )

    def _fit_on_device(
        self, X, y, X_val, y_val, n_epochs, batch_size, lr, min_lr, warmup_epochs,
        weight_decay, early_stop_patience, optimizer, momentum, augment, seed,
        verbose, print_every,
    ):
        if on_gpu():
            self.dtype = np.float32
            cast_params(self.parameters())
        gpu_enter(self, self._move)
        try:
            return self._fit_loop(
                X, y, X_val, y_val, n_epochs, batch_size, lr, min_lr, warmup_epochs,
                weight_decay, early_stop_patience, optimizer, momentum, augment, seed,
                verbose, print_every,
            )
        finally:
            gpu_exit(self, self._move)

    def _fit_loop(
        self, X, y, X_val, y_val, n_epochs, batch_size, lr, min_lr, warmup_epochs,
        weight_decay, early_stop_patience, optimizer, momentum, augment, seed,
        verbose, print_every,
    ):
        X = np.asarray(X, dtype=self.dtype)
        y = as_device(y)
        rng = host_rng(seed)
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = list(self.parameters())
        state = {key: (np.zeros_like(getattr(layer, p)), np.zeros_like(getattr(layer, p)))
                 for key, layer, p in params}
        step = 0
        steps_per_epoch = int(np.ceil(len(y) / batch_size))
        total_steps = n_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        best_val, best_snap, wait = float("inf"), None, 0

        for epoch in range(n_epochs):
            perm = np.asarray(rng.permutation(len(y)))
            loss_sum = np.zeros((), dtype=self.dtype)
            correct = np.zeros((), dtype=np.int64)
            for start in range(0, len(y), batch_size):
                idx = perm[start : start + batch_size]
                if step < warmup_steps:
                    lr_t = lr * (step + 1) / warmup_steps
                else:
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    lr_t = min_lr + (lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))
                step += 1

                xb = X[idx] if augment is None else augment(X[idx], rng).astype(self.dtype)
                loss, probs = self.loss_and_backward(xb, y[idx])
                loss_sum += loss * len(idx)
                correct += np.sum(np.argmax(probs, axis=1) == y[idx])

                bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
                for key, layer, p in params:
                    param = getattr(layer, p)
                    grad = getattr(layer, "d" + p)
                    m, v = state[key]
                    if p == "W" and weight_decay > 0.0:
                        param *= 1.0 - lr_t * weight_decay
                    if optimizer == "sgd":
                        m *= momentum
                        m += grad
                        param -= lr_t * m
                        continue
                    m *= b1
                    m += (1.0 - b1) * grad
                    v *= b2
                    v += (1.0 - b2) * grad * grad
                    param -= lr_t * (m / bc1) / (np.sqrt(v / bc2) + eps)

            self.lr_history_.append(float(lr_t))
            self.loss_.append(float(loss_sum) / len(y))
            self.accuracy_.append(float(correct) / len(y) * 100.0)
            msg = f"    Epoch {epoch + 1:2d}/{n_epochs} | lr = {lr_t:.5f} | loss = {self.loss_[-1]:.4f} | train_acc = {self.accuracy_[-1]:6.2f}%"

            if X_val is not None:
                vloss, vacc = self.evaluate(X_val, y_val)
                self.val_loss_.append(vloss)
                self.val_accuracy_.append(vacc)
                msg += f" | val_loss = {vloss:.4f} | val_acc = {vacc:6.2f}%"
                if early_stop_patience is not None:
                    if vloss < best_val - 1e-5:
                        best_val, best_snap, wait = vloss, self._snapshot(), 0
                        self.best_epoch_ = epoch
                    else:
                        wait += 1
            if verbose and ((epoch + 1) % print_every == 0 or epoch == 0):
                print(msg)
            if early_stop_patience is not None and wait >= early_stop_patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch + 1} (best={self.best_epoch_ + 1})")
                break

        if best_snap is not None:
            self._restore(best_snap)
        return self


def gradient_check(model: Sequential, x, y, n_checks: int = 8, eps: float = 1e-5, seed: int = 0) -> dict:
    """Compare backprop gradients with central differences on random entries of every parameter.

    Use a float64 model with dropout=0 so the training-mode forward is deterministic.
    """
    rng = host_rng(seed)
    gpu_enter(model, model._move)
    try:
        return _gradient_check(model, x, y, rng, n_checks, eps)
    finally:
        gpu_exit(model, model._move)


def _gradient_check(model, x, y, rng, n_checks, eps) -> dict:
    model.loss_and_backward(x, y)
    analytic = {key: getattr(layer, "d" + p).copy() for key, layer, p in model.parameters()}

    def loss_at():
        probs = softmax(model.forward(x, training=False))
        return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))

    report = {}
    for key, layer, p in model.parameters():
        param = getattr(layer, p)
        flat_idx = rng.choice(param.size, size=min(n_checks, param.size), replace=False)
        errs = []
        for fi in flat_idx:
            idx = tuple(int(v) for v in _numpy.unravel_index(int(fi), param.shape))
            orig = param[idx].copy()
            param[idx] = orig + eps
            lp = loss_at()
            param[idx] = orig - eps
            lm = loss_at()
            param[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[key][idx]
            errs.append(abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        report[key] = float(max(errs))
    return report
