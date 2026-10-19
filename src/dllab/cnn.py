"""From-scratch convolutional network (NumPy only, NCHW layout).

Conv2D uses im2col so that convolution becomes one matrix multiply; its backward pass
reuses the same columns for dW and scatters dcols back with col2im.
"""
from __future__ import annotations

import numpy as np


def im2col(x: np.ndarray, k: int, pad: int):
    """(N, C, H, W) -> (N*Ho*Wo, C*k*k) patch matrix for a stride-1 k x k window."""
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    ho, wo = h + 2 * pad - k + 1, w + 2 * pad - k + 1
    s0, s1, s2, s3 = xp.strides
    windows = np.lib.stride_tricks.as_strided(
        xp, shape=(n, c, k, k, ho, wo), strides=(s0, s1, s2, s3, s2, s3), writeable=False
    )
    cols = windows.transpose(0, 4, 5, 1, 2, 3).reshape(n * ho * wo, c * k * k)
    return cols, ho, wo


def col2im(dcols: np.ndarray, x_shape: tuple, k: int, pad: int, ho: int, wo: int) -> np.ndarray:
    """Inverse scatter of im2col: overlapping patches accumulate their gradients."""
    n, c, h, w = x_shape
    d = dcols.reshape(n, ho, wo, c, k, k).transpose(0, 3, 4, 5, 1, 2)
    dxp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=dcols.dtype)
    for i in range(k):
        for j in range(k):
            dxp[:, :, i : i + ho, j : j + wo] += d[:, :, i, j]
    return dxp[:, :, pad : pad + h, pad : pad + w] if pad else dxp


class Conv2D:
    param_names = ("W", "b")

    def __init__(self, c_in: int, c_out: int, k: int, pad: int, rng, dtype):
        self.k, self.pad = k, pad
        self.W = (rng.randn(c_out, c_in, k, k) * np.sqrt(2.0 / (c_in * k * k))).astype(dtype)
        self.b = np.zeros(c_out, dtype=dtype)

    def forward(self, x, training: bool):
        cols, ho, wo = im2col(x, self.k, self.pad)
        self._cache = (x.shape, cols, ho, wo)
        f = self.W.shape[0]
        out = cols @ self.W.reshape(f, -1).T + self.b
        return out.reshape(x.shape[0], ho, wo, f).transpose(0, 3, 1, 2)

    def backward(self, dout):
        x_shape, cols, ho, wo = self._cache
        f = self.W.shape[0]
        d = dout.transpose(0, 2, 3, 1).reshape(-1, f)
        self.dW = (d.T @ cols).reshape(self.W.shape)
        self.db = d.sum(axis=0)
        return col2im(d @ self.W.reshape(f, -1), x_shape, self.k, self.pad, ho, wo)


class ReLU:
    param_names = ()

    def forward(self, x, training: bool):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class MaxPool2:
    """2x2 max pooling, stride 2. The gradient goes only to the arg-max of each window."""

    param_names = ()

    def forward(self, x, training: bool):
        n, c, h, w = x.shape
        ho, wo = h // 2, w // 2
        win = x.reshape(n, c, ho, 2, wo, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, ho, wo, 4)
        idx = win.argmax(axis=-1)
        self._cache = (x.shape, idx)
        return np.take_along_axis(win, idx[..., None], axis=-1)[..., 0]

    def backward(self, dout):
        (n, c, h, w), idx = self._cache
        ho, wo = h // 2, w // 2
        dwin = np.zeros((n, c, ho, wo, 4), dtype=dout.dtype)
        np.put_along_axis(dwin, idx[..., None], dout[..., None], axis=-1)
        return dwin.reshape(n, c, ho, wo, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


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
        scale = np.sqrt(2.0 / n_in) if init == "he" else np.sqrt(2.0 / (n_in + n_out))
        self.W = (rng.randn(n_in, n_out) * scale).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)

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
        self._rng = np.random.RandomState(seed)

    def forward(self, x, training: bool):
        if not training or self.p <= 0.0:
            self._mask = None
            return x
        self._mask = (self._rng.rand(*x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self._mask

    def backward(self, dout):
        return dout if self._mask is None else dout * self._mask


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def augment_flip_crop(x, rng, pad: int = 2):
    """Random horizontal flip + random crop after zero padding (NCHW batch)."""
    n, _, h, w = x.shape
    flip = rng.rand(n) < 0.5
    x = np.where(flip[:, None, None, None], x[..., ::-1], x)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    dy, dx = rng.randint(0, 2 * pad + 1, size=(2, n))
    return np.stack([xp[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w] for i in range(n)])


class SimpleCNN:
    """[conv3x3 -> ReLU -> maxpool2] x len(channels) -> FC(hidden) -> ReLU -> Dropout -> FC(n_classes).

    Trained with softmax cross-entropy, AdamW (decoupled weight decay on W only),
    linear warmup + cosine learning-rate decay, and validation early stopping.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (8, 16),
        hidden: int = 64,
        n_classes: int = 10,
        image_size: int = 28,
        in_channels: int = 1,
        dropout: float = 0.2,
        seed: int = 42,
        dtype=np.float32,
    ):
        if image_size % 2 ** len(channels):
            raise ValueError(f"image_size must be divisible by {2 ** len(channels)} ({len(channels)} 2x2 poolings)")
        rng = np.random.RandomState(seed)
        self.dtype = dtype
        self.layers, self.layer_names = [], []
        c_prev = in_channels
        for k, c in enumerate(channels, start=1):
            self.layers += [Conv2D(c_prev, c, 3, 1, rng, dtype), ReLU(), MaxPool2()]
            self.layer_names += [f"conv{k}", f"relu{k}", f"pool{k}"]
            c_prev = c
        flat = c_prev * (image_size // 2 ** len(channels)) ** 2
        self.layers += [
            Flatten(),
            Dense(flat, hidden, rng, dtype),
            ReLU(),
            Dropout(dropout, seed + 7),
            Dense(hidden, n_classes, rng, dtype, init="xavier"),
        ]
        self.layer_names += ["flatten", "fc1", "relu_fc", "dropout", "fc2"]
        self.n_conv = len(channels)
        self.loss_: list[float] = []
        self.accuracy_: list[float] = []
        self.val_loss_: list[float] = []
        self.val_accuracy_: list[float] = []
        self.lr_history_: list[float] = []
        self.best_epoch_: int | None = None

    def parameters(self):
        for name, layer in zip(self.layer_names, self.layers):
            for p in layer.param_names:
                yield f"{name}.{p}", layer, p

    @property
    def n_params(self) -> int:
        return int(sum(getattr(layer, p).size for _, layer, p in self.parameters()))

    def forward(self, x, training: bool = False):
        a = np.asarray(x, dtype=self.dtype)
        for layer in self.layers:
            a = layer.forward(a, training)
        return a

    def backward(self, dlogits):
        d = dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d)

    def loss_and_backward(self, x, y):
        """Forward in training mode, softmax-CE loss, then backprop; returns (loss, probs)."""
        probs = _softmax(self.forward(x, training=True))
        m = len(y)
        loss = float(-np.mean(np.log(probs[np.arange(m), y] + 1e-12)))
        d = probs.copy()
        d[np.arange(m), y] -= 1.0
        self.backward(d / m)
        return loss, probs

    def feature_maps(self, x) -> dict[str, np.ndarray]:
        """Activations after each conv block's ReLU (relu1, relu2, ...) for visualization."""
        a = np.asarray(x, dtype=self.dtype)
        maps = {}
        last = f"relu{self.n_conv}"
        for name, layer in zip(self.layer_names, self.layers):
            a = layer.forward(a, False)
            if name.startswith("relu") and name[4:].isdigit():
                maps[name] = a
            if name == last:
                break
        return maps

    def predict_proba(self, X, batch_size: int = 500):
        out = [_softmax(self.forward(X[i : i + batch_size])) for i in range(0, len(X), batch_size)]
        return np.concatenate(out, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X, y) -> tuple[float, float]:
        probs = self.predict_proba(X)
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
        augment=None,
        seed: int = 0,
        verbose: bool = False,
    ):
        """`augment(batch, rng) -> batch` is applied to each training batch (not to validation)."""
        X = np.asarray(X, dtype=self.dtype)
        rng = np.random.RandomState(seed)
        b1, b2, eps = 0.9, 0.999, 1e-8
        state = {key: (np.zeros_like(getattr(layer, p)), np.zeros_like(getattr(layer, p)))
                 for key, layer, p in self.parameters()}
        step = 0
        steps_per_epoch = int(np.ceil(len(y) / batch_size))
        total_steps = n_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        best_val, best_snap, wait = float("inf"), None, 0

        for epoch in range(n_epochs):
            perm = rng.permutation(len(y))
            loss_sum, correct = 0.0, 0
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
                correct += int(np.sum(np.argmax(probs, axis=1) == y[idx]))

                bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
                for key, layer, p in self.parameters():
                    param = getattr(layer, p)
                    grad = getattr(layer, "d" + p)
                    m, v = state[key]
                    m *= b1
                    m += (1.0 - b1) * grad
                    v *= b2
                    v += (1.0 - b2) * grad * grad
                    if p == "W" and weight_decay > 0.0:
                        param *= 1.0 - lr_t * weight_decay
                    param -= (lr_t * (m / bc1) / (np.sqrt(v / bc2) + eps)).astype(param.dtype)

            self.lr_history_.append(float(lr_t))
            self.loss_.append(loss_sum / len(y))
            self.accuracy_.append(correct / len(y) * 100.0)
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
            if verbose:
                print(msg)
            if early_stop_patience is not None and wait >= early_stop_patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch + 1} (best={self.best_epoch_ + 1})")
                break

        if best_snap is not None:
            self._restore(best_snap)
        return self


def gradient_check(model: SimpleCNN, x, y, n_checks: int = 8, eps: float = 1e-5, seed: int = 0) -> dict:
    """Compare backprop gradients with central differences on random entries of every parameter.

    Use a float64 model with dropout=0 so the training-mode forward is deterministic.
    """
    rng = np.random.RandomState(seed)
    model.loss_and_backward(x, y)
    analytic = {key: getattr(layer, "d" + p).copy() for key, layer, p in model.parameters()}

    def loss_at():
        probs = _softmax(model.forward(x, training=False))
        return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))

    report = {}
    for key, layer, p in model.parameters():
        param = getattr(layer, p)
        flat_idx = rng.choice(param.size, size=min(n_checks, param.size), replace=False)
        errs = []
        for fi in flat_idx:
            idx = np.unravel_index(fi, param.shape)
            orig = param[idx]
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
