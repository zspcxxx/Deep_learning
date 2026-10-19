"""Vanilla RNN, LSTM and GRU with hand-written backpropagation through time (NumPy only).

Inputs are (N, T, D) arrays (usually one-hot), hidden states (N, T, H). A RecurrentNet puts a
Dense read-out on top: either on the last step only ("last", sequence classification) or on every
step ("all", next-token prediction with an optional padding mask).
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class RNNCell:
    """h_t = tanh(x_t Wx + h_{t-1} Wh + b)."""

    param_names = ("Wx", "Wh", "b")

    def __init__(self, n_in: int, hidden: int, rng, dtype):
        self.H = hidden
        self.Wx = (rng.randn(n_in, hidden) * np.sqrt(1.0 / n_in)).astype(dtype)
        self.Wh = (rng.randn(hidden, hidden) * np.sqrt(1.0 / hidden)).astype(dtype)
        self.b = np.zeros(hidden, dtype=dtype)

    def init_state(self, n: int):
        return np.zeros((n, self.H), dtype=self.Wx.dtype)

    def step(self, x, h):
        return np.tanh(x @ self.Wx + h @ self.Wh + self.b)

    def forward(self, X):
        n, t_len, _ = X.shape
        h = self.init_state(n)
        hs = np.empty((n, t_len, self.H), dtype=self.Wx.dtype)
        for t in range(t_len):
            h = self.step(X[:, t], h)
            hs[:, t] = h
        self._cache = (X, hs)
        return hs

    def backward(self, dhs):
        """dhs: dL/dh_t from the read-out at every step. Returns dL/dX (N, T, D)."""
        X, hs = self._cache
        n, t_len, _ = X.shape
        self.dWx, self.dWh, self.db = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
        dX = np.empty_like(X)
        dh_next = np.zeros((n, self.H), dtype=X.dtype)
        for t in reversed(range(t_len)):
            h_prev = hs[:, t - 1] if t > 0 else np.zeros_like(dh_next)
            da = (dhs[:, t] + dh_next) * (1.0 - hs[:, t] ** 2)
            self.dWx += X[:, t].T @ da
            self.dWh += h_prev.T @ da
            self.db += da.sum(axis=0)
            dX[:, t] = da @ self.Wx.T
            dh_next = da @ self.Wh.T
        return dX


class LSTMCell:
    """Gates [i, f, o, g] from one fused matmul; c_t = f*c_{t-1} + i*g, h_t = o*tanh(c_t)."""

    param_names = ("Wx", "Wh", "b")

    def __init__(self, n_in: int, hidden: int, rng, dtype, forget_bias: float = 1.0):
        self.H = hidden
        self.Wx = (rng.randn(n_in, 4 * hidden) * np.sqrt(1.0 / n_in)).astype(dtype)
        self.Wh = (rng.randn(hidden, 4 * hidden) * np.sqrt(1.0 / hidden)).astype(dtype)
        self.b = np.zeros(4 * hidden, dtype=dtype)
        self.b[hidden : 2 * hidden] = forget_bias

    def init_state(self, n: int):
        z = np.zeros((n, self.H), dtype=self.Wx.dtype)
        return z, z.copy()

    def _gates(self, x, h):
        H = self.H
        z = x @ self.Wx + h @ self.Wh + self.b
        return _sigmoid(z[:, :H]), _sigmoid(z[:, H : 2 * H]), _sigmoid(z[:, 2 * H : 3 * H]), np.tanh(z[:, 3 * H :])

    def step(self, x, state):
        h, c = state
        i, f, o, g = self._gates(x, h)
        c = f * c + i * g
        return o * np.tanh(c), c

    def forward(self, X):
        n, t_len, _ = X.shape
        h, c = self.init_state(n)
        hs = np.empty((n, t_len, self.H), dtype=self.Wx.dtype)
        cache = []
        for t in range(t_len):
            i, f, o, g = self._gates(X[:, t], h)
            c_prev, h_prev = c, h
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            hs[:, t] = h
            cache.append((h_prev, c_prev, i, f, o, g, tc))
        self._cache = (X, cache)
        return hs

    def backward(self, dhs):
        X, cache = self._cache
        n, t_len, _ = X.shape
        self.dWx, self.dWh, self.db = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
        dX = np.empty_like(X)
        dh_next = np.zeros((n, self.H), dtype=X.dtype)
        dc_next = np.zeros_like(dh_next)
        for t in reversed(range(t_len)):
            h_prev, c_prev, i, f, o, g, tc = cache[t]
            dh = dhs[:, t] + dh_next
            dc = dc_next + dh * o * (1.0 - tc ** 2)
            dz = np.concatenate([
                dc * g * i * (1.0 - i),
                dc * c_prev * f * (1.0 - f),
                dh * tc * o * (1.0 - o),
                dc * i * (1.0 - g ** 2),
            ], axis=1)
            self.dWx += X[:, t].T @ dz
            self.dWh += h_prev.T @ dz
            self.db += dz.sum(axis=0)
            dX[:, t] = dz @ self.Wx.T
            dh_next = dz @ self.Wh.T
            dc_next = dc * f
        return dX


class GRUCell:
    """PyTorch-style GRU: update z, reset r, candidate n.

    z = σ(x Wxz + h Whz + bz)
    r = σ(x Wxr + h Whr + br)
    n = tanh(x Wxn + r ⊙ (h Whn + bn))
    h_t = (1 − z) ⊙ n + z ⊙ h_{t-1}

    A large update-gate bias makes z start near 1, so the previous state is copied
    and gradients can travel back through the additive path (same role as LSTM forget bias).
    """

    param_names = ("Wx", "Wh", "b")

    def __init__(self, n_in: int, hidden: int, rng, dtype, update_bias: float = 0.0):
        self.H = hidden
        self.Wx = (rng.randn(n_in, 3 * hidden) * np.sqrt(1.0 / n_in)).astype(dtype)
        self.Wh = (rng.randn(hidden, 3 * hidden) * np.sqrt(1.0 / hidden)).astype(dtype)
        self.b = np.zeros(3 * hidden, dtype=dtype)
        self.b[:hidden] = update_bias

    def init_state(self, n: int):
        return np.zeros((n, self.H), dtype=self.Wx.dtype)

    def _parts(self, x, h):
        H = self.H
        z_lin = x @ self.Wx[:, :H] + h @ self.Wh[:, :H] + self.b[:H]
        r_lin = x @ self.Wx[:, H : 2 * H] + h @ self.Wh[:, H : 2 * H] + self.b[H : 2 * H]
        u = h @ self.Wh[:, 2 * H :] + self.b[2 * H :]
        z, r = _sigmoid(z_lin), _sigmoid(r_lin)
        n = np.tanh(x @ self.Wx[:, 2 * H :] + r * u)
        return z, r, n, u

    def step(self, x, h):
        z, r, n, _ = self._parts(x, h)
        return (1.0 - z) * n + z * h

    def forward(self, X):
        n_batch, t_len, _ = X.shape
        h = self.init_state(n_batch)
        hs = np.empty((n_batch, t_len, self.H), dtype=self.Wx.dtype)
        cache = []
        for t in range(t_len):
            z, r, n, u = self._parts(X[:, t], h)
            h_prev = h
            h = (1.0 - z) * n + z * h
            hs[:, t] = h
            cache.append((h_prev, z, r, n, u))
        self._cache = (X, cache)
        return hs

    def backward(self, dhs):
        X, cache = self._cache
        n_batch, t_len, _ = X.shape
        H = self.H
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)
        dX = np.empty_like(X)
        dh_next = np.zeros((n_batch, H), dtype=X.dtype)
        for t in reversed(range(t_len)):
            h_prev, z, r, n, u = cache[t]
            dh = dhs[:, t] + dh_next
            da_z = dh * (h_prev - n) * z * (1.0 - z)
            da_n = dh * (1.0 - z) * (1.0 - n ** 2)
            du = da_n * r
            da_r = (da_n * u) * r * (1.0 - r)
            self.dWx[:, :H] += X[:, t].T @ da_z
            self.dWx[:, H : 2 * H] += X[:, t].T @ da_r
            self.dWx[:, 2 * H :] += X[:, t].T @ da_n
            self.dWh[:, :H] += h_prev.T @ da_z
            self.dWh[:, H : 2 * H] += h_prev.T @ da_r
            self.dWh[:, 2 * H :] += h_prev.T @ du
            self.db[:H] += da_z.sum(axis=0)
            self.db[H : 2 * H] += da_r.sum(axis=0)
            self.db[2 * H :] += du.sum(axis=0)
            dX[:, t] = da_z @ self.Wx[:, :H].T + da_r @ self.Wx[:, H : 2 * H].T + da_n @ self.Wx[:, 2 * H :].T
            dh_next = (
                dh * z
                + da_z @ self.Wh[:, :H].T
                + da_r @ self.Wh[:, H : 2 * H].T
                + du @ self.Wh[:, 2 * H :].T
            )
        self.dh0 = dh_next
        return dX


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class RecurrentNet:
    """cell ("rnn" | "lstm" | "gru") + Dense read-out, trained with softmax cross-entropy, Adam and gradient clipping."""

    def __init__(self, cell: str, n_in: int, hidden: int, n_out: int, seed: int = 42, dtype=np.float64,
                 forget_bias: float = 1.0, update_bias: float = 0.0):
        rng = np.random.RandomState(seed)
        self.kind = cell
        if cell == "lstm":
            self.cell = LSTMCell(n_in, hidden, rng, dtype, forget_bias=forget_bias)
        elif cell == "gru":
            self.cell = GRUCell(n_in, hidden, rng, dtype, update_bias=update_bias)
        elif cell == "rnn":
            self.cell = RNNCell(n_in, hidden, rng, dtype)
        else:
            raise ValueError(f"unknown cell {cell!r}")
        self.Wy = (rng.randn(hidden, n_out) * np.sqrt(1.0 / hidden)).astype(dtype)
        self.by = np.zeros(n_out, dtype=dtype)
        self.dtype = dtype
        self.loss_: list[float] = []
        self.grad_norm_: list[float] = []

    def parameters(self):
        for p in self.cell.param_names:
            yield f"cell.{p}", self.cell, p
        yield "out.Wy", self, "Wy"
        yield "out.by", self, "by"

    @property
    def n_params(self) -> int:
        return int(sum(getattr(obj, p).size for _, obj, p in self.parameters()))

    def logits(self, X, mode: str = "last"):
        hs = self.cell.forward(np.asarray(X, dtype=self.dtype))
        return (hs[:, -1] if mode == "last" else hs) @ self.Wy + self.by, hs

    def loss_and_backward(self, X, Y, mode: str = "last", mask=None):
        """mode "last": Y (N,) labels for the final step. mode "all": Y (N, T) next-token labels, mask (N, T)."""
        logits, hs = self.logits(X, mode)
        probs = _softmax(logits)
        if mode == "last":
            n = len(Y)
            loss = float(-np.mean(np.log(probs[np.arange(n), Y] + 1e-12)))
            dlog = probs.copy()
            dlog[np.arange(n), Y] -= 1.0
            dlog /= n
            self.dWy = hs[:, -1].T @ dlog
            self.dby = dlog.sum(axis=0)
            dhs = np.zeros_like(hs)
            dhs[:, -1] = dlog @ self.Wy.T
        else:
            mask = np.ones(Y.shape, dtype=self.dtype) if mask is None else mask.astype(self.dtype)
            count = mask.sum()
            n, t_len = Y.shape
            p_true = probs[np.arange(n)[:, None], np.arange(t_len)[None, :], Y]
            loss = float(-(np.log(p_true + 1e-12) * mask).sum() / count)
            dlog = probs.copy()
            dlog[np.arange(n)[:, None], np.arange(t_len)[None, :], Y] -= 1.0
            dlog *= mask[..., None] / count
            self.dWy = np.einsum("nth,ntv->hv", hs, dlog)
            self.dby = dlog.sum(axis=(0, 1))
            dhs = dlog @ self.Wy.T
        dX = self.cell.backward(dhs)
        return loss, probs, dX

    def predict(self, X):
        return np.argmax(self.logits(X, "last")[0], axis=1)

    def sequence_loss(self, X, Y, mask, batch_size: int = 1000) -> float:
        """Mean next-token cross-entropy (nats per token) over the masked steps, forward only."""
        total, count = 0.0, 0.0
        for s in range(0, len(Y), batch_size):
            logits, _ = self.logits(X[s : s + batch_size], "all")
            p = _softmax(logits)
            y, m = Y[s : s + batch_size], mask[s : s + batch_size]
            p_true = p[np.arange(len(y))[:, None], np.arange(y.shape[1])[None, :], y]
            total += float(-(np.log(p_true + 1e-12) * m).sum())
            count += float(m.sum())
        return total / count

    def forget_gates(self, X) -> np.ndarray:
        """LSTM only: forget-gate activations f_t, shape (N, T, H)."""
        self.cell.forward(np.asarray(X, dtype=self.dtype))
        return np.stack([c[3] for c in self.cell._cache[1]], axis=1)

    def update_gates(self, X) -> np.ndarray:
        """GRU only: update-gate activations z_t, shape (N, T, H)."""
        self.cell.forward(np.asarray(X, dtype=self.dtype))
        return np.stack([c[1] for c in self.cell._cache[1]], axis=1)

    def fit(self, batch_fn, n_steps: int, lr: float = 3e-3, clip: float = 1.0, mode: str = "last",
            log_every: int = 0, log_fn=None):
        """batch_fn(step) -> (X, Y) or (X, Y, mask). Adam with global gradient-norm clipping."""
        b1, b2, eps = 0.9, 0.999, 1e-8
        state = {k: (np.zeros_like(getattr(o, p)), np.zeros_like(getattr(o, p))) for k, o, p in self.parameters()}
        for step in range(1, n_steps + 1):
            batch = batch_fn(step)
            loss, *_ = self.loss_and_backward(*batch[:2], mode=mode, mask=batch[2] if len(batch) > 2 else None)
            grads = {k: getattr(o, "d" + p) for k, o, p in self.parameters()}
            norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
            scale = min(1.0, clip / (norm + 1e-12)) if clip else 1.0
            lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * (step - 1) / n_steps))
            bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
            for k, o, p in self.parameters():
                g = grads[k] * scale
                m, v = state[k]
                m *= b1
                m += (1.0 - b1) * g
                v *= b2
                v += (1.0 - b2) * g * g
                getattr(o, p)[...] -= lr_t * (m / bc1) / (np.sqrt(v / bc2) + eps)
            self.loss_.append(loss)
            self.grad_norm_.append(norm)
            if log_every and (step % log_every == 0 or step == 1) and log_fn:
                log_fn(step, loss, norm)
        return self

    def sample(self, start: int, stop: int, n: int, temperature: float = 1.0, max_len: int = 20, seed: int = 0):
        """Generate n token sequences, feeding back each sampled token, until `stop` or max_len."""
        rng = np.random.RandomState(seed)
        vocab = self.Wy.shape[1]
        out = []
        for _ in range(n):
            state = self.cell.init_state(1)
            tok, seq = start, []
            for _ in range(max_len):
                x = np.zeros((1, vocab), dtype=self.dtype)
                x[0, tok] = 1.0
                state = self.cell.step(x, state)
                h = state[0] if isinstance(state, tuple) else state
                p = _softmax((h @ self.Wy + self.by)[0] / temperature)
                tok = int(rng.choice(vocab, p=p))
                if tok == stop:
                    break
                seq.append(tok)
            out.append(seq)
        return out


def gradient_check(model: RecurrentNet, X, Y, mode: str = "last", mask=None, n_checks: int = 8,
                   eps: float = 1e-5, seed: int = 0) -> dict:
    """Central differences vs BPTT on random entries of every parameter (use float64)."""
    rng = np.random.RandomState(seed)
    model.loss_and_backward(X, Y, mode=mode, mask=mask)
    analytic = {k: getattr(o, "d" + p).copy() for k, o, p in model.parameters()}
    report = {}
    for k, o, p in model.parameters():
        param = getattr(o, p)
        errs = []
        for fi in rng.choice(param.size, size=min(n_checks, param.size), replace=False):
            idx = np.unravel_index(fi, param.shape)
            orig = param[idx]
            param[idx] = orig + eps
            lp = model.loss_and_backward(X, Y, mode=mode, mask=mask)[0]
            param[idx] = orig - eps
            lm = model.loss_and_backward(X, Y, mode=mode, mask=mask)[0]
            param[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[k][idx]
            errs.append(abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        report[k] = float(max(errs))
    return report


def input_grad_norms(model: RecurrentNet, X, Y) -> np.ndarray:
    """‖dL/dx_t‖ for each step t (loss on the last step): how much each input still matters at the end."""
    _, _, dX = model.loss_and_backward(X, Y, mode="last")
    return np.linalg.norm(dX, axis=2).mean(axis=0)
