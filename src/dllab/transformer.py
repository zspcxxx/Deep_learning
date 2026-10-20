"""A small Transformer encoder assembled from the parts built in lessons 01–14 (NumPy only).

Layers follow the nn.py convention: forward(x, training) / backward(dout), parameters listed in
`param_names` with gradients stored as d<name>, composite layers expose `children`.

    tokens → Embedding (+ learned positions) → n × Block → LayerNorm → Linear → softmax per position
    Block (pre-LN): x = x + Dropout(MultiHeadSelfAttention(LN(x)))
                    x = x + Dropout(FFN(LN(x)))          FFN = Linear → GELU → Linear

Every part can be switched off (positions, residual, LayerNorm, ...) so lesson 15 can remove
one at a time and see which earlier lesson's failure comes back.
"""
from __future__ import annotations

import numpy as _numpy

from dllab.nn import Dense, Dropout, ReLU
from dllab.xp import as_device, as_param, asnumpy, cast_params, host_np, host_rng, np, prefer_cpu_if_small


def _on_small(method):
    def wrapped(self, *args, **kwargs):
        with prefer_cpu_if_small(self.n_params):
            return method(self, *args, **kwargs)
    return wrapped


def make_sort_dataset(n: int, seq_len: int, vocab: int, seed: int) -> tuple:
    """X: n random digit sequences (repeats allowed); Y: the same sequences sorted ascending."""
    rng = _numpy.random.RandomState(seed)
    X = rng.randint(0, vocab, size=(n, seq_len)).astype(_numpy.int64)
    return X, _numpy.sort(X, axis=1)


def _softmax(z, axis: int = -1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _one_hot(tokens, vocab: int, dtype):
    tokens = np.asarray(tokens)
    out = np.zeros(tokens.shape + (vocab,), dtype=dtype)
    np.put_along_axis(out, tokens[..., None], 1.0, axis=-1)
    return out


# ---------------------------------------------------------------- layers


class Embedding:
    param_names = ("E",)

    def __init__(self, vocab: int, d: int, rng, dtype, std: float):
        self.vocab = vocab
        self.E = as_param(rng.randn(vocab, d) * std, dtype)

    def forward(self, tokens, training: bool):
        self._onehot = _one_hot(tokens, self.vocab, self.E.dtype)
        return self._onehot @ self.E

    def backward(self, dout):
        d = dout.shape[-1]
        self.dE = self._onehot.reshape(-1, self.vocab).T @ dout.reshape(-1, d)
        return None


class PositionalEmbedding:
    """Learned vector per position. Attention alone is permutation-invariant (lesson 14)."""

    param_names = ("P",)

    def __init__(self, seq_len: int, d: int, rng, dtype, std: float):
        self.P = as_param(rng.randn(seq_len, d) * std, dtype)

    def forward(self, x, training: bool):
        return x + self.P[None, : x.shape[1]]

    def backward(self, dout):
        self.dP = np.zeros_like(self.P)
        self.dP[: dout.shape[1]] = dout.sum(axis=0)
        return dout


class Linear:
    """Dense on the last axis, so (N, T, d_in) works as well as (N, d_in)."""

    param_names = ("W", "b")

    def __init__(self, n_in: int, n_out: int, rng, dtype, init_scale: float = 1.0, bias: bool = True):
        self.W = as_param(rng.randn(n_in, n_out) * init_scale * (1.0 / n_in) ** 0.5, dtype)
        self.b = as_param(host_np.zeros(n_out), dtype) if bias else None
        if not bias:
            self.param_names = ("W",)

    def forward(self, x, training: bool):
        self._x = x
        return x @ self.W if self.b is None else x @ self.W + self.b

    def backward(self, dout):
        n_in, n_out = self.W.shape
        self.dW = self._x.reshape(-1, n_in).T @ dout.reshape(-1, n_out)
        if self.b is not None:
            self.db = dout.reshape(-1, n_out).sum(axis=0)
        return dout @ self.W.T


class LayerNorm:
    """Normalise each token's feature vector (lesson 07 normalises across the batch instead)."""

    param_names = ("gamma", "beta")

    def __init__(self, d: int, dtype, eps: float = 1e-5):
        self.gamma = as_param(host_np.ones(d), dtype)
        self.beta = as_param(host_np.zeros(d), dtype)
        self.eps = eps

    def forward(self, x, training: bool):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._inv = 1.0 / np.sqrt(var + self.eps)
        self._xh = (x - mu) * self._inv
        return self.gamma * self._xh + self.beta

    def backward(self, dout):
        xh, inv = self._xh, self._inv
        d = dout.shape[-1]
        self.dgamma = (dout * xh).reshape(-1, d).sum(axis=0)
        self.dbeta = dout.reshape(-1, d).sum(axis=0)
        dxh = dout * self.gamma
        return inv * (dxh - dxh.mean(axis=-1, keepdims=True) - xh * (dxh * xh).mean(axis=-1, keepdims=True))


class GELU:
    """tanh approximation, as in lesson 09."""

    param_names = ()
    _C = (2.0 / _numpy.pi) ** 0.5

    def forward(self, x, training: bool):
        self._x = x
        self._t = np.tanh(self._C * x * (1.0 + 0.044715 * x * x))
        return 0.5 * x * (1.0 + self._t)

    def backward(self, dout):
        x, t = self._x, self._t
        dt = (1.0 - t * t) * self._C * (1.0 + 3 * 0.044715 * x * x)
        return dout * (0.5 * (1.0 + t) + 0.5 * x * dt)


class MultiHeadSelfAttention:
    """softmax(Q Kᵀ / √d_head) V per head (lesson 14), heads concatenated and projected by W_o."""

    param_names = ()

    def __init__(self, d: int, n_heads: int, rng, dtype, init_scale: float = 1.0):
        if d % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dh = n_heads, d // n_heads
        # A key bias only shifts every score of a query equally, which softmax ignores (zero gradient).
        self.q = Linear(d, d, rng, dtype, init_scale, bias=False)
        self.k = Linear(d, d, rng, dtype, init_scale, bias=False)
        self.v = Linear(d, d, rng, dtype, init_scale, bias=False)
        self.o = Linear(d, d, rng, dtype, init_scale)
        self.children = [("q", self.q), ("k", self.k), ("v", self.v), ("o", self.o)]
        self.weights = None

    def _split(self, x):
        n, t, _ = x.shape
        return x.reshape(n, t, self.h, self.dh).transpose(0, 2, 1, 3)

    def _merge(self, x):
        n, _, t, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(n, t, self.h * self.dh)

    def forward(self, x, training: bool):
        Q = self._split(self.q.forward(x, training))
        K = self._split(self.k.forward(x, training))
        V = self._split(self.v.forward(x, training))
        A = _softmax(np.matmul(Q, np.swapaxes(K, -1, -2)) / self.dh ** 0.5)
        self._cache = (Q, K, V, A)
        self.weights = A
        return self.o.forward(self._merge(np.matmul(A, V)), training)

    def backward(self, dout):
        Q, K, V, A = self._cache
        dctx = self._split(self.o.backward(dout))
        dA = np.matmul(dctx, np.swapaxes(V, -1, -2))
        dV = np.matmul(np.swapaxes(A, -1, -2), dctx)
        dS = A * (dA - np.sum(dA * A, axis=-1, keepdims=True)) / self.dh ** 0.5
        dQ = np.matmul(dS, K)
        dK = np.matmul(np.swapaxes(dS, -1, -2), Q)
        return (self.q.backward(self._merge(dQ)) + self.k.backward(self._merge(dK))
                + self.v.backward(self._merge(dV)))


class Identity:
    param_names = ()

    def forward(self, x, training: bool):
        return x

    def backward(self, dout):
        return dout


class FeedForward:
    """The per-token MLP of lesson 02: Linear → activation → Linear. activation="linear" removes the bend."""

    param_names = ()

    def __init__(self, d: int, d_ff: int, rng, dtype, activation: str = "gelu", init_scale: float = 1.0):
        self.fc1 = Linear(d, d_ff, rng, dtype, init_scale)
        self.act = {"gelu": GELU, "relu": ReLU, "linear": Identity}[activation]()
        self.fc2 = Linear(d_ff, d, rng, dtype, init_scale)
        self.children = [("fc1", self.fc1), ("fc2", self.fc2)]

    def forward(self, x, training: bool):
        return self.fc2.forward(self.act.forward(self.fc1.forward(x, training), training), training)

    def backward(self, dout):
        return self.fc1.backward(self.act.backward(self.fc2.backward(dout)))


class Block:
    """Pre-LN Transformer block. residual=False drops both shortcuts (lesson 11's plain deep net)."""

    param_names = ()

    def __init__(self, d: int, n_heads: int, d_ff: int, rng, dtype, *, residual: bool, layer_norm: bool,
                 dropout: float, activation: str, init_scale: float, seed: int):
        self.residual, self.layer_norm = residual, layer_norm
        self.ln1 = LayerNorm(d, dtype) if layer_norm else None
        self.attn = MultiHeadSelfAttention(d, n_heads, rng, dtype, init_scale)
        self.ln2 = LayerNorm(d, dtype) if layer_norm else None
        self.ffn = FeedForward(d, d_ff, rng, dtype, activation, init_scale)
        self.drop1 = Dropout(dropout, seed)
        self.drop2 = Dropout(dropout, seed + 1)
        self.children = [(n, c) for n, c in (("ln1", self.ln1), ("attn", self.attn), ("ln2", self.ln2),
                                             ("ffn", self.ffn)) if c is not None]

    def forward(self, x, training: bool):
        h = self.ln1.forward(x, training) if self.ln1 else x
        a = self.drop1.forward(self.attn.forward(h, training), training)
        x1 = x + a if self.residual else a
        h2 = self.ln2.forward(x1, training) if self.ln2 else x1
        f = self.drop2.forward(self.ffn.forward(h2, training), training)
        return x1 + f if self.residual else f

    def backward(self, dout):
        dh2 = self.ffn.backward(self.drop2.backward(dout))
        dx1 = self.ln2.backward(dh2) if self.ln2 else dh2
        if self.residual:
            dx1 = dx1 + dout
        dh = self.attn.backward(self.drop1.backward(dx1))
        dx = self.ln1.backward(dh) if self.ln1 else dh
        return dx + dx1 if self.residual else dx


# ---------------------------------------------------------------- models


def _walk(prefix, layer):
    for p in layer.param_names:
        yield f"{prefix}.{p}", layer, p
    for cname, child in getattr(layer, "children", []):
        yield from _walk(f"{prefix}.{cname}", child)


def _cross_entropy(logits, Y):
    """Mean softmax cross-entropy over every (sample, position); returns loss, probs, dlogits."""
    probs = _softmax(logits)
    n, t = Y.shape
    rows, cols = np.arange(n)[:, None], np.arange(t)[None, :]
    loss = float(-np.mean(np.log(probs[rows, cols, Y] + 1e-12)))
    d = probs.copy()
    d[rows, cols, Y] -= 1.0
    return loss, probs, d / (n * t)


class _Model:
    def parameters(self):
        for name, layer in self.layers:
            yield from _walk(name, layer)

    @property
    def n_params(self) -> int:
        return int(sum(getattr(o, p).size for _, o, p in self.parameters()))

    @_on_small
    def loss_and_backward(self, X, Y, training: bool = True) -> float:
        Y = as_device(Y)
        loss, _, d = _cross_entropy(self.forward(X, training), Y)
        self.backward(d)
        return loss

    @_on_small
    def predict(self, X, batch_size: int = 1000):
        return _numpy.concatenate([asnumpy(np.argmax(self.forward(X[s : s + batch_size], False), axis=-1))
                                   for s in range(0, len(X), batch_size)])

    @_on_small
    def evaluate(self, X, Y, batch_size: int = 1000) -> dict:
        """loss, token accuracy (%), whole-sequence accuracy (%) in inference mode."""
        loss_sum, pred = 0.0, []
        for s in range(0, len(X), batch_size):
            yb = as_device(Y[s : s + batch_size])
            loss, probs, _ = _cross_entropy(self.forward(X[s : s + batch_size], False), yb)
            loss_sum += loss * len(yb)
            pred.append(asnumpy(np.argmax(probs, axis=-1)))
        pred = _numpy.concatenate(pred)
        Y = _numpy.asarray(Y)
        return {
            "loss": loss_sum / len(X),
            "token_acc": float((pred == Y).mean() * 100),
            "seq_acc": float((pred == Y).all(axis=1).mean() * 100),
        }


class TinyTransformer(_Model):
    def __init__(self, vocab: int, seq_len: int, d_model: int = 32, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.0, *, positional: bool = True, residual: bool = True,
                 layer_norm: bool = True, activation: str = "gelu", init_scale: float = 1.0,
                 seed: int = 0, dtype=np.float64):
        rng = host_rng(seed)
        self.config = dict(vocab=vocab, seq_len=seq_len, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                           d_ff=d_ff, dropout=dropout, positional=positional, residual=residual,
                           layer_norm=layer_norm, activation=activation, init_scale=init_scale)
        self.emb = Embedding(vocab, d_model, rng, dtype, std=0.5)
        self.pos = PositionalEmbedding(seq_len, d_model, rng, dtype, std=0.5) if positional else None
        self.blocks = [Block(d_model, n_heads, d_ff, rng, dtype, residual=residual, layer_norm=layer_norm,
                             dropout=dropout, activation=activation, init_scale=init_scale, seed=seed + 10 * i)
                       for i in range(n_layers)]
        self.ln_f = LayerNorm(d_model, dtype) if layer_norm else None
        self.head = Linear(d_model, vocab, rng, dtype)
        self.layers = [("emb", self.emb)]
        if self.pos:
            self.layers.append(("pos", self.pos))
        self.layers += [(f"block{i}", b) for i, b in enumerate(self.blocks)]
        if self.ln_f:
            self.layers.append(("ln_f", self.ln_f))
        self.layers.append(("head", self.head))

    def forward(self, X, training: bool = False):
        a = as_device(X)
        for _, layer in self.layers:
            a = layer.forward(a, training)
        return a

    def backward(self, dlogits):
        d = dlogits
        for _, layer in reversed(self.layers):
            d = layer.backward(d)

    @_on_small
    def attention_maps(self, X) -> list:
        """Per block, attention weights (N, heads, T, T): row = query position, column = key position."""
        self.forward(X, training=False)
        return [asnumpy(b.attn.weights) for b in self.blocks]


class MLPBaseline(_Model):
    """Lesson 02 / 09 MLP on the flattened one-hot sequence: no weight sharing across positions."""

    def __init__(self, vocab: int, seq_len: int, hidden: tuple = (256, 256), dropout: float = 0.0,
                 seed: int = 0, dtype=np.float64):
        rng = host_rng(seed)
        self.vocab, self.seq_len = vocab, seq_len
        sizes = [vocab * seq_len, *hidden]
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers += [(f"fc{i}", Dense(a, b, rng, dtype)), (f"relu{i}", ReLU())]
            if dropout > 0:
                self.layers.append((f"drop{i}", Dropout(dropout, seed + i)))
        self.layers.append(("out", Dense(sizes[-1], vocab * seq_len, rng, dtype, init="xavier")))

    def forward(self, X, training: bool = False):
        X = as_device(X)
        a = _one_hot(X, self.vocab, self.layers[0][1].W.dtype).reshape(len(X), -1)
        for _, layer in self.layers:
            a = layer.forward(a, training)
        return a.reshape(len(X), self.seq_len, self.vocab)

    def backward(self, dlogits):
        d = dlogits.reshape(len(dlogits), -1)
        for _, layer in reversed(self.layers):
            d = layer.backward(d)


# ---------------------------------------------------------------- training


def lr_at(step: int, total: int, lr: float, warmup: int, min_lr: float) -> float:
    """Linear warmup, then cosine decay to min_lr (lesson 09)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + float(_numpy.cos(_numpy.pi * progress)))


def train(model: _Model, X_tr, Y_tr, X_val, Y_val, *, steps: int, batch_size: int = 128, lr: float = 3e-3,
          min_lr: float = 1e-4, warmup: int = 100, weight_decay: float = 0.01, clip: float = 1.0,
          eval_every: int = 100, patience: int | None = 6, seed: int = 0, log_fn=None) -> dict:
    """AdamW (decay on weight matrices only) + warmup/cosine + gradient clipping + validation early stopping.

    Returns the history; the model is left at the step with the lowest validation loss.
    """
    with prefer_cpu_if_small(model.n_params):
        return _train(model, X_tr, Y_tr, X_val, Y_val, steps, batch_size, lr, min_lr, warmup,
                      weight_decay, clip, eval_every, patience, seed, log_fn)


def _train(model, X_tr, Y_tr, X_val, Y_val, steps, batch_size, lr, min_lr, warmup, weight_decay, clip,
           eval_every, patience, seed, log_fn):
    cast_params(model.parameters())
    rng = host_rng(seed)
    params = list(model.parameters())
    b1, b2, eps = 0.9, 0.999, 1e-8
    state = {k: (np.zeros_like(getattr(o, p)), np.zeros_like(getattr(o, p))) for k, o, p in params}
    hist = {"loss": [], "lr": [], "grad_norm": [], "val": [], "best_step": None, "stopped_at": None}
    best, best_snap, wait = float("inf"), None, 0
    for step in range(steps):
        idx = rng.randint(0, len(X_tr), size=batch_size)
        loss = model.loss_and_backward(X_tr[idx], Y_tr[idx], training=True)
        grads = {k: getattr(o, "d" + p) for k, o, p in params}
        norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
        scale = min(1.0, clip / (norm + 1e-12)) if clip else 1.0
        lr_t = lr_at(step, steps, lr, warmup, min_lr)
        bc1, bc2 = 1.0 - b1 ** (step + 1), 1.0 - b2 ** (step + 1)
        for k, o, p in params:
            param, g = getattr(o, p), grads[k] * scale
            m, v = state[k]
            if p == "W" and weight_decay > 0:
                param *= 1.0 - lr_t * weight_decay
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * g * g
            param -= lr_t * (m / bc1) / (np.sqrt(v / bc2) + eps)
        hist["loss"].append(loss)
        hist["lr"].append(lr_t)
        hist["grad_norm"].append(norm)

        if (step + 1) % eval_every == 0 or step + 1 == steps:
            ev = model.evaluate(X_val, Y_val)
            hist["val"].append({"step": step + 1, **ev})
            if log_fn:
                log_fn(step + 1, float(_numpy.mean(hist["loss"][-eval_every:])), lr_t, ev)
            if not _numpy.isfinite(ev["loss"]):
                hist["stopped_at"] = step + 1
                break
            if ev["loss"] < best - 1e-5:
                best, wait, hist["best_step"] = ev["loss"], 0, step + 1
                best_snap = {k: getattr(o, p).copy() for k, o, p in params}
            elif patience is not None:
                wait += 1
                if wait >= patience:
                    hist["stopped_at"] = step + 1
                    break
    if best_snap is not None:
        for k, o, p in params:
            getattr(o, p)[...] = best_snap[k]
    return hist


def gradient_check(model: _Model, X, Y, n_checks: int = 4, eps: float = 1e-5, seed: int = 0) -> dict:
    """Central differences vs backprop on random entries of every parameter (float64, dropout 0)."""
    rng = host_rng(seed)
    with prefer_cpu_if_small(model.n_params):
        model.loss_and_backward(X, Y, training=False)
        analytic = {k: getattr(o, "d" + p).copy() for k, o, p in model.parameters()}

        def loss_at():
            return _cross_entropy(model.forward(X, False), as_device(Y))[0]

        report = {}
        for k, o, p in model.parameters():
            param = getattr(o, p)
            errs = []
            for fi in rng.choice(param.size, size=min(n_checks, param.size), replace=False):
                idx = tuple(int(v) for v in _numpy.unravel_index(int(fi), param.shape))
                orig = param[idx].copy()
                param[idx] = orig + eps
                lp = loss_at()
                param[idx] = orig - eps
                lm = loss_at()
                param[idx] = orig
                num = (lp - lm) / (2 * eps)
                ana = float(analytic[k][idx])
                errs.append(abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
            report[k] = float(max(errs))
        return report
