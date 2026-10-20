"""Scaled dot-product attention and a content-addressable lookup model (NumPy only).

Lesson 13 puts Bahdanau attention inside a seq2seq decoder. This module isolates the
Query / Key / Value view used by Transformers: score = Q Kᵀ / √d, weights = softmax(score),
context = weights V. A mean-pool baseline shows why soft lookup matters as memory grows.
"""
from __future__ import annotations

import numpy as _numpy

from dllab.xp import as_device, as_param, asnumpy, cast_params, host_np, host_rng, np, prefer_cpu_if_small


def _on_small(method):
    def wrapped(self, *args, **kwargs):
        with prefer_cpu_if_small(self.n_params):
            return method(self, *args, **kwargs)
    return wrapped


def _softmax(z, axis: int = -1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def one_hot(tokens, vocab: int, dtype) -> np.ndarray:
    tokens = np.asarray(tokens)
    out = np.zeros(tokens.shape + (vocab,), dtype=dtype)
    np.put_along_axis(out, tokens[..., None], 1.0, axis=-1)
    return out


def scaled_dot_product_attention(Q, K, V):
    """Q (N, Tq, d), K (N, Tk, d), V (N, Tk, dv) → context (N, Tq, dv), weights (N, Tq, Tk)."""
    d = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / (float(d) ** 0.5)
    weights = _softmax(scores, axis=-1)
    context = np.matmul(weights, V)
    return context, weights


def make_lookup_batch(n: int, n_slots: int, n_keys: int, n_values: int, rng) -> tuple:
    """Each sample is n_slots unique (key, value) pairs; query one key, target its value."""
    if n_keys < n_slots:
        raise ValueError("n_keys must be >= n_slots so keys can be unique inside a sample")
    # Host NumPy only: lesson RNGs are RandomState, and CuPy cannot stack those arrays.
    keys = _numpy.stack([rng.choice(n_keys, size=n_slots, replace=False) for _ in range(n)]).astype(_numpy.int64)
    values = _numpy.asarray(rng.randint(0, n_values, size=(n, n_slots)), dtype=_numpy.int64)
    pick = _numpy.asarray(rng.randint(0, n_slots, size=n))
    query = keys[_numpy.arange(n), pick]
    target = values[_numpy.arange(n), pick]
    return keys, values, query, target


class LookupNet:
    """Read a value from a bag of key–value slots: mean pool vs scaled dot-product attention."""

    def __init__(self, n_keys: int, n_values: int, d_model: int = 32, mode: str = "attention",
                 seed: int = 0):
        if mode not in {"mean", "attention"}:
            raise ValueError(f"unknown mode {mode!r}")
        rng = host_rng(seed)
        dtype = np.float64
        self.n_keys = n_keys
        self.n_values = n_values
        self.d_model = d_model
        self.mode = mode
        self.dtype = dtype
        scale = (1.0 / d_model) ** 0.5
        in_dim = n_keys + n_values
        self.We = as_param(rng.randn(in_dim, d_model) * scale, dtype)
        self.be = as_param(host_np.zeros(d_model), dtype)
        self.Wq_in = as_param(rng.randn(n_keys, d_model) * scale, dtype)
        self.bq_in = as_param(host_np.zeros(d_model), dtype)
        if mode == "attention":
            self.Wq = as_param(rng.randn(d_model, d_model) * scale, dtype)
            self.Wk = as_param(rng.randn(d_model, d_model) * scale, dtype)
            self.Wv = as_param(rng.randn(d_model, d_model) * scale, dtype)
            self.Wo = as_param(rng.randn(d_model, n_values) * scale, dtype)
        else:
            self.Wo = as_param(rng.randn(2 * d_model, n_values) * scale, dtype)
        self.bo = as_param(host_np.zeros(n_values), dtype)
        self.loss_: list[float] = []
        self.grad_norm_: list[float] = []
        self._last_weights = None

    def parameters(self):
        yield "emb.We", self, "We"
        yield "emb.be", self, "be"
        yield "emb.Wq_in", self, "Wq_in"
        yield "emb.bq_in", self, "bq_in"
        if self.mode == "attention":
            yield "att.Wq", self, "Wq"
            yield "att.Wk", self, "Wk"
            yield "att.Wv", self, "Wv"
        yield "out.Wo", self, "Wo"
        yield "out.bo", self, "bo"

    @property
    def n_params(self) -> int:
        return int(sum(getattr(obj, p).size for _, obj, p in self.parameters()))

    def _embed_slots(self, keys, values):
        k = one_hot(keys, self.n_keys, self.dtype)
        v = one_hot(values, self.n_values, self.dtype)
        x = np.concatenate([k, v], axis=-1)
        return x @ self.We + self.be, x

    def _embed_query(self, query):
        q = one_hot(query, self.n_keys, self.dtype)
        return q @ self.Wq_in + self.bq_in, q

    @_on_small
    def logits(self, keys, values, query):
        keys, values, query = as_device(keys), as_device(values), as_device(query)
        slots, slot_in = self._embed_slots(keys, values)
        q_emb, q_in = self._embed_query(query)
        if self.mode == "mean":
            pooled = slots.mean(axis=1)
            h = np.concatenate([pooled, q_emb], axis=1)
            logits = h @ self.Wo + self.bo
            self._cache = ("mean", slot_in, q_in, slots, q_emb, pooled, h, None)
            self._last_weights = None
            return logits
        Q = (q_emb @ self.Wq)[:, None, :]
        K = slots @ self.Wk
        V = slots @ self.Wv
        context, weights = scaled_dot_product_attention(Q, K, V)
        h = context[:, 0]
        logits = h @ self.Wo + self.bo
        self._cache = ("attention", slot_in, q_in, slots, q_emb, Q, K, V, weights, context, h)
        self._last_weights = weights
        return logits

    @_on_small
    def loss_and_backward(self, keys, values, query, target):
        target = as_device(target)
        logits = self.logits(keys, values, query)
        probs = _softmax(logits, axis=-1)
        n = len(target)
        loss = float(-np.mean(np.log(probs[np.arange(n), target] + 1e-12)))
        dlog = probs.copy()
        dlog[np.arange(n), target] -= 1.0
        dlog /= n

        mode = self._cache[0]
        if mode == "mean":
            _, slot_in, q_in, slots, q_emb, pooled, h, _ = self._cache
            self.dWo = h.T @ dlog
            self.dbo = dlog.sum(axis=0)
            dh = dlog @ self.Wo.T
            d_pooled, dq_emb = dh[:, : self.d_model], dh[:, self.d_model :]
            d_slots = np.broadcast_to(d_pooled[:, None, :] / slots.shape[1], slots.shape).copy()
            self.dWq_in = q_in.T @ dq_emb
            self.dbq_in = dq_emb.sum(axis=0)
            self.dWe = np.einsum("ntd,nth->dh", slot_in, d_slots)
            self.dbe = d_slots.sum(axis=(0, 1))
            return loss

        _, slot_in, q_in, slots, q_emb, Q, K, V, weights, context, h = self._cache
        self.dWo = h.T @ dlog
        self.dbo = dlog.sum(axis=0)
        dh = dlog @ self.Wo.T
        d_context = dh[:, None, :]
        # context = weights @ V  →  dV = weightsᵀ @ d_context, d_weights = d_context @ Vᵀ
        dV = np.matmul(np.swapaxes(weights, -1, -2), d_context)
        d_weights = np.matmul(d_context, np.swapaxes(V, -1, -2))
        # softmax Jacobian: d_scores = w ⊙ (d_w − sum_k d_w_k w_k)
        sum_dw = np.sum(d_weights * weights, axis=-1, keepdims=True)
        d_scores = weights * (d_weights - sum_dw)
        d_scores = d_scores / (float(Q.shape[-1]) ** 0.5)
        # scores = Q @ Kᵀ
        dQ = np.matmul(d_scores, K)
        dK = np.matmul(np.swapaxes(d_scores, -1, -2), Q)
        # Q = q_emb @ Wq, etc.
        self.dWq = q_emb.T @ dQ[:, 0]
        dq_from_att = dQ[:, 0] @ self.Wq.T
        self.dWk = np.einsum("ntd,nth->dh", slots, dK)
        d_slots = dK @ self.Wk.T
        self.dWv = np.einsum("ntd,nth->dh", slots, dV)
        d_slots = d_slots + dV @ self.Wv.T
        self.dWq_in = q_in.T @ dq_from_att
        self.dbq_in = dq_from_att.sum(axis=0)
        self.dWe = np.einsum("ntd,nth->dh", slot_in, d_slots)
        self.dbe = d_slots.sum(axis=(0, 1))
        return loss

    @_on_small
    def predict(self, keys, values, query):
        return asnumpy(np.argmax(self.logits(keys, values, query), axis=1))

    @_on_small
    def accuracy(self, keys, values, query, target) -> float:
        pred = self.predict(keys, values, query)
        return float(np.mean(pred == _numpy.asarray(target)) * 100)

    @_on_small
    def attention_weights(self, keys, values, query) -> np.ndarray:
        """Softmax weights over slots, shape (N, T). Mean mode returns uniform weights."""
        self.logits(keys, values, query)
        if self._last_weights is None:
            n, t_len = np.asarray(keys).shape
            return _numpy.full((n, t_len), 1.0 / t_len)
        return asnumpy(self._last_weights[:, 0])

    def fit(self, batch_fn, n_steps: int, lr: float = 3e-3, clip: float = 1.0,
            log_every: int = 0, log_fn=None):
        with prefer_cpu_if_small(self.n_params):
            return self._fit_steps(batch_fn, n_steps, lr, clip, log_every, log_fn)

    def _fit_steps(self, batch_fn, n_steps, lr, clip, log_every, log_fn):
        cast_params(self.parameters())
        self.dtype = self.We.dtype
        b1, b2, eps = 0.9, 0.999, 1e-8
        state = {k: (np.zeros_like(getattr(o, p)), np.zeros_like(getattr(o, p))) for k, o, p in self.parameters()}
        self.loss_.clear()
        for step in range(1, n_steps + 1):
            keys, values, query, target = batch_fn(step)
            loss = self.loss_and_backward(keys, values, query, target)
            grads = {k: getattr(o, "d" + p) for k, o, p in self.parameters()}
            norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
            scale = min(1.0, clip / (norm + 1e-12)) if clip else 1.0
            lr_t = lr * 0.5 * (1.0 + float(np.cos(np.pi * (step - 1) / n_steps)))
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


def gradient_check(model: LookupNet, keys, values, query, target, n_checks: int = 6,
                   eps: float = 1e-5, seed: int = 0) -> dict:
    rng = host_rng(seed)
    model.loss_and_backward(keys, values, query, target)
    analytic = {k: getattr(o, "d" + p).copy() for k, o, p in model.parameters()}
    report = {}
    for k, o, p in model.parameters():
        param = getattr(o, p)
        errs = []
        for fi in rng.choice(param.size, size=min(n_checks, param.size), replace=False):
            idx = tuple(int(v) for v in _numpy.unravel_index(int(fi), param.shape))
            orig = param[idx].copy()
            param[idx] = orig + eps
            lp = model.loss_and_backward(keys, values, query, target)
            param[idx] = orig - eps
            lm = model.loss_and_backward(keys, values, query, target)
            param[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[k][idx]
            errs.append(abs(num - ana) / max(1e-6, abs(num) + abs(ana)))
        report[k] = float(max(errs))
    return report
