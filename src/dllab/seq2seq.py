"""Encoder–decoder sequence-to-sequence models (NumPy only).

The encoder is a GRU that reads the source. The decoder is a second GRU that emits the
target one token at a time, trained with teacher forcing. Two ways to pass the source:

- bottleneck: the decoder's initial state is the encoder's last hidden vector
- attention: at every decoder step a context vector is a weighted sum of every encoder state
"""
from __future__ import annotations

import numpy as np

from dllab.rnn import GRUCell


def _softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def one_hot(tokens: np.ndarray, vocab: int) -> np.ndarray:
    out = np.zeros(tokens.shape + (vocab,), dtype=np.float64)
    np.put_along_axis(out, tokens[..., None], 1.0, axis=-1)
    return out


class EncoderDecoder:
    """GRU encoder + GRU decoder. `attention=False` is the fixed-vector seq2seq model."""

    def __init__(self, vocab: int, hidden: int, attention: bool = False, update_bias: float = 3.0,
                 seed: int = 0):
        rng = np.random.RandomState(seed)
        dtype = np.float64
        self.vocab = vocab
        self.hidden = hidden
        self.attention = attention
        dec_in = vocab + hidden if attention else vocab
        self.encoder = GRUCell(vocab, hidden, rng, dtype, update_bias=update_bias)
        self.decoder = GRUCell(dec_in, hidden, rng, dtype, update_bias=update_bias)
        self.Wy = (rng.randn(hidden, vocab) * np.sqrt(1.0 / hidden)).astype(dtype)
        self.by = np.zeros(vocab, dtype=dtype)
        if attention:
            scale = np.sqrt(1.0 / hidden)
            self.Wa = (rng.randn(hidden, hidden) * scale).astype(dtype)
            self.Ua = (rng.randn(hidden, hidden) * scale).astype(dtype)
            self.va = (rng.randn(hidden) * scale).astype(dtype)
        self.loss_: list[float] = []
        self.grad_norm_: list[float] = []

    def parameters(self):
        for name, cell in (("enc", self.encoder), ("dec", self.decoder)):
            for p in cell.param_names:
                yield f"{name}.{p}", cell, p
        yield "out.Wy", self, "Wy"
        yield "out.by", self, "by"
        if self.attention:
            yield "att.Wa", self, "Wa"
            yield "att.Ua", self, "Ua"
            yield "att.va", self, "va"

    @property
    def n_params(self) -> int:
        return int(sum(getattr(obj, p).size for _, obj, p in self.parameters()))

    def _unroll(self, cell: GRUCell, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        n_batch, t_len, _ = X.shape
        hs = np.empty((n_batch, t_len, cell.H), dtype=X.dtype)
        cache = []
        for t in range(t_len):
            z, r, n, u = cell._parts(X[:, t], h)
            h_prev = h
            h = (1.0 - z) * n + z * h
            hs[:, t] = h
            cache.append((h_prev, z, r, n, u))
        cell._cache = (X, cache)
        return hs

    def _encode(self, src: np.ndarray) -> np.ndarray:
        return self.encoder.forward(one_hot(src, self.vocab))

    def _attend(self, h: np.ndarray, enc_hs: np.ndarray):
        pre = enc_hs @ self.Wa + (h @ self.Ua)[:, None, :]
        energy = np.tanh(pre)
        score = energy @ self.va
        alpha = _softmax(score)
        ctx = np.einsum("ns,nsh->nh", alpha, enc_hs)
        return alpha, ctx, energy

    def _forward_decoder(self, dec_in: np.ndarray, enc_hs: np.ndarray) -> np.ndarray:
        """Teacher-forced decoder. Attention, when enabled, reads every encoder state at each step."""
        n_batch, t_len = dec_in.shape
        tokens = one_hot(dec_in, self.vocab)
        h = enc_hs[:, -1]
        hs = np.empty((n_batch, t_len, self.hidden), dtype=np.float64)
        cache = []
        att_cache = []
        if self.attention:
            X = np.empty((n_batch, t_len, self.vocab + self.hidden), dtype=np.float64)
            for t in range(t_len):
                alpha, ctx, energy = self._attend(h, enc_hs)
                x = np.concatenate([tokens[:, t], ctx], axis=1)
                X[:, t] = x
                z, r, n, u = self.decoder._parts(x, h)
                h_prev = h
                h = (1.0 - z) * n + z * h
                hs[:, t] = h
                cache.append((h_prev, z, r, n, u))
                att_cache.append((h_prev, alpha, energy))
        else:
            X = tokens
            hs = self._unroll(self.decoder, X, h)
            self._att_cache = []
            return hs
        self.decoder._cache = (X, cache)
        self._att_cache = att_cache
        return hs

    def logits_teacher(self, src: np.ndarray, dec_in: np.ndarray):
        enc_hs = self._encode(src)
        hs = self._forward_decoder(dec_in, enc_hs)
        self._enc_hs = enc_hs
        return hs @ self.Wy + self.by, hs

    def loss_and_backward(self, src: np.ndarray, dec_in: np.ndarray, target: np.ndarray):
        logits, hs = self.logits_teacher(src, dec_in)
        probs = _softmax(logits)
        n_batch, t_len = target.shape
        idx_n = np.arange(n_batch)[:, None]
        idx_t = np.arange(t_len)[None, :]
        loss = float(-np.mean(np.log(probs[idx_n, idx_t, target] + 1e-12)))
        dlog = probs.copy()
        dlog[idx_n, idx_t, target] -= 1.0
        dlog /= n_batch * t_len
        self.dWy = np.einsum("nth,ntv->hv", hs, dlog)
        self.dby = dlog.sum(axis=(0, 1))
        dhs = dlog @ self.Wy.T
        if self.attention:
            self._backward_attention(dhs)
        else:
            self.decoder.backward(dhs)
            denc = np.zeros_like(self._enc_hs)
            denc[:, -1] = self.decoder.dh0
            self.encoder.backward(denc)
            self._zero_att_grads()
        return loss

    def _zero_att_grads(self):
        if self.attention:
            self.dWa = np.zeros_like(self.Wa)
            self.dUa = np.zeros_like(self.Ua)
            self.dva = np.zeros_like(self.va)

    def _attend_backward(self, dctx, h, enc_hs, alpha, energy):
        dalpha = np.einsum("nh,nsh->ns", dctx, enc_hs)
        denc = alpha[:, :, None] * dctx[:, None, :]
        dscore = alpha * (dalpha - np.sum(dalpha * alpha, axis=1, keepdims=True))
        denergy = dscore[:, :, None] * self.va[None, None, :]
        self.dva += np.einsum("nsh,ns->h", energy, dscore)
        dpre = denergy * (1.0 - energy ** 2)
        self.dWa += np.einsum("nsh,nsk->hk", enc_hs, dpre)
        self.dUa += np.einsum("nh,nsk->hk", h, dpre)
        denc += np.einsum("nsk,hk->nsh", dpre, self.Wa)
        dh = np.einsum("nsk,hk->nh", dpre, self.Ua)
        return dh, denc

    def _backward_attention(self, dhs):
        X, cache = self.decoder._cache
        n_batch, t_len, _ = X.shape
        H = self.decoder.H
        V = self.vocab
        cell = self.decoder
        cell.dWx = np.zeros_like(cell.Wx)
        cell.dWh = np.zeros_like(cell.Wh)
        cell.db = np.zeros_like(cell.b)
        self._zero_att_grads()
        denc = np.zeros_like(self._enc_hs)
        dh_next = np.zeros((n_batch, H), dtype=X.dtype)
        for t in reversed(range(t_len)):
            h_prev, z, r, n, u = cache[t]
            dh = dhs[:, t] + dh_next
            da_z = dh * (h_prev - n) * z * (1.0 - z)
            da_n = dh * (1.0 - z) * (1.0 - n ** 2)
            du = da_n * r
            da_r = (da_n * u) * r * (1.0 - r)
            cell.dWx[:, :H] += X[:, t].T @ da_z
            cell.dWx[:, H : 2 * H] += X[:, t].T @ da_r
            cell.dWx[:, 2 * H :] += X[:, t].T @ da_n
            cell.dWh[:, :H] += h_prev.T @ da_z
            cell.dWh[:, H : 2 * H] += h_prev.T @ da_r
            cell.dWh[:, 2 * H :] += h_prev.T @ du
            cell.db[:H] += da_z.sum(axis=0)
            cell.db[H : 2 * H] += da_r.sum(axis=0)
            cell.db[2 * H :] += du.sum(axis=0)
            dX = da_z @ cell.Wx[:, :H].T + da_r @ cell.Wx[:, H : 2 * H].T + da_n @ cell.Wx[:, 2 * H :].T
            dh_gru = (
                dh * z
                + da_z @ cell.Wh[:, :H].T
                + da_r @ cell.Wh[:, H : 2 * H].T
                + du @ cell.Wh[:, 2 * H :].T
            )
            h_att, alpha, energy = self._att_cache[t]
            dh_att, denc_t = self._attend_backward(dX[:, V:], h_att, self._enc_hs, alpha, energy)
            denc += denc_t
            dh_next = dh_gru + dh_att
        denc[:, -1] += dh_next
        self.encoder.backward(denc)

    def decode(self, src: np.ndarray, sos: int, eos: int):
        """Greedy decode. Returns tokens (N, S+1) and, for the attention model, alphas (N, S+1, S)."""
        enc_hs = self._encode(src)
        n_batch, src_len, _ = enc_hs.shape
        h = enc_hs[:, -1]
        tok = np.full(n_batch, sos, dtype=np.int64)
        outs = np.empty((n_batch, src_len + 1), dtype=np.int64)
        alphas = np.zeros((n_batch, src_len + 1, src_len), dtype=np.float64) if self.attention else None
        for t in range(src_len + 1):
            if self.attention:
                alpha, ctx, _ = self._attend(h, enc_hs)
                alphas[:, t] = alpha
                x = np.concatenate([one_hot(tok, self.vocab), ctx], axis=1)
            else:
                x = one_hot(tok, self.vocab)
            h = self.decoder.step(x, h)
            tok = np.argmax(h @ self.Wy + self.by, axis=1)
            outs[:, t] = tok
        return outs, alphas

    def exact_match(self, src: np.ndarray, target: np.ndarray, sos: int, eos: int) -> float:
        pred, _ = self.decode(src, sos, eos)
        return float(np.mean(np.all(pred == target, axis=1)) * 100)

    def fit(self, batch_fn, n_steps: int, lr: float = 3e-3, clip: float = 1.0,
            log_every: int = 0, log_fn=None):
        b1, b2, eps = 0.9, 0.999, 1e-8
        state = {k: (np.zeros_like(getattr(o, p)), np.zeros_like(getattr(o, p))) for k, o, p in self.parameters()}
        self.loss_.clear()
        for step in range(1, n_steps + 1):
            src, dec_in, target = batch_fn(step)
            loss = self.loss_and_backward(src, dec_in, target)
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


def make_reverse_batch(n: int, length: int, n_sym: int, sos: int, eos: int, rng) -> tuple:
    """Source digits, decoder inputs (SOS + reversed[:-1] + last reversed? ), targets (reversed + EOS)."""
    src = rng.randint(0, n_sym, size=(n, length))
    rev = src[:, ::-1]
    target = np.concatenate([rev, np.full((n, 1), eos, dtype=np.int64)], axis=1)
    dec_in = np.concatenate([np.full((n, 1), sos, dtype=np.int64), rev], axis=1)
    return src, dec_in, target


def gradient_check(model: EncoderDecoder, src, dec_in, target, n_checks: int = 6,
                   eps: float = 1e-5, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    model.loss_and_backward(src, dec_in, target)
    analytic = {k: getattr(o, "d" + p).copy() for k, o, p in model.parameters()}
    report = {}
    for k, o, p in model.parameters():
        param = getattr(o, p)
        errs = []
        for fi in rng.choice(param.size, size=min(n_checks, param.size), replace=False):
            idx = np.unravel_index(int(fi), param.shape)
            orig = param[idx]
            param[idx] = orig + eps
            lp = model.loss_and_backward(src, dec_in, target)
            param[idx] = orig - eps
            lm = model.loss_and_backward(src, dec_in, target)
            param[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = analytic[k][idx]
            errs.append(abs(num - ana) / max(1e-6, abs(num) + abs(ana)))
        report[k] = float(max(errs))
    return report
